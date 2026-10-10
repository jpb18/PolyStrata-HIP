# Viabilidade do port do motor PolyStrata (`strata-poly`) para AMD RX 9060 XT

Auditoria só de leitura, feita sobre o commit `27f9ec9d` (ramo `main`, 2026-10-10). Não foi alterado código,
configuração, dependência nem script. Este ficheiro é o único criado.

**Hardware de destino:** Radeon RX 9060 XT 16 GB (RDNA4, gfx1200), Ryzen 5 5600X (6 núcleos / 12 threads, AVX2,
sem AVX-512), 32 GB de RAM (DDR4), Windows.
**Objetivo:** modelos MoE Qwen 35B-A3B (Qwen3.6-35B-A3B, Ornith 1.5) em Q3/Q4, até 48K de contexto, mantendo a
gestão de especialistas entre CPU e GPU.

Legenda usada ao longo do documento:

- **[CONFIRMADO]**: verificado no código deste repositório (ficheiro e linha indicados) ou em documentação oficial
  da AMD.
- **[DOC]**: afirmado na documentação do repositório (medições de terceiros ou dos mantenedores), não reproduzido
  aqui.
- **[ESTIMATIVA]**: cálculo meu a partir de números do repositório; não medido.
- **[A VALIDAR]**: precisa de experiência no hardware de destino.

---

## 1. Resumo executivo e veredicto

**Veredicto: viável, com risco moderado, e sem bloqueador técnico encontrado no código.** O que falta não é uma
reimplementação: é ligar o motor `strata-poly` à infraestrutura HIP que o repositório já tem para o outro motor
(`strata`), completar meia dúzia de nomes no shim CUDA→HIP, e depois medir. A incerteza principal não está no
código, está na plataforma: HIP no Windows para gfx1200 através de builds *nightly* (TheRock) e a latência de
sincronização CPU↔GPU por camada nesse ambiente, da qual depende diretamente a velocidade de escrita (decode) do
desenho atual.

Pontos que sustentam o veredicto:

1. **Os modelos 35B-A3B só correm no `strata-poly`, e esse motor é hoje só CUDA.** [CONFIRMADO] O alvo CMake
   `strata-poly` só existe com `STRATA_ENABLE_CUDA AND NOT STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906`
   (`CMakeLists.txt:1533`). O setup recusa-o em AMD: `poly_fits()` devolve `not hip and ...` e `poly_verdict()`
   diz "does not run on AMD cards (its engine is a CUDA program)" (`setup.py:4808-4816`). O backend HIP que existe
   e funciona é o do motor `strata`, que corre **Qwen3.8-Flash-Next** (arquitetura `qwen4exp`, 512 especialistas
   por camada), não Qwen3.6-35B-A3B (`qwen35moe`, que só aparece em `src/poly/`).
2. **O código do `strata-poly` é pequeno e já escrito no estilo que o HIP aceita.** [CONFIRMADO] 8 651 linhas em
   `src/poly/`, das quais 1 462 de kernels (`kernels.cu`) e 110 de wrapper MMQ (`mmq.cu`). Os kernels assumem
   warps de 32 (`kernels.cu:18`), que é o wave32 de RDNA; usam apenas `__shfl_*_sync` com máscara completa,
   `__half2float`/`__float2half`, `__popc` e memória partilhada pequena (o maior bloco é 32 KiB,
   `kernels.cu:25,855`, abaixo dos 64 KiB de LDS). Não há `mma.sync`, `cp.async`, `ldmatrix`, `asm` PTX, WMMA nem
   cooperative groups.
3. **As peças pesadas já correm em HIP no motor `strata`.** [CONFIRMADO] Os produtos quantizados de decode
   (`kernels::native_mmvq`, `native_mmvq_slots`, `native_quantize_q8_1`) vêm de `src/kernels/cuda/native_mmvq.cu`,
   que compila em HIP wave32 (só gfx906 é excluído de `native_mmvq_slots`, `native_mmvq.cu:2633`). Os produtos de
   prefill vêm do MMQ do llama.cpp (`ggml-cuda/mmq.cuh`), que o motor `strata` já compila em HIP com
   `GGML_USE_HIP=1` (`CMakeLists.txt:1316-1360`) e que foi validado em gfx1201 [DOC] e gfx1200 (Linux) [DOC]. Os
   especialistas na CPU são `ggml-cpu` puro (`cpu_experts.cpp:19`), independentes da GPU.
4. **Das 63 identificações CUDA distintas usadas em `src/poly/`, só 5 faltam no shim HIP** [CONFIRMADO]:
   `cublasSgemm`, `cublasSgemmStridedBatched`, `cublasGemmStridedBatchedEx`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP` e
   `cudaGraphInstantiateWithFlags` (o shim, `include/strata/hip_compat/cuda_runtime.h` e `cublas_v2.h`, mapeia as
   restantes). Todas têm equivalente direto em HIP/hipBLAS.
5. **O que pode tirar a vantagem sobre o llama.cpp é a plataforma, não o código.** O decode do `strata-poly` espera
   pela GPU **em cada router** (uma ida e volta CPU↔GPU por camada MoE, `engine.cpp:1141`), e a documentação diz
   que a escrita "is bound by the engine's thread, not by the card" [DOC, `docs/BACKENDS.md`]. No Windows com HIP
   a documentação do repositório já registou eventos HIP que não medem bem as cópias e um erro HIP "fantasma"
   deixado pelo hipBLAS [DOC, `docs/AMD_HIP.md` #247, #325, #380]. A latência real de lançamento, de
   `hipEventSynchronize` e de replay de hipGraph no gfx1200/Windows **não está medida em lado nenhum** e decide se
   a vantagem se mantém. Isto tem de ser medido antes de escrever código (Fase 0).

**Resposta à questão principal.** Sim, é realisticamente possível portar o `strata-poly` para a RX 9060 XT
mantendo as otimizações que o distinguem (cache dinâmica de especialistas em VRAM, especialistas em falta
calculados na CPU em paralelo, cópia antecipada no prefill, *windows* de tokens adivinhados com o bloco MTP do
ficheiro, CUDA/HIP graphs). As dependências de CUDA **não** justificam outra abordagem: são finas, quase todas já
abstraídas, e a única biblioteca externa (cuBLAS) tem tradução direta. O que **não** está demonstrado é que, no
Windows, o resultado seja mais rápido do que o llama.cpp com `--n-cpu-moe`; isso só se sabe medindo. A abordagem
de menor risco é: medir primeiro (llama.cpp Vulkan/HIP e microbenchmarks HIP no próprio PC), portar depois em HIP,
de forma opt-in e incremental. Um port para Vulkan do `strata-poly` seria uma reescrita de todos os kernels e não é
recomendado; o llama.cpp com Vulkan é a referência a bater e o plano B.

---

## 2. Mapa da arquitetura relevante

### 2.1 Motores e backends do repositório

| Componente | Onde | Modelos | Backends de GPU no código | Estado em AMD |
| --- | --- | --- | --- | --- |
| `strata` (motor original) | `src/core`, `src/kernels`, `src/prefill`, `src/program`, `include/strata` | Qwen3.8-Flash-Next e variantes (packs `qwen4exp`) | CUDA; HIP wave32 (`STRATA_ENABLE_HIP`, `cmake/hip_backend.cmake`); HIP gfx906 wave64 (`STRATA_HIP_GFX906`) | Funcional. gfx1100/gfx1201 validados pelos mantenedores, gfx1200 por um utilizador **em Linux** [DOC `docs/AMD_HIP.md`] |
| `strata-poly` | `src/poly/`, `include/strata/poly/kernels.hpp` | GLM 5.3 Flash (`glm5-next`), **Qwen3.6-35B-A3B e Ornith 1.5 (`qwen35moe`)**, Gemma 4 26B-A4B | Só CUDA (`CMakeLists.txt:1530-1571`) | **Não compila para AMD** [CONFIRMADO] |
| SYCL (Intel Arc) | `sycl/` | Qwen3.8 | SYCL | Fora de âmbito |
| `strata-vision` | `tools/vision/` | codificador de imagens (llama.cpp) | CUDA ou CPU | Em AMD só CPU, e só Linux [DOC] |
| Servidor e web app | `serve/` | todos | nenhum (processo filho que fala por linhas de texto) | Independente do backend [DOC `docs/BACKENDS.md`, "How a backend works"] |
| Instalador | `setup.py` | todos | escolhe e compila/descarrega o motor | Para AMD instala só `strata` (`setup.py:4808-4816`) |

O servidor fala com o motor por um protocolo de texto (`GEN`, `T`, `DONE`), por isso um `strata-poly` compilado
para HIP é transparente para o servidor e para a web app. [DOC `docs/BACKENDS.md`]

### 2.2 Estrutura do `strata-poly`

```
main.cpp          protocolo serve, amostragem, guesses (n-gram/sufixo), conversas, Windows power throttling
engine.cpp/.hpp   Engine: GGUF mapeado, pesos densos em VRAM, cache de especialistas, CPU experts,
                  prefill (read(), moe_rows(), staging), windows de tokens adivinhados, graphs, estado
qwen35.cpp/.hpp   Qwen35: gated delta net (3 em 4 camadas), atenção com KV fp16 (1 em 4), routing, bloco MTP
glm5next.cpp      Glm5Next (não é alvo aqui, mas compila no mesmo executável)
gemma4.cpp        Gemma4 (idem)
kernels.cu        kernels float: norms, rope, delta net, atenção (decode e prefill), swiglu, soma de experts...
mmq.cu/.hpp       wrapper dos kernels MMQ do llama.cpp (ggml-cuda) para o prefill
cpu_experts.cpp   especialistas na CPU com ggml-cpu, pool de threads próprio
conversations.hpp cache de conversas na RAM (só host)
guesses.hpp       guesses a partir de repetições (só host)
```

Dependências externas ligadas ao alvo (`CMakeLists.txt:1553-1558`): `strata_poly_mmq` (MMQ do llama.cpp),
`strata_kernels` (kernels partilhados com `strata`, inclui `native_mmvq`), `strata_spec` (política de drafts,
só host), `ggml-cpu`, `ggml-base`, cuBLAS, cudart.

### 2.3 Fluxo de um token (decode) no `strata-poly`

[CONFIRMADO, `engine.cpp:1520-1671` e `1137-1236`]

1. `run()` sincroniza o stream principal, chama `follow()` (aplica trocas da cache já chegadas, pede novas), copia
   o embedding para a GPU.
2. Por camada: `layer_head()` (atenção ou delta net até ao router) e `moe_shared()` — gravados como graph e
   repetidos quando `Layer::fixed` (`engine.cpp:1559-1569`).
3. `moe_router()` copia as pontuações do router e a entrada dos especialistas para memória *pinned*; `moe()` faz
   `cudaEventSynchronize(ev_)` (**ida e volta por camada**), faz o routing na CPU, separa especialistas em cache
   (GPU) dos que não estão (CPU).
4. Os da cache são lançados 8 a 8 (`native_mmvq_slots`) **a partir de uma thread do pool da CPU**, enquanto as
   outras threads calculam os especialistas em falta com ggml-cpu (`engine.cpp:1180-1230`).
5. `moe_add()` copia a soma da CPU para a GPU; `layer_tail()` (graph) fecha a camada.
6. Com *windows* de tokens adivinhados e especialistas na CPU, o motor escolhe entre "inteira" e "em duas metades
   desfasadas" (uma thread do pool alimenta a GPU com a outra metade), medindo ambas (`pick_way()`,
   `engine.cpp:1601-1663`).

### 2.4 Fluxo do prefill (`read()`/`moe_rows()`)

[CONFIRMADO, `engine.cpp:1246-1486`, `1748-1888`] As chamadas de um prompt atravessam uma camada umas atrás das
outras; os especialistas fora da cache passam uma vez por camada por um *staging buffer* em VRAM, alimentado por
dois buffers *pinned* de 512 MiB alternados (`runtime.hpp:37`, `engine.cpp:970`) e um stream próprio de cópia
antecipada (`up_`). Os produtos são MMQ do llama.cpp; a atenção do prompt usa `cublasGemmStridedBatchedEx` em
FP16 com acumulação FP32 (`kernels.cu:1406-1421`).

---

## 3. Inventário das dependências de CUDA em `src/poly/` e classificação

Contagem por grep de todos os identificadores CUDA/cuBLAS em `src/poly/*.{cpp,cu,hpp}`. [CONFIRMADO]

Classes:
**A** = portável diretamente com HIP (o shim existente já trata);
**B** = adaptação moderada (pequena alteração de shim, build ou um `#ifdef`);
**C** = reimplementação específica para AMD (só para recuperar desempenho, não para funcionar);
**D** = potencial bloqueador técnico.

### 3.1 Runtime, memória, streams, eventos, graphs

| Dependência | Uso no poly | Shim HIP existente | Classe | Notas |
| --- | --- | --- | --- | --- |
| `cudaMalloc`/`cudaFree`/`cudaMemset(Async)` | arenas de pesos, cache de especialistas, scratch (`engine.cpp:241-249`, `522-523`) | sim (`cuda_runtime.h:45,82,104-105`) | A | |
| `cudaMallocHost` / `cudaFreeHost` | todos os buffers *pinned* (`engine.cpp:251-256`): 2×512 MiB de staging, até 512 MiB de resíduo (`engine.cpp:1795`), 16 buffers de troca, router, logits | sim, via `hipHostMalloc` (`cuda_runtime.h:83,127-129`) | A/B | Volume *pinned* total ~1,1 a 1,7 GiB [ESTIMATIVA]. A doc diz que grandes alocações *pinned* podem falhar em ROCm [DOC `AMD_HIP.md`, "Build"]; no Windows existe um teto de 8 GiB para o arena do outro motor [DOC]. **[A VALIDAR]** no Windows. |
| `cudaMemcpy` síncrono de memória mapeada (mmap do GGUF) para VRAM | enchimento da cache: 3 cópias por especialista (`engine.cpp:542-547`) | sim | B | Funciona; velocidade de cópias *pageable* a partir de um ficheiro mapeado no Windows HIP desconhecida (afeta só o tempo de carga). Em Linux já houve bloqueios com cópias *pageable* grandes (`GPU_PINNED_MIN_XFER_SIZE`, #920) [DOC]. |
| `cudaMemcpyAsync` H2D/D2H com *pinned* | router, CPU experts, staging, trocas (`engine.cpp:1130-1131`, `1240`, `1266-1271`, `649`) | sim | A | |
| `cudaMemGetInfo` | dimensionar a cache (`engine.cpp:469`, `484`) | sim, e no Windows é redirecionado para `strata::hip_compat::mem_get_info` que desconta o orçamento WDDM (`cuda_runtime.h:84-95`) | B | A função está em `src/core/device.cu:168-190`; o alvo `strata-poly` tem de a ligar. Sem isto, a cache enche a placa para lá do que o Windows dá e a velocidade cai (41→30 tok/s numa RX 6800) [DOC #380]. |
| Streams (`cudaStreamCreateWithFlags(NonBlocking)`), 4 streams: principal, cópia antecipada, resíduo, trocas (`engine.cpp:564,781,977,988`) | sim | A | |
| Eventos (`Record`, `Synchronize`, `Query`, `WaitEvent`, `DisableTiming`) | sincronização router↔CPU, staging, trocas | sim | A | O poly **não** usa `cudaEventElapsedTime` (mede com `steady_clock`), por isso o problema dos eventos HIP no Windows (#380) não o afeta. [CONFIRMADO] |
| Stream capture + graphs (`BeginCapture(ThreadLocal)`, `EndCapture`, `GraphInstantiateWithFlags`, `GraphLaunch`, `GraphExecDestroy`) | um graph por camada × nº de tokens × tamanho da atenção (`engine.cpp:1497-1517`, `1543-1569`, `1703-1708`) | parcial: falta o nome `cudaGraphInstantiateWithFlags` (o shim tem `cudaGraphInstantiate` com flags, `cuda_runtime.h:136-138`) | B | Uma linha no shim. O motor `strata` já usa capture/graphs em HIP (`src/core/graph.cpp`). Captura pode ocorrer numa thread do pool (`engine.cpp:1636-1645`); ThreadLocal é suportado em HIP. Desempenho do replay de hipGraph no Windows **[A VALIDAR]**. |
| `cudaGetLastError` após lançamentos (`mmq.cu:22-25`) | verificação de erro do MMQ | sim | B | No Windows o hipBLAS pode deixar `hipErrorInvalidValue` pendente após GEMMs FP16/BF16 bem sucedidos; o motor `strata` absorve-o (`src/prefill/gemm.cu:88-101`). O poly faz GEMMs FP16 (`kernels.cu:1127`, `1406`, `1416`) e logo a seguir o MMQ chama `cudaGetLastError` → lançaria uma exceção falsa. É preciso o mesmo `absorb` em `blas_ck`. |

### 3.2 Kernels próprios (`src/poly/kernels.cu`)

| Dependência | Onde | Classe | Notas |
| --- | --- | --- | --- |
| Warp de 32 lanes, `__shfl_xor_sync`/`__shfl_down_sync`/`__shfl_sync` com máscara `0xffffffff` | `kernels.cu:18,30,48,54,538,750,824,969` | A | O shim implementa as variantes `_sync` sobre wave32 e aborta se a máscara não for completa (`intrinsics.hpp:304-348`). Todas as utilizações do poly usam máscara completa. [CONFIRMADO] |
| Blocos até 1 024 threads | `kernels.cu:88`, `955-975`, `attn_value_kernel` 32×32 (`1360-1363`) | A | RDNA suporta 1 024 work-items por workgroup. |
| Memória partilhada estática (máx. 32 KiB em `attn_value_kernel`, `kernels.cu:855`) e dinâmica pequena (`idx_select_kernel`, só GLM) | | A | Dentro dos 64 KiB de LDS de RDNA. |
| `half`, `__half2float`, `__float2half`, `__ushort_as_half` | KV fp16, atenção | A | `hip/hip_fp16.h` via shim (`hip_compat/cuda_fp16.h`). |
| `__popc` | `kernels.cu:988` | A | Existe em HIP. |
| Nenhum PTX, `mma.sync`, `cp.async`, `ldmatrix`, WMMA, cooperative groups, `__launch_bounds__` | — | — | [CONFIRMADO] por grep. |
| Kernels de decode da atenção (`attend`, `kernels.cu:1342`) e da delta net (`gdn_step`, `conv4_silu`, `kernels.cu:1443-1462`) | | A (funcional) / C (desempenho) | Escritos e afinados para NVIDIA (p. ex. "sum over values in eight shares", "a warp a column"). Corretos em wave32; o desempenho em RDNA4 **[A VALIDAR]**. |

### 3.3 Bibliotecas externas

| Dependência | Onde | Classe | Notas |
| --- | --- | --- | --- |
| `cublasCreate`/`SetStream`/`Destroy` | `kernels.cu:1112-1137` | A | Shim → hipBLAS. |
| `cublasSgemm`, `cublasSgemmStridedBatched` (FP32) | `gemm_f32` (`kernels.cu:1151-1162`), GLM `dsa_attend`/`idx_score_batch` | B | Faltam no shim; `hipblasSgemm`/`hipblasSgemmStridedBatched` são 1:1. Para Qwen3.6 só `gemm_f32` importa (projeções BF16/F32 do prompt, `engine.cpp:1096-1103`). |
| `cublasGemmStridedBatchedEx` FP16→FP32 com `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | atenção do prompt (`kernels.cu:1406-1421`), *warm-up* (`kernels.cu:1127`) | B (funcional) / C (desempenho) | `hipblasGemmStridedBatchedEx` existe; `TENSOR_OP` → `HIPBLAS_GEMM_DEFAULT`. A AMD não garante kernels rápidos para FP16-in/FP32-out em todas as arquiteturas (em gfx1030 foram 6,6× mais lentos, #835 [DOC]). Em gfx1200 **[A VALIDAR]**. Se for lento, um kernel de atenção WMMA (há precedente para gfx12 em `qsa_prompt_attn`, `STRATA_HIP_WMMA` [DOC]) seria classe C. |
| MMQ do llama.cpp (`ggml-cuda/mmq.cuh`, `quantize.cu`), revisão fixada `3cf0325` (`CMakeLists.txt:1183`) | `mmq.cu` | B | O `strata` já compila este código em HIP. O poly precisa de mais instâncias de tipo do que o `strata` compila em HIP: `strata` tem `q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0` (+`q4_k q5_k q5_1 q6_k` com `STRATA_MMQ_KQUANTS`) (`CMakeLists.txt:1250`, `1338-1342`); o poly lista também `q4_0 q4_1 q5_0 q2_k q3_k iq1_s` (`CMakeLists.txt:1539-1541`, `mmq.cu:16-18`). Os UD-Q3_K_M e UD-Q4_K_M da Unsloth usam Q3_K/Q4_K/Q5_K/Q6_K nos especialistas [DOC `BACKENDS.md`]. Compilar mais instâncias é trivial; que o caminho MMQ de RDNA4 desta revisão seja rápido (WMMA vs dp4a) **[A VALIDAR]**. |
| `ggml_backend_cuda_context`, `ggml_cuda_info()` | `mmq.cu:29,51` | A | Fornecidos por `src/prefill/ggml_cuda_host.cu`, que já tem ramos `GGML_USE_HIP` (`ggml_cuda_host.cu:23,74`). |
| `ggml-cpu` (especialistas na CPU) | `cpu_experts.cpp` | A | Independente da GPU. Com AVX2 sem AVX-512, como o 5600X, é o mesmo caminho que o portátil de teste (i7-14700HX, AVX2). |
| `strata_kernels` (`native_mmvq`, `native_mmvq_slots`, `native_quantize_q8_1`, `native_q8_1_bytes`) | decode | A | Já compilados em HIP; `native_mmvq_slots` só exclui gfx906 (`native_mmvq.cu:2632-2637`). Os caminhos CUDA-only de `native_mmvq.cu` (`#if !defined(__HIPCC__)`, linhas 31, 1440, 2676) não são usados pelo poly. |

### 3.4 Sistema operativo e build

| Dependência | Onde | Classe | Notas |
| --- | --- | --- | --- |
| Condição do alvo `strata-poly` | `CMakeLists.txt:1533` | B | Hoje exclui HIP. Um opt-in novo (ex.: `STRATA_POLY_HIP`) que crie `strata_poly_mmq` como fonte HIP (como o ramo `elseif(STRATA_PREFILL_MMQ)` de `CMakeLists.txt:1316-1360`) e ligue `strata_hip_runtime` + hipBLAS. `hip_backend.cmake:107-113` já marca todos os `src/*.cu` como HIP, incluindo `src/poly/*.cu`. |
| `#include <cuda_runtime.h>` em `.cpp` (`engine.cpp:15`, `runtime.hpp:8`) | | A | O shim é força-incluído em todas as fontes C++ e HIP (`hip_backend.cmake:93-105`). |
| Linux-only (`sched_*`, `/proc/self/numa_maps`) | `engine.cpp:29-32`, `53-69`, `79-150` | A | Já protegido por `#ifdef __linux__`. |
| Windows (`PROCESS_POWER_THROTTLING`) | `main.cpp:33`, `775-777` | A | Já existe (o poly corre em Windows com CUDA [DOC]). |
| Compilador no Windows: clang do ROCm (TheRock) para host e device, MSVC só para linker/SDK | `tools/hip/build_windows.bat` | B | O script atual só constrói `strata` e `strata-device` (`build_windows.bat`, passo 3). Precisa de incluir o alvo novo e o pacote (`package_windows.py`). |
| Instalador | `setup.py:4808-4816` e `poly_setup()` (`setup.py:4927`) | B | Fora do âmbito técnico do motor; última fase. |

### 3.5 Resumo da classificação

| Classe | Itens |
| --- | --- |
| A — direto | ~90% das chamadas: memória, streams, eventos, cópias, kernels próprios, `native_mmvq*`, ggml-cpu, MMQ (código), cuBLAS create/stream |
| B — adaptação moderada | 5 nomes em falta no shim; alvo CMake opt-in; instâncias MMQ extra; ligação a `device.cu` (WDDM); absorção do erro pendente do hipBLAS; build/pacote Windows |
| C — específico AMD, só para desempenho | eventualmente: atenção do prompt em WMMA se o hipBLAS FP16→FP32 for lento em gfx1200; re-afinação dos kernels de decode da atenção e da delta net; redução de idas e voltas CPU↔GPU por camada se a latência HIP/Windows for alta |
| D — bloqueador | **nenhum encontrado no código.** Riscos de plataforma (secção 5) podem tornar-se bloqueadores *de desempenho*, não de funcionalidade. |

---

## 4. Componentes HIP existentes reutilizáveis

| Componente | Ficheiro | Para quê no port |
| --- | --- | --- |
| Shim CUDA runtime → HIP (wave32) | `include/strata/hip_compat/cuda_runtime.h` | quase todo o runtime do poly, sem tocar no código do poly |
| Shim cuBLAS → hipBLAS (com fallback rocBLAS para hipBLAS < 1.0) | `include/strata/hip_compat/cublas_v2.h` | `GemmEx` e tipos; acrescentar os 3 nomes em falta |
| Intrínsecos wave32 (`dp4a` → `sudot4` em gfx1200, `__byte_perm` → `v_perm_b32`, SWAR de bytes, `shfl_*_sync`) | `include/strata/hip_compat/intrinsics.hpp` | kernels quantizados partilhados (gfx1200 está explicitamente coberto, linha 226) |
| Configuração CMake HIP, verificação de arquitetura, força-inclusão do shim | `cmake/hip_backend.cmake` | gfx1200 já está em `_strata_hip_community` (linha 14) |
| `native_mmvq`, `native_mmvq_slots`, `native_quantize_q8_1` | `src/kernels/cuda/native_mmvq.cu` | todos os produtos quantizados de decode do poly |
| MMQ do llama.cpp compilado em HIP + host shim | `CMakeLists.txt:1316-1360`, `src/prefill/ggml_cuda_host.cu` | prefill do poly |
| `mem_get_info` com orçamento WDDM e retry do primeiro `hipMemGetInfo` | `src/core/device.cu:163-190` | dimensionar a cache no Windows |
| Absorção do erro pendente do hipBLAS no Windows | `src/prefill/gemm.cu:88-101` | copiar o padrão para `blas_ck` em `kernels.cu` |
| Verificação de arquitetura do binário à partida | `src/core/device.cu`, `generate.cpp:2132-2145` | mensagem clara em vez de "invalid device function" |
| Build Windows com TheRock + pacote com DLLs do ROCm, `amdhip64_7.dll` ao lado do exe | `tools/hip/build_windows.bat`, `tools/hip/package_windows.py` | distribuir `strata-poly.exe` |
| Kernel de atenção WMMA para gfx12 (`STRATA_HIP_WMMA`) | `qsa_prompt_attn` [DOC] | precedente/inspiração se a atenção do prompt via hipBLAS for lenta |
| Ferramenta de paridade de especialistas | `src/poly/expert_parity.cpp` (`poly_expert_parity MODEL.gguf`) | teste de aceitação dos produtos GPU vs CPU vs referência FP64, sem precisar do motor completo |

O que **não** é reutilizável diretamente: o caminho de especialistas do `strata` (packs `qwen4exp`, arena de
especialistas, `doorbell`, `router_top10`). São de outro desenho; o poly tem o seu.

---

## 5. Bloqueadores e riscos técnicos

Ordenados por impacto no objetivo ("manter a vantagem sobre o llama.cpp").

### R1 — Latência por camada CPU↔GPU no Windows com HIP (alto, [A VALIDAR])

O decode faz, por camada MoE: cópia D2H das pontuações do router → `hipEventSynchronize` → routing na CPU →
lançamentos dos especialistas em cache (de uma thread do pool) → cópia H2D da soma da CPU (`engine.cpp:1127-1243`).
Qwen3.6 tem 40 camadas MoE (41 com o bloco MTP: "10,496 experts" = 41 × 256, [DOC `BACKENDS.md`]). Com CUDA a
documentação mede ~35 lançamentos pequenos por camada e diz que a thread espera pela GPU em cada router [DOC].
Se, no Windows/WDDM com HIP, cada sincronização custar dezenas de µs a mais e cada lançamento fora de graph for
mais caro, isso multiplica-se por 40 por token. Nenhum documento do repositório mede isto para HIP/Windows.
**É o principal risco ao desempenho e o primeiro a medir (Fase 0).** Mitigação possível (classe C): routing na
GPU, como o motor `strata` faz com `router_top10` em HIP [DOC], para não precisar da ida e volta quando todos os
especialistas da camada estão em cache.

### R2 — Pilha HIP no Windows para gfx1200 (médio-alto)

- **Oficial [CONFIRMADO, AMD]:** a página de requisitos do HIP SDK para Windows (ROCm 7.2) lista a RX 9060 XT
  (gfx1200) com runtime, HIP SDK e debugger suportados, **só em Windows 11**
  ([system requirements 7.2](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.2/reference/system-requirements.html)).
  A mesma documentação diz que o HIP SDK para Windows **não suporta a linguagem HIP do CMake**, e que o hipBLASLt
  no Windows é suportado em gfx1101
  ([component support 7.2](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.2/conceptual/component-support.html)).
- **O que o repositório usa no Windows:** não o HIP SDK oficial, mas as *wheels nightly* do TheRock
  (`10.2.0a20260930`, índice `whl-next`), com o clang do ROCm a compilar host e device e a linguagem HIP do CMake
  (`build_windows.bat`). O TheRock descreve-se como estando em "early preview". Isto é **suporte experimental**,
  não oficial. O pacote inclui hipBLAS/rocBLAS/hipBLASLt para as arquiteturas escolhidas [DOC `AMD_HIP.md`,
  "Windows"].
- **O que está validado no Windows com AMD:** uma RX 9070 XT (gfx1201) compilada no próprio PC, Coder IQ1_M,
  ctest 42/46 [DOC]. A documentação diz que **o zip pronto ainda não correu um modelo numa placa discreta**
  [DOC `AMD_HIP.md`, "What is validated (0.1.34)"]. A RX 9060 XT só foi validada **em Linux**, com ROCm 7.2
  [DOC `AMD_HIP.md`, "Community-validated cards"]. **A combinação gfx1200 + Windows não tem nenhum relatório,
  mesmo para o motor `strata`.**
- **Problemas Windows já conhecidos** [DOC]: DLL `amdhip64_7.dll` do driver em System32 a sobrepor-se à do pacote
  (#468, #461); `hipMemGetInfo` que não conta a memória do ambiente de trabalho (#380); erro pendente do hipBLAS
  após GEMMs FP16/BF16 (#247); `hipHostGetDevicePointer` que devolve o ponteiro do host e cópias D2D para memória
  mapeada que não chegam (o poly não usa isto); eventos HIP que medem velocidades impossíveis (o poly não os usa
  para medir).
- **Alternativa de menor risco de plataforma:** Linux nativo (dual boot) com ROCm 7.x, onde gfx1200 já correu o
  motor `strata` com ctest 32/32 [DOC]. WSL2 não foi avaliado aqui **[A VALIDAR]**.

### R3 — Desempenho do hipBLAS FP16→FP32 em gfx1200 (médio, [A VALIDAR])

A atenção do prompt (`attend_batch`) foi reescrita para usar *tensor cores* via cuBLAS e daí veio grande parte do
ganho em prompts longos (Qwen3.6 a ler 30 000 tokens: 3 835 → 5 114 tok/s na RTX 5090) [DOC `BACKENDS.md`,
"What was tuned"]. A ordem dos blocos (múltiplo de 16 linhas) foi afinada para o cuBLAS. Se o rocBLAS/hipBLASLt
não tiver um kernel rápido para esta forma em gfx1200, o prefill com contexto longo (48K) é o que mais sofre. Há
precedente de lentidão de 6,6× em gfx1030 para exatamente este tipo de GEMM [DOC #835].

### R4 — Largura de banda da RAM e núcleos do 5600X (médio, inerente ao hardware)

Os especialistas fora da VRAM são lidos da RAM à velocidade da memória [DOC: "The CPU's 11 ms are memory"]. O
5600X tem DDR4 em dois canais (até ~51 GB/s teóricos com DDR4-3200) e 6 núcleos; o portátil de referência
(i7-14700HX, DDR5) usou 13 threads. Isto afeta igualmente o llama.cpp com `--n-cpu-moe`, por isso não muda a
comparação, mas baixa o teto absoluto. [ESTIMATIVA] Um especialista Q3_K de Qwen3.6 tem ~1,4 MB
(14,6 GB / 10 496); com 8 especialistas por camada e 40 camadas, cada 10% de falhas na cache são ~45 MB por token
lidos da RAM, ~1 ms por token a ~40 GB/s efetivos.

### R5 — Orçamento de VRAM no Windows com 16 GB e 48K (baixo-médio)

[ESTIMATIVA, a partir de `setup.py:378-387`] Qwen3.6 UD-Q3_K_M ocupa ~4,8 GB de VRAM antes de qualquer
especialista, a 32K, medido numa RTX 5090 com CUDA. Cada token de contexto custa 26 624 B (`row_bytes`), por isso
+16K para chegar a 48K são ~0,4 GiB. Com reserva de 1 GiB (o motor usa 1/16 da placa, entre 512 e 1 024 MiB,
`engine.cpp:485-486`) e o ambiente de trabalho do Windows na mesma placa, sobram ~8 a 9 GiB para especialistas:
~55-62% dos 14,6 GB de especialistas do UD-Q3_K_M (no portátil de 8 GB o perfil levou a taxa de acerto a 49-55%
com só ~19% dos especialistas em VRAM [DOC]). UD-Q2_K_XL (10,8 GB) caberia quase todo; UD-Q4_K_M (20,1 GB),
~45%. O consumo próprio do runtime HIP, do rocBLAS/hipBLAS e dos graphs no Windows é desconhecido
**[A VALIDAR]**. RAM: UD-Q4_K_M a 48K com 32 GB está no limite, como no portátil (31,3 GiB de 31,8 [DOC]).

### R6 — Ordem das somas e paridade (baixo)

O repositório não promete bits idênticos entre backends [DOC `AMD_HIP.md`]; a verificação do poly é feita contra
o llama.cpp com tolerância de probabilidades [DOC `BACKENDS.md`, "How each backend was checked"]. O port pode
usar exatamente essa metodologia como critério de aceitação.

### R7 — Não determinismo raro em HIP (baixo)

"Rarely (about 1 start in 10) a HIP run's greedy output differs from another start's ... not yet explained"
[DOC `AMD_HIP.md`, RDNA4]. Pode aparecer também no poly; os testes de aceitação devem repetir arranques.

---

## 6. Análise por área pedida

### 6.1 Expert caching e offloading CPU/GPU

[CONFIRMADO] Tudo isto é lógica de host + cópias + `native_mmvq_slots`, portanto HIP-portável:

- Enchimento por "uso por byte" a partir do perfil (`fill_expert_cache`, `engine.cpp:481-558`).
- Trocas durante a resposta: uma thread (`follow_loop`, `engine.cpp:629-661`) copia do ficheiro mapeado para um
  buffer *pinned* e para o *slot* num stream próprio; `follow()` decide pares entrada/saída (`engine.cpp:663-760`).
- Especialistas em falta calculados na CPU por ggml-cpu enquanto uma thread do pool lança os em cache
  (`engine.cpp:1180-1230`).
- Redimensionamento da cache quando a VRAM livre desce (`mind_vram`, `engine.cpp:608-614`).

Pontos de atenção específicos de AMD/Windows: o dimensionamento depende de `cudaMemGetInfo` corrigido pelo WDDM
(classe B); os lançamentos a partir de outra thread para o mesmo stream são suportados em HIP, mas o custo por
lançamento é a incógnita R1.

### 6.2 Quantizações

[CONFIRMADO] Decode: `native_mmvq_supported` aceita Q4_0, Q5_0, Q5_1, Q8_0, Q2_K–Q6_K, IQ4_NL, IQ4_XS, IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S/IQ1_M (tipos 2,6,7,8,10–14,16–18,20–23,29,42; `native_mmvq.cu:2576-2580`);
tudo isto compila em HIP. Prefill: MMQ do llama.cpp, com as instâncias listadas na secção 3.3. Pesos densos F32 e
BF16 por `gemv_*`/`gemm_f32`. KV em FP16. Os ficheiros alvo (Unsloth UD-Q3_K_M, UD-Q2_K_XL, UD-Q4_K_M; Ornith
APEX) estão todos dentro destes tipos [DOC `BACKENDS.md`]. Q4_1 não tem caminho de decode quantizado (não é usado
por estes ficheiros).

### 6.3 Speculative decoding

[CONFIRMADO] Três fontes: o bloco MTP dentro do GGUF (`draft()`, `engine.cpp:2018-2060`), guesses por repetição
(`guesses.hpp`, só host) e a política adaptativa (`src/spec/draft_policy.cpp`, só host). A verificação usa
*windows* de até 8 tokens com graphs por tamanho, *snapshots* do estado recorrente (`kSnapPlaces`,
`runtime.hpp`) e a escolha "inteira vs metades" medida em tempo real (`pick_way`/`took_way`). Nada disto tem
código específico de NVIDIA. O ganho medido no portátil de 8 GB foi de 55-56 → 75-80 tok/s para Qwen3.6 UD-Q3_K_M
[DOC]. A política mede ela própria se adivinhar compensa, por isso adapta-se a uma plataforma com outra latência.

### 6.4 KV cache e contexto longo

[CONFIRMADO] KV em FP16, sem quantização nem paginação, alocado inteiro para `max_context` (`qwen35.cpp:122-135`).
Só 1 em 4 camadas tem atenção; o resto é delta net com estado fixo. A 48K o custo é ~1,2 GiB [ESTIMATIVA, via
`row_bytes`], comportável numa placa de 16 GB. O scratch de pontuações é limitado por `kScoreFloats`
(`qwen35.cpp:138`, `159`). O decode depois de prompts longos dependeu de duas otimizações de atenção feitas no
portátil ("An answer's attention scores read a key side by side", "A window's tokens read the keys and values
once") [DOC]; são kernels próprios (classe A funcional, C para afinação). A atenção do prompt depende do hipBLAS
(R3). O motor `strata` em AMD tem KV int8 e paginação; o poly não, e não precisa para 48K.

### 6.5 Transferências PCIe e sincronização

[CONFIRMADO] Prefill: staging com dois buffers *pinned* de 512 MiB alternados e um stream de cópia antecipada que
corre enquanto a camada faz a atenção (`stage`, `engine.cpp:1246-1274`; decisão de reter um "round" medida em
tempo real). Decode: cópias pequenas por camada (router e soma da CPU). [A VALIDAR] largura de banda H2D real no
5600X: a placa é PCIe 5.0 x16, mas o 5600X dá PCIe 4.0 (ou 3.0 em chipset A520). A documentação do repositório
avisa que no Windows os eventos HIP não medem bem cópias; o poly mede com relógio do host, por isso não é afetado.

### 6.6 Kernels específicos NVIDIA

[CONFIRMADO] No poly, nenhum: sem PTX, sem `mma`, sem WMMA, sem `cp.async`, sem *Programmatic Dependent
Launch*. A única dependência "de tensor cores" é indireta, via `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (que em hipBLAS é
só `DEFAULT`). Os caminhos CUDA-only de `native_mmvq.cu` (interleaved, `q8_1_il`) não são chamados pelo poly. O
MMQ do llama.cpp tem os seus próprios caminhos por arquitetura, já exercitados em RDNA4 pelo motor `strata`.

---

## 7. Plano de implementação incremental

Princípios, tirados do `AGENTS.md`: cada fase é opt-in; o build CUDA por omissão fica byte-idêntico; cada fase
compila todos os backends que toca; nenhum *default* muda sem medição.

Complexidade relativa: **S** (pequena), **M** (média), **L** (grande), **XL** (muito grande). Sem prazos.

### Fase 0 — Medir antes de portar (sem alterar o repositório) · S

Objetivo: decidir com números se o port pode bater o llama.cpp nesta máquina, e se o Windows serve.

1. Correr `engine\strata-device.exe --list-devices` e `--selftest` do zip HIP oficial na RX 9060 XT
   (procedimento em `docs/AMD_HIP.md`, "Reporting a Windows AMD run"). Isto valida a pilha TheRock em gfx1200 no
   Windows sem tocar no poly.
2. Linha de base: llama.cpp (Vulkan e, se possível, HIP) com o mesmo GGUF Qwen3.6 UD-Q3_K_M, ver secção 9.
3. Microbenchmarks HIP no próprio PC (um programa à parte, fora do repositório): latência de lançamento de um
   kernel vazio; ida e volta "cópia D2H de 1 KiB + `hipEventSynchronize`"; replay de um hipGraph com ~35 kernels
   pequenos; `hipHostMalloc` de 1,5 GiB; H2D *pinned* e a partir de ficheiro mapeado.

Critério para avançar: (a) o selftest passa; (b) a ida e volta × 40 camadas cabe numa fração pequena do tempo por
token do llama.cpp (por exemplo, abaixo de 25%). Se (b) falhar no Windows, repetir em Linux antes de desistir.

### Fase 1 — Compilar `strata-poly` para HIP (gfx1200), sem mudar comportamento · M

- Opção CMake nova e desligada por omissão (ex.: `STRATA_POLY_HIP`) que, com `STRATA_ENABLE_HIP`, cria
  `strata_poly_mmq` como fonte HIP (copiando o padrão de `CMakeLists.txt:1316-1360`, com as instâncias
  `q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs`) e
  liga `strata-poly` a `strata_hip_runtime`, hipBLAS e à unidade que define `hip_compat::mem_get_info`.
- Shim: `cublasSgemm`, `cublasSgemmStridedBatched`, `cublasGemmStridedBatchedEx`,
  `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, `cudaGraphInstantiateWithFlags`.
- `blas_ck`/`mmq ck`: absorver o `hipErrorInvalidValue` pendente só em `__HIPCC__ && _WIN32`, como em
  `src/prefill/gemm.cu:88-101`.
- `build_windows.bat`: alvo opcional `strata-poly`.

Aceitação: compila em Windows (TheRock) e Linux (ROCm 7) para gfx1200; o build CUDA por omissão produz o mesmo
binário de antes (comparar hashes); `ctest` do HIP sem regressões; `poly_workers_test` e `poly_guesses_test`
passam.

### Fase 2 — Paridade numérica dos produtos · S

- Compilar e correr `poly_expert_parity Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` em gfx1200: `native_mmvq`,
  `native_mmvq_slots`, MMQ e CPU contra a referência FP64 de ggml, para cada tipo e forma do ficheiro.

Aceitação: a ferramenta passa (é o seu próprio critério de tolerância); `native_mmvq_slots` dá os mesmos bits que
um especialista por lançamento (já é verificado pela ferramenta).

### Fase 3 — Inferência básica correta · M

Configuração mínima: `graphs=false`, `window=1`, sem draft, `expert_swaps=0`, contexto 8K, prompt em chamadas
curtas. Primeiro com `gpu_experts=false` (todos os especialistas na CPU: isola a GPU às camadas densas), depois
com a cache estática.

Aceitação (metodologia de `docs/BACKENDS.md`, "How each backend was checked"): greedy, 64 tokens após 5 prompts de
25 a 8 000 tokens, ids comparados com o llama.cpp no mesmo ficheiro e as 5 probabilidades mais altas por passo.
Objetivo: divergências apenas em quase-empates (as duas melhores dentro de 0,1), diferença máxima de
probabilidade da mesma ordem que a da tabela do repositório (≤ 0,15 para Qwen3.6 UD-Q3_K_M). 5 arranques para
apanhar R7.

### Fase 4 — Gestão de especialistas completa no Windows · M

Ligar `expert_swaps=16`, perfil de especialistas, reserva por tamanho da placa, `mind_vram`. Confirmar a linha de
orçamento WDDM no log e que o ambiente de trabalho não força a placa a mover memória para a RAM.

Aceitação: com o mesmo perfil copiado e a cache como carregada, duas execuções dão os mesmos tokens (critério do
repositório); taxa de acerto da cache e `swapped_experts` reportados; sem quedas de velocidade ao longo de 10
respostas seguidas; VRAM livre nunca abaixo de 1/4 da reserva.

### Fase 5 — Graphs, windows e speculative decoding · M

Ligar `graphs=true`, `window` por omissão, bloco MTP, guesses por repetição, modo "metades".

Aceitação: com e sem windows, respostas de 300 tokens com os mesmos tokens (critério do repositório); com todos os
especialistas na CPU, idem; taxa de acerto das guesses comparável à do portátil (75-90% [DOC]); a política
desliga o draft quando não compensa. Medir decode com e sem graphs para quantificar R1.

### Fase 6 — Prefill rápido · M (L se for preciso C)

Ligar MMQ do prompt, staging com cópia antecipada, `read()` de várias chamadas, atenção do prompt via hipBLAS.

Aceitação: a verificação interna "read de várias chamadas = chamadas uma a uma, byte a byte" (descrita em
`BACKENDS.md`) passa em 2 000, 8 000 e 30 000 tokens; velocidade de leitura medida em 2K/8K/30K/45K.

### Fase 7 — Contexto de 48K · S

Contexto 49 152, prompt de ~45 000 tokens (`--prompt-tokens`), com UD-Q3_K_M e UD-Q4_K_M.

Aceitação: arranca e responde; tokens após o prompt iguais aos do llama.cpp até quase-empate; pico de VRAM e de
RAM registado; sem paginação para a RAM do sistema (WDDM) nem leitura do SSD no meio da resposta.

### Fase 8 (opcional) — Otimizações específicas de RDNA4 · L a XL

Só se as Fases 5-7 mostrarem o gargalo: (a) routing na GPU para camadas com todos os especialistas em cache
(R1); (b) atenção do prompt em WMMA gfx12 (R3), aproveitando o padrão do `qsa_prompt_attn` WMMA do motor
`strata`; (c) re-afinação dos kernels de decode da atenção e da delta net para RDNA4.

Aceitação: cada uma A/B contra a anterior no mesmo dia, 3 execuções, mesmos tokens ou divergência só em
quase-empates; só fica se for mais rápida.

### Fase 9 — Instalador · M

`poly_fits`/`poly_verdict` deixam de recusar HIP quando o motor existe; `build_windows.bat`/`package_windows.py`
incluem `strata-poly.exe`; `--check` reporta. Aceitação: `python tools/test_setup_<...>.py` relevantes passam sem
GPU; instalação limpa num PC Windows com a RX 9060 XT.

### Dependências entre fases

```
F0 ──► F1 ──► F2 ──► F3 ──► F4 ──► F5 ──► F7
                       └──────────► F6 ──┘
                                         └──► F8 (opcional) ──► F9
```

F4/F5 e F6 são independentes entre si depois de F3.

---

## 8. Critérios de aceitação transversais

- Build CUDA por omissão inalterado (hash do binário antes/depois), como exige `AGENTS.md`.
- Build HIP de Linux e de Windows compilados para cada mudança que toque `.cu`.
- Nenhuma alteração de *default* sem números do mesmo dia e o hardware em que foram medidos.
- Testes existentes: `ctest` HIP, `poly_workers_test`, `poly_guesses_test`, `glm_conversations_test`.
- Paridade: `poly_expert_parity` + comparação de 64 tokens com o llama.cpp.
- Robustez: 5 arranques por configuração; 10 pedidos seguidos via servidor; conversa retomada de um checkpoint
  igual a uma conversa nova (testes descritos em `BACKENDS.md`).
- Relatório final no formato de `docs/COMMUNITY_BENCHMARKS.md`, dizendo o que não foi testado.

---

## 9. Estratégia de benchmark contra llama.cpp

### 9.1 Configurações a comparar (mesma máquina, mesmo dia, mesmo driver)

| Rótulo | Motor | Backend | Colocação dos especialistas |
| --- | --- | --- | --- |
| L-VK | llama.cpp (revisão registada) | Vulkan | `-ngl 99` + `--n-cpu-moe N` (ou `-ot exps=CPU` parcial), N mínimo que cabe em 16 GB ao contexto testado |
| L-HIP | llama.cpp | HIP/ROCm | idem |
| P-HIP | `strata-poly` portado | HIP | cache dinâmica + CPU, perfil "quente" (depois de uma execução) |
| P-HIP-frio | idem | HIP | sem perfil (primeira execução) |
| P-HIP-nospec | idem | HIP | `window=1`, sem draft (isola o ganho do speculative) |

Mesmo GGUF (Qwen3.6-35B-A3B UD-Q3_K_M; repetir com UD-Q4_K_M e UD-Q2_K_XL), mesmo contexto alocado (32K e 48K),
greedy, mesmo número de threads da CPU (testar 5, 6 e 11 no 5600X e fixar a melhor por motor).

### 9.2 Métricas, separadas

| Métrica | Como medir | Comprimentos |
| --- | --- | --- |
| **Prefill** (tok/s) | llama.cpp: `llama-bench -p 512,2048,8192,30000` e `-d` para profundidade; poly: `python polystrata.py benchmark --model qwen3.6 --prompt-tokens N --json` | 512, 2K, 8K, 30K, 45K |
| **Decode** (tok/s) | llama.cpp: `llama-bench -n 128 -d 0,2048,8192,30000,45000`; poly: o mesmo benchmark (velocidade após cada prompt) | idem, separado em "conversa curta" e "após prompt longo" |
| **Tempo até ao 1.º token** | benchmark do poly; `llama-server` com o mesmo prompt | 2K, 30K, 45K |
| **VRAM** | pico via log do motor (`STRATA_MEMORY_DEBUG=1` no poly) e Gestor de Tarefas/`hipInfo`; registar a VRAM ocupada pelo ambiente de trabalho antes de começar | por contexto |
| **RAM** | pico do *working set* do processo + cache de ficheiros do sistema | por quantização |
| **Taxa de acerto da cache / guesses** | contadores do poly | — |

### 9.3 Disciplina

- 3 execuções por ponto, mediana; descartar a primeira leitura depois de carregar o modelo (a doc mostra que é
  fria [DOC `AMD_HIP.md`, "Tuning table"]).
- A/B intercalado (A, B, A, B), não em blocos.
- Registar driver Adrenalin, versão do ROCm (TheRock), revisão do llama.cpp, BIOS (Resizable BAR, PCIe 4.0 vs
  3.0), memória (DDR4 a que velocidade), modo de energia do Windows.
- Para decode, distinguir "prompt curto" de "após 30K/45K", porque o comportamento da atenção e da cache muda.
- Reportar também os tokens produzidos (as respostas divergem; uma resposta mais curta muda a média).

### 9.4 O que conta como "manter a vantagem"

A documentação do repositório só compara com o llama.cpp quando **o modelo inteiro cabe na placa** (RTX 5090:
~180 tok/s do llama.cpp vs 323 do poly com guesses [DOC]) e nota que o llama.cpp "cannot do that on a card the
model does not fit". **Não há no repositório nenhuma medição do llama.cpp com `--n-cpu-moe` numa placa onde o
modelo não cabe.** Por isso a vantagem no cenário da RX 9060 XT (16 GB, UD-Q3_K_M de 17,1 GB) é uma hipótese, e é
exatamente o que L-VK vs P-HIP mede. Proposta de critério: decode P-HIP ≥ 1,3× L-VK em conversa curta e ≥ 1,0×
após 45K, com prefill ≥ 0,8× L-VK.

---

## 10. HIP vs Vulkan para este hardware

| Critério | Port HIP do `strata-poly` | Port Vulkan do `strata-poly` | llama.cpp Vulkan (sem port) |
| --- | --- | --- | --- |
| Código a escrever | shim + build + correções Windows; kernels reutilizados | reescrever `kernels.cu` (1 462 linhas), `native_mmvq*` (2 857 linhas partilhadas), MMQ (ou reaproveitar shaders do ggml-vulkan), substituir cuBLAS, streams/eventos/graphs por command buffers e semáforos | nenhum |
| Infraestrutura existente no repo | grande (secção 4) | nenhuma | — |
| Maturidade no Windows para gfx1200 | HIP SDK oficial lista gfx1200; o repo usa TheRock nightly (preview); zero relatórios gfx1200+Windows | Vulkan é o caminho de produção do driver Adrenalin | maduro, muito usado em RDNA4 no Windows [conhecimento geral, A VALIDAR na máquina] |
| Matrizes RDNA4 (WMMA) | acessíveis por intrínsecos HIP (precedente no repo) e pelo rocBLAS | via `VK_KHR_cooperative_matrix` | já usa cooperative matrix onde existe [A VALIDAR] |
| Custo de submissão por camada (R1) | incógnita no Windows | command buffers pré-gravados são baratos; mas a ida e volta CPU↔GPU por camada continua a existir no desenho do poly | não tem ida e volta por camada no caminho com `--n-cpu-moe` (o grafo do ggml separa por backend) |
| Gestão dinâmica de especialistas | preservada | teria de ser reescrita sobre Vulkan | **não tem**: colocação estática |
| Risco total | moderado | alto | baixo (não tem as otimizações) |

Conclusão: **Vulkan não tem vantagem técnica suficiente para justificar reescrever o motor.** A sua vantagem
real, maturidade no Windows, aproveita-se usando o llama.cpp Vulkan como linha de base e plano B. Se a Fase 0
mostrar que o HIP no Windows tem latências que matam o desenho do poly, a alternativa seguinte é o port HIP em
**Linux**, não um port Vulkan.

---

## 11. Afirmações do repositório que divergem do código ou entre si

1. `AGENTS.md`: "`strata-poly` ... on Linux with one NVIDIA card". O código e outros documentos dizem que também
   corre em **Windows** com NVIDIA: `main.cpp:33,775` (código Windows), `setup.py` (`"windows": True` para
   `qwen36`, `ornith`, `gemma4`), `docs/BACKENDS.md` (secção do portátil Windows). O `AGENTS.md` está
   desatualizado nesse ponto.
2. `docs/AMD_HIP.md`, primeira frase: "PolyStrata runs on AMD Radeon cards through its HIP backend, the same engine
   as on NVIDIA". Verdadeiro para o motor `strata` (Qwen3.8), **falso para o `strata-poly`** e portanto para
   Qwen3.6/Ornith/Gemma 4/GLM. O próprio código e o `setup.py:4813-4816` dizem o contrário; o `README.md`
   (linhas 234-258) diz corretamente "Not yet: AMD cards" para esses modelos. Quem ler só o `AMD_HIP.md` pode
   concluir mal.
3. `docs/AMD_HIP.md` lista a RX 9060 XT como validada, e a secção Windows diz "set up like an NVIDIA one". A
   validação da RX 9060 XT foi **em Linux** (ROCm 7.2, Ryzen 9 7950X), e a própria página diz que o zip de
   Windows ainda não correu um modelo numa placa discreta. Não há nenhum relatório de gfx1200 no Windows.
4. `README.md:197` diz "Windows 10 / 11"; a matriz oficial da AMD para RDNA4 no HIP SDK lista só **Windows 11**.
   Com TheRock pode funcionar em Windows 10, mas não é suportado oficialmente.
5. Os números de VRAM "antes de qualquer especialista" usados pelo `setup.py` (`vram_beside_gb`) foram medidos numa
   RTX 5090 com CUDA (`setup.py:362`); não se aplicam diretamente a HIP no Windows.

---

## 12. Confirmado vs. por validar

**Confirmado (código ou documentação oficial):**

- `strata-poly` não compila para HIP; é a condição de CMake que o impede, não um uso de API sem equivalente.
- O código CUDA do poly usa só APIs com tradução HIP direta; 5 nomes faltam no shim.
- Os kernels do poly são wave32, sem PTX/WMMA/`mma`; memória partilhada ≤ 32 KiB.
- Os kernels quantizados partilhados e o MMQ do llama.cpp já compilam em HIP e cobrem gfx1200.
- Os especialistas na CPU são ggml-cpu, independentes da GPU.
- O decode tem uma sincronização CPU↔GPU por camada MoE.
- A AMD lista oficialmente gfx1200 no HIP SDK para Windows 11 (runtime, SDK, debugger).
- O HIP SDK oficial não suporta a linguagem HIP do CMake; o repositório contorna isso com TheRock.

**Por validar experimentalmente:**

- Pilha TheRock em gfx1200 no Windows (nenhum relatório).
- Latência de lançamento, de `hipEventSynchronize` e de replay de hipGraph no Windows (R1).
- Desempenho do `hipblasGemmStridedBatchedEx` FP16→FP32 em gfx1200 (R3).
- Desempenho do MMQ do llama.cpp em gfx1200 com as instâncias K-quant.
- Alocação de ~1,5 GiB *pinned* e cópias a partir de ficheiro mapeado no Windows HIP.
- VRAM consumida pelo runtime HIP, bibliotecas e graphs no Windows.
- Desempenho real do llama.cpp Vulkan com `--n-cpu-moe` nesta máquina, que é a referência a bater.
- Se o port mantém a vantagem do poly sobre o llama.cpp num cenário em que o modelo não cabe na VRAM (o
  repositório não tem essa comparação nem com NVIDIA).

---

## 13. Recomendação

1. **Começar pela Fase 0, sem escrever código no repositório.** Correr o selftest do zip HIP, medir o llama.cpp
   Vulkan com `--n-cpu-moe` e os microbenchmarks de latência HIP no próprio PC. É barato e decide tudo o resto.
2. **Se a Fase 0 passar: portar em HIP, opt-in, pelas Fases 1-7.** É a abordagem de menor risco (reutiliza o
   shim, os kernels quantizados, o MMQ, o build Windows e as correções Windows já existentes) e a de melhor
   potencial (mantém intactos a cache dinâmica, o cálculo na CPU em paralelo, o staging do prefill e o
   speculative decoding). A Fase 1 a 3 é um volume de trabalho pequeno a médio.
3. **Se a latência HIP no Windows for o problema:** fazer o mesmo port em Linux (onde o gfx1200 já correu o motor
   `strata`), e só depois considerar a Fase 8a (routing na GPU) para reduzir as idas e voltas.
4. **Não portar para Vulkan.** Usar o llama.cpp Vulkan como referência e como solução imediata enquanto o port não
   provar ser melhor.
5. Se o objetivo pudesse ser "um modelo Qwen MoE grande com o motor otimizado já hoje", o motor `strata` com
   Qwen3.8-Flash-Next Coder IQ1_M já corre na RX 9060 XT (27 tok/s de decode e 540 tok/s de prompt a 2,4K, em
   Linux, contra 20 e 450 do llama.cpp HIP no mesmo PC [DOC]). Não é um 35B-A3B, e no Windows também está por
   validar nessa placa.

---

## Fontes externas

- [AMD — System requirements (Windows), ROCm 7.2](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.2/reference/system-requirements.html)
- [AMD — ROCm components supported in HIP SDK, ROCm 7.2](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.2/conceptual/component-support.html)
- [AMD HIP SDK for Windows](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
- [AMD — Install hipBLASLt (menciona as nightlies do TheRock)](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/install/install.html)
- [OneTrainer #1381 — relato de utilizador: RDNA4 no Windows só via nightlies do TheRock (não oficial)](https://github.com/Nerogar/OneTrainer/issues/1381)
- [llama.cpp commit 3cf0325 (revisão fixada pelo repositório)](https://github.com/ggml-org/llama.cpp/commit/3cf03257f219afbe7334045ff7c6a06ac68c627d)
