# Port HIP do `strata-poly` — implementação

Ramo `hip-poly`, commits sobre `27f9ec9d` (`git log 27f9ec9d..hip-poly`). Não foi feito push nem aberto PR.
Referência técnica: [AMD_PORT_FEASIBILITY.md](AMD_PORT_FEASIBILITY.md).

**Onde foi testado:** este PC, que é o hardware de destino. RX 9060 XT 16 GB (gfx1200, driver Adrenalin
32.0.31041.1004), Ryzen 5 5600X (6 núcleos / 12 threads), 31,9 GB de RAM, Windows 11 Pro. ROCm 10.2.0a20260930
(wheels nightly TheRock, a mesma versão do motor HIP que já usas em `C:\strata`). O PC estava em uso normal:
Firefox, ChatGPT e outros programas ocupavam ~13 GB de RAM e parte da VRAM.

**Modelo testado:** `unsloth/Qwen3.5-35B-A3B-GGUF`, ficheiro `Qwen3.5-35B-A3B-Q3_K_M.gguf` (16 356 375 168 bytes,
SHA-256 `5607c8fc…ab1b2d`, igual ao do Hugging Face). Neste ficheiro os especialistas estão em IQ3_XXS (gate/up)
e IQ4_XS (down), os pesos densos em Q8_0 e a saída em Q6_K. O ficheiro **não** tem bloco MTP.

---

## 1. Estado

**O port é funcional e foi validado nesta RX 9060 XT.** O `strata-poly` compila com HIP, inicializa a placa,
carrega o GGUF, faz prefill e decode sem erros e divide os especialistas entre GPU e CPU (82–99% dos especialistas
usados vêm da VRAM, conforme o prompt; o resto é calculado na CPU). Gera as mesmas tokens que o llama.cpp no mesmo ficheiro, dentro
do critério do repositório (`docs/BACKENDS.md`, "How each backend was checked"). Responde de forma coerente
através do servidor (API OpenAI com e sem streaming, e API Anthropic).

Validado nesta máquina, com este modelo:

| Requisito | Estado | Como foi verificado |
| --- | --- | --- |
| Compilar com HIP | ✅ | `tools\hip\build_windows.bat poly` (gfx1200), sem erros |
| Inicializar a RX 9060 XT | ✅ | `strata-device --list-devices`, `hip_poly_probe` |
| Carregar o GGUF Qwen3.5-35B-A3B | ✅ | 9–26 s; 7 134 dos 10 240 especialistas em VRAM (9,1 GiB) |
| Prefill e decode sem erros | ✅ | prompts de 22 a 44 419 tokens, contexto 49 152 |
| Validação numérica | ✅ | `poly_expert_parity` e `poly_llama_parity.py` (secção 4) |
| Especialistas em CPU e GPU sem falhas de sincronização | ✅ | modo misto, só CPU, trocas ativas, janelas em metades |
| Expert caching (cache seguindo a resposta) | ✅ | 380–1 350 trocas por resposta, tokens iguais |
| Sobreposição GPU/CPU | ✅ | caminho das janelas "em metades" (thread do pool lança a GPU) |
| Speculative decoding | ⚠️ parcial | palpites por repetição sim; o bloco MTP **não** foi testado (o ficheiro não tem) |
| Graph capture/replay | ✅ | com e sem `--no-graphs`, os mesmos tokens |
| Contexto longo (48K) | ✅ | 44 419 tokens com `--max-context 49152`, pelo servidor |

---

## 2. Alterações efetuadas

### 2.1 Integração CMake (commit `9d177ceb`)

- **Nova opção `STRATA_POLY_HIP`** (OFF por omissão, exige `STRATA_ENABLE_HIP=ON`, `STRATA_NATIVE_EXPERTS=ON` e
  hipBLAS ≥ 1.0). Com ela, o bloco do `strata-poly` em `CMakeLists.txt` tem um ramo HIP:
  - as fontes `.cu` do poly já eram marcadas como HIP por `cmake/hip_backend.cmake`;
  - o motor recebe o shim CUDA→HIP como o `strata`;
  - a biblioteca `strata_poly_mmq` (MMQ do llama.cpp) é compilada como HIP com `GGML_USE_HIP=1` e **sem** o shim,
    tal como a `strata_mmq` do motor `strata`;
  - liga a `ggml-base`, `hip::host` e `roc::hipblas`.
- O ramo CUDA executa exatamente os mesmos comandos de antes; a restrição foi substituída por uma escolha
  CUDA/HIP, não removida.
- Novos alvos, só em HIP: `hip_poly_probe` e `poly_kernel_bench`.

### 2.2 Compatibilidade CUDA/HIP

- **Shim** (`include/strata/hip_compat/`): `cublasSgemm`, `cublasSgemmStridedBatched`,
  `cublasGemmStridedBatchedEx`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (→ `HIPBLAS_GEMM_DEFAULT`; a AMD não tem um
  algoritmo "tensor op" separado, a biblioteca escolhe os kernels de matriz) e `cudaGraphInstantiateWithFlags`.
  O hipBLAS ≥ 1.0 tem a mesma ordem de argumentos que o cuBLAS; o CMake recusa a versão antiga 0.x, onde estas
  funções precisariam de embrulhos rocBLAS que não pude testar.
- **Erro residual do hipBLAS no Windows** (#247, #325), em `src/poly/kernels.cu`:
  - depois de um produto bem sucedido, um `hipErrorInvalidValue` pendente é limpo; qualquer outro erro é lançado;
  - **para não esconder erros reais**, antes de cada produto o último erro é verificado (`blas_before`): um erro
    de um lançamento anterior é reportado como tal, em vez de ser engolido;
  - só em `STRATA_USE_HIP && _WIN32`.
- **`hipErrorNotReady`:** verifiquei se uma consulta de trabalho por terminar deixa `hipErrorNotReady` como último
  erro. Não deixa (`hip_poly_probe`), por isso não foi preciso alterar nada.

### 2.3 Particularidades Windows/HIP medidas e corrigidas (commits `7a316e2a`, `473a6476`, `f862e0f1`)

Cada correção nasceu de uma medição neste PC e tem um fallback:

| Problema medido | Medição | Correção | Fallback |
| --- | --- | --- | --- |
| Uma cópia pequena (`hipMemcpyAsync`) entre dois kernels custa a vez do copy engine | kernel + D2H 1 KiB: **~100 µs**; a mesma cópia feita por um kernel: **3–4 µs**; logits 0,95 MiB: 211 vs 90 µs; 8 MiB: 734 vs 653 µs | As cópias do stream do token (pontuações do router, entrada dos especialistas, soma da CPU, embeddings, posições, logits) passam a ser feitas por um kernel (`copy_words`) que lê/escreve a memória pinned diretamente; os streams laterais (staging, `read()`, trocas da cache) mantêm o copy engine | `STRATA_POLY_COPY_KERNELS=0`. No Linux está desligado por omissão (`=1` liga), porque não foi medido |
| Kernel de 8 especialistas por lançamento com linhas curtas | 8 × IQ4_XS down 512→2048: **92 µs (46 GB/s)**; só 16 das 128 threads de cada bloco trabalhavam | Em HIP, linhas cujas threads cabem em meio warp têm meio warp cada, 8 linhas por bloco, com **as mesmas somas pela mesma ordem**: **16 µs (279 GB/s)**; 20 de 20 formatos/formas dão os mesmos bits que um lançamento por matriz | o kernel geral (formatos/formas que não cabem em meio warp) |
| Orçamento WDDM de VRAM muda com o uso | budget de 15 416 MiB até ~10 GiB ocupados, **13 937 MiB** a partir daí; o `hipMalloc` aceita até 19,9 GiB (excesso vai para a RAM). A cache era dimensionada pelo valor maior, perdia a reserva, e `mind_vram()` esvaziava-a e enchia-a de novo **antes de cada pedido (4 s cada)** | Depois de alocar a cache, volta-se a medir e a cache encolhe o que falta à reserva, até a reserva existir (máx. 3 voltas). Só Windows HIP | `STRATA_WDDM_BUDGET=0` (cálculo sem orçamento) |
| Primeiro produto float do hipBLAS | **6 375 ms** no primeiro SGEMM de um processo | `blas_create()` faz um SGEMM pequeno ao arrancar, ao lado do produto de halves que já fazia: o primeiro SGEMM real passa a **0,5 ms** | — (custo passa para o tempo de carga) |

O "deferred loading" do HIP (`HIP_ENABLE_DEFERRED_LOADING=0`) **não** é alternativa: com ele desligado, o runtime
falha a carregar símbolos do rocSOLVER.

### 2.4 Diagnóstico e ferramentas

- `STRATA_POLY_PROFILE=1` (só HIP): tempo de cada parte de uma chamada de um token, camada a camada.
- `strata-poly --tokens @FICHEIRO` (`src/poly/main.cpp`): ids de um prompt longo a partir de um ficheiro. A linha
  de comandos do Windows limita-se a 32 767 caracteres, ~4 000 tokens. É a única alteração que muda o binário CUDA;
  é opt-in.
- `src/poly/hip_probe.cu` (`hip_poly_probe`, `hip_poly_probe vram`) e `src/poly/kernel_bench.cu`
  (`poly_kernel_bench`): medições sem modelo.
- `tools/hip/build_windows.bat poly [tests]`, `tools/hip/poly_windows.bat`, `tools/hip/poly_config.py`,
  `tools/hip/poly_llama_parity.py`, `tools/hip/poly_serve_timing.py` (e `tools/hip/_rocm_env.py`, que põe o ROCm
  no PATH do motor que estes dois scripts arrancam).
- `docs/AMD_HIP.md`: secção "strata-poly on AMD".

---

## 3. Ficheiros modificados e criados

| Ficheiro | Alteração |
| --- | --- |
| `CMakeLists.txt` | opção `STRATA_POLY_HIP`, ramo HIP do `strata-poly` e da `strata_poly_mmq`, alvos de diagnóstico |
| `include/strata/hip_compat/cublas_v2.h` | 4 nomes cuBLAS |
| `include/strata/hip_compat/cuda_runtime.h` | `cudaGraphInstantiateWithFlags` |
| `include/strata/poly/kernels.hpp` | `copy_words` (só HIP) |
| `src/poly/kernels.cu` | `copy_words`; erro residual do hipBLAS; aquecimento do SGEMM (só HIP) |
| `src/poly/engine.cpp` | `POLY_COPY` nas cópias do stream do token; reserva da cache no Windows; perfilador (só HIP) |
| `src/kernels/cuda/native_mmvq.cu` | kernel de meio warp para linhas curtas em `native_mmvq_slots` (só HIP) |
| `src/poly/main.cpp` | `--tokens @FICHEIRO` |
| `src/poly/hip_probe.cu`, `src/poly/kernel_bench.cu` | novos (diagnóstico) |
| `tools/hip/build_windows.bat` | modo `poly` (o modo por omissão não mudou) |
| `tools/hip/poly_windows.bat`, `poly_config.py`, `poly_llama_parity.py`, `poly_serve_timing.py`, `_rocm_env.py` | novos |
| `docs/AMD_HIP.md` | secção nova e nota na abertura |

**Caminho CUDA:**
- Todas as alterações ao motor estão dentro de `#if defined(STRATA_USE_HIP)` (ou `__HIPCC__`), ou são macros que
  no CUDA se expandem no mesmo texto de antes (`POLY_COPY` → `ck(cudaMemcpyAsync(...))`, `STRATA_BLAS_BEFORE` →
  `((void) 0)`).
- A exceção é `--tokens @FICHEIRO`.
- **Não compilei o caminho CUDA** (este PC não tem placa NVIDIA nem CUDA Toolkit). O `AGENTS.md` pede que se
  compile cada backend tocado antes de revisão; isso fica por fazer numa máquina com CUDA.

---

## 4. Testes executados e resultados

### 4.1 Compilação e testes sem modelo

| Teste | Resultado |
| --- | --- |
| Build `poly` (strata-poly, parity, testes, probe, bench) | OK, sem erros |
| Build completo do diretório (inclui o motor `strata` HIP e ~60 programas de teste) | OK |
| `ctest` completo nesta GPU (73 testes) | **71 passam** (5 dos quais saltados por falta de hardware ou tabela). Falham 2, já documentados em `docs/AMD_HIP.md` como esperados no Windows: `hip_handoff` (cópia para memória mapeada no Windows) e `ple_parity` (falta o ficheiro Q2_0) |
| Testes do poly e das conversas (`poly_workers_test`, `poly_guesses_test`, 7 testes de conversas) | 9/9 |
| `tools/test_setup_*.py` que leem a documentação (8) e `test_polystrata_names.py` | OK |

O motor `strata` (Qwen3.8) compilado desta árvore **não** foi corrido de ponta a ponta. O modelo Coder em
"resident" precisa de ~26 GB de RAM e o PC tinha ~13 GB ocupados por outros programas. O código que lhe toca
(macros novas do shim, que não usa, e um ramo de `launch_slots`, que só o poly chama) está coberto pelos testes de
paridade partilhados, que passaram.

### 4.2 Paridade numérica dos kernels

- **`poly_expert_parity` no GGUF real:**
  - erro relativo vs FP64: GPU 0,012–0,016, CPU 0,029–0,039;
  - dois especialistas num lançamento dão os mesmos bits que um;
  - todas as matrizes do ficheiro (Q8_0, Q6_K, IQ3_XXS, IQ4_XS) com vetor e com lote (MMQ) a ~0,011.
- **`poly_kernel_bench`:** `native_mmvq_slots` contra um lançamento por matriz, bit a bit, em 10 formatos
  (Q4_0, Q5_0, Q8_0, IQ4_NL, Q2_K–Q6_K, IQ4_XS) × n_in 256 e 512: **20/20 iguais**.

### 4.3 Paridade com o llama.cpp

Método de `docs/BACKENDS.md`: greedy, 64 tokens, as 5 probabilidades mais altas da referência em cada passo.
Referência: o llama.cpp HIP (build Unsloth b11160) que já estava neste PC, `--cpu-moe`, mesmo GGUF, mesmos ids.

| Configuração do strata-poly | short (22) | code (870) | doc (3 069) | doc (11 481) | Diferença máxima |
| --- | --- | --- | --- | --- | --- |
| `--window 1 --no-graphs` | 19/64 * | 64/64 | 64/64 | 64/64 | 0,095 |
| `--window 1` (graphs) | 19/64 * | 64/64 | 64/64 | 64/64 | 0,095 |
| idem, depois de todas as correções | 64/64 | 64/64 | 64/64 | 64/64 | 0,141 |
| todos os especialistas na CPU | 64/64 | 64/64 | 64/64 | 64/64 | 0,107 |
| copy engine (`STRATA_POLY_COPY_KERNELS=0`) | 19/64 * | 64/64 | 64/64 | 64/64 | 0,076 |
| omissão (janelas, palpites, graphs, trocas), só ids | 19/64 * | 64/64 | 64/64 | 64/64 | — |

\* Parte num quase empate: o llama.cpp tem 0,535 / 0,465 e o strata-poly 0,51 / 0,49 entre os mesmos dois tokens.

A diferença máxima sobe para 0,141 depois da correção da reserva (menos especialistas em VRAM, logo mais na CPU,
cujos produtos diferem nos últimos bits). Fica dentro do que o repositório reporta para o Qwen3.6 (0,08–0,15).

### 4.4 Servidor

Pela API OpenAI (normal e streaming) e pela API Anthropic:
- uma explicação correta em português de Portugal;
- uma função Python correta;
- uma resposta Anthropic com raciocínio.

---

## 5. Desempenho medido

Uma execução de cada, neste PC, com outros programas abertos. **Não são benchmarks controlados**; servem para
orientar.

### 5.1 Decode: evolução

`strata-poly --tokens`, prompt curto (22 tokens), `--window 1`:

| Estado | tok/s |
| --- | ---: |
| Primeira compilação funcional, 12 threads | 25–27,5 |
| 6 threads | 32–34 |
| + cópias por kernel e kernel de meio warp | 47–49 (64 tokens) |
| Final | 50,6 (128 tokens) |

O perfilador mostra onde ficou o tempo de um token (128 tokens, com as sincronizações do perfilador):
- ~0,31–0,34 ms/camada até ao router e especialista partilhado; perto do custo dos seus kernels, ~0,21 ms de pesos
  mais ~0,05 ms de sincronização;
- ~0,17–0,26 ms/camada nos especialistas roteados;
- 1,5 ms de saída e logits (a matriz de saída Q6_K tem 417 MB).

### 5.2 Comparação com o llama.cpp HIP

Mesmo GGUF, mesmos prompts, contexto 16 384, 6 threads, greedy, 128 tokens:

| | Decode, prompt curto | Decode depois de 11,5K | Prefill 11 481 tokens |
| --- | ---: | ---: | ---: |
| llama.cpp HIP b11160, melhor `--n-cpu-moe` (8; testei 6, 8, 10, 14, 20, 26) | 23,4–24,7 | 23,5–23,7 | 752–847 |
| strata-poly HIP, `--window 1` | 54,1 / 67,3 ** | 55,1 / 61,5 ** | 1 525 |
| strata-poly HIP, omissão (palpites) | 52,4 / 70,0 ** | 59,6 / 65,4 ** | 1 427 |

\*\* 1.ª / 2.ª passagem no mesmo processo: na 2.ª a cache de especialistas já seguiu a resposta anterior.

**Esta comparação não é com o llama.cpp Vulkan.** Não havia um build Vulkan neste PC; os comandos estão na
secção 8.

### 5.3 Pelo servidor, até 48K

`python polystrata.py benchmark --prompt-tokens 2000,8000,24000,34000`, contexto 49 152, 256 tokens de resposta:

| Prompt | Tokens | TTFT | Prefill | Decode | Especialistas da VRAM | Palpites certos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 21 | 0,25 s | 84 | 58,7 | 93,6% | 44,7% |
| long-2000 | 2 743 | 2,49 s | 1 105 | 62,7 | 95,1% | 47,9% |
| long-8000 | 10 615 | 6,13 s | 1 401 | 60,5 | 97,3% | 45,0% |
| long-24000 | 31 352 | 17,8 s | 1 190 | 52,7 | 98,3% | 45,5% |
| long-34000 | 44 419 | 14,2 s | 966 | 50,7 | 98,7% | 56,9% |

Os prompts longos partilham o início do texto, por isso parte de cada um já estava lida; o prefill conta só o que
foi lido. RAM da máquina no pico: 28,9–29,7 GiB (máquina inteira). A coluna de VRAM do benchmark vem vazia no
Windows AMD (não há leitura de sensores lá).

### 5.4 Microbenchmarks (`hip_poly_probe`)

| Medição | Valor |
| --- | --- |
| Lançamento de um kernel (custo no host) | 0,67 µs |
| Lançamento de um kernel, até ter corrido (seguidos) | 1,16 µs |
| Ida e volta (kernel + 1 KiB D2H + espera de evento) | mediana 140 µs (111–436) |
| 35 kernels + espera | 109 µs diretos, 113 µs como graph |
| Memória pinned | 3 × 512 MiB em 279 ms |
| H2D a partir de memória pinned | 14,2 GB/s |

---

## 6. Componentes ainda não validados

- **Bloco MTP / draft model** (`--draft-model`): o ficheiro testado não o tem. Os palpites por repetição foram
  testados.
- **Quantizações K-quant nos especialistas** (Q3_K, Q4_K, Q5_K, Q6_K) **num modelo real**, e portanto o **Q4**
  pedido: o "Q3_K_M" da Unsloth para o Qwen3.5 usa IQ3_XXS/IQ4_XS. Os kernels K-quant só foram comparados entre si,
  bit a bit (secção 4.2), não contra uma referência. Um Q4_K_M (22 GB) não cabia no disco (restam ~21 GB) nem com
  folga na RAM com os outros programas abertos.
- **Qwen3.6, Ornith 1.5, Gemma 4, GLM 5.3 Flash** com HIP: compilam no mesmo executável, mas não foram corridos.
- **Imagens** (`strata-vision`), **várias GPUs**, **Linux com HIP**, outras placas AMD.
- **O build CUDA** depois destas alterações (sem toolkit nem placa NVIDIA aqui).
- **Instalação pelo `setup.py`:** continua a recusar o `strata-poly` em AMD. Usa-se `tools/hip/poly_config.py`.
- **Uso prolongado:** horas de servidor, muitos pedidos e conversas estacionadas não foram testados.

## 7. Limitações e problemas possíveis

1. **ROCm no Windows por nightlies do TheRock** (pré-lançamento). Uma atualização do driver ou das wheels pode
   mudar o comportamento: o atraso do copy engine e o orçamento WDDM foram medidos nesta versão.
2. **Orçamento de VRAM do Windows:**
   - com o processo a usar mais de ~10 GiB, o Windows dá-lhe ~13,9 GiB dos 16,3; a cache de especialistas fica
     com ~9,1 GiB;
   - com `STRATA_WDDM_BUDGET=0` a cache chega a 11,4 GiB, mas o decode ficou mais lento nessa medição
     (1,20–1,30 s contra 1,05–1,19 s por 64 tokens), coerente com memória empurrada para a RAM.
3. **Threads:** o motor usa por omissão todas as CPUs lógicas (12); neste 5600X 6 threads foram mais rápidas
   (32–34 contra 27,5 tok/s). Usa `--threads 6` (o `poly_config.py` deixa escolher).
4. **RAM:** com 32 GB e ~13 GB ocupados por outros programas, o ficheiro de 16,4 GB não cabe inteiro na cache de
   ficheiros; tempos de carga de 9 a 26 s.
5. **Tempo de carga:** o enchimento da cache faz cópias síncronas a partir do ficheiro mapeado (`fill_expert_cache`).
6. **O `--top-probs` com janelas** imprime os últimos logits do motor, não os de cada token aceite: para comparar
   probabilidades usar `--window 1` (ou `--ids-only` no harness). Isto vem do motor, não do port.
7. **Ganho das janelas de palpites** (sem MTP) é pequeno ou nulo neste modelo: na secção 5.2 as duas
   configurações ficam a ±10% uma da outra, para um lado e para o outro.
8. **Não determinismo:** com especialistas na CPU, duas execuções podem diferir num quase empate (como em CUDA);
   não observei divergências fora de quase empates.

---

## 8. Comandos para validação na RX 9060 XT

Na raiz do repositório (PowerShell ou cmd). O primeiro build instala o ROCm em `.rocm-win`, ~4,9 GB para gfx1200.

```bat
set STRATA_HIP_ARCHS=gfx1200
set STRATA_GGML_DIR=C:\strata\third_party\llama.cpp
tools\hip\build_windows.bat poly
```

`STRATA_GGML_DIR` é opcional: sem ele o CMake descarrega o llama.cpp na revisão fixada. A cópia em `C:\strata` é
essa revisão (3cf0325).

Testes sem modelo:

```bat
tools\hip\poly_windows.bat probe
tools\hip\poly_windows.bat tests
tools\hip\poly_windows.bat cmd build-hip-poly-win\poly_kernel_bench.exe
```

Paridade dos kernels no GGUF:

```bat
tools\hip\poly_windows.bat parity C:\Users\user\models\Qwen3.5-35B-A3B-Q3_K_M.gguf
```

Paridade com o llama.cpp. As referências ficam guardadas em `refs.json`; a 2.ª linha só lê esse ficheiro. Estes
scripts usam só a biblioteca padrão do Python e põem eles próprios o ROCm de `.rocm-win` no PATH do motor:

```bat
python tools\hip\poly_llama_parity.py --engine build-hip-poly-win\strata-poly.exe --model C:\Users\user\models\Qwen3.5-35B-A3B-Q3_K_M.gguf --llama-server C:\Users\user\.unsloth\llama.cpp\build\bin\Release\llama-server.exe --refs refs.json --engine-args="--window 1 --threads 6"
python tools\hip\poly_llama_parity.py --engine build-hip-poly-win\strata-poly.exe --model C:\Users\user\models\Qwen3.5-35B-A3B-Q3_K_M.gguf --refs refs.json --ids-only --engine-args="--threads 6"
python tools\hip\poly_serve_timing.py --engine build-hip-poly-win\strata-poly.exe --model C:\Users\user\models\Qwen3.5-35B-A3B-Q3_K_M.gguf --refs refs.json --repeat 2 --max-new 128 --engine-args="--threads 6"
```

O servidor e o `polystrata.py` precisam dos pacotes de `requirements.txt` (`jinja2` e outros), que o Python do
sistema não tem: o `polystrata.py` arranca o servidor com o mesmo Python com que é chamado. Uma vez, na raiz do
repositório (é o que o setup faz; `.venv` está no `.gitignore`):

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Servidor, web app e benchmark, sempre com o Python do `.venv`:

```bat
.venv\Scripts\python tools\hip\poly_config.py --model C:\Users\user\models\Qwen3.5-35B-A3B-Q3_K_M.gguf --name qwen3.5-35b-a3b-q3_k_m-hip --context 49152 --threads 6
.venv\Scripts\python polystrata.py --model qwen3.5-35b-a3b-q3_k_m-hip
.venv\Scripts\python polystrata.py benchmark --model qwen3.5-35b-a3b-q3_k_m-hip --prompt-tokens 2000,8000,24000,34000 --json bench.json
```

Onde vai o tempo de um token:

```bat
set STRATA_POLY_PROFILE=1
tools\hip\poly_windows.bat run --model C:\Users\user\models\Qwen3.5-35B-A3B-Q3_K_M.gguf --tokens "9707,11,1879" --max-new 128 --window 1 --threads 6
```

**Comparação com o llama.cpp Vulkan.** Usa o zip `llama-bXXXX-bin-win-vulkan-x64.zip` das releases do llama.cpp
(não foi descarregado nem testado aqui), mesmo GGUF, mesmas threads:

```bat
llama-bench -m C:\Users\user\models\Qwen3.5-35B-A3B-Q3_K_M.gguf -ngl 99 --n-cpu-moe 8 -t 6 -p 512,2048,8192 -n 128 -d 0,8192,32768
```

A estratégia completa (prefill, decode, memória e contexto, separados) está na secção 9 do relatório de
viabilidade. Compara no mesmo dia, intercalado, 3 execuções, e repete para alguns valores de `--n-cpu-moe`.

---

## 9. Próximas otimizações recomendadas

Por ordem de evidência medida:

1. **Evitar a ida e volta ao host por camada** quando todos os especialistas escolhidos estão na cache. O routing
   faz-se no host depois de copiar as pontuações. A ida e volta mede ~140 µs (secção 5.4) e os especialistas
   roteados ocupam 0,17–0,26 ms/camada, para ~40 µs de kernels. O caminho é routing na GPU (como o `router_top10`
   do `strata`) mais um lançamento de especialistas que leia os ids da memória da placa. É a maior margem restante
   no decode.
2. **Threads por omissão:** contar só os núcleos físicos neste tipo de CPU (6 em vez de 12 deu +18–24%). Fica a
   medir noutras CPUs antes de mudar a omissão.
3. **Prefill:** um perfil do lote de prompt (atenção hipBLAS FP16, MMQ em gfx1200, delta net) ainda não foi feito.
   O WMMA de gfx12 do `strata` (`qsa_prompt_attn`) é o precedente se a atenção for a parte maior.
4. **Carga:** encher a cache por buffers pinned e cópias assíncronas em vez de `cudaMemcpy` síncrono de memória
   mapeada.
5. **Integração no `setup.py`:** aceitar AMD para os modelos do `strata-poly` quando o motor HIP existe, escrever
   `--threads` e incluir o `strata-poly.exe` no zip HIP.
6. **Validar** um Q4_K_M (K-quants), um GGUF com MTP (Qwen3.6 da Unsloth) e o build CUDA.
