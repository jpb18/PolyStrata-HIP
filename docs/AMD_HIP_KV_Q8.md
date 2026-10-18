# KV cache Q8 no strata-poly HIP (Qwen3.6): auditoria, arquitetura e plano

**Estado operacional, 18/10/2026:** Q8 promovido para uso local por escolha;
FP16 mantém-se por omissão. T4/T5/semelhança/T7/T9/T11 aceites manualmente.
O âmbito e os resultados históricos abaixo mantêm as datas das respetivas decisões.
Referências antigas a NOT YET APPROVED foram substituídas pela
[promoção atual](../bench/results/2026-10-18-hip-kv-q8-promotion/README.md).
Ver [arranque e limites](HIP_LAUNCH_KV.md).


**Atualização de política, 17/10/2026:** o utilizador aprovou o critério
[T4-v2-2026-10-17](AMD_HIP_KV_Q8_T4_V2.md): `ΔNLL + 2 × EP ≤ 0,02 nats/token`,
com os mesmos limites KL/top-1. Esta versão aplica-se ao T4; não altera T5–T11 nem promove Q8.
Os resultados antigos mantêm a versão que usaram. O documento anterior a esta edição está
[arquivado por bytes](../bench/results/2026-10-17-hip-kv-q8-t4-v2/provenance/AMD_HIP_KV_Q8.v1.md).

**Decisão de revisão T4 v2: [MANUALLY APPROVED](AMD_HIP_KV_Q8_MANUAL_APPROVAL.md)**,
por instrução explícita do utilizador. A exceção numérica de prosa-1K fica aceite e deixa
de bloquear por T4; medições e resultados automáticos históricos permanecem registados.
T7, T9 e T11 continuam separados; a promoção geral Q8 não foi aprovada.

**Avaliação comparativa, 17/10/2026:** a versão
[KV-Q8-paired-v1-2026-10-17](AMD_HIP_KV_Q8_PAIRED_TESTS.md) passa a separar resultados
absolutos do modelo, diferenças funcionais Q8/FP16 e verificações de integração.
T5 é diagnóstico textual, T7 é eficiência/desempenho e T6/T8 usam pares sob os mesmos
controlos. Os números e critérios antigos continuam nos relatórios históricos; os estados
planeados neste documento abaixo são contexto histórico onde não houver atualização explícita.

Fase de investigação e planeamento, para aprovação antes de qualquer implementação. Não se alterou código, não se
compilou nada, não se tocou nos modelos, perfis, launchers nem no servidor em execução.

**Estado (13/10/2026):**
- As fases 0, 1 e 2 estão feitas: linha de base FP16, seleção `--kv` com Q8 ainda recusado, e os kernels Q8
  isolados.
- A Porta A deu positiva, com reservas.
- Resultados e próxima fase em
  [bench/results/2026-10-13-hip-kv-q8](../bench/results/2026-10-13-hip-kv-q8/README.md).
- Fase 2.5 feita: o despacho dos kernels rápidos nas janelas a 128K/262K (a "observação lateral" da secção 1.4),
  como opção `STRATA_POLY_ATTN_FIT=1`, validada para entrar com a fase 3:
  [bench/results/2026-10-13-hip-attn-dispatch](../bench/results/2026-10-13-hip-attn-dispatch/README.md).
- Fase 3 feita: a KV Q8 integrada no Qwen3.6 (`--kv q8_0`, experimental; FP16 continua a omissão e igual ao bit),
  nos perfis 48K, 128K e 262K, com P1. Fora do launcher. A fase 4 não começou:
  [bench/results/2026-10-13-hip-kv-q8-phase3](../bench/results/2026-10-13-hip-kv-q8-phase3/README.md).

Marcas usadas:
- **(código)**: lido no código deste ramo (`hip-poly`), não medido.
- **(medido)**: número de um relatório ou log que já existia, com a origem indicada.
- **(estimativa)**: conta feita a partir de números medidos; tem de ser confirmada.

PC de referência: RX 9060 XT 16 GB (gfx1200), Ryzen 5 5600X, 32 GB de RAM, Windows 11, ROCm HIP;
Qwen3.6-35B-A3B UD-Q3_K_M nos perfis 48K, 128K, 262K e 262K streaming.

## Respostas curtas

| Pergunta | Resposta |
|---|---|
| A estimativa de 2 992 MiB a 262K está certa? | **Sim, para Q8_0 com blocos de 32 e escala FP16.** 11 caches × 2 (K, V) × 2 cabeças × (256 códigos + 8 escalas × 2 B) = 11 968 B por posição; × 262 144 = 2 992 MiB, contra 5 632 em FP16. **(código + medido)** |
| O que já existe para reutilizar? | No motor `strata` (Qwen3.8 QSA): KV INT8 com grupos de 64 (`kv_q8.hpp`), medida como indistinguível de FP16 no seu modelo. No `strata-poly`: o caminho do KV streaming que copia blocos de linhas para um *stage* FP16 antes dos produtos hipBLAS. São o padrão, não o código a copiar. |
| Arquitetura recomendada | **Q8 lido diretamente pelos kernels de decode** (variantes Q8 dos kernels atuais) e, no prefill, **desquantização por blocos para um stage FP16** que os mesmos produtos hipBLAS leem. Sem bibliotecas novas. |
| Quanta VRAM fica livre? | 48K: ~479 MiB; 128K: ~1 304; 262K: ~2 624; 262K streaming: ~690 de VRAM e ~2 145 de RAM fixada. **(estimativa)** |
| E em experts? | Se a VRAM libertada for para a cache: +~360 / +~980 / +~1 980 / +~520 experts (60 → 64 %, 48 → 57 %, 27 → 46 %, 57 → 62 % dos 10 496). **(estimativa)** |
| Q8 será mais rápido? | Não há garantia. A atenção lê ~47 % menos bytes, o que conta a 128K e 262K; a 48K o efeito deve ser pequeno, e a 2K a atenção depende do número de kernels, não dos bytes. O prefill paga uma desquantização estimada em ~1 % a 200K. Tem de ser medido. |
| FP16 continua igual? | Sim: os kernels FP16 não mudam de texto, o FP16 é a omissão, e cada fase verifica logits iguais ao bit contra o binário operacional. |
| Recomendação | **Avançar**, por fases, com duas portas de decisão: depois dos kernels isolados (fase 2) e depois dos benchmarks (fase 5). Q8 com KV streaming fica para uma fase condicional; até lá o motor recusa a combinação. |

---

## 1. Estado atual da KV cache

### 1.1 Geometria (Qwen3.6-35B-A3B)

- 40 camadas: 30 gated delta net (GDN) e 10 de atenção completa (`full_attention_interval` 4); mais 1 bloco MTP com
  atenção, que tem a sua própria cache. **11 caches KV.**
  ([AMD_QWEN36_IMPLEMENTATION.md](reports/AMD_QWEN36_IMPLEMENTATION.md) §2.2 e §7.1)
- Atenção: 16 cabeças de consulta, **2 cabeças KV, 256 dimensões**, K normalizada por cabeça (`k_norm`), RoPE parcial
  (64 de 256 dimensões).
- Uma linha de K (ou de V) por posição e camada: 2 × 256 halves = 1 024 B. K + V: 2 048 B por camada, **22 528 B por
  posição** nas 11 caches.
- Confirmado pelos logs de carga: 1 056 MiB a 49 152 posições, 2 816 a 131 072, 5 632 a 262 144 **(medido**,
  [AMD_QWEN36_Q3_MIGRATION.md](reports/AMD_QWEN36_Q3_MIGRATION.md) §7.3).

O estado recorrente do GDN (`state_`, 251 MiB com as cópias para as janelas) não é KV, não cresce com o contexto e
**fica fora deste trabalho**.

### 1.2 Alocação **(código)**

- [`Qwen35::alloc_caches()`](../src/poly/qwen35.cpp:240): um único `dmalloc` para todas as caches; cada camada de
  atenção tem `B.keys` e `B.vals` (`uint16_t*`, FP16), `kv_split_ × n_kv × head_dim` cada, linha a linha
  ([qwen35.hpp:68](../src/poly/qwen35.hpp:68)).
- [`Qwen35::plan_kv()`](../src/poly/qwen35.cpp:127) decide `kv_split_` (posições na VRAM) uma vez, no arranque:
  todas, `--kv-resident N` ou `auto`. Usa `sizeof(uint16_t)` diretamente.
- Ordem em [`Engine::load()`](../src/poly/engine.cpp:1222): `alloc()` → `alloc_caches()`
  ([engine.cpp:1425](../src/poly/engine.cpp:1425)) → scratch → **só depois** `fill_expert_cache()`
  ([engine.cpp:1292](../src/poly/engine.cpp:1292)), que mede a VRAM livre e enche a cache de experts até à reserva.
  **A VRAM que a KV não usar passa sozinha para os experts.** É o mecanismo do ganho indireto; também quer dizer que,
  sem mais nada, o processo continua a ocupar a mesma VRAM total.
- `Engine::budget()` → `Qwen35::kv_budget()` ([qwen35.cpp:230](../src/poly/qwen35.cpp:230)) e a linha
  `VRAM MiB: ... KV ...` do arranque ([main.cpp:1132](../src/poly/main.cpp:1132)).

### 1.3 Escrita de K/V **(código)**

[`Qwen35::attn_store()`](../src/poly/qwen35.cpp:468): `lin(wk)` → `head_norm_rope` (norma por cabeça e RoPE, em
floats) → `lin(wv)` → `store_f16` ou, numa gravação de grafo (`recording_`), `store_f16_at`, que lê a posição de
`at_` na placa quando o grafo é repetido ([kernels.cu:417-436](../src/poly/kernels.cu:417)). Com streaming, as linhas
a partir de `kv_split_` vão para a RAM fixada e, dentro de uma cabeça de poucos tokens, também para a cópia
antecipada (`pre_k_`, `pre_v_`).

Cada posição é escrita de forma independente das outras. É isto que torna corretos `window()`/`keep()` (as linhas de
palpites rejeitados são simplesmente reescritas depois) e que uma quantização **por linha** preserva.

### 1.4 Leitura **(código)**

| Caminho | Quando | Kernels | O que leem |
|---|---|---|---|
| Decode rápido Qwen | chamadas de < 32 tokens, `dim` 256, 8 cabeças por grupo, sem janela/anel/`whole`/streaming, e o scratch tem lugar para as partes ([kernels.cu:1951-1962](../src/poly/kernels.cu:1951)) | `attn_score_rows_kernel`, `attn_softmax_kernel`, `attn_value_part_kernel<8|16>`, `attn_value_sum_kernel` | K e V em FP16, 8 B (scores) e 16 B (valores) por lane |
| Decode genérico | o resto (inclui streaming e `STRATA_POLY_ATTN=0`) | `attn_score_kernel<SPILL>`, `attn_softmax_kernel`, `attn_value_kernel<SPILL>` | FP16, VRAM ou RAM via `spill_row()` |
| Prefill | chamadas de ≥ 32 tokens | [`attend_batch()`](../src/poly/kernels.cu:2010): `cublasGemmStridedBatchedEx` FP16 × FP16 → FP32 (hipBLAS), softmax *online* por blocos de linhas | a cache FP16 **onde está**; com streaming, um bloco que toca a RAM é copiado para `kstage`/`vstage` por `kv_gather_kernel` ([kernels.cu:2054](../src/poly/kernels.cu:2054)) |

Uma propriedade que os kernels de decode garantem e que tem de continuar: **um token dá os mesmos bits sozinho ou
dentro de uma janela** (verificado por `poly_kernel_bench attn-decode`).

**Observação lateral (código, não medido).** O caminho rápido só é usado quando a chamada tem menos tokens do que
`chunk_` (com `c == chunk` a condição de [kernels.cu:1962](../src/poly/kernels.cu:1962) nunca se verifica).
`chunk_ = 16 Mi / (16 × contexto)` dá 21 a 48K, 8 a 128K e **4 a 262K**: a 262K, uma janela MTP de 4 ou mais tokens
usa os kernels genéricos, que a fase de decode mediu como ~2× mais lentos para 4 tokens a 45K. As otimizações de decode
foram medidas só a 48K. Não é deste trabalho, mas tem duas consequências aqui: o Q8 precisa também das variantes
genéricas, e a linha de base a 262K tem de ser medida, não extrapolada.

### 1.5 Contexto, checkpoints e reutilização de prefixos **(código)**

- `save_state()`/`load_state()` guardam só o estado recorrente; as linhas da KV ficam no motor.
- As conversas estacionadas ([conversations.hpp](../src/poly/conversations.hpp)) copiam linhas com `save_rows()` /
  `load_rows()` ([qwen35.cpp:358-404](../src/poly/qwen35.cpp:358)), `row_bytes()` cada, para RAM do processo. Não há
  formato em disco: o formato da KV é fixo durante a vida do processo, por isso nunca se misturam linhas FP16 e Q8.

### 1.6 MTP e janelas **(código)**

O bloco MTP é a camada `n_layer_` com cache própria; `block_attend()`/`block_rows()` chamam os mesmos `attn()` /
`attn_store()` ([qwen35.cpp:600](../src/poly/qwen35.cpp:600)). Os palpites são verificados por `window()` (até 8
tokens, grafos gravados por tamanho). Suportar Q8 em `attn()` e `attn_store()` cobre o MTP sem código próprio.

### 1.7 KV streaming **(código + medido)**

Perfil 262K streaming: 49 152 posições na VRAM (1 056 MiB), 212 992 em RAM fixada mapeada (4 576 MiB), e 458 MiB de
*stage*: a cópia antecipada de uma camada (`pre_k_`/`pre_v_`, 416 MiB) e o stage do prefill (`kstage`/`vstage`,
21 840 linhas, 43 MiB). Decode a 261K: 3,6 tok/s contra 14,1 com a KV toda na VRAM
([AMD_QWEN36_Q3_MIGRATION.md](reports/AMD_QWEN36_Q3_MIGRATION.md) §7.1).

### 1.8 Interação com a cache de experts **(código + medido)**

`resize_cache()` não toca na KV. A KV é decidida antes dos experts, por isso cada MiB poupado na KV é um MiB a mais
para experts, salvo se a reserva (`--vram-reserve-mib`) subir. O tamanho de um expert no UD-Q3_K_M é, em média,
**1,326 MiB** (3 818 MiB / 2 883 experts a 262K; 8 396 / 6 329 a 48K).

## 2. Suporte existente para quantização reutilizável

| O quê | Onde | Reutilizável? |
|---|---|---|
| KV INT8 do motor `strata`: escala FP16 por 64 valores e por cabeça; `kv_append_q8_*` (quantiza na escrita) e `kv_gather_q8_step` (desquantiza as células escolhidas para scratch FP16; a atenção não muda) | [kv_q8.hpp](../include/strata/kernels/kv_q8.hpp), [kv_q8.cu](../src/kernels/cuda/kv_q8.cu), `src/core/layer.cpp` | **A matemática e o desenho, não o código.** O desenho "só muda o armazenamento" funciona lá porque a atenção QSA lê poucas células escolhidas. A atenção do Qwen3.6 é densa: desquantizar todo o contexto por token custaria mais do que ler FP16 (secção 3.1). Os kernels assumem páginas, `QsaShapes` e terminam com `std::exit(1)`, o que não respeita o protocolo fail-closed do poly |
| Qualidade medida desse INT8 | [bench/results/2026-09-27-kv-q4](../bench/results/2026-09-27-kv-q4/README.md) | **Sim, como referência de tolerâncias.** Mesmo conjunto de experts, só a KV muda, FP16 como referência: INT8 tem ΔNLL de −0,013 a +0,012 nats/token (dentro de ~2 erros-padrão), KL média 0,009–0,014 até 1K e 0,114 a 8K, top-1 igual em 92–98 %. O Q4_0 deu +0,078 e +0,112 nats/token em documentos, e foi julgado uma perda real. Outro modelo (Qwen3.8 QSA), CUDA |
| Q4_0 com Hadamard e K8V4 | motor `strata` | Não nesta fase (precisão medida pior; pedido é 8 bits) |
| Referência de quantização Q8_0 do ggml (`quantize_row_q8_0_ref`) | `third_party/llama.cpp-3cf0325/ggml` (já ligado ao poly) | **Sim**: referência na CPU para testar códigos e escalas ao bit |
| `kv_gather_kernel` + `kstage`/`vstage` do streaming | [kernels.cu:445](../src/poly/kernels.cu:445), [kernels.cu:2054](../src/poly/kernels.cu:2054) | **Sim, é o padrão do prefill Q8**: em vez de copiar o bloco, desquantiza-o para o mesmo tipo de stage, e os produtos hipBLAS não mudam |
| Ferramentas | `tools/hip/kv_parity.py` (logits passo a passo, KL, Δp, quase-empates), `poly_llama_parity.py`, `kv_needles.py`, `bench_compare.py`, `bench_prefill.py`, `decode_trace.py`, `poly_api_check.py`, `poly_kernel_bench attn-decode`, `test_gpu_failure.py` + `tests/poly_gpu_failure_mock/` | **Sim**, com extensões (fase 1 e 2). `tools/kv_precision_compare.py`, citado no relatório do Q4, não existe neste checkout; o `strata-poly` também não tem modo *teacher-forced* |

Nada no `strata-poly` lê KV quantizada hoje.

## 3. Arquitetura recomendada

### 3.1 Abordagens comparadas

| | A. Só armazenamento: desquantizar para FP16 antes de cada atenção (o desenho do `strata`) | **B. Decode lê Q8 diretamente; prefill desquantiza por blocos para um stage FP16 (recomendada)** | C. Q8 de ponta a ponta com WMMA int8 do gfx1200 | D. FP8 E4M3 |
|---|---|---|---|---|
| Decode | Lê 1,06 B + escreve 2 B + relê 2 B por valor: **~2,5× os bytes do FP16** | Lê ~1,06 B por valor: **~0,53× os bytes do FP16** | Igual a B | ~0,5× |
| Memória extra | Scratch FP16 do contexto inteiro de uma camada (512 MiB a 262K) | Stage do prefill (16 MiB com limite de 8 192 linhas) | Nenhum | Nenhum |
| Prefill | Igual a B | GEMM hipBLAS atual sobre o stage; ~1 % a mais a 200K (secção 6) | Kernels WMMA int8 novos; Q e pesos da softmax também quantizados (erro maior) | Precisa de GEMM FP8 (hipBLASLt no Windows gfx1200 por confirmar) ou desquantização como B |
| Kernels novos | Poucos | Variantes Q8 de 4 kernels de atenção, escrita e desquantização | Muitos, sem referência no repositório | Como B, mais a dependência de biblioteca |
| Precisão | = B | Erro por valor ≤ 0,44 % do máximo do bloco de 32 | Pior que B (Q e pesos também em 8 bits) | 3 bits de mantissa: erro relativo até ~6 % por valor |
| Referência externa | — | llama.cpp `-ctk q8_0 -ctv q8_0` usa a mesma quantização | — | — |
| Veredicto | Rejeitada: anula o ganho no decode | **Recomendada** | Fora da primeira versão | Fora da primeira versão |

### 3.2 Formato

**Semântica Q8_0 do ggml**: blocos de 32 valores; `d = max|x| / 127` em float, `q = round(x / d)` (com `1/d`, como o
ggml), escala guardada em FP16; valor lido = `q × d_fp16`. Uma linha de 32 valores nunca atravessa duas cabeças nem
duas posições.

Erro máximo por valor: `d/2` do arredondamento mais `127 × d × 2⁻¹¹` da escala em FP16 = **0,562 d = 0,44 % do
máximo do bloco**. Este limite é testável (secção 8, T1).

Disposição na memória, **por cabeça**: os 256 códigos e logo a seguir as 8 escalas.

```
linha de K (ou V), uma posição, uma camada: 544 B
| cabeça 0: 256 × int8 | 8 × fp16 | cabeça 1: 256 × int8 | 8 × fp16 |
  0                     256        272                    528       544
```

- 272 e 544 são múltiplos de 16: códigos e escalas ficam alinhados a 16 B em todas as linhas (cargas de 8 e 16 B).
- Uma linha continua contígua: `save_rows`/`load_rows`, a cópia antecipada (`ahead()`), `spill_row()` e
  `kv_gather` funcionam com a mesma lógica, mudando só o número de bytes por linha.
- O bloco `block_q8_0` do ggml (34 B com a escala à frente) desalinha os códigos: não serve para cargas vetoriais.

Alternativas de agrupamento:

| Agrupamento | B por posição (11 caches) | 262K | Nota |
|---|---:|---:|---|
| FP16 | 22 528 | 5 632 MiB | atual |
| **Q8_0, blocos de 32 (recomendado)** | **11 968** | **2 992 MiB** | compatível com llama.cpp; um *outlier* da K afeta 32 valores |
| INT8, grupos de 64 (motor `strata`) | 11 616 | 2 904 MiB | qualidade medida noutro modelo; 88 MiB a menos a 262K |
| INT8, uma escala por cabeça (256) | 11 352 | 2 838 MiB | mais sensível aos canais grandes da K |
| FP8 E4M3 sem escala | 11 264 | 2 816 MiB | ver 3.1 |

Escolho blocos de 32 pela referência externa (permite comparar com o llama.cpp na mesma quantização) e pela
robustez a canais grandes da K, que os modelos Qwen costumam ter, por 88 MiB a 262K. Os grupos de 64 continuam a ser
uma alternativa barata se a fase 2 mostrar erro igual.

### 3.3 K e V

O utilizador escolhe FP16 ou Q8 para as duas. Internamente, K e V têm **tipos independentes** (`kv_k`, `kv_v`):
o kernel de pontuações depende só da K e o dos valores só da V, por isso as combinações não custam kernels a mais.
Serve para os testes separarem o erro da K do da V (opções de diagnóstico, secção 3.10) e deixa aberta uma variante
"K FP16, V Q8" se a K se mostrar sensível. O bloco MTP usa o mesmo formato que o resto (decisão D7).

### 3.4 Escrita

Kernel novo `store_q8` (e `store_q8_at`, para os grafos gravados, com a variante que escreve a parte em RAM): uma
warp de 32 lanes por bloco, um valor por lane, máximo por *shuffle* (desempate determinístico, como foi corrigido no
Q4), `q` e escala escritos pela mesma warp. Quantiza diretamente dos floats de `k_`/`v_`, sem passar por FP16.
Custo: 1 024 tokens × 1 KiB por camada num lote de prefill. Insignificante.

### 3.5 Leitura no decode

Variantes Q8 dos quatro kernels que leem a cache, com a mesma organização de threads:

| Kernel FP16 | Variante Q8 | O que muda |
|---|---|---|
| `attn_score_rows_kernel` | `attn_score_rows_q8_kernel` | Cada lane lê 8 códigos (8 B) em vez de 2 × 4 halves; 4 lanes partilham uma escala; a soma de um bloco é multiplicada pela escala antes de `heads_sum()`. As queries continuam em registos |
| `attn_value_part_kernel<Q>` | `attn_value_part_q8_kernel<Q>` | Cada lane lê os seus 8 valores como 8 B + 1 escala (em vez de 16 B) |
| `attn_score_kernel<SPILL>` | `<SPILL, Q8>` | 4 códigos (4 B) por passo em vez de 4 halves; a escala de 8 lanes |
| `attn_value_kernel<SPILL>` | `<SPILL, Q8>` | 1 B + escala partilhada pela warp em vez de 2 B |

Softmax e `attn_value_sum_kernel` não mudam. Sem buffers temporários. A propriedade "um token sozinho ou numa janela
dá os mesmos bits" mantém-se, porque a ordem das somas de cada token continua a não depender dos outros.

### 3.6 Leitura no prefill

`attend_batch()` ganha um ramo Q8: cada bloco de linhas é desquantizado para um stage FP16 (`kv_dequant_kernel`, o
mesmo desenho de `kv_gather_kernel`, sabendo da divisão VRAM/RAM) e os dois produtos hipBLAS leem o stage, tal como já
leem `kstage`/`vstage` no streaming. O tamanho do bloco passa a ser `min(o que cabe no scratch, linhas do stage)`.
Proposta: stage de **8 192 linhas** (2 × 8 192 × 1 KiB = 16 MiB); com lotes de 1 024 tokens os blocos já têm ~672
linhas, por isso o limite só toca chamadas de 32 a ~1 000 tokens. Sem bibliotecas novas; `CUDA_R_16F` como hoje.

### 3.7 MTP, janelas e `keep()`

Cobertos por `attn()`/`attn_store()` (secção 1.6). A quantização é por linha, por isso uma linha reescrita depois de um
palpite rejeitado não deixa rasto. A taxa de aceitação do MTP é um critério de aceitação (T7), não uma suposição.

### 3.8 Checkpoints e conversas

`row_bytes()` passa a 11 968 B com Q8. `save_rows`/`load_rows` copiam as linhas Q8 tal como estão (códigos e
escalas): uma conversa estacionada volta ao bit. Efeito lateral: o mesmo `--conversation-cache-mib` guarda ~1,9× mais
tokens.

### 3.9 KV streaming

Com a disposição da secção 3.2, o streaming Q8 reutiliza `spill_row()`, `ahead()`, a verificação de RAM fixada e
`save_rows`; precisa das variantes `SPILL` dos kernels genéricos e de `kv_dequant_kernel` sobre a parte em RAM. É o
caminho com mais casos e o menos necessário: a 262K, a KV Q8 inteira (2 992 MiB) fica abaixo do que o perfil 262K
FP16 já usa hoje. Proposta: fase 7, condicional; **até lá, `--kv q8_0` com `--kv-resident` diferente de `all` é
recusado no arranque**, com uma mensagem clara.

### 3.10 Seleção no motor

- `Options`: `KvType kv_k = kF16, kv_v = kF16` ([engine.hpp:35](../src/poly/engine.hpp:35)).
- CLI: `--kv fp16|q8_0` (o mesmo nome de opção do motor `strata`, que usa `--kv int8|q4_0|k8v4`). `--kv int8` é
  recusado de propósito: é outro formato. Opções de teste, documentadas como tal: `--kv-k`, `--kv-v`.
- Validação **antes de qualquer alocação**, em `Engine::load()`: arquitetura sem suporte (Gemma 4, GLM), geometria fora
  do que os kernels Q8 aceitam (`head_dim % 32`), Q8 com streaming (até à fase 7). Cada caso é um erro de arranque
  (`strata-poly: ...`, código 1), **nunca uma passagem silenciosa a FP16**.
- Identificação: a linha `INFO` passa a ter `kv=fp16|q8_0 kv_bytes_per_position=22528|11968`; a linha `VRAM MiB` do
  arranque diz o formato. O servidor já lê qualquer `chave=valor` do `INFO` (`self.info`), e a página web já mostra
  `eng.kv` ([app.js:406](../serve/web/app.js:406)); falta o rótulo de `q8_0`.
- Um erro HIP num kernel Q8 segue a política existente: `gpu_fatal()`, processo novo, que arranca **com a mesma
  configuração** (logo, com Q8).

### 3.11 Launcher e configuração

**Onde fica o formato.** Numa configuração de execução própria, como os perfis: `tools/hip/poly_profile.py` ganha
`--kv fp16|q8`, e Q8 escreve `strata-<modelo>-hip[-perfil]-q8.json` com `--kv q8_0` nos `args` (por `--extra`, que
`poly_config.py` já aceita). Razões:
- os ficheiros FP16 de hoje não mudam;
- o nome do modelo, o log e `/v1/models` dizem `-q8`;
- o reinício depois de uma falha usa a mesma configuração, por isso não há queda silenciosa para FP16;
- `/api/models/load` e o `switch_on_request` já trocam entre configurações por nome.

A expressão `SUFFIX` de `poly_profile.py` ([poly_profile.py:31](../tools/hip/poly_profile.py:31)) tem de aprender o
sufixo `-q8`, senão `--list` mostra as configurações Q8 como modelos novos. A reutilização de uma configuração já
escrita tem de verificar também `--kv`.

**Menu** (em `START-POLY-HIP.bat`, depois do contexto e antes de `poly_profile.py`, portanto antes de o motor alocar a
KV):

```
  PolyStrata no AMD - formato da KV cache?

   1  FP16   precisao original, maior utilizacao de VRAM      (KV 5 632 MiB a 262K)
   2  Q8     menor utilizacao de VRAM, quantizacao de 8 bits  (KV 2 992 MiB a 262K)

  Escolhe 1-2 (em 20 s fica a 1):
```

A linha de cada opção mostra a KV do perfil escolhido (tabela da secção 5).

Regras propostas:
- **FP16 é a omissão.** Q8 só por escolha: tecla no menu, argumento ou variável.
- Argumentos: `START-POLY-HIP.bat 262k q8`, `... fp16` (sem menu de formato); variável `POLY_KV=q8|fp16`;
  `check` mostra o formato que usaria. Precedência: argumento > `POLY_KV` > menu > preferência guardada > FP16.
- **Preferência guardada**: uma escolha feita no menu grava `build-hip-poly-win\last-kv.txt` e passa a ser a omissão
  do menu seguinte. Para o tempo esgotado não gravar nada, o menu usa uma tecla escondida como omissão
  (`choice /c 12T /t 20 /d T`): `errorlevel 3` = tempo esgotado → preferência guardada ou FP16, sem gravar.
- **Arranques não interativos nunca esperam**: com o perfil dado por argumento, com `POLY_CONFIG` ou em `check`, o
  menu de formato não aparece (argumento, `POLY_KV`, preferência guardada ou FP16). Se `choice` falhar (sem consola),
  fica FP16.
- `262k-stream` + Q8, até à fase 7: o launcher diz que não está disponível e sugere o perfil 262K (onde Q8 cabe na
  VRAM). O motor recusa também: a verificação do launcher só evita um arranque falhado.
- Um servidor já ativo com outro formato é tratado como hoje com outro modelo: troca pelo nome da configuração.
- Não corre `setup.py`; não muda os modelos nem os perfis FP16.

**Servidor e métricas**: nenhuma mudança obrigatória para funcionar. Para identificar: rótulo `q8_0` em `app.js`;
uma métrica Prometheus (`strata_engine_kv_bits` 16|8 ou um rótulo `kv`); `bench_compare.py`, `bench_prefill.py`,
`kv_needles.py` gravam `info.kv` nos seus JSON. O provider do OpenCode
([opencode-polystrata-hip.json](../tools/hip/opencode-polystrata-hip.json)): a entrada `current` serve sem mudanças;
entradas `-q8` só na fase 8. Nota: com `switch_on_request`, um cliente que peça o nome do modelo FP16 troca o
servidor para FP16. É uma troca pedida e registada, não uma queda silenciosa, mas tem de estar documentada.

### 3.12 O que não muda

Os kernels FP16 (o texto deles não é tocado; o Q8 são funções e templates novos), o GDN e o
`gdn_state_reg_kernel`, as otimizações de decode (routing na placa, MMVQ, cabeça do draft a 65 536 IDs), o MoE, o
MMQ, o routing, os grafos (só ganham variantes de lançamento) e a política fail-closed.

## 4. Ficheiros e kernels a alterar

| Ficheiro | Função / símbolo | Alteração | Tipo |
|---|---|---|---|
| `src/poly/engine.hpp` | `Options`, `Budget` | `kv_k`, `kv_v`; formato no `Budget` | local |
| `src/poly/engine.cpp` | `Engine::load()` | validação do formato antes de `alloc()`; hash opcional do conjunto de experts em cache (teste) | local |
| `src/poly/main.cpp` | parser, `usage()`, `serve()` (`INFO`), linha `VRAM MiB` | `--kv`, `--kv-k`, `--kv-v`; `kv=`; opções de teste `--force-tokens`, `--dump-kv` | local |
| `src/poly/qwen35.hpp` | `Block` | `keys`/`vals` (e `_h`, `_hp`) passam a bytes, com o passo de linha | local |
| `src/poly/qwen35.cpp` | `plan_kv`, `alloc_caches`, `kv_budget`, `row_bytes`, `save_rows`, `load_rows`, `attn_store`, `attn`, `spill`, `ahead` | bytes por linha pelo tipo; stage de desquantização; escolha FP16/Q8 nos lançamentos | local (o FP16 continua a chamar as mesmas funções) |
| `src/poly/gemma4.cpp`, `glm5next.cpp` | — | nada: `Engine::load()` recusa Q8 nestas arquiteturas | — |
| `include/strata/poly/kernels.hpp` | `KvSpill`, novas declarações | descrição da disposição Q8; `store_q8`, `store_q8_at`, `attend_q8`, `attend_batch_q8` | novo |
| `src/poly/kernels.cu` | `store_q8_kernel`, `store_q8_at_kernel<SPILL>`, `kv_dequant_kernel` | escrita e desquantização | **kernels novos** |
| `src/poly/kernels.cu` | `attn_score_rows_q8_kernel`, `attn_value_part_q8_kernel<Q>`, `attn_score_kernel<SPILL,Q8>`, `attn_value_kernel<SPILL,Q8>` | leitura Q8 no decode | **variantes** dos existentes |
| `src/poly/kernels.cu` | `attend_batch` (ramo Q8 numa função separada que partilha o resto) | desquantizar cada bloco para o stage; mesmos produtos hipBLAS | alteração localizada |
| `src/poly/kernel_bench.cu` | `attn-decode-q8`, `attn-batch-q8`, `kv-q8-quant` | até 262 144 posições; exatidão; mínimo pela banda; janela = sozinho; KV real de `--dump-kv` | novo |
| `src/poly/conversations_test.cpp` | — | `row_bytes` diferente | teste |
| `tools/hip/poly_profile.py` | `PROFILES`, `SUFFIX`, `main()` | `--kv`, sufixo `-q8` | local |
| `START-POLY-HIP.bat` | menu, argumentos | menu de formato, `q8`/`fp16`, `POLY_KV`, `last-kv.txt` | local |
| `serve/web/app.js`, `serve/prometheus.py` | rótulo, métrica | `q8_0` | local |
| `tools/hip/kv_q8_compare.py` (novo), `kv_parity.py`, `kv_needles.py`, `bench_compare.py` | — | comparação *teacher-forced*; registo do formato | ferramentas |
| `tools/hip/test_poly_profile_kv.py` (novo) | — | testes do `poly_profile.py` sem GPU (como `tools/test_setup_*.py`) | teste |
| `tests/poly_gpu_failure_mock/`, `tools/hip/test_gpu_failure.py`, `serve/test_fatal_recovery.py` | — | falha num kernel Q8; o reinício mantém `kv=q8_0` | teste |

Nenhuma dependência nova: os kernels Q8 são HIP/CUDA simples (warps de 32, como o gfx1200 corre), e o prefill usa o
mesmo hipBLAS. `kernels.cu` também compila para CUDA (o `strata-poly` é um programa CUDA por omissão): a regra do
[AGENTS.md](../AGENTS.md) obriga a compilar esse backend, o que não é possível neste PC (risco R8).

## 5. Estimativas de memória

### 5.1 Por perfil (geometria real; MiB)

| | 48K | 128K | 262K | 262K streaming |
|---|---:|---:|---:|---:|
| Posições na VRAM / na RAM | 49 152 / 0 | 131 072 / 0 | 262 144 / 0 | 49 152 / 212 992 |
| **FP16** KV na VRAM | 1 056 | 2 816 | 5 632 | 1 056 |
| FP16 stage (cópia antecipada + prefill) | 0 | 0 | 0 | 416 + 43 = 459 |
| FP16 KV na RAM fixada | 0 | 0 | 0 | 4 576 |
| **Q8** códigos na VRAM | 528 | 1 408 | 2 816 | 528 |
| Q8 escalas e metadados na VRAM | 33 | 88 | 176 | 33 |
| **Q8 KV na VRAM** | **561** | **1 496** | **2 992** | **561** |
| Q8 stage de desquantização (limite 8 192 linhas) | 16 | 16 | 16 | 0 (usa o de 43) |
| Q8 cópia antecipada | — | — | — | 221 |
| Q8 KV na RAM fixada | 0 | 0 | 0 | **2 431** |
| **VRAM libertada** | **~479** | **~1 304** | **~2 624** | **~690** |
| RAM fixada libertada | 0 | 0 | 0 | ~2 145 |

Contas: 22 528 e 11 968 B por posição (secção 3.2). Alinhamento: as linhas de 544 B já são múltiplos de 16, por isso
o alinhamento não acrescenta nada; o `dmalloc` arredonda a 256 B uma vez. Buffers temporários: só o stage do prefill;
o decode lê Q8 sem buffers. **(estimativa sobre geometria verificada)**

### 5.2 Para onde vai a VRAM libertada

Com a omissão de hoje (`fill_expert_cache()` enche até à reserva), a 1,326 MiB por expert **(estimativa)**:

| | 48K | 128K | 262K | 262K streaming |
|---|---:|---:|---:|---:|
| Experts na VRAM, FP16 **(medido, Q3 migração §7.3)** | 6 329 (60,3 %) | 5 004 (47,7 %) | 2 883 (27,5 %) | 5 961 (56,8 %) |
| Experts a mais com Q8 | +~361 | +~983 | +~1 979 | +~520 |
| Experts na VRAM, Q8 | ~6 690 (63,7 %) | ~5 987 (57,0 %) | ~4 862 (46,3 %) | ~6 481 (61,7 %) |

Cuidado com a escala: nos logs do perfil 48K a cache de experts variou entre 7 755 e 8 396 MiB de um arranque para
outro (VRAM ocupada por outros programas); nos de 262K, entre ~3 500 e 3 818 MiB. **O ganho a 48K (~479 MiB) é menor
do que essa variação**: as medições têm de fixar a reserva (secção 8.2).

### 5.3 Reservada contra residente, e o WDDM

- O que muda com certeza é a **memória reservada** (`hipMalloc`). O que fica **residente** é outra coisa no WDDM:
  a [investigação de residência](reports/AMD_HIP_VRAM_RESIDENCY.md) mediu, a 262K em repouso, 10 971 MiB dedicados contra
  12 437 alocados, e uma das hipóteses é que as páginas da KV nunca escritas não estejam residentes.
- Se essa hipótese se confirmar, a KV FP16 de 262K só ocupa VRAM física à medida que a conversa cresce. Com Q8 e a
  VRAM libertada entregue aos experts, **os experts ficam residentes logo no arranque** (são escritos ao carregar).
  Consequência: numa conversa curta, Q8 pode ter **mais** VRAM dedicada residente do que FP16, não menos. Com o
  contexto cheio as duas chegam ao mesmo total.
- Por isso a "redução da pressão WDDM" só acontece se a VRAM libertada **não** for para os experts. Há duas políticas,
  a medir na fase 5 (decisão D3):
  - **P1 – experts** (o comportamento automático): mais velocidade esperada a 128K/262K; o processo ocupa o mesmo.
  - **P2 – reserva**: `--vram-reserve-mib` sobe na configuração Q8 pela poupança; o processo a 262K desce de ~12 437 para
    ~9 813 MiB alocados (12,1 → 9,6 GiB), abaixo dos ~10 GiB a partir dos quais o orçamento do Windows baixou nas medições anteriores
    (15 416 → 13 937 MiB, [docs/AMD_HIP.md](AMD_HIP.md)). Hipótese a medir, não promessa.
- Os 4 `GPU_FATAL` de hoje no perfil 262K (espera de 1 s pelos experts da CPU,
  [AMD_HIP_VRAM_RESIDENCY.md](reports/AMD_HIP_VRAM_RESIDENCY.md) §2) acontecem com 27 % dos experts na VRAM. Com P1, menos
  experts correm na CPU. Pode reduzir essas esperas; não está demonstrado.
- Este trabalho não implementa gestão adaptativa: a KV continua decidida uma vez no arranque.

## 6. Impacto esperado no desempenho

### 6.1 O que se espera, e porquê **(estimativa)**

**Decode.** Cada token lê K e V de todas as posições em 10 camadas (e 1 por palpite do MTP). Bytes por camada e
token: FP16 2 048 × P, Q8 1 088 × P. A ~280 GB/s (derivado da linha de 45K de "ler K e V uma vez", 331 µs, em
[AMD_HIP_DECODE_OPT.md](reports/AMD_HIP_DECODE_OPT.md) §3.3), o mínimo pela banda:

| Contexto | FP16, por camada | Q8, por camada | Diferença numa janela (10 camadas) |
|---|---:|---:|---:|
| 45K | 331 µs (medido) | ~176 µs | ~1,5 ms |
| 131K | ~0,96 ms | ~0,51 ms | ~4,5 ms |
| 261K | ~1,91 ms | ~1,01 ms | ~9,0 ms |

O kernel FP16 rápido gasta 1,3× o seu mínimo a 45K (432 contra 331 µs, comentário de
[kernels.cu:939-948](../src/poly/kernels.cu:939)). Se o Q8 mantiver essa eficiência, o ganho real é desta ordem;
cargas de 8 B em vez de 16 B podem baixá-la, e é isso que a fase 2 mede primeiro. A 2K a atenção custa ~35 µs contra
15 µs de mínimo: é o número de kernels que manda, e o Q8 não deve ajudar (pode piorar ligeiramente).
A 262K há ainda o efeito indireto: +~1 980 experts na VRAM (27 → 46 %), menos trabalho na CPU, que hoje domina o
tempo a esse contexto. O efeito combinado não se deixa prever com os números que existem: só se mede.

**Prefill.** Por lote de 1 024 tokens e camada na posição P, a desquantização lê 1 088 × P B e escreve 2 048 × P B: a
200K, ~628 MB, ~2,2 ms; ~25 ms por lote nas ~11 camadas. O lote médio a 262K demora ~2 s (506 tok/s medidos), por isso
**~1 %**. A 8K, desprezável. Os produtos hipBLAS não mudam. Mais experts na VRAM também reduzem os que passam pelo
staging no prefill, o que pode compensar.

**TTFT** segue o prefill. **Aceitação do MTP**: o bloco de palpites lê KV Q8; pode mudar ligeiramente.

### 6.2 Como medir

| Métrica | Ferramenta / fonte | Como |
|---|---|---|
| Prefill completo | `tools/hip/bench_prefill.py`, `kv_needles.py` (1.º pedido) | 8K, 32K, 65K, 131K, 261K; mediana de 3, alternando FP16 e Q8 |
| Prefill incremental | `bench_compare.py` (pedidos consecutivos no mesmo processo) e métricas `reread_tokens`, `new_prompt_tokens`, `prefill_ms` do `DONE` | conversa de vários turnos, mesmo processo |
| TTFT | `ttft_ms` do `DONE` ([docs/INFERENCE_METRICS.md](INFERENCE_METRICS.md)) | por pedido |
| Decode | `bench_compare.py` (2K/8K/31K/45K, 128 tokens); `kv_needles.py` (resumo de 128 tokens a 65K/131K/261K); `decode_trace.py` / `STRATA_POLY_TRACE=2` (tempo da atenção por camada) | MTP ligado e `--window 1` |
| Kernels | `poly_kernel_bench attn-decode[-q8]`, `attn-batch-q8` | 2K a 262K, 1/2/4/8 tokens, contra o mínimo pela banda |
| Aceitação do MTP | `DONE`: palpites aceites / oferecidos | os mesmos prompts |
| Hit rate da cache de experts | `DONE`: experts da VRAM / total | idem |
| Experts residentes | `INFO expert_slots`, linha `loaded ... routed experts in VRAM` | por arranque |
| VRAM dedicada e partilhada | PDH `\GPU Process Memory(pid_*)\Dedicated Usage`, `Shared Usage`, `Total Committed` (como na investigação de residência) | em repouso, depois de 8K, 131K, 261K, a cada 2 s |
| CPU e RAM | `/v1/status` → `hardware` (cpu, ram_used); working set e commit do processo | durante o decode |

Regras: o mesmo dia, o mesmo binário para as duas variantes (o novo, em `--kv fp16` e `--kv q8_0`), alternado ABAB,
aquecimento não contado, cópia do perfil de experts (o motor escreve no perfil: as corridas não podem mexer no
operacional), PC sem outro `strata-poly` ativo. As duas políticas P1 e P2 medidas em separado.

## 7. Plano faseado

A ordem proposta difere da sugerida em três pontos:
1. **Começa por uma linha de base** (fase 0): a 262K não há números de decode depois das otimizações recentes, e o
   ruído entre processos tem de ser conhecido antes de se fixar tolerâncias.
2. **Kernels antes do armazenamento**: guardar Q8 sem kernels que o leiam não se pode testar, e o desempenho dos
   kernels é a primeira porta de decisão. Os kernels testam-se isolados no `poly_kernel_bench` (com KV real
   descarregada na fase 1).
3. **Benchmarks antes do launcher, e streaming no fim e condicional**: o menu só vale a pena se o Q8 ganhar; o
   streaming Q8 é o caminho com mais casos e, com Q8, o menos necessário.

Durante todas as fases: o executável operacional (`build-hip-poly-win\strata-poly.exe`) e as configurações FP16 não
são tocados. O trabalho compila em `build-hip-poly-q8-win\`; as configurações de teste apontam para esse executável.

```
0 base ─► 1 seleção ─► 2 kernels ─► [porta A] ─► 3 integração ─► 4 correção ─► 5 benchmarks ─► [porta B]
                                                                                                │
                                                     8 operacional ◄── 6 launcher ◄─────────────┘
                                                     7 streaming Q8 (condicional, depois de 5)
```

### Fase 0 – Linha de base e ruído

- **Objetivo:** números FP16 do mesmo dia nos 4 perfis; ruído entre processos com experts diferentes na VRAM.
- **Ficheiros:** nenhum do motor. Resultados em `bench/results/<data>-hip-kv-q8-baseline/`.
- **Trabalho:** `bench_compare.py` a 48K; `kv_needles.py` a 65K/131K/261K (prefill e decode); `decode_trace.py` a
  131K e 261K (confirma ou não a observação do caminho genérico, secção 1.4); `poly_kernel_bench attn-decode`; PDH
  em repouso e depois de cada prompt; `kv_parity.py` FP16 contra FP16 com `--vram-reserve-mib` diferente (logo outros
  experts na VRAM) a 8K e 32K: distribuição de KL e Δp por passo.
- **Complexidade:** baixa (1–2 dias de máquina).
- **Riscos:** precisa do servidor parado; outros programas mudam a VRAM livre.
- **Testes obrigatórios:** os acima, 3 repetições.
- **Aceitação:** relatório com medianas e dispersão; limiar de ruído (p50/p99 de KL e Δp) que as tolerâncias usam.
- **Rollback:** não se aplica.

### Fase 1 – Infraestrutura de seleção (sem Q8)

- **Objetivo:** a opção existe, é validada e registada; o FP16 continua igual ao bit.
- **Ficheiros:** `engine.hpp` (`Options`), `engine.cpp` (`load()`), `main.cpp` (parser, `usage`, `INFO`, linha `VRAM`),
  `qwen35.cpp` (`kv_budget`).
- **Alterações:** `--kv fp16|q8_0`, `--kv-k`, `--kv-v`; `q8_0` ainda recusado ("ainda não implementado"); arquiteturas
  sem suporte recusadas; `INFO kv=fp16 kv_bytes_per_position=22528`; opções de teste: `--force-tokens @FICHEIRO`
  (alimenta tokens dados em vez dos amostrados e escreve os logits de cada passo com `--logits-out`), `--dump-kv
  FICHEIRO` (`save_rows` depois do prompt) e `STRATA_POLY_CACHE_HASH=1` (hash do conjunto de experts em cache, para
  confirmar que duas corridas têm os mesmos).
- **Complexidade:** baixa a média.
- **Riscos:** acrescentar `kv=` ao `INFO` muda bytes do protocolo no caminho por omissão (decisão D6; a aritmética não
  muda).
- **Testes obrigatórios:** logits ao bit contra o binário operacional (mesmos experts, `--window 1` e MTP por
  omissão); `conversations_test`, `guesses_test`; `--kv int8`/valor desconhecido → código 2; `--kv q8_0` → erro claro,
  o servidor mostra o erro e não arranca FP16.
- **Aceitação:** FP16 ao bit; tempos dentro de ±2 % (mediana de 3).
- **Rollback:** reverter o commit; o executável operacional nunca foi substituído.

### Fase 2 – Kernels Q8 isolados — porta de decisão A

- **Objetivo:** kernels corretos e rápidos antes de qualquer integração.
- **Ficheiros:** `kernels.hpp`, `kernels.cu` (secção 4), `kernel_bench.cu`.
- **Alterações:** `store_q8[_at]`, `kv_dequant_kernel`, as 4 variantes de leitura, o ramo Q8 de `attend_batch`.
- **Complexidade:** **média a alta** (a fase maior).
- **Riscos:** eficiência das cargas de 8 B no gfx1200; pressão de registos no kernel de pontuações; tocar sem querer
  no código FP16 (mitigação: funções novas; `git diff` dos kernels FP16 vazio; opcionalmente, ISA igual com
  `--save-temps`).
- **Testes obrigatórios:** T1, T2, T3 (secção 8); janela = sozinho ao bit; contextos até 262 144; KV real de
  `--dump-kv` (32K, todas as 11 camadas).
- **Aceitação (porta A):** decode Q8 ≤ 1,35× o seu mínimo pela banda a ≥ 32K e mais rápido do que FP16 a ≥ 8K;
  a 2K, não mais de 10 % mais lento; atenção do prefill Q8 ≤ 1,05× a FP16 a ≥ 32K; T1–T3 cumpridos. Se falhar: rever
  a disposição ou parar aqui, sem custo para o motor.
- **Rollback:** código sem uso pelo motor; reverter.

### Fase 3 – Armazenamento e integração no Qwen35 (sem streaming)

- **Objetivo:** `--kv q8_0` funciona de ponta a ponta nos perfis 48K, 128K e 262K.
- **Ficheiros:** `qwen35.hpp`, `qwen35.cpp` (secção 4), `engine.cpp` (validação).
- **Alterações:** alocação e `plan_kv` por bytes de linha (o stage entra em `later`); stage de desquantização;
  `kv_budget`; `row_bytes`/`save_rows`/`load_rows`; despacho em `attn_store()` e `attn()` (MTP incluído); Q8 +
  `--kv-resident` ≠ `all` recusado.
- **Complexidade:** média.
- **Riscos:** os grafos gravados com tamanhos (`room()`): as variantes Q8 têm de ser graváveis como as FP16; contas
  de bytes em `ahead()`/`save_rows` (preparadas, mas só exercitadas na fase 7).
- **Testes obrigatórios:** arranque nos 3 perfis com a VRAM da tabela 5.1 (±1 MiB); greedy curto; `--force-tokens`;
  conversa estacionada e retomada (T9); `RESUME`; MTP ligado; FP16 ainda ao bit (T11).
- **Aceitação:** os testes acima; nenhum `GPU_FATAL` em 1 hora de uso misto.
- **Rollback:** sem `--kv q8_0` o código Q8 não corre; reverter.

### Fase 4 – Correção, qualidade e estabilidade

- **Objetivo:** a campanha da secção 8.
- **Ficheiros:** `tools/hip/kv_q8_compare.py` (novo), extensões de `kv_parity.py`, `kv_needles.py`, mocks de falha.
- **Complexidade:** média (sobretudo tempo de máquina: os 261K levam ~9 min de prefill cada).
- **Riscos:** confundir o erro do Q8 com o dos experts (mitigação: secção 8.2).
- **Testes obrigatórios:** T1–T11.
- **Aceitação:** todos os critérios; os casos amarelos (secção 8.3) discutidos e aceites por escrito.
- **Rollback:** não muda nada operacional.

### Fase 5 – Benchmarks e decisão — porta de decisão B

- **Objetivo:** decidir se o Q8 se oferece, e com que política de VRAM.
- **Trabalho:** a matriz da secção 6.2: 4 perfis (o streaming só em FP16 até à fase 7) × FP16 / Q8-P1 / Q8-P2.
- **Complexidade:** média (tempo de máquina).
- **Aceitação (porta B):** Q8 oferecido se, a 128K ou 262K, o decode for ≥ 10 % mais rápido ou a VRAM dedicada
  residente com o contexto cheio descer ≥ 1 GiB (P2), sem perder mais de 3 % de prefill; a 48K, sem perda > 3 % em
  nada. Se só a 128K/262K ganhar, o menu recomenda Q8 só nesses perfis.
- **Rollback:** nada a desfazer.

### Fase 6 – Launcher e configuração

- **Objetivo:** escolher o formato no arranque (secção 3.11).
- **Ficheiros:** `poly_profile.py`, `START-POLY-HIP.bat`, `app.js`, `prometheus.py`, `tools/hip/test_poly_profile_kv.py`.
- **Complexidade:** baixa a média (o `.bat` com `EnableDelayedExpansion` é o mais frágil).
- **Riscos:** um menu que bloqueie um arranque automático; uma configuração FP16 reescrita (mitigação: só se escrevem
  ficheiros `-q8`; teste que compara os FP16 antes e depois).
- **Testes obrigatórios:** testes Python sem GPU de `poly_profile.py` (sufixos, `--list`, reutilização, recusa
  stream+Q8); `START-POLY-HIP.bat check` com cada combinação de argumentos/`POLY_KV`; arranque sem consola (stdin
  nulo) não espera; tempo esgotado não grava a preferência; os ficheiros FP16 ficam iguais ao byte.
- **Aceitação:** todos; FP16 continua a omissão sem interação nenhuma.
- **Rollback:** reverter os dois ficheiros; apagar `strata-*-q8.json` e `last-kv.txt`.

### Fase 7 – Q8 com KV streaming (condicional)

- **Objetivo:** `262k-stream` com Q8 (VRAM ~825 MiB de KV e stage, RAM fixada 2 431 MiB).
- **Ficheiros:** `kernels.cu` (variantes `SPILL` Q8, `kv_dequant_kernel` sobre a RAM), `qwen35.cpp` (`ahead`,
  `before_head`/`after_head`, a verificação de RAM, levantar a recusa), `poly_profile.py`.
- **Complexidade:** média a alta.
- **Riscos:** os caminhos com mais casos (cópia antecipada, escrita dupla na RAM e na cópia).
- **Testes obrigatórios:** `kv_parity.py` Q8 todo na VRAM contra Q8 em streaming **ao bit** (a mesma garantia que o
  streaming FP16 dá hoje); needles a 261K; decode a 65K e 261K.
- **Aceitação:** paridade ao bit; decode a 261K não pior do que o streaming FP16.
- **Rollback:** voltar a recusar a combinação.

### Fase 8 – Integração operacional controlada

- **Objetivo:** o novo executável passa a ser o operacional, com FP16 por omissão.
- **Trabalho:** cópia de segurança `build-hip-poly-win\strata-poly-pre-q8.exe`; repetir T11 com o executável
  definitivo; gerar as configurações `-q8` só quando o utilizador escolhe; entradas `-q8` no provider do OpenCode
  (opcional); documentação em [docs/AMD_HIP.md](AMD_HIP.md); um período com Q8 só por escolha antes de qualquer
  recomendação no menu.
- **Complexidade:** baixa.
- **Riscos:** substituir o executável com o servidor ativo (fazê-lo parado).
- **Aceitação:** FP16 ao bit e com os tempos da fase 0; Q8 com os números da fase 5.
- **Rollback:** repor `strata-poly-pre-q8.exe`; as configurações FP16 nunca mudaram.

## 8. Validação e rollback

### 8.1 Testes e tolerâncias

| # | Teste | Critério | Porquê este limite |
|---|---|---|---|
| T1 | Quantização: códigos e escalas contra `quantize_row_q8_0_ref` do ggml (10⁶ blocos aleatórios + linhas reais de `--dump-kv`; blocos a zero; valores grandes) | iguais ao bit; erro por valor ≤ 0,562 d | limite analítico da secção 3.2 |
| T2 | Kernels Q8 de decode e prefill contra uma referência FP64 **sobre os valores desquantizados** | erro máximo ≤ 1e-4 × max\|saída\| a 262K; janela = sozinho ao bit | isola o kernel do formato: só a ordem das somas em float, que cresce com o número de linhas |
| T3 | Erro de quantização ao nível do kernel, KV real (32K, 11 camadas): saída da atenção Q8 contra FP16 | RMS relativo ≤ 1 % por camada; registar por camada e por profundidade | provisório; a camada pior diz se a K ou a V precisa de outro tratamento |
| T4 | *Teacher-forced* (`kv_q8_compare.py`: prompt lido em prefill, depois 512 tokens forçados em decode), mesmos experts, `--window 1`; prosa, código, conversa; 1K, 8K, 32K, 128K, 255K | **ΔNLL + 2 × EP ≤ +0,02 nats/token** ([T4-v2-2026-10-17](AMD_HIP_KV_Q8_T4_V2.md)); KL média ≤ 0,03 até 1K e ≤ 0,15 a ≥ 8K; top-1 igual ≥ 95 % até 1K e ≥ 90 % a ≥ 8K | v2 aprovada pelo utilizador em 17/10: limita a diferença incluindo 2 EP face à margem tolerada, em vez de exigir proximidade a zero. EP é descritivo nos tokens correlacionados. Mantém a margem absoluta, KL e top-1 derivados dos ensaios INT8/Q4 anteriores; resultados v1 são preservados |
| T5 | Greedy (`kv_parity.py --comparison-mode q8-relative --ref "--kv fp16" --test "--kv q8_0"`), 128 tokens, mesmos controlos | diagnóstico textual, com revisão funcional por caso; os limites antigos 90 % / gaps 0,1 e 0,2 ficam métricas legadas, sem decidir qualidade por igualdade de palavras | [Política comparativa](AMD_HIP_KV_Q8_PAIRED_TESTS.md): uma trajetória diferente não prova regressão funcional; integridade/execução inválida continua a impedir conclusões |
| T6 | Needles (`kv_needles.py`) a 8K, 32K, 65K, 131K, 261K | comparar a recuperação das mesmas notas por par Q8/FP16; reportar ambos corretos, ambos falham e discordâncias; o plano define os casos esperados | falhas partilhadas não são atribuídas a Q8; falhas só Q8 requerem revisão; os 24/24 originais ficam registados |
| T7 | Eficiência MTP e desempenho | reportar aceites/ofertados, delta em pontos e tempos comparáveis separadamente; −2 pontos fica diagnóstico legado, sem FAIL de qualidade; desempenho INCONCLUSIVE sem metrologia/séries comparáveis | [Política comparativa](AMD_HIP_KV_Q8_PAIRED_TESTS.md): texto e schedules influenciam a métrica; aceitação não é qualidade nem throughput |
| T8 | Código e ferramentas | validadores por braço mantidos; decisão funcional emparelhada por tarefa, destacando falhas só Q8 sem compensar por ganhos; API15/JSON ficam verificações separadas de integração | falhas partilhadas não demonstram regressão Q8; o limiar antigo Q8 ≥ FP16−1 fica legado; integrações inválidas e dados em falta nunca são aprovação |
| T9 | Reutilização e reconstrução | conversa estacionada e retomada, `RESUME`, e o mesmo prompt lido de novo: Q8 contra Q8 **ao bit** | a quantização é determinística e por linha |
| T10 | Fail-closed | falha injetada em `store_q8` e `attend_q8` (mock) → `ERR GPU_FATAL`, código 70, `/health` 503, reinício com `INFO kv=q8_0` | nenhuma queda silenciosa para FP16 |
| T11 | Não regressão do FP16 | logits ao bit contra o executável operacional (mesmos experts, MTP e `--window 1`); tempos ±2 % | regra do [AGENTS.md](../AGENTS.md): o caminho por omissão fica igual ao byte |

### 8.2 Isolar o erro do Q8 (determinismo)

A variabilidade entre processos já observada vem dos experts: um expert calculado na CPU arredonda de outra maneira
do que na VRAM, e o Q8 muda quantos experts cabem. Sem controlo, a comparação FP16/Q8 mede as duas coisas juntas.

1. **Os mesmos experts nos dois braços:** cópia do perfil de experts (só de leitura para o teste); `--expert-swaps 0`;
   `--vram-reserve-mib` do braço Q8 = o do FP16 + a VRAM libertada (secção 5.1), para a cache ficar do mesmo tamanho;
   confirmação pelo `STRATA_POLY_CACHE_HASH` e por `expert_slots`.
2. **Braço puro** (≤ 8K): `--no-gpu-experts`: todos os experts na CPU, logo a mesma conta nos dois braços.
3. `--window 1` para os logits (as janelas do MTP somam noutra ordem); o MTP mede-se à parte (T7).
4. **K e V em separado:** `--kv-k q8_0` só e `--kv-v q8_0` só, para atribuir o erro.
5. **Braço de ruído:** FP16 contra FP16 com reservas diferentes. Dá a escala do que os utilizadores já têm hoje entre
   arranques. Um Q8 cujo erro fique abaixo desse ruído não é distinguível em uso.
6. O mesmo executável para FP16 e Q8 nas comparações de qualidade; o executável operacional só entra em T11.

### 8.3 Leitura dos resultados

- **Verde:** todos os critérios.
- **Amarelo:** T4 acima dos limites absolutos mas abaixo do braço de ruído (5): discutir antes de avançar.
- **Vermelho:** T6 abaixo de 24/24, T9/T10/T11 falham, ou ΔNLL > +0,05: parar; experimentar "K FP16, V Q8" ou
  grupos de 64 antes de desistir.

### 8.4 Rollback

- O FP16 é a omissão e os seus kernels não mudam: desligar o Q8 é não pedir `--kv q8_0`.
- Executável de desenvolvimento em `build-hip-poly-q8-win\`; o operacional só é substituído na fase 8, com cópia
  `strata-poly-pre-q8.exe`.
- Configurações Q8 em ficheiros `-q8` à parte; apagá-los (e `last-kv.txt`) repõe o launcher.
- Cada fase é um commit (ou PR) separado e reversível; nenhuma muda modelos nem perfis FP16.

### 8.5 Resultado, classificação e impacto na campanha (T5 não bloqueante)

Esta secção conserva a política e os estados históricos de T5. A decisão das campanhas novas
usa [a avaliação comparativa](AMD_HIP_KV_Q8_PAIRED_TESTS.md); os quatro fingerprints revistos
e a aprovação manual do ponto funcional mantêm-se. Não regenerar relatórios históricos
com os novos critérios para os apresentar como resultados originais.

O T5 foi o único teste que parou a campanha de CP2: uma separação com diferença 0,3602 (> 0,2) no prompt
`cp2-t5-07-portuguese`, **determinista** — a confirmação isolada tem os logits FP16 e Q8 iguais ao bit nos 128
passos ([cp2-t5-incident.json](../bench/results/2026-10-14-hip-kv-q8-phase4/cp2-t5-incident.json)). Parar tudo
por causa disso misturava três coisas diferentes. Passam a ser três, separadas:

| Conceito | Valores | O que diz |
|---|---|---|
| **Resultado** | PASS / FAIL / NOT_RUN / ERROR | o critério estrito do T5 (0,1 / 0,2), **sem o mover** |
| **Classificação** | NONE / KNOWN_DIVERGENCE / NEW_DIVERGENCE / NONFINITE / INTEGRITY_ERROR / EXEC_ERROR / CONFIG_MISMATCH | o que é a divergência, ou porque não há resultado utilizável |
| **Impacto (gate)** | BLOCKING / NON_BLOCKING | o efeito na campanha e, mais tarde, na promoção |

Regras (`bench/results/2026-10-14-hip-kv-q8-phase4/p4_t5_policy.py`):

- PASS → NON_BLOCKING.
- FAIL + KNOWN_DIVERGENCE → NON_BLOCKING. **O T5 continua FAIL**: uma divergência conhecida não baixa o critério.
- FAIL + NEW_DIVERGENCE → NON_BLOCKING, mas registada à parte e com revisão obrigatória antes da promoção. Não se
  confunde com a conhecida.
- NONFINITE, INTEGRITY_ERROR, CONFIG_MISMATCH e EXEC_ERROR → BLOCKING. Uma saída não finita, um erro de execução
  ou uma inconsistência de configuração continuam a parar a campanha (o fail-closed mantém-se).
- NOT_RUN → BLOCKING: um teste que não correu nunca é declarado PASS.

A divergência conhecida identifica-se por uma **impressão digital** — caso, protocolo, referências e a posição
onde as duas variantes se separam — e nunca pela igualdade exata de um `ref_gap` em vírgula flutuante; os números
observados ficam preservados à parte (`p4_known_divergences.json`). Assim a mesma divergência reconhece-se mesmo
que um valor seja impresso com outro arredondamento, e uma separação na mesma posição mas noutro caso ou protocolo
é tratada como nova.

Os nomes dos testes não mudam: **T7 = aceitação do MTP, T8 = código e ferramentas, T9 = reutilização e
reconstrução** (§8.1). "Qualidade funcional" e "estabilidade operacional" são nomes de **portão** que agrupam
esses testes, não testes T novos nem redefinições de T7/T8/T9.

Relatório: `python p4_promotion.py` escreve `PROMOTION.md` (ficheiro novo; não reescreve `cp2-results.md` nem
qualquer artefacto histórico) e distingue resultados históricos de campanha nova. Estado atual:

```
T5-Strict: FAIL
T5-Classification: KNOWN_DIVERGENCE
T5-Gate: NON_BLOCKING
Functional Quality: PENDING
Operational Stability: PENDING
Q8 Promotion: NOT YET APPROVED
```

T7 e T9 da campanha existente podem ser corridos depois desta alteração, sem voltar a parar no T5.

## 9. Riscos e limitações

| # | Risco | Efeito | Mitigação |
|---|---|---|---|
| R1 | Precisão em contexto longo (recuperação com atenção muito concentrada) | notas perdidas a 131K/261K | T6 obrigatório; variante K FP16; grupos de 32 |
| R2 | Kernel Q8 de decode menos eficiente (cargas de 8 B) | ganho menor ou perda a contextos curtos | porta A; ajustar lanes por linha |
| R3 | Regressão do FP16 | o perfil atual fica pior | kernels FP16 intactos; T11 em todas as fases |
| R4 | Mais VRAM residente no arranque com P1 (secção 5.3) | mais despromoção WDDM em conversas curtas | medir PDH; política P2 |
| R5 | Caminho genérico a 262K (secção 1.4) | Q8 precisa das variantes genéricas; ganhos a 262K podem vir mascarados | variantes genéricas na fase 2; medir com `STRATA_POLY_TRACE=2` |
| R6 | Streaming Q8 | complexidade | fase 7 condicional; recusa explícita até lá |
| R7 | Confundir o erro do Q8 com o dos experts | conclusões erradas | secção 8.2 |
| R8 | `kernels.cu` também é CUDA | a compilação NVIDIA pode partir | compilar CUDA noutra máquina ou CI antes de um PR; código Q8 sem intrínsecos AMD |
| R9 | O `INFO` muda no caminho por omissão | protocolo diferente ao byte | chave aditiva, lida genericamente; decisão D6 |
| R10 | Um cliente pede o nome FP16 com `switch_on_request` | o servidor troca de formato | documentar; a troca é pedida e registada |
| R11 | Os ganhos de 48K estão dentro da variação de arranque | medições inconclusivas | reserva fixa nas medições |

Limitações desta fase: nenhuma medição nova; as estimativas de memória são exatas pela geometria, mas as de
desempenho são limites pela banda e não velocidades; a qualidade de Q8 neste modelo não foi medida (a única evidência
é a do INT8 noutro modelo); a residência física (secção 5.3) depende das experiências E2 da investigação de VRAM.

## 10. Recomendação

**Avançar com a arquitetura B**, Q8_0 com blocos de 32 e disposição por cabeça, pela ordem da secção 7, parando nas
duas portas:
- **Porta A** (depois dos kernels isolados): só se integra se o decode Q8 for mais rápido do que FP16 a ≥ 8K e o erro
  em KV real estiver dentro de T1–T3.
- **Porta B** (depois dos benchmarks): só se oferece no launcher se houver ganho medido, no mínimo a 128K/262K.

O valor esperado está nos perfis longos: a 262K, ~2,6 GiB de VRAM e metade dos bytes da atenção; a 128K, ~1,3 GiB.
A 48K o ganho é pequeno (~0,5 GiB, dentro da variação entre arranques). O FP16 fica a omissão e igual ao bit.

### Decisões a aprovar

| # | Decisão | Recomendação |
|---|---|---|
| D1 | Formato | Q8_0, blocos de 32, escala FP16, disposição por cabeça (alternativa: grupos de 64) |
| D2 | Opção | `--kv fp16\|q8_0`; `INFO kv=` |
| D3 | VRAM libertada | P1 (experts) por omissão no Q8; P2 medida na fase 5 e oferecida se ganhar em estabilidade |
| D4 | Preferência | a escolha no menu é guardada e passa a omissão do menu; o tempo esgotado não grava; arranques com argumentos não mostram o menu |
| D5 | Streaming | fase 7, condicional; até lá o motor recusa Q8 com `--kv-resident` |
| D6 | `kv=fp16` no `INFO` do caminho por omissão | sim (chave aditiva; a aritmética não muda) |
| D7 | KV do bloco MTP | Q8 como as outras camadas; T7 decide |

## O que não foi feito

- Nenhuma alteração de código, compilação, descarga, nem mudança de modelos, perfis, launchers ou configurações.
- Nenhuma medição nova: o servidor operacional não foi interrompido. Os números marcados **(medido)** vêm dos
  relatórios e logs citados.
- A observação do caminho genérico a 262K (secção 1.4) vem só da leitura do código.
- Não se verificou a disponibilidade de GEMM FP8/int8 no hipBLAS(Lt) do Windows para o gfx1200: não é precisa na
  arquitetura recomendada.
