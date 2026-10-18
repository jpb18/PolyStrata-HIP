> Historical measurement report. Use the [current setup instructions](../../README.md). Personal paths were generalized for publication; measurements were preserved. Raw campaign links may require the local evidence archive.

# PolyStrata HIP: profiling e otimização do decode do Qwen3.6-35B-A3B UD-Q3_K_M

Ramo `hip-poly`, sobre `7023db0c` (fim da migração para o Qwen3.6 Q3, [AMD_QWEN36_Q3_MIGRATION.md](AMD_QWEN36_Q3_MIGRATION.md)).
**Não foi feito commit nem push**: as alterações estão na cópia de trabalho (secção 9).

**Onde foi medido:** este PC. RX 9060 XT 16 GB (gfx1200, 32 CUs), Ryzen 5 5600X (6 núcleos), 31,9 GB de RAM,
Windows 11 Pro, ROCm 10.2.0a20260930 (TheRock), em uso normal: outros programas ocupavam ~3,1–4,8 GB de VRAM
(o `RadeonSoftware`, browsers, o ChatGPT e o Claude abertos). Dados, scripts e logs:
`bench/results/2026-10-12-hip-decode-opt/`.

**Referência (antes):** o binário do início desta fase, guardado como `build-hip-poly-win/strata-poly-base.exe`
(o da migração, o mesmo que mediu 91,0 / 86,2 / 70,6 / 62,4 tok/s a 2K / 8K / 31K / 45K em
[AMD_EMPERO_QWEN38.md](AMD_EMPERO_QWEN38.md)), medido de novo no mesmo dia, alternado com o novo.

---

## 1. Resumo

As três rondas desta fase mediram, a quente com MTP, 92,43 → 129,21 tok/s a 2K (+39,79%),
84,72 → 110,57 a 8K (+30,51%), 69,88 → 102,79 a 31K (+47,10%) e 63,62 → 94,62 a 45K (+48,73%).
São resultados históricos do método abaixo, com o perfil de especialistas operacional a atualizar os seus contadores.

A validação final foi interrompida por um WATCHDOG AMD, acompanhado de uma geração B-31K incoerente.
O teste `--no-draft` da cadeia anterior não chegou a executar por erro de argumentos. Não se declara este
binário pronto para produção com base nestas medições. A continuação, a investigação do incidente, os testes
completados e os A/B do executável identificado por hash estão em
[AMD_HIP_DECODE_VALIDATION.md](AMD_HIP_DECODE_VALIDATION.md).

## 2. Método

- **Benchmark de decode e prefill:** `tools/hip/bench_compare.py`, o método das fases anteriores. Os mesmos prompts de
  ~2K, 8K, 31K e 45K tokens (`2026-10-10-hip-poly-vs-llama-vulkan/prompts.json`), contexto 49 152, 6 threads, 128
  tokens greedy. Cada corrida carrega o motor, lê o conjunto A (a frio) e o conjunto B (a quente). Houve um
  aquecimento não contado de cada binário, e depois 3 rondas alternadas: base, novo, novo `--window 1`, base
  `--window 1`. Os valores são a mediana das 3 (`scripts/bench_chain.sh`, `bench.json`, `bench-summary.md`).
- **Decisões intermédias:** A/B alternados com 3 rondas (`ab-*.json`) através de `tools/hip/decode_trace.py`
  (novo, secção 3.1). O ruído entre processos é de ±7 %: uma corrida isolada não chega para decidir nada.
- **Precisão:**
  - `tools/hip/poly_llama_parity.py` contra as referências do llama.cpp HIP b11160 da migração (A: `--cpu-moe`,
    B: `--n-cpu-moe 22`), com os mesmos 5 prompts (22 a 35 575 tokens) e o critério das fases anteriores
    (`scripts/parity_chain.sh`, `parity-*.json`);
  - verificações bit a bit de cada kernel novo no `poly_kernel_bench`;
  - `STRATA_POLY_CARD_ROUTE_CHECK=1`.
- **Profiling:** dois diagnósticos novos, opt-in e sem efeito quando desligados (secção 3.1).

## 3. Profiling: onde estava o tempo

### 3.1 Instrumentação nova

| O quê | Como se liga | O que mede |
| --- | --- | --- |
| Traço por rondas (`src/poly/main.cpp`, `Trace`) | `STRATA_POLY_TRACE=1` | Depois de cada resposta, na stderr: palpites do MTP (chamadas, ms), `eval` de um token, janelas de verificação por tamanho (ms, tokens escritos), `keep`, amostragem, espera do host pelo routing, tempo dos especialistas na CPU. Cada parte é uma chamada que já espera pela placa: medir não muda o que corre |
| Tempo na placa por parte (`src/poly/engine.cpp`, `GpuTrace`) | `STRATA_POLY_TRACE=2` | Eventos HIP entre as partes de cada camada (cabeça delta net, cabeça de atenção, especialista partilhado, especialistas, cauda, logits) e de cada palpite (entrada, atenção, MoE, cabeça de saída), por tamanho de chamada; e quanto tempo a placa esperou pelos especialistas da CPU (`wait_add`) |
| `tools/hip/decode_trace.py` | — | Um processo, um aquecimento, os prompts pedidos com o traço; guarda tokens, velocidades e o traço em JSON |
| `poly_kernel_bench attn-decode` | — | A atenção de uma chamada de 1, 2, 3, 4 e 8 tokens a 2K–45K, contra o mínimo que a banda permite, e a verificação de que cada token dá os mesmos bits na janela e sozinho |
| `poly_kernel_bench cols` | — | Se os produtos de vários tokens de uma vez dão a cada token os bits que tem sozinho: Q8_0, Q6_K, router F32, delta net |

Custo dos eventos: no Windows cada `hipEventRecord` entre kernels custa ~40 µs, por isso o `STRATA_POLY_TRACE=2`
infla os tempos absolutos (um token passa de ~13 para ~19 ms). Serve para comparar partes e tamanhos, não para
medir velocidade.

### 3.2 O ponto de partida (binário base, modo por omissão com MTP, B-2k, um processo a quente)

`STRATA_POLY_TRACE=1` (`trace.json`, `base-trace`): 128 tokens em 1 641 ms (12,8 ms/token), 55 rondas.

| Parte | ms | % | Por chamada |
| --- | ---: | ---: | --- |
| Janelas de verificação (2–4 tokens) | 1 150 | 70 % | janela de 3 tokens: 27,4 ms; de 4: 38,6 ms |
| Palpites do MTP | 239 | 15 % | 2,42 ms |
| `eval` de um token | 244 | 15 % | 14,3 ms |
| Amostragem, `keep`, o resto | 8 | 0,5 % | |
| **Do total: o host à espera do router** | **1 043** | **64 %** | |

O decode de um token (`--window 1`) estava perto do limite de banda; as janelas não.

| Por token, a 2K | Bytes lidos | Tempo à banda medida (~280 GB/s) | Medido |
| --- | ---: | ---: | ---: |
| Pesos densos (Q8_0) + cabeça de saída (Q6_K, 417 MB) | ~1,95 GB | ~7,0 ms | |
| 8 especialistas × 40 camadas | ~0,43 GB | ~1,5 ms | |
| **Um token** | ~2,4 GB | **~8,5 ms** | 12,6–13,2 ms |
| **Janela de 4** (densos uma vez, até 32 especialistas por camada) | ~3,1 GB | **~11 ms** | **38,6 ms** |

Uma janela de 4 tokens custava 3× um token. A causa: com mais de um token por chamada, cada uma das 40 camadas MoE
esperava no host pelo router (~60–190 µs de ida e volta no Windows, medido pelo `hip_poly_probe`), encaminhava no
host, e só depois lançava os especialistas. A placa ficava parada entre camadas. A fase 2 tinha tirado esta
espera aos tokens sozinhos; as janelas do MTP continuavam com ela (recomendação nº 1 de
[AMD_QWEN36_IMPLEMENTATION.md](AMD_QWEN36_IMPLEMENTATION.md), secção 12).

### 3.3 Atenção a contextos longos

`poly_kernel_bench attn-decode` (`attn-decode-before.txt`), µs por camada de atenção, cache na VRAM:

| Contexto | 1 token | 4 tokens | 8 tokens | Ler K e V uma vez |
| --- | ---: | ---: | ---: | ---: |
| 2K | 31 | 84 | 152 | 15 |
| 8K | 106 | 315 | 606 | 60 |
| 31K | 467 | 1 271 | 2 399 | 228 |
| 45K | 648 | 1 893 | 3 612 | 331 |

O custo crescia com os tokens da chamada. Separado por kernel, a 45K:

| Kernel | 1 token | 4 tokens | 8 tokens |
| --- | ---: | ---: | ---: |
| pontuações | 256 | 819 | 1 606 |
| softmax | 70 | 109 | 167 |
| valores | 379 | 937 | 1 733 |

- **Pontuações:** liam a K uma vez, mas reliam os 8 KiB de queries do grupo para cada linha de 512 bytes da cache, e
  somavam os produtos de cada cabeça entre lanes com 5 shuffles. A 45K, isso custava mais do que ler a K.
- **Valores:** cada bloco tinha as queries de um só token, por isso uma janela de n tokens lia a V n vezes, e passava
  os pesos de cada linha pela warp, um shuffle por query. Um token sozinho tinha só 16 blocos para os 32 CUs.

Com a nova velocidade das janelas, a 45K a atenção passou a ser ~40 % do tempo de uma janela
(`STRATA_POLY_TRACE=2`, `gtrace-default`).

### 3.4 Palpites do MTP

`STRATA_POLY_TRACE=2`: um palpite custava ~2,3 ms na placa, dos quais **1,46–1,56 ms são a cabeça de saída**:
248 320 linhas de Q6_K, 417 MB lidos para escolher um token. O MoE do bloco encaminhava no host (0,49 ms com a ida e
volta).

### 3.5 Lançamentos de kernels

`hip_poly_probe`: 0,57 µs por lançamento no host; 1,15 µs por kernel minúsculo em sequência; 35 kernels
dependentes + espera: 127 µs, com ou sem graph (3,6 µs cada). Uma chamada do Qwen3.6 tem ~1 250 kernels (30–35
por camada). São 2–4 ms por chamada que não dependem dos bytes lidos.

## 4. O que ficou implementado

- Routing HIP das janelas de até oito tokens, com oito slots por token e até 64 slots nos kernels MMVQ.
- Routing do bloco draft na GPU, com a soma dos especialistas da CPU entregue por sequência.
- Atenção Qwen com queries reutilizadas e soma dos valores em partes; o softmax continua causal.
- Cabeça de saída do draft sobre 65 536 IDs, mantendo a verificação no vocabulário completo.
- Traços opt-in por ronda e por parte na GPU; controlos para voltar aos caminhos anteriores.

As dimensões, publicação CPU/GPU, timeout e controlo da cache foram auditados na continuação. Não foram
acrescentadas novas otimizações nem revertidas indiscriminadamente as existentes.

## 5. Correção e incidentes

O log confirma 35 309 routings sem divergências de seleção e diferença máxima de pesos 1,19e-7. A corrida
B-31K incoerente tinha **55,08 tok/s**, 3/26 palpites corretos, e terminou às 18:42:47, segundos depois do
`LiveKernelEvent 141` das 18:42. O routing desses mesmos scores estar certo não garante que as ativações
anteriores, os pesos ou a KV tenham sobrevivido ao reset. A causa inicial do WATCHDOG ainda não foi estabelecida.

As duas repetições imediatas foram normais. Na continuação, 40 respostas de 256 tokens, incluindo 20 B-31K,
também foram normais; 287 044 routings verificados não divergiram. A comparação direta sem draft produziu os
mesmos 64 tokens nos prompts curto, de código e de 35 575 tokens, com todos os logits finitos. Há diferenças
numéricas e variações de redação entre estados da cache, inclusive no binário base.

Ver os logs, tolerâncias, limitações e as três falhas preexistentes da suite no relatório de continuação.

## 6. Resultados históricos

`bench/results/2026-10-12-hip-decode-opt/bench.json` e `bench-summary.md` guardam todas as corridas, medianas,
intervalos, prefill, TTFT, picos de RAM/VRAM e MTP. As experiências `ab-draftcard.json` e `ab-dvocab.json` são
traços intermédios e não se somam aos ganhos do benchmark. `STRATA_POLY_TRACE=2` tem custo de eventos medido e
não serve para uma comparação de velocidade absoluta.

O último benchmark terminou antes da compilação final de 18:39, sem guardar hash por corrida. Os A/B finais
da continuação usam o mesmo workload e ferramenta, mas fixam uma cópia dos contadores de especialistas antes
de cada processo e guardam os hashes. As campanhas ficam separadas.

## 7. Estado de entrega

As otimizações ficam preservadas e o binário base continua disponível. O relatório de continuação estabelece
o que foi medido e o risco residual depois de um timeout do driver. Não há commit, push ou alteração do driver.

## 8. Controlos existentes

| Controlo | Efeito |
| --- | --- |
| `STRATA_POLY_CARD_WINDOW=0` | Janelas encaminhadas no host |
| `STRATA_POLY_CARD_DRAFT=0` | Bloco draft encaminhado no host |
| `STRATA_POLY_CARD_ROUTE=0` | Routing no host também para tokens individuais |
| `STRATA_POLY_ATTN=0` | Atenção anterior |
| `STRATA_POLY_ATTN_FIT=1` | Experimental (fase 2.5 da KV Q8): as janelas usam os kernels rápidos enquanto as pontuações e as somas parciais cabem no scratch real; ver [bench/results/2026-10-13-hip-attn-dispatch](../../bench/results/2026-10-13-hip-attn-dispatch/README.md) |
| `STRATA_POLY_DRAFT_VOCAB=0` | Vocabulário completo nos palpites do draft |
| `--window 1` | Sem janelas de palpites |
| `--no-draft` | Não executa o bloco draft; os palpites por repetição continuam disponíveis |

## 9. Âmbito dos ficheiros

O decode alterou `src/poly/engine.cpp`, `engine.hpp`, `kernels.cu`, `main.cpp`, `qwen35.hpp`, `runtime.hpp`,
`src/kernels/cuda/native_mmvq.cu`, `iq_kernels.cu` e os dois headers de kernels, além de `kernel_bench.cu` e
`tools/hip/decode_trace.py`. A cópia de trabalho inclui ainda alterações de experiências anteriores com Empero
e outros modelos; não pertencem todas a esta fase. O diff inicial e o manifesto da continuação preservam essa
distinção. Só HIP/gfx1200 foi executado neste PC; não há validação CUDA/SYCL destas alterações.
