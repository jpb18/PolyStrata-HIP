# Port HIP do `strata-poly` — fase 3: KV streaming para contextos longos

Ramo `hip-poly`, commits sobre `d947977c` (`git log d947977c..hip-poly`). Não foi feito push nem aberto PR.
Fases anteriores: [AMD_PORT_IMPLEMENTATION.md](AMD_PORT_IMPLEMENTATION.md), [AMD_PORT_PHASE2.md](AMD_PORT_PHASE2.md).

**Onde foi medido:** este PC. RX 9060 XT 16 GB (gfx1200, Adrenalin 32.0.31041.1004), Ryzen 5 5600X (6 núcleos),
31,9 GB de RAM, Windows 11 Pro, ROCm 10.2.0a20260930 (TheRock). Modelo: `Qwen3.5-35B-A3B-Q3_K_M.gguf` (Unsloth), o
mesmo das fases 1 e 2 (contexto nativo do ficheiro: 262 144). O PC estava em uso normal: outros programas ocupavam
4,2–4,4 GB de VRAM (mais do que os 2,4–3,0 GB da fase 2) e deixavam 18–24 GB de RAM disponível antes de cada
corrida (registado por corrida nos JSON). Dados: `bench/results/2026-10-11-hip-poly-kv-streaming/`.

---

## 1. Resumo executivo

- **Implementado e validado no hardware: KV streaming opt-in para Qwen3.5/3.6** (`--kv-resident N|auto|all`). As
  primeiras N posições de cada camada de atenção ficam na VRAM e as restantes em RAM fixada. Os mesmos kernels leem
  as duas partes, e a parte na RAM chega à VRAM por DMA uma camada à frente. O resultado é **igual bit a bit** ao da
  KV toda na VRAM: 32/32 passos com os 248 320 logits iguais, em prompts de 8K e 31K, com especialistas na CPU.
- **Critério mínimo cumprido: 128K reais com KV streaming na RX 9060 XT.** Documentos de 130 900 tokens, com 81 920
  posições por camada na RAM (1,6 GiB), e 3 notas no início, no meio e no fim: **18 de 18 encontradas** em 3
  repetições, mais a pergunta pelas três ao mesmo tempo. Sem out-of-memory e sem aumento da paginação. Pela API
  OpenAI do servidor, um pedido de 108 264 tokens encontrou a nota do meio.
- **Segunda meta cumprida: 261 000 tokens** (o contexto nativo do ficheiro é 262 144), com streaming (4,1 GiB da KV
  na RAM) e com a KV toda na VRAM: 12 de 12 notas encontradas em cada modo, em 2 repetições.
- **A conclusão mais importante:** neste modelo a atenção é **densa** (cada token lê todas as posições), e a KV de
  262K em fp16 (5 GiB) **cabe na VRAM** de 16 GB. Uma posição na RAM custa ~24× uma na VRAM em cada token: 14,2 GB/s
  de PCIe contra 300 GB/s. Por isso:
  - **a KV toda na VRAM é a configuração mais rápida em todos os comprimentos acima de ~32K**. Decode a 131K:
    29,4 tok/s residente contra 8,1 com streaming a partir de 48K; a 261K: 16,7 contra 2,8;
  - o streaming só ganha abaixo do corte (+8–11% a 8K, por ter mais especialistas na VRAM) e serve para quando a
    KV não cabe: placas mais pequenas, VRAM ocupada por outros programas, contextos maiores do que a VRAM;
  - `--kv-resident auto` escolhe por isso tudo na VRAM sempre que cabe, o que neste PC é sempre, até 262 144.
- **Antes desta fase, o caminho para 128K/262K já existia**: `--max-context` maior com a KV toda na VRAM. Faltava
  ter verificado que funcionava com contexto realmente preenchido, e foi isso que se fez: 261 000 tokens, decode de
  16,7 tok/s, prefill de 477 tok/s (9 min).
- **Sem regressão no caminho por omissão:** logits iguais bit a bit ao binário anterior (`d947977c`). Decode e
  prefill a 2K e 31K iguais dentro do ruído (3 corridas alternadas de cada binário).
- **Duas regressões do próprio streaming encontradas e corrigidas:** a leitura zero-copy (6,6–7,3 GB/s no caminho
  crítico) foi substituída pelo pré-carregamento por DMA (64K: 13,4 → 25,2 tok/s; 131K: 3,5 → 8,1); e as esperas
  entre streams em cabeças que não tocam na RAM custavam 5–12% (corrigido: 8K 56,9 → 65,0 tok/s).

---

## 2. O que se mediu antes de escolher a arquitetura

### 2.1 O modelo

O Qwen3.5-35B-A3B tem 40 camadas: 30 de **gated delta net** (estado recorrente de tamanho fixo, 62 MiB no total,
independente do contexto) e **10 de atenção completa** (16 cabeças de consulta, 2 de chave/valor, 256 dimensões,
fp16). Só estas 10 têm cache KV: 2 KiB por posição e camada, **20 KiB por posição** no total.

| Contexto | KV completa (fp16) |
| ---: | ---: |
| 49 152 | 0,94 GiB |
| 131 072 | 2,50 GiB |
| 262 144 | 5,00 GiB |

O estado recorrente não é uma cache KV: não cresce com o contexto, fica sempre na VRAM e não foi tocado (os
checkpoints da conversa continuam a guardá-lo como antes).

### 2.2 O motor atual já carregava 128K e 262K

Antes de mudar código: `--max-context 131072` e `262144` carregam e geram com a KV toda na VRAM. O custo é a cache
de especialistas, que fica com o que sobra (prompt curto, `--window 1`, mesmo processo de medição):

| Contexto | Especialistas na VRAM | Decode (prompt de 6 tokens) |
| ---: | ---: | ---: |
| 49 152 | 6 709 (8,5 GiB) | 39,9 tok/s (1.ª carga, ficheiro frio) |
| 131 072 | 5 480 (7,0 GiB) | 65,6 tok/s |
| 262 144 | 3 517 (4,5 GiB) | 28,9 tok/s |

Em 262K as tokens geradas diferiam das de 48K na 2.ª token. Verificado: com todos os especialistas na CPU
(`--no-gpu-experts`) 48K e 262K dão probabilidades iguais; com especialistas na GPU é um quase-empate (0,0455 /
0,0410) num prompt sem template, onde a cache de especialistas diferente decide. Não é um erro da atenção.

### 2.3 Largura de banda (`hip_poly_probe kv`)

| Leitura de 512 MiB | GB/s |
| --- | ---: |
| VRAM, por kernel | 300 |
| Memória fixada (RAM), por kernel, zero-copy (16, 8 ou 4 bytes por thread) | 12,6 |
| Memória fixada → VRAM, copy engine (peças de 1 a 512 MiB) | 13,9–14,2 |

O Windows fixou 7 GiB em peças de 1 GiB sem recusar (1,7 s).

### 2.4 A consequência para o desenho

A atenção do Qwen3.5 é **densa**: cada token lê **todas** as posições da KV, em todas as 10 camadas. Ao contrário
da atenção esparsa (QSA) do motor Strata original, não há blocos "frios" que fiquem na RAM sem custo:

- uma posição na RAM custa 20 KiB pelo PCIe **por token**: a 12,6 GB/s, 1,6 µs, contra 0,07 µs na VRAM (~24×);
- um especialista que a cache de especialistas perde custa só quando é escolhido (8 de 256 por camada), e é
  calculado na CPU.

Por byte de VRAM, uma posição da KV é lida mais vezes do que qualquer especialista que a cache dispensa para lhe
dar lugar. **Mover a KV para a RAM só pode tornar o decode mais rápido enquanto a conversa não chega às posições
na RAM** (há mais especialistas na VRAM: +3–11% medido abaixo dos 32K, secção 9); a partir daí é sempre mais lento.
Serve quando a KV não cabe, ou para reservar um contexto máximo grande sem pagar por ele nas conversas curtas. Foi isso que se implementou: um mecanismo de capacidade, opt-in, que dá os
mesmos resultados bit a bit, com o modo totalmente residente como omissão e como `auto` sempre que cabe.

---

## 3. Arquitetura implementada

### 3.1 Residência parcial "prefixo na VRAM"

Com `--kv-resident N` (N < contexto), cada camada de atenção guarda as posições `[0, N)` na VRAM, como antes, e
as posições `[N, contexto)` em **memória fixada mapeada** (`cudaHostAlloc(..., Mapped)`, ponteiro de dispositivo
por `cudaHostGetDevicePointer`), na mesma disposição (linha a linha, `n_kv × head_dim` halves por linha).

- A cópia na RAM é **a autoritativa** para as posições ≥ N (como no Strata original, onde a cópia no host é a
  autoritativa); não há duas cópias que possam divergir.
- Os endereços são fixos desde a carga: os graphs gravados (uma gravação por camada e por tamanho de sala) podem
  conter estes ponteiros e ser repetidos em qualquer posição.
- Nenhuma posição é descartada nem resumida: a atenção lê sempre todas as posições `[0, pos]`, exatamente como com
  a KV na VRAM (`attn_rows()` não mudou).

### 3.2 Decode e janelas de tokens adivinhados (chamadas de < 32 tokens)

Duas formas, ambas com os **mesmos kernels e a mesma ordem das somas** (resultado bit a bit igual ao residente):

1. **Pré-carregamento por DMA (omissão quando há streaming).** Um buffer de staging na VRAM com o tamanho da parte
   na RAM de **uma** camada. Depois da cabeça de cada camada (`Engine::after_head`, fora de qualquer gravação), o
   motor enfileira num stream lateral a cópia da parte na RAM da **próxima** camada de atenção (copy engine), a
   seguir a um evento do stream principal; antes da cabeça de atenção (`before_head`) o stream principal espera o
   evento da cópia. A cópia sobrepõe-se às 3 camadas delta-net e aos MoE que há entre duas camadas de atenção, e
   a da 1.ª camada de atenção da chamada seguinte sobrepõe-se ao fim desta (logits, amostragem).
   - A token nova é escrita na RAM **e** no staging (`attn_store`).
   - Se o staging não for da camada, ou lhe faltarem linhas, a cópia é feita na hora (contada como "tardia").
   - Qualquer escrita na RAM fora destes ganchos (prefill, leitura de prompt por `read()`, `load_rows` de uma
     conversa estacionada) invalida o staging.
   - Todas as cabeças de poucas tokens usam o staging, em qualquer posição: uma gravação que lesse a RAM
     diretamente não manteria o staging atualizado. Uma cabeça cujas linhas estão todas abaixo do corte não espera
     pela cópia (não lê nem escreve o staging).
2. **Zero-copy** (`STRATA_POLY_KV_AHEAD=0`): os kernels `attn_score` e `attn_value` leem as linhas ≥ N pelo
   ponteiro de dispositivo da memória fixada, no caminho crítico (6,3–7,3 GB/s efetivos, secção 9.3).

Os kernels foram tornados templates num `bool SPILL`: com a KV toda na VRAM o código é o de antes.

### 3.3 Prefill (chamadas de ≥ 32 tokens)

`attend_batch` já fazia atenção por blocos de linhas com softmax online (máximo e soma por token e cabeça,
reescalamento dos blocos anteriores). Com streaming, um bloco que chega a posições ≥ N é primeiro copiado inteiro
(parte da VRAM e parte da RAM, por um kernel `kv_gather` de 16 bytes por thread) para um stage de VRAM, e os
produtos do hipBLAS leem o stage. **A partição em blocos é a mesma** que com a KV na VRAM, por isso o resultado é o
mesmo bit a bit. O stage tem o tamanho do maior bloco possível (`attend_batch_block()`): 42 MiB.

### 3.4 Modo automático e limites

- `--kv-resident all` (omissão, igual a não passar nada): a alocação e os kernels de antes.
- `--kv-resident auto`: todas as posições na VRAM se couberem ao lado dos pesos densos, do scratch (≈1,5 MiB por
  token do lote de prompt), do stage, das gravações e da reserva; senão, as que couberem (múltiplo de 256). Neste
  PC escolhe **sempre tudo na VRAM** até 262 144 (verificado em 48K e 262K).
- `--kv-ram-mib N`: o máximo de RAM que a parte fixada pode ocupar. Por omissão, o que deixa
  `--kv-ram-floor-mib` (4 096) de RAM física disponível no momento da alocação.
- Se a parte na RAM não couber, o motor **recusa na carga** com uma mensagem que diz o contexto que cabe com essas
  definições, em vez de deixar o Windows paginar.

### 3.5 Relatório de memória na carga

Cada carga escreve onde ficou a memória (`Engine::budget()`), por exemplo, contexto 262 144, `--kv-resident 49152`:

```
strata-poly: KV streaming: positions 0 to 49151 of the attention caches in VRAM (0.94 GiB), 49152 to 262143 in pinned RAM (4.06 GiB), a layer's of them copied to VRAM ahead of its attention
strata-poly: VRAM MiB: dense weights 1910, expert cache 7947, KV 960 (49152 positions), KV stage 458, recurrent state 251, scratch and the rest 810; 1027 free.  Pinned RAM MiB: KV 4160 (212992 positions), the rest 1078; RAM available 4827 MiB
```

e no fim do processo as cópias feitas (`... copies of a layer's rows in RAM to VRAM, ... made when the head came,
... GB copied`).

---

## 4. Componentes do Strata original: reutilizados e adaptados

O motor `strata` (Qwen3.8, `src/core/layer.cpp`, `include/strata/kernels/kv_stream.hpp`,
`src/kernels/cuda/kv_stream.cu`, `docs/KV_PREFETCH.md`) faz KV streaming para as camadas **QSA**, que são
esparsas: um indexador escolhe blocos de 4 células, e só esses são lidos.

| Componente do original | No `strata-poly` |
| --- | --- |
| Cópia autoritativa da KV em memória fixada mapeada, com a mesma disposição da cache residente | **Reutilizado o desenho** (`hmalloc_mapped`, disposição linha a linha do `strata-poly`) |
| Escritores escrevem sempre a cópia do host (e a VRAM só se residente), nada a invalidar depois de rascunhos rejeitados nem ao retomar conversas | **Reutilizado o princípio**: a token nova vai para a RAM; o staging do pré-carregamento é escrito também e invalidado quando a RAM muda por outro caminho |
| Staging de uma camada da cópia do host para a VRAM no prompt (`kv_stage_from_host`) | **Adaptado**: em vez da camada inteira, só o bloco de linhas da atenção por blocos que chega à RAM (`kv_gather`), mantendo a partição |
| Pré-carregar a camada seguinte num stream próprio (`STRATA_KV_PREFETCH`, `docs/KV_PREFETCH.md`, no prefill) | **Adaptado ao decode**: a parte na RAM da próxima camada de atenção, com eventos de "usado" e "pronto", fora dos graphs |
| Mapa de residência (page table, CLOCK, `kv_stream_resolve` dentro dos graphs) | **Não reutilizado**: a atenção densa lê todos os blocos em todos os tokens; não há conjunto quente a manter |
| Formatos int8 / q4_0 / K8V4 da KV | **Não reutilizados**: o `strata-poly` guarda a KV em fp16; mudar o formato mudaria os resultados, o que o pedido exclui |
| Kernels QSA (`qsa_decode_attn_*`, `kv_gather_*`) | **Não reutilizados**: o `strata-poly` tem os seus kernels de atenção densa (`attn_score`, `attn_value`, `attend_batch`), que foram estendidos |
| `--kv-resident N` como opção | **Mesmo nome e significado** (células que ficam na VRAM) |

---

## 5. Alterações ao motor PolyStrata

| Ficheiro | Alteração |
| --- | --- |
| `include/strata/poly/kernels.hpp`, `src/poly/kernels.cu` | `KvSpill`; `store_f16`/`store_f16_at` com parte na RAM; `attn_score`/`attn_value` com `template <bool SPILL>`; `attend_batch` com o stage dos blocos; `kv_gather_kernel`; `attend_batch_block()` |
| `src/poly/engine.hpp`, `engine.cpp` | `Options::kv_resident`, `kv_ram_mib`, `kv_ram_floor_mib`; `Budget` e `Engine::budget()`; `hmalloc_mapped()`; ganchos `before_head`/`after_head` à volta da cabeça de cada camada em `run()` |
| `src/poly/qwen35.hpp`, `qwen35.cpp` | plano de residência (`plan_kv`, `auto`), alocação dividida e verificação da RAM, `spill()`, pré-carregamento (`ahead`, `before_head`, `after_head`), `save_rows`/`load_rows` das duas partes, estatísticas |
| `src/poly/main.cpp` | `--kv-resident N\|auto\|all`, `--kv-ram-mib`, `--kv-ram-floor-mib`, relatório de memória, `--logits-out FICHEIRO` (logits de cada passo, para comparar bit a bit) |
| `src/poly/hip_probe.cu` | `hip_poly_probe kv`: largura de banda VRAM / zero-copy / copy engine, quanto o Windows fixa |
| `src/poly/kernel_bench.cu` | `poly_kernel_bench attn`: atenção de decode e de lote de 8K a 262K, residente / dividida / tudo na RAM, com verificação bit a bit |
| `tools/hip/kv_parity.py` | o mesmo prompt com duas configurações (ou dois binários), logits de cada passo comparados |
| `tools/hip/kv_needles.py` | recuperação de agulhas a 5 %, 50 % e 95 % de documentos até 261 000 tokens, pelo protocolo `--serve`, com tempos e memória |
| `tools/hip/kv_summary.py` | as tabelas deste relatório a partir dos JSON |
| `tools/hip/poly_config.py` | `--kv-resident`, `--kv-ram-mib` na configuração de execução do servidor |

Só o Qwen3.5/3.6 (`qwen35`) tem streaming; Gemma 4 e GLM 5.3 Flash usam os kernels com os novos parâmetros por
omissão (`spill = nullptr`), ou seja, o código de antes.

**Caminho CUDA:** as alterações não estão sob `STRATA_USE_HIP` (o mecanismo é independente do backend), mas o
caminho por omissão (`--kv-resident all`) executa o código de antes: os kernels com `SPILL = false` são o texto
anterior, e `store_f16` sem parte na RAM faz o mesmo lançamento. **O build CUDA não foi compilado** (não há toolkit
nem placa NVIDIA neste PC).

---

## 6. Gestão e orçamentos de memória

### 6.1 O que ocupa o quê (relatório de carga, uma corrida de cada configuração)

| MiB | 48K, KV na VRAM | 128K, KV na VRAM | 128K, streaming a partir de 49 152 | 262K, KV na VRAM | 262K, streaming a partir de 49 152 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pesos densos (VRAM) | 1 910 | 1 910 | 1 910 | 1 910 | 1 910 |
| Cache de especialistas (VRAM) | 8 405 | 6 910 | 8 325 | 4 724 | 8 189 |
| KV na VRAM | 960 | 2 560 | 960 | 5 120 | 960 |
| Staging da KV (VRAM: bloco do prefill + uma camada para o pré-carregamento) | 0 | 0 | 202 | 0 | 458 |
| Estado recorrente e cópias para as janelas (VRAM) | 251 | 251 | 251 | 251 | 251 |
| Scratch, gravações, resto (VRAM) | 809 | 810 | 810 | 810 | 810 |
| Livre (reserva) | 1 021 | 1 006 | 948 | 1 010 | 972 |
| KV na RAM fixada | 0 | 0 | 1 600 | 0 | 4 160 |
| Outra RAM fixada (staging de especialistas, cópias) | 1 078 | 1 078 | 1 078 | 1 078 | 1 078 |
| RAM disponível no sistema depois da carga | 9 106 | 8 884 | 4 157 | 12 734 | 4 475 |

Especialistas executados na CPU: os que não estão na cache, lidos do GGUF mapeado (16,4 GB). Ocupam cache de
ficheiros do Windows, não memória privada. A RAM disponível depois da carga varia com o que o leitor de páginas
(`map_pages`) já leu do ficheiro.

### 6.2 Orçamentos e limites

- **VRAM:** o Windows dá ao processo um orçamento de 15 416 MiB (fase 1). O motor aloca os pesos densos, a KV na
  VRAM, o staging, o estado e o scratch, e a cache de especialistas fica com o resto menos a reserva (512–1 024
  MiB). Com `auto`, a KV só sai da VRAM se não couber com a reserva e o scratch.
- **RAM fixada:** `--kv-ram-mib` limita a parte da KV na RAM. Sem ele, a alocação tem de deixar
  `--kv-ram-floor-mib` (4 096) de RAM física disponível. A memória fixada não é paginada, por isso o limite existe
  para que a cache de ficheiros do GGUF e os outros programas tenham onde ficar.
- **Memória reservada pelo Windows e por outros programas:** neste PC, 4,2–4,4 GB de VRAM e 10–13 GB de RAM.
  Medido antes de cada corrida (JSON).
- **262K com streaming neste PC:** 4,1 GiB fixados, 4,0–4,5 GB de RAM disponível depois da carga. Cabe, mas fica no
  limite de 4 GiB. Mais 32K de contexto já não caberiam sem outros programas fechados ou um `--kv-ram-floor-mib`
  menor.

### 6.3 Paginação e memória comprometida

Medidos antes e depois de cada corrida da matriz (secção 9). Mediana; ficheiro de paginação usado e commit charge do
Windows:

| Configuração | Commit antes / depois MiB | Ficheiro de paginação antes / depois MiB | Pico de RAM privada / working set MiB | Pico de VRAM dedicada / partilhada MiB |
| --- | --- | --- | --- | --- |
| 48K, KV na VRAM | 32 877 / 33 035 | 3 168 / 3 165 | 11 876 / 18 378 | 12 900 / 1 946 |
| 128K, KV na VRAM | 31 328 / 30 780 | 2 966 / 2 812 | 11 353 / 18 978 | 13 166 / 2 490 |
| 128K, streaming | 31 989 / 32 055 | 2 715 / 3 001 | 14 221 / 19 878 | 13 029 / 4 092 |
| 262K, KV na VRAM | 29 606 / 29 730 | 2 489 / 2 116 | 13 644 / 18 842 | 13 253 / 2 106 |
| 262K, streaming | 31 084 / 30 400 | 2 682 / 2 659 | 16 122 / 19 706 | 12 988 / 6 268 |

- O ficheiro de paginação **não cresceu** em nenhuma configuração (variação de −373 a +286 MiB, também com outros
  programas a mexer).
- O commit charge ficou em 29–35 GB de um limite de 65 GB.
- A "VRAM partilhada" do streaming inclui a KV fixada (+1,6 GiB a 128K, +4,1 GiB a 262K). Não é transbordo da VRAM.

---

## 7. Limites de contexto efetivamente alcançados

| Contexto | Modo | Estado | Prova |
| ---: | --- | --- | --- |
| 49 152 (atual) | KV na VRAM | ✅ preservado | 18/18 notas a 8K, 32K e 45K; sem regressão (secção 9.1) |
| 131 072 | KV na VRAM | ✅ validado | 24/24 notas a 130 900 tokens (4 repetições) |
| 131 072 | streaming a partir de 49 152 | ✅ **validado** (critério mínimo) | 18/18 notas a 130 900 tokens (3 repetições); paridade bit a bit e contra o llama.cpp (secção 8); API com 108 264 tokens |
| 262 144 | KV na VRAM | ✅ validado | 12/12 notas a 261 000 tokens (2 repetições; e 6/6 numa corrida anterior) |
| 262 144 | streaming a partir de 49 152 | ✅ validado | 12/12 notas a 261 000 tokens (2 repetições, pré-carregamento; e 6/6 numa corrida em zero-copy) |
| > 262 144 | — | não tentado | é o contexto nativo do ficheiro; sem YaRN, como pedido |

Em cada comprimento as perguntas foram feitas uma a uma (nota a 5%, 50% e 95% do documento) e depois todas
juntas. As respostas são greedy, sem pensamento. Uma nota conta como encontrada quando o código está na resposta.
Todas as respostas foram exatamente o código pedido (ver `answer` nos JSON). As 2.ª a 4.ª perguntas reutilizam o
documento, lendo de novo só a partir do último checkpoint antes da pergunta.

O estado recorrente não limita o contexto: tem tamanho fixo (62 MiB) e foi o mesmo em todas as medições.

---

## 8. Resultados de paridade

Ferramenta: `tools/hip/kv_parity.py` (greedy, `--window 1`, os logits completos de cada passo por
`--logits-out`). "Bit a bit" quer dizer os 248 320 logits de cada passo com os mesmos bits.

### 8.1 Exata: especialistas na CPU (`--no-gpu-experts`)

Com todos os especialistas na CPU a cache de especialistas não muda entre execuções, e o motor é determinístico:
qualquer diferença viria da KV.

| Prompt | Contexto | Configuração contra a KV toda na VRAM | Passos iguais bit a bit | Maior diferença de logit / probabilidade / KL |
| --- | ---: | --- | ---: | --- |
| B-8k (8 235) | 16 384 | `--kv-resident 1024`, zero-copy | 32/32 | 0 / 0 / 0 |
| B-8k | 16 384 | `--kv-resident 4100` (corte fora de um bloco), zero-copy | 32/32 | 0 / 0 / 0 |
| B-8k | 16 384 | `--kv-resident 1024`, pré-carregamento | 32/32 | 0 / 0 / 0 |
| B-8k | 16 384 | `--kv-resident 4100`, pré-carregamento | 32/32 | 0 / 0 / 0 |
| B-8k | 16 384 | `--kv-resident 4100`, depois da correção das esperas | 32/32 | 0 / 0 / 0 |
| B-8k | 16 384 | `--kv-resident 12000` (tudo abaixo do corte: o caminho sem esperas) | 32/32 | 0 / 0 / 0 |
| A-31k (31 296) | 65 536 | `--kv-resident 8192` (23 136 posições na RAM no fim), pré-carregamento | 32/32 | 0 / 0 / 0 |
| B-2k (2 068) | 49 152 | binário novo, omissão, contra o binário de `d947977c` | 32/32 | 0 / 0 / 0 |
| B-2k | 49 152 | binário novo, `--kv-resident auto`, contra `d947977c` | 32/32 | 0 / 0 / 0 |

Nestes prompts o prefill passa pela atenção por blocos com o stage (lotes de 1 024 tokens que leem a RAM) e o
decode pelos kernels de decode. Os mesmos testes cobrem o estado recorrente: logits iguais bit a bit depois de
31 296 tokens e 32 passos implicam o mesmo estado das 30 camadas delta-net em cada passo.

Modo por omissão (janelas de tokens adivinhados e o caminho "em metades", que entrega a placa a outra thread):
prompt A-8k, 96 tokens, especialistas na CPU, `--kv-resident 1024` com pré-carregamento contra tudo na VRAM: **as
mesmas 96 tokens**.

Kernels isolados (`poly_kernel_bench attn`, dados aleatórios, 16 cabeças, 2 de chave/valor, 256 dimensões): uma
token de decode e um lote de 1 024 tokens, com a KV a partir de 49 152 na RAM e toda na RAM, contra a KV toda na
VRAM, em contextos de 8 192, 32 768, 49 152, 65 536, 131 072 e 262 144: **iguais bit a bit em todos os casos**.

### 8.2 Com especialistas na GPU (o caminho normal)

Com especialistas na GPU, a cache de especialistas é diferente em cada execução (o tamanho depende da VRAM livre
no momento, e as trocas seguem a resposta), e um especialista calculado na CPU difere nos últimos bits do mesmo na
GPU (fase 2, secção 4.3). Por isso a comparação justa é com a variação entre duas execuções iguais:

Prompt A-31k (31 296 tokens), contexto 65 536, 64 passos:

| Contra uma execução com a KV toda na VRAM | Primeira token diferente | Maior diferença de probabilidade | Média | KL máximo |
| --- | --- | ---: | ---: | ---: |
| `--kv-resident 16384` (15K posições na RAM, pré-carregamento) | passo 26, quase-empate | 0,061 | 0,0042 | 0,0078 |
| outra execução igual, KV toda na VRAM | passo 26, quase-empate | 0,057 | 0,0038 | 0,0070 |

A diferença do streaming é do tamanho da variação do próprio modo residente. Dentro do critério das fases 1 e 2
(tokens iguais ou primeira diferença num quase-empate; diferença máxima ≤ 0,15).

### 8.3 Estabilidade numérica do softmax

O softmax não mudou: a decode usa o mesmo kernel (máximo, exponenciais, soma, divisão), e o prefill o mesmo
softmax online por blocos (máximo corrente, reescalamento por `exp(antes - agora)`), com a mesma partição em
blocos. Nas referências: maior logit entre 20,1 e 37,8 por passo, entropia entre 0 e 2,9 nats, todos os logits
finitos, em todos os prompts (2K a 31K). Em contextos longos (até 261 000 tokens) o comportamento observável é a
recuperação de informação (secção 7): as agulhas do início, do meio e do fim são encontradas.

### 8.4 Contra o llama.cpp

Referências guardadas na fase 2 (llama.cpp HIP b11160, mesmos ids): `refs-llama-hip-gpu.json` (B: especialistas
na GPU, 4 prompts de 22 a 11 481 tokens) e `refs-llama-hip-cpu-30k.json` (A30: 35 575 tokens). O PolyStrata com
`--window 1` e a maior parte da KV na RAM:

| Referência | Configuração | Tokens iguais | Maior diferença de probabilidade | Média | Fase 2 (KV na VRAM) |
| --- | --- | --- | ---: | ---: | ---: |
| B, 4 prompts | `--kv-resident 1024` | 64/64 em todos | 0,078 (passo 17 do doc-8k) | 0,0010–0,0056 | 0,042 |
| A30 (35 575) | `--kv-resident 8192` (~27 400 posições na RAM) | 64/64 | 0,098 (num quase-empate da referência: 0,419 / 0,412) | 0,0058 | 0,079 |

Dentro da tolerância do harness (≤ 0,15) e da dispersão entre configurações do próprio llama.cpp (até 0,122,
fase 2). A diferença para a fase 2 é do tamanho da variação que a cache de especialistas introduz entre execuções
(secção 8.2): os especialistas usados são os mesmos, mas quais são calculados na CPU muda.


---

## 9. Benchmarks

### 9.1 Caminho por omissão: antes e depois

Binário de `d947977c` (compilado numa worktree à parte) contra o desta fase, 3 corridas alternadas de cada,
contexto 49 152, 256 tokens, prompts A-2k e A-31k da fase 2 (mediana; entre parênteses as 3 corridas):

| | Antes | Depois |
| --- | ---: | ---: |
| Decode A-2k, `--window 1` | 74,1 (70,1–74,6) | 75,7 (71,9–76,4) |
| Decode A-31k, `--window 1` | 58,2 (55,0–60,4) | 60,0 (58,4–60,3) |
| Decode A-2k, omissão (palpites) | 74,2 (71,8–74,2) | 74,0 (69,0–74,2) |
| Decode A-31k, omissão | 57,6 (56,6–58,2) | 57,5 (56,7–58,0) |
| Prefill A-31k, `--window 1` / omissão | 1 342 / 1 387 | 1 387 / 1 377 |

Sem regressão. As diferenças estão dentro da dispersão entre corridas. Os valores são mais baixos do que os
80 tok/s da fase 2 porque outros programas ocupavam mais 1,3–2 GB de VRAM nesse dia.

### 9.2 Matriz: decode, prefill e TTFT por comprimento

`tools/hip/kv_needles.py`, um processo por configuração, alternadas. Para cada comprimento: o 1.º pedido lê o
documento inteiro (TTFT e prefill); depois 3 perguntas e um **resumo de 128 tokens** reutilizam-no (decode medido
no resumo; TTFT de reutilização nas 4). Mediana de 2 a 4 repetições.

**Decode (tok/s, resumo de 128 tokens):**

| Prompt | 48K, KV na VRAM | 128K, KV na VRAM | 128K, streaming | 128K, streaming zero-copy | 262K, KV na VRAM | 262K, streaming |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 192 | **68,2** | 60,4 | 65,0 | 66,4 | 52,9 | 58,5 |
| 32 768 | **54,9** | 54,1 | 53,3 | 54,0 | 49,0 | 50,7 |
| 45 000 / 49 152 | **54,1** (45K) | 50,1 | 44,9 | 49,6 | — | — |
| 65 536 | — | **43,4** | 25,2 | 13,4 | 39,3 | 25,2 |
| 130 900 / 131 072 | — | **29,4** | 8,1 | 3,5 | 26,0 | 6,9 |
| 261 000 | — | — | — | — | **16,7** | 2,8 |

"Streaming" é `--kv-resident 49152` com pré-carregamento; zero-copy é o mesmo com `STRATA_POLY_KV_AHEAD=0` (1
corrida).

**Prefill do 1.º pedido (tok/s) e TTFT (s):**

| Prompt | 48K, KV na VRAM | 128K, KV na VRAM | 128K, streaming | 262K, KV na VRAM | 262K, streaming |
| ---: | --- | --- | --- | --- | --- |
| 8 192 | 1 518 / 5,4 | 1 381 / 5,9 | 1 221 / 6,7 | 1 242 / 6,6 | 640 / 12,8 * |
| 32 768 | 1 389 / 23,6 | 1 321 / 24,8 | 1 261 / 26,0 | 1 304 / 25,1 | 1 202 / 27,3 |
| 45 000 / 49 152 | 1 264 / 35,6 | 1 183 / 41,5 | 1 109 / 44,3 | — | — |
| 65 536 | — | 1 059 / 61,9 | 978 / 67,0 | 1 065 / 61,6 | 985 / 66,5 |
| 130 900 / 131 072 | — | 750 / 174,5 | 688 / 190,4 | 758 / 173,2 | 686 / 191,0 |
| 261 000 | — | — | — | 477 / 548,1 | 428 / 610,1 |

\* O 1.º pedido de cada processo inclui aquecimentos e varia nos dois modos (128K na VRAM: 5,7–11,4 s). A 262K
com streaming foi lento nas duas corridas. Coincide com a RAM disponível depois da carga (4,0–4,5 GB, contra 12,7
residente): hipótese, não verificada, de que a cache de ficheiros do GGUF fica mais pequena e o 1.º prefill lê
páginas do disco. No 2.º documento as velocidades igualam-se.

O prefill com streaming custa 4–10% (os blocos que chegam à RAM são copiados para o stage; −10% a 261K). No micro-benchmark, um
lote de 1 024 tokens a 262K demora 297,8 ms com a KV na VRAM e 329,8 ms dividida.

**Reutilização do contexto entre perguntas (TTFT, s):** 1,4–2,2 a 8K, 2,0–2,6 a 32K, 2,9–3,4 a 64K, 4,4–5,1 a
128K–131K, 4,1–4,4 a 261K, nos dois modos. Corresponde a ler de novo ~900–2 048 tokens desde o último checkpoint;
as corridas da matriz não usaram `--turn-token`.

### 9.3 Transferências, "acertos" e latência da atenção

| Prompt (corte em 49 152) | Posições por camada na RAM | Lidas da VRAM | Bytes pelo PCIe por token de decode | Decode medido | Ritmo efetivo do PCIe |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65 536 | 16 384 | 75 % | 0,34 GB | 25,2 tok/s | 8,5 GB/s |
| 130 900 | 81 748 | 37,5 % | 1,67 GB | 8,1 tok/s | 13,5 GB/s |
| 261 000 | 211 848 | 18,8 % | 4,34 GB | 2,8 tok/s | 12,3 GB/s |

A 131K e 261K o decode com streaming está **limitado pelo copy engine** (14,2 GB/s medidos): é o limite físico
deste PCIe 4.0, não do código. Volume total copiado por corrida da matriz: ~300 GB (128K) e ~1 035 GB (262K). Cópias
feitas na hora em vez de à frente: 4,6–5,3%.

Especialistas servidos da VRAM no resumo (mediana): 48K 94–98%; 128K na VRAM 90–96%; 128K streaming 94–98%; 262K
na VRAM **78–87%**; 262K streaming 93–98%.

Latência de **uma** camada de atenção (`poly_kernel_bench attn`; o modelo tem 10 por token):

| Contexto | Decode, KV na VRAM | Decode, a partir de 49 152 na RAM (zero-copy) | Cópia à frente de uma camada (copy engine, calculado) | Lote de 1 024, VRAM / dividida |
| ---: | ---: | ---: | ---: | --- |
| 8 192 | 0,11 ms | — | — | 10,6 / — ms |
| 32 768 | 0,50 ms | — | — | 38,2 / — ms |
| 65 536 | 0,96 ms | 5,3 ms | 2,4 ms | 74,5 / 77,0 ms |
| 131 072 | 2,07 ms | 23,7 ms | 11,8 ms | 148,6 / 161,6 ms |
| 262 144 | 4,31 ms | 60,2 ms | 30,6 ms | 297,8 / 329,8 ms |

A atenção de decode com a KV na VRAM lê a ~125 GB/s (537 MB numa camada a 262K em 4,3 ms), menos de metade dos 300
GB/s da placa. É a maior margem que resta para contextos longos (secção 15).

### 9.4 Leitura dos números

- **Até 48K:** o contexto de 49 152 continua a ser o mais rápido.
- **Até 128K:** a KV toda na VRAM com contexto de 131 072 custa 1–11% nas conversas curtas (menos especialistas
  na VRAM) e é 1,7× (64K) a 3,6× (131K) mais rápida do que o streaming nas longas.
- **Até 262K:** a KV toda na VRAM custa 11–22% nas conversas curtas, mas é 1,6× (64K) a 6× (261K) mais rápida
  nas longas. O streaming a partir de 48K ganha 3–11% abaixo dos 32K.

---

## 10. Regressões encontradas e resolvidas

| Problema | Onde | Correção |
| --- | --- | --- |
| O decode com streaming por zero-copy rendia 6,6 GB/s (metade da sonda) e ficava no caminho crítico: 12,7 tok/s a 64K com só 16K posições na RAM (contra 36 residente) | medição, `kv_needles` e `poly_kernel_bench attn` | pré-carregamento por DMA da parte na RAM, uma camada de atenção à frente, num stream lateral (secção 3.2); 64K: 12,7 → 25,3 tok/s |
| As esperas entre streams em todas as cabeças de atenção, também quando a chamada não tocava na RAM, tornavam um contexto de 131 072 com streaming 5–12% mais lento do que com a KV na VRAM nos prompts de 8K a 49K, apesar de ter ~1 000 especialistas a mais na VRAM | matriz (`matrix-before-sync-fix/`) | uma cabeça cujas linhas estão todas abaixo do corte não espera (commit da correção); 8K: 56,9 → 65,0 tok/s, 32K: 50,7 → 53,3 |
| Cópias "tardias" (feitas quando a cabeça chega, sem sobreposição): 4,6–5,3% | estatísticas no fim do processo | aceites: são a 1.ª camada de atenção depois de um prefill, ou as metades das janelas a alternar camadas |
| Script de medição: um prompt de 131 000 tokens mais um resumo de 128 não cabia num contexto de 131 072; o motor recusou corretamente (`ERR`), mas o script perdia a corrida | `tools/hip/kv_needles.py` | a resposta é limitada ao que cabe, e o JSON é escrito mesmo quando um pedido falha |
| Benchmark de regressão: um caminho relativo com `/` não era encontrado pelo `CreateProcess` | uso das ferramentas | caminhos absolutos; `kv_parity.py` resolve-os |

Nenhuma regressão foi encontrada no caminho por omissão (secção 9.1).

---

## 11. Funcionalidades preservadas e fallbacks

| Funcionalidade | Com a KV toda na VRAM (omissão) | Com KV streaming | Como foi verificado |
| --- | --- | --- | --- |
| Routing na placa (fase 2) | inalterado | ativo; as cópias à frente são enfileiradas antes de o token ser entregue à placa | decode e paridade com especialistas na GPU |
| Cache de especialistas e trocas | inalterada | a cache fica com a VRAM que a KV deixa (+3,7 GiB a 262K) | relatório de memória, `experts_vram` |
| Sobreposição CPU/GPU (janelas em metades) | inalterada | ativa; o staging é recarregado quando as metades alternam camadas | mesmas 96 tokens (secção 8.1) |
| Cópias por kernel (`copy_words`) | inalteradas | inalteradas; as cópias da KV usam o copy engine num stream próprio | — |
| Graphs | inalterados | ativos: os ponteiros da RAM e do staging são fixos; as esperas por eventos ficam fora das gravações | todas as medições com graphs |
| Speculative decoding (palpites por repetição) | inalterado | ativo; as linhas de palpites rejeitados são reescritas como antes | modo por omissão, mesmas tokens |
| `--turn-token` e checkpoints do prompt | inalterados | ativos | `kv_needles`: cada pergunta seguinte lê só 900–2 048 tokens |
| Conversas estacionadas (`--conversation-cache-mib`) | inalteradas | `save_rows`/`load_rows` copiam as duas partes; o staging é invalidado | duas conversas alternadas, mesmas tokens ao voltar (secção 13) |
| API OpenAI/Anthropic, OpenCode/OpenChamber | inalteradas | `--kv-resident` passa pela configuração (`poly_config.py`) | 12/12 verificações de `poly_api_check.py` com contexto 131 072 e `--kv-resident 49152`; um pedido de 108 264 tokens pela API encontrou a nota |
| Blocos MTP / modelo de rascunho | inalterados | leem a sua parte na RAM por zero-copy (sem staging) | **não testado** (o ficheiro não tem MTP) |

Fallbacks:
- `--kv-resident all` (omissão): o caminho de antes, verificado bit a bit contra o binário anterior.
- `STRATA_POLY_KV_AHEAD=0`: streaming sem pré-carregamento (zero-copy), sem o buffer de staging (menos VRAM, decode
  mais lento).
- Se o sistema não fixar a memória pedida, ou não houver RAM, a carga falha com uma mensagem que diz o contexto que
  cabe; não há degradação silenciosa nem paginação.
- Gemma 4 e GLM 5.3 Flash: sem streaming (só `qwen35` o implementa); `--kv-resident` é ignorado por eles.

---

## 12. Instruções de configuração

Compilar como na fase 2:

```bat
set STRATA_GGML_DIR=C:\Users\user\Documents\PolyStrata\third_party\llama.cpp-3cf0325
set STRATA_HIP_ARCHS=gfx1200
tools\hip\build_windows.bat poly
```

Opções do `strata-poly`:

| Opção | Efeito |
| --- | --- |
| (nenhuma) ou `--kv-resident all` | toda a KV na VRAM, como antes |
| `--kv-resident auto` | toda na VRAM quando cabe (neste PC sempre, até 262 144), senão o máximo que cabe |
| `--kv-resident N` | as primeiras N posições na VRAM, o resto na RAM fixada |
| `--kv-ram-mib N` | o máximo de RAM que a parte na RAM pode fixar |
| `--kv-ram-floor-mib N` | sem `--kv-ram-mib`: RAM física que tem de ficar disponível (omissão 4 096) |
| `STRATA_POLY_KV_AHEAD=0` | sem pré-carregamento: a atenção lê a RAM pelo PCIe |

Configurações recomendadas neste PC (secção 9):

- **Até 48K, o mais rápido:** `--context 49152` como até aqui.
- **Até 128K com a KV toda na VRAM:** `--context 131072`. As conversas curtas ficam 1–11% mais lentas do que com
  48K (menos especialistas na VRAM); as longas são as mais rápidas possíveis.
- **Até 262K:** `--context 262144` com a KV toda na VRAM é o mais rápido a partir de ~32K, e as conversas curtas
  perdem 11–22% (a cache de especialistas fica com ~4,7 GiB). Com `--kv-resident 49152` as conversas até 32K ganham
  3–11%, mas as posições acima de 48K pagam o PCIe (25 tok/s a 64K, 6,9 a 131K, 2,8 a 261K).

Servidor (configuração de execução):

```bat
.venv\Scripts\python tools\hip\poly_config.py --model C:\Users\user\models\Qwen3.5-35B-A3B-Q3_K_M.gguf --context 262144 --threads 6 --kv-resident 49152 --data C:\Users\user\models\qwen35-pack --name qwen3.5-35b-a3b-q3_k_m-hip-262k
.venv\Scripts\python serve\server.py --engine strata --config strata-qwen3.5-35b-a3b-q3_k_m-hip-262k.json --port 8090
```

Verificações:

```bat
.venv\Scripts\python tools\hip\kv_parity.py --engine build-hip-poly-win\strata-poly.exe --model C:\Users\user\models\Qwen3.5-35B-A3B-Q3_K_M.gguf --prompt ids.txt --ctx 16384 --args "--threads 6 --no-gpu-experts" --test "--kv-resident 1024"
.venv\Scripts\python tools\hip\kv_needles.py --engine build-hip-poly-win\strata-poly.exe --model C:\Users\user\models\Qwen3.5-35B-A3B-Q3_K_M.gguf --tokenizer C:\Users\user\models\qwen35-pack\tokenizer --lengths 32768,131072 --ctx 131072 --summary 128 --engine-args "--threads 6 --kv-resident 49152"
tools\hip\poly_windows.bat cmd build-hip-poly-win\poly_kernel_bench.exe attn
tools\hip\poly_windows.bat cmd build-hip-poly-win\hip_poly_probe.exe kv
```

---

## 13. Testes executados

| Teste | Resultado |
| --- | --- |
| Compilação HIP (`build_windows.bat poly` e `poly tests`) | sem erros |
| ctest, 103 testes | 91 passam, 9 saltados, 3 falham: os mesmos 3 da fase 2 (`expert_cache_segmented_test`, `hip_handoff`, `ple_parity`), fora do código alterado (`ctest.txt`) |
| `hip_poly_probe kv` | VRAM 300 GB/s, zero-copy 12,6–13,2, copy engine 14,2; 7 GiB fixados |
| `poly_kernel_bench attn` (8K–262K, 3 divisões) | iguais bit a bit em todos os casos |
| `kv_parity.py`, especialistas na CPU (2K, 8K, 31K; cortes 1024, 4100, 8192, 12000; zero-copy e pré-carregamento) | logits iguais bit a bit em todos os passos |
| `kv_parity.py`, binário anterior contra o novo (omissão e `auto`) | iguais bit a bit |
| Modo por omissão (janelas, metades), 96 tokens | as mesmas tokens |
| `kv_parity.py`, especialistas na GPU, 31K | dentro da variação entre execuções residentes |
| `poly_llama_parity.py` com streaming, 5 prompts até 35 575 tokens | PASS, 64/64 em todos |
| Conversas estacionadas com a KV dividida (2 × 8K, corte a 3000) | voltam da RAM (`restored ... in 60 ms`) com as mesmas tokens |
| `kv_needles.py`: 48K, 128K e 262K, KV na VRAM e streaming, 2–4 repetições | todas as notas encontradas (secção 7) |
| Benchmark de regressão, 3 × 4 corridas alternadas | sem regressão |
| `poly_api_check.py` com streaming (131 072, `--kv-resident 49152`) | 12/12 |
| Pedido de 108 264 tokens pela API OpenAI com streaming | nota encontrada, TTFT 135 s |
| `--kv-resident auto` a 48K e 262K | tudo na VRAM, como previsto |

**Distinção pedida:**

| | Implementado | Compilado | Testado | Validado no hardware |
| --- | --- | --- | --- | --- |
| Residência parcial, kernels com parte na RAM, stage do prefill | ✅ | ✅ HIP | ✅ | ✅ |
| Pré-carregamento por DMA | ✅ | ✅ HIP | ✅ | ✅ |
| `auto`, limites de RAM, recusa na carga | ✅ | ✅ HIP | `auto` ✅; a recusa por falta de RAM **não foi provocada** | parcial |
| Conversas estacionadas com a KV dividida | ✅ | ✅ HIP | ✅ | ✅ |
| Build CUDA | ✅ (código comum) | ❌ não compilado | ❌ | ❌ |
| Blocos MTP com a KV dividida | ✅ (zero-copy) | ✅ HIP | ❌ (o ficheiro não tem MTP) | ❌ |
| Linux com HIP | — | ❌ | ❌ | ❌ |

---

## 14. Limitações conhecidas

1. **Neste modelo, o streaming é sempre mais lento do que a KV na VRAM acima de ~32K.** O decode com posições na
   RAM está limitado pelo PCIe (14,2 GB/s): ~1,6 µs por posição e por token. Não é uma forma de acelerar contextos
   longos, mas de os permitir quando a KV não cabe.
2. **A RX 9060 XT usa PCIe 4.0 neste PC** (o Ryzen 5 5600X não tem 5.0). Numa plataforma PCIe 5.0 o mesmo código
   deverá ter perto do dobro da largura de banda. Não medido.
3. **262K com streaming deixa ~4 GB de RAM disponível** (no limite por omissão). O 1.º pedido de cada processo foi
   2× mais lento nessas condições; ver a nota da secção 9.2.
4. **A recusa por falta de RAM não foi provocada.** O código calcula e recusa, mas nenhuma medição ficou abaixo do
   limite.
5. **Só Qwen3.5/3.6.** Gemma 4 e GLM 5.3 Flash não têm streaming.
6. **Build CUDA não compilado**, e Linux/HIP não testado.
7. **Blocos MTP / modelo de rascunho** com a KV dividida: implementados (leem a sua parte por zero-copy), não
   testados.
8. **O 1.º pedido de um processo varia** em ambos os modos (aquecimentos, páginas do ficheiro): as tabelas usam a
   mediana e mostram o intervalo.
9. **Medido num PC em uso**, com 4,2–4,4 GB de VRAM e 10–13 GB de RAM ocupados por outros programas. Valores
   absolutos diferentes noutras condições.

---

## 15. Próximos passos recomendados

Por ordem de ganho esperado, com os números que o justificam:

1. **Atenção de decode com a KV na VRAM a contextos longos.** Lê a ~125 GB/s, de 300 possíveis: 4,3 ms por camada
   e 43 ms por token a 262K. Um kernel "split-K" (flash-decoding: várias partes do contexto em paralelo, juntas
   por softmax online) pode aproximar-se do pico. Estimativa: decode a 261K de 16,7 para ~25–30 tok/s. Muda a
   ordem das somas, por isso tem de ser validado como uma alteração de omissão (medida e paridade).
2. **Atenção da parte na RAM calculada pela CPU** em paralelo com a GPU (resultados parciais juntos por softmax
   online, como no ponto 1). Somaria a largura de banda da RAM do CPU à do PCIe. Precisa de uma ida e volta ao host
   por camada de atenção, como os especialistas da CPU, e muda a ordem das somas. Só faz sentido para o streaming.
3. **KV em 8 bits** (como o `--kv` do motor Strata): metade da memória, ou o dobro do contexto na mesma VRAM ou
   PCIe. Muda os valores, por isso tem de ser opt-in e medido em perplexidade e recuperação.
4. **Integração no `setup.py` / `START-POLY-HIP.bat`:** escolher o contexto e `--kv-resident auto` pela VRAM e RAM
   livres.
5. **Validar** um ficheiro com MTP, o build CUDA, e uma plataforma PCIe 5.0.
