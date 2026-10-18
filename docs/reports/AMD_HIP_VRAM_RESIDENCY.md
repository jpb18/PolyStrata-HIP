> Historical measurement report. Use the [current setup instructions](../../README.md). Personal paths were generalized for publication; measurements were preserved. Raw campaign links may require the local evidence archive.

# Residência adaptativa da VRAM no strata-poly HIP (Windows/WDDM): investigação

Fase de investigação. Não se alterou código. O que está marcado **(código)** foi lido no código e não foi medido. O que
está marcado **(medido)** foi lido neste PC a 2026-10-13: RX 9060 XT (16 304 MiB), driver 32.0.31041.1004,
Windows 11 Pro 26200, com o servidor a correr Qwen3.6-35B-A3B UD-Q3_K_M, `--max-context 262144`. Não se fez nenhuma
experiência que alocasse VRAM, porque o motor do servidor estava ativo e uma alocação podia provocar a expulsão que
se quer estudar.

## Respostas curtas

| Pergunta | Resposta |
|---|---|
| A cache de experts pode ser redimensionada com o servidor ativo? | **Sim, já pode, entre pedidos.** `strata-poly` aceita o comando `VRAM [reserve_mib]` (`POST /v1/vram`) e chama `mind_vram()` antes de cada pedido. Ambos usam `Engine::resize_cache()`. **(código)** |
| A KV cache e o estado recorrente são preservados? | **Sim.** `resize_cache()` só liberta e volta a encher `exp_` e as tabelas de slots; KV, estado recorrente, snapshots, scratch, staging e grafos ficam como estão. **(código)** |
| É seguro tal como está? | **Não totalmente.** Os experts que saem da VRAM passam para a CPU sem que as suas páginas sejam lidas de novo, e o routing na placa continua ligado. Num log deste PC, um encolhimento foi seguido de `GPU_FATAL` (espera >1 s pelos experts da CPU). |
| A DXGI distingue memória alocada de residente? | **Não por si só.** `QueryVideoMemoryInfo` dá o orçamento e o uso do próprio processo por grupo de segmentos. A residência física por processo vem dos contadores `GPU Process Memory` (PDH) ou de `D3DKMTQueryStatistics`. |
| A VRAM pode ser recuperada sem reiniciar o motor? | **Encolher: sim** (já funciona). **Voltar a crescer: só até ao tamanho com que o motor arrancou** (`full_bytes_`). Falta medir se `hipFree` devolve de facto a memória ao WDDM no Windows, ou se o runtime a guarda em cache. |
| Recuperação dinâmica (a meio de um pedido)? | **Viável em princípio**, com uma arena em segmentos (`hipMalloc` por segmento, sem VMM) e o mecanismo de trocas que já existe (`follow()`). Exige fases e testes que ainda não existem. |

## 1. O que já existe

### Motor (`src/poly/`)

- `Engine::fill_expert_cache()` ([engine.cpp:865](../../src/poly/engine.cpp:865)) mede a VRAM livre, reserva
  `reserve_mib_` (por omissão 1/16 da placa, entre 512 e 1024 MiB), faz **um único** `cudaMalloc` para a arena e copia
  os experts do ficheiro mapeado, ordenados por uso por byte (perfil + uso desta execução). Só no Windows HIP, um
  ciclo de até 3 rondas ([engine.cpp:918](../../src/poly/engine.cpp:918)) encolhe a arena até a reserva existir com a
  cache já alocada, porque o orçamento do Windows baixa quando o processo passa de ~10 GiB. Medido com
  `hip_poly_probe vram` (2026-10-18, 262K, KV Q8): com 887 MiB retidos pelo processo de teste o Windows retira 1 493 MiB
  ao resto, de 984 a 10 240 MiB retidos retira 887, de 10 752 MiB em diante retira ~2 225 (orçamento 15 271 → 13 933; a
  primeira medição desta placa deu 15 416 → 13 937). Note-se que a linha `Windows budgets N` sai ao arrancar, antes das
  alocações, e por isso não pode mostrar o degrau. Ver [docs/AMD_HIP.md](../AMD_HIP.md).
- `Engine::resize_cache(mib)` ([engine.cpp:1021](../../src/poly/engine.cpp:1021)): sincroniza `st_`, espera pelas trocas em
  curso do thread `follow_loop()`, sincroniza o stream das trocas, põe todos os slots a null, faz `cudaFree(exp_)`,
  e chama `fill_expert_cache()` com a nova reserva, depois `upload_slots()` (tabela de slots na placa, HIP) e
  `start_swaps()`. Libertar antes de alocar é bom sob pressão: o pico não sobe.
- `Engine::mind_vram()` ([engine.cpp:1054](../../src/poly/engine.cpp:1054)): antes de cada pedido
  ([main.cpp:807](../../src/poly/main.cpp:807)), se a VRAM livre for menos de 1/4 da reserva, refaz a cache.
- Comando `VRAM` ([main.cpp:736](../../src/poly/main.cpp:736)): só é lido entre pedidos (o ciclo de `serve()` é serial).
  Uma exceção chama `gpu_fatal()` no HIP: o processo termina e o servidor arranca outro
  ([docs/HIP_GPU_FAILURES.md](../HIP_GPU_FAILURES.md)).
- A linha `INFO` do arranque diz sempre `vram_elastic=1` ([main.cpp:711](../../src/poly/main.cpp:711)). O servidor
  envia `POST /v1/vram` a qualquer motor com o método `vram` ([server.py:3040](../../serve/server.py:3040)) e reaplica a
  última reserva depois de um reinício ([server.py:3032](../../serve/server.py:3032)). Nota:
  [docs/VRAM_ELASTIC.md](../VRAM_ELASTIC.md) descreve o motor Qwen (`--vram-elastic`, só NVIDIA) e diz que o HIP
  fica desligado; isso não se aplica ao `strata-poly`, que aceita o comando sem flag e não está documentado lá.

### O que um redimensionamento não toca **(código)**

KV (`alloc_caches()`, e em Qwen35 a divisão VRAM/RAM decidida uma vez em `plan_kv()`,
[qwen35.cpp:127](../../src/poly/qwen35.cpp:127)), `state_` (estado recorrente), `snap_` (estados das janelas), `hn_`,
scratch, `stage_`, `pos_`, as conversas estacionadas (em RAM, `conversations.hpp`) e os grafos gravados.

Os grafos não guardam ponteiros de slots: o que é gravado (`heads`, `card_heads`, `shared`, `tail`,
[engine.cpp:2401](../../src/poly/engine.cpp:2401)) vai até ao router ou começa depois dos experts. O caminho na placa lê os
slots através da tabela `dslot_`; o caminho do host lê `L.slot[e]` em cada chamada; as `runs_` de prompt
([engine.cpp:2264](../../src/poly/engine.cpp:2264)) juntam slots contíguos por aritmética de ponteiros, e por isso
aceitam uma arena que não seja contígua (apenas partem as execuções nas fronteiras). Isto é o que torna possível uma
arena em segmentos (secção 6).

### Lacunas encontradas **(código)**

1. **Páginas dos experts despromovidos.** Um expert copiado para a VRAM liberta as suas páginas do working set
   (`drop_pages()`, `VirtualUnlock`, [engine.cpp:184](../../src/poly/engine.cpp:184)). No arranque, `map_pages()`
   ([engine.cpp:197](../../src/poly/engine.cpp:197)) volta a ler só os experts fora da cache, e enquanto isso
   `warming_` desliga o routing na placa ([engine.cpp:1344](../../src/poly/engine.cpp:1344),
   [card_takes](../../src/poly/engine.cpp:1952)). `resize_cache()` não faz nenhuma das duas coisas: os experts que um
   encolhimento manda para a CPU podem vir do disco, a placa espera no máximo 1 s (`wait_ticks_`,
   [engine.cpp:1867](../../src/poly/engine.cpp:1867)) e `card_gave_up()` ([engine.cpp:2060](../../src/poly/engine.cpp:2060)) acaba
   em `GPU_FATAL`.
2. **Reconstrução completa.** Encolher 100 MiB volta a copiar do ficheiro todos os experts que ficam (cujas páginas
   também foram libertadas). Medido antes: ~4 s por reconstrução no RX 9060 XT ([docs/AMD_HIP.md](../AMD_HIP.md)).
3. **Teto de crescimento.** A arena nunca passa de `full_bytes_`, o tamanho do arranque. Um motor que arrancou com
   a VRAM ocupada por outro programa não recupera essa VRAM quando ela fica livre; só um reinício o faz.
4. **Sem verificação de RAM.** Encolher passa bytes de experts da VRAM para a RAM (page cache) e para a CPU, e nada
   verifica se a RAM disponível chega.
5. **Só entre pedidos.** Um pedido longo (prompt de 262K, ou uma resposta longa) não tem nenhuma vigilância.
6. **Um só sinal.** `vram_free_mib()` = `hipMemGetInfo` livre menos o que o orçamento DXGI retém
   ([device.cu:188](../../src/core/device.cu:188)). Não vê `CurrentUsage`, o grupo `NON_LOCAL` nem a residência física.

## 2. Evidência deste PC

### Um encolhimento seguido de falha **(medido, log histórico)**

[`bench/results/2026-10-13-hip-prefill-kernels/tmp/contaminated/campaign2-first/G0-r1.engine.log`](../../bench/results/2026-10-13-hip-prefill-kernels/tmp/contaminated/campaign2-first/G0-r1.engine.log),
numa execução posta de lado porque outro motor corria ao mesmo tempo (RAM livre ~200 MB, segundo o README dessa fase):

```
VRAM MiB: dense weights 1961, expert cache 6163, KV 2816 (131072 positions), ... 1037 free. ... RAM available 2615 MiB
the card had little VRAM left: the expert cache 4648 -> 2351 experts, 0 MiB free now
STRATA_POLY_CARD_ROUTE: the card waited more than a second for the CPU's experts; ...
GPU_FATAL ... the card waited more than a second for the CPU's experts; ...
```

`mind_vram()` cortou metade da cache e mesmo assim ficaram 0 MiB livres (o outro processo tinha a VRAM); os 2 297
experts passados para a CPU vieram de páginas que não estavam em RAM, e o processo morreu. Um caso, com RAM
esgotada: não prova a causa, mas é exatamente a lacuna 1 com a lacuna 4.

### O servidor de hoje **(medido, log e contadores)**

O log `strata-qwen3.6-35b-a3b-ud-q3_k_m-hip-262k.log` tem 6 arranques entre as 14:44 e as 15:57 e **4 `GPU_FATAL`
pela mesma espera de 1 s, sem nenhum redimensionamento**. Cada arranque: denso 1961, experts ~3500, KV 5632 (262 144
posições), estado 251, resto 1002, ~1060 MiB livres; 2 561 a 2 792 de 10 496 experts em VRAM. A espera de 1 s já é
um problema por si: qualquer política que mande mais experts para a CPU tem de não o agravar.

Contadores PDH do processo ativo (PID 89880), 3 amostras a 2 s, todas iguais:

| Contador | MiB |
|---|---|
| `GPU Process Memory` Dedicated Usage (= Local Usage) | 10 971 |
| `GPU Process Memory` Shared Usage (= Non Local Usage) | 3 747 |
| `GPU Process Memory` Total Committed | 14 687 |
| `GPU Adapter Memory` Dedicated Usage (toda a placa) | 13 213 |
| VRAM que o motor diz ter alocado (log) | 12 437 |
| RAM fixada que o motor diz ter (log) | 1 087 |

A memória dedicada residente (10 971) é **~1 470 MiB menor** do que a que o motor alocou na VRAM, embora a placa
tenha ~3 GiB que ninguém usa. A partilhada (3 747) é ~2 660 MiB maior do que a RAM fixada do motor. Não sei qual
destas é a explicação: (a) alocações despromovidas para memória do sistema; (b) páginas da KV de 262K nunca
tocadas, que o WDDM conta como comprometidas mas não residentes; (c) memória interna do runtime HIP que eu não
conto. É uma amostra, de um processo em repouso. É precisamente a diferença entre "alocado" e "residente" que esta
fase devia medir, e a experiência E2 (secção 7) distingue as três.

## 3. Detetar pressão no Windows

O WDDM não recusa uma alocação acima da VRAM: aceita-a e move memória para o sistema
([device.cu:107](../../src/core/device.cu:107): 30 em vez de 41 tok/s num RX 6800). O OOM não é um sinal; é preciso medir.

| Sinal | Fonte | O que diz | O que não diz |
|---|---|---|---|
| `Budget` (LOCAL) | `IDXGIAdapter3::QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL)` (já usado em `device.cu`) | Quanto o Windows quer que este processo mantenha em VRAM; desce quando outros programas pedem ou quando o processo cresce | O que está residente |
| `CurrentUsage` (LOCAL) | idem | Uso do processo nesse grupo. `CurrentUsage > Budget` = o processo está acima do orçamento e é candidato a despromoção | Que a despromoção já aconteceu |
| `CurrentUsage` (NON_LOCAL) | `QueryVideoMemoryInfo(..., NON_LOCAL)` | Memória do sistema visível à placa: RAM fixada do motor e, se houver, alocações despromovidas | Separar uma da outra (o motor sabe a sua RAM fixada: `Budget.pinned_other + kv_host`) |
| Mudança de orçamento | `RegisterVideoMemoryBudgetChangeNotificationEvent` | Um evento quando o orçamento muda, sem polling | Falta confirmar que dispara num processo só HIP (E3) |
| Residência por processo | PDH `\GPU Process Memory(pid_<PID>_luid_<LUID>_phys_0)\Dedicated Usage`, `Shared Usage`, `Total Committed` (existem neste PC) | Bytes do processo em segmentos dedicados vs. do sistema; é o que o Gestor de Tarefas mostra | Que alocação foi movida; inclui memória do runtime |
| Residência por segmento | `D3DKMTQueryStatistics` (gdi32, `d3dkmthk.h` do WDK) | Estatísticas por processo e segmento (comprometido, residente) | Semidocumentado; campos a confirmar no SDK antes de usar |
| Sintoma | Tempo da placa nos grafos fixos por token (`STRATA_POLY_TRACE`, métricas de inferência) | Pesos densos ou KV lidos por PCIe tornam as camadas fixas mais lentas, com o mesmo trabalho | A causa |
| RAM | `GlobalMemoryStatusEx().ullAvailPhys` | Se encolher a cache é possível sem mandar experts para o disco | Quais páginas de experts estão em RAM |

**Alocado vs residente, na prática.** O motor conhece o que alocou (`Engine::budget()`). Proposta: no `READY`,
guardar `base = Dedicated Usage − VRAM alocada pelo motor` (a parte do runtime); depois,
`défice = VRAM alocada + base − Dedicated Usage`. Um défice acima de, por exemplo, 256 MiB durante várias
amostras seguidas indica memória do motor fora da VRAM. Antes de confiar nisto, a E2 tem de mostrar que o défice
é ~0 num motor sem pressão (o que hoje **não** se verifica: secção 2).

`QueryVideoMemoryInfo` só responde pelo processo que chama, por isso o orçamento e o uso têm de vir do motor; os
contadores PDH podem ser lidos pelo servidor Python pelo PID do motor, sem tocar no motor.

## 4. Recuperar VRAM sem reiniciar

- **Dar VRAM a outro programa:** já funciona por `resize_cache()`. Falta medir se `hipFree` devolve a memória ao
  WDDM (a `Dedicated Usage` do processo desce) ou se o runtime HIP a guarda em cache; o comentário de
  [device.cu:117](../../src/core/device.cu:117) diz que o runtime guarda memória libertada. Se a guardar, o encolhimento
  não serve a outro programa (E1).
- **Voltar a crescer:** funciona até `full_bytes_`. Passar desse teto é uma mudança pequena (o teto passa a ser
  todos os experts, limitado pelo orçamento), mas um crescimento que passe o orçamento leva à despromoção; por isso
  o crescimento tem de olhar para `Budget − CurrentUsage`, não só para o livre do HIP.
- **KV em VRAM:** não é redimensionável a quente. `plan_kv()` fixa `kv_split_` no arranque, os grafos de atenção
  são gravados por tamanho (`head_key()`) e mover linhas exige copiar a KV. Não proponho isto.
- **VMM do HIP** (`hipMemAddressReserve`/`hipMemCreate`/`hipMemMap`, o que o motor Qwen usa no CUDA): não sei se o
  HIP do Windows o suporta; verificar com `hipDeviceAttributeVirtualMemoryManagementSupported` (E5). Não é preciso:
  vários `hipMalloc` de segmentos chegam, porque os slots são ponteiros absolutos (secção 1).
- **Reinício controlado:** perde a KV, o estado e as conversas estacionadas (estão na RAM do processo), mas é o
  único caminho que repõe tudo. Já existe (`/v1/unload` + `/v1/load`, ou o reinício por falha).

## 5. Arquitetura proposta

Tudo opt-in; sem a opção, o caminho de hoje fica igual ao byte (regra do [AGENTS.md](../../AGENTS.md)).

### Fase 0: medir (sem mudar o caminho por omissão)

As experiências da secção 7, num PC sem outro motor ativo. Decidem se as fases seguintes valem a pena.

### Fase 1: monitorização (só leitura)

- **Motor:** um comando `MEM` entre pedidos (mesmo padrão de `VRAM`) que responde
  `budget_local, usage_local, usage_nonlocal, hip_free, vram_alloc, pinned_alloc, ram_avail, expert_slots`. Opcional,
  atrás de uma opção: um thread que acorda pelo evento de orçamento (ou a cada 1 s) e só escreve no stderr quando o
  estado muda. Nada de linhas novas no stdout durante um pedido: o servidor faz o routing do stdout.
- **Servidor:** lê os contadores PDH do PID do motor (via `pdh.dll` por `ctypes`, nada para instalar) a cada 2 s, e
  expõe em `GET /v1/status` → `vram.residency` e em Prometheus (`serve/prometheus.py`). Estados:

| Estado | Condição (com histerese: N amostras seguidas) |
|---|---|
| OK | `usage_local ≤ budget − reserva` e défice < 256 MiB |
| TIGHT | `budget − usage_local < reserva/2`, ou o orçamento desceu |
| OVER | `usage_local > budget` |
| DEMOTED | défice ≥ 256 MiB, ou tempo dos grafos fixos > 1,3× a base (limiares a fixar pela fase 0) |
| RAM_LOW | `ram_avail` < `--conversation-cache-min-free-mib` + bytes que um encolhimento mandaria para a CPU |

### Fase 2: recuperação controlada (entre pedidos)

Mudanças no motor, todas dentro de `resize_cache()`:

1. Depois de encolher, os experts que saíram da VRAM são lidos para a RAM (`map_pages()` só com esses intervalos) e
   `warming_` fica ligado até acabar, como no arranque. Corrige a lacuna 1.
2. Encolher é recusado (`ERR`, motor continua) quando a RAM disponível não cobre os bytes que passam para a CPU mais
   o piso de RAM. Corrige a lacuna 4.
3. Teto de crescimento opcional acima de `full_bytes_`, limitado por `Budget − CurrentUsage − reserva`.

Política no servidor (Python, testável sem GPU como `tools/test_setup_*.py`):

- TIGHT/OVER → `VRAM <reserva maior>` no próximo intervalo entre pedidos.
- OK estável durante 60 s e com `Budget − usage_local > reserva + 512 MiB` → crescer de volta.
- No máximo um redimensionamento por 60 s (uma reconstrução custa ~4 s; o caso de 4 s *por pedido* já aconteceu).
- DEMOTED que persiste depois de um encolhimento, ou `ERR` de redimensionamento → reinício controlado quando o
  motor estiver livre, com a mensagem no log. Nunca a meio de uma resposta.
- RAM_LOW → não encolher; registar e deixar o Windows decidir (despromover VRAM é mais lento mas não mata o motor;
  um expert lido do disco com o routing na placa mata).

### Fase 3: recuperação dinâmica (a meio de um pedido), se a fase 0 a justificar

- **Arena em segmentos:** `exp_` passa a uma lista de `hipMalloc` de, por exemplo, 256 a 512 MiB, cada segmento
  cheio com experts de uma ou poucas camadas, por ordem. As `runs_` partem nas fronteiras; o caminho na placa e o
  do host não mudam.
- **Encolher um segmento** num limite de token (depois de `sync(st_)`): parar o `follow_loop()` e esperar pelo stream
  das trocas; pôr a null os slots do segmento no host e na tabela `dslot_` (`slot_changed()`); sincronizar; só
  então `hipFree`. Ligar `warming_` e ler as páginas desses experts. Sem reconstrução: os outros segmentos ficam.
- **Crescer um segmento:** `hipMalloc`, e os slots novos são preenchidos pelo mecanismo de trocas que já existe
  (`follow()`, pinned → stream próprio → `slot_changed()`), que já corre a meio de uma resposta.
- O monitor do motor pede a ação por uma flag atómica; o ciclo de geração executa-a no limite de token seguinte.

### Invariantes de segurança (todas as fases)

1. Só a arena de experts e as tabelas de slots mudam. KV, estado recorrente, snapshots, scratch, staging e grafos
   nunca são libertados nem mudados por esta funcionalidade.
2. Ordem fixa: slots a null no host e na placa → sincronizar `st_` e o stream das trocas → `hipFree`. Nenhum kernel em
   voo pode ler um segmento libertado.
3. Libertar antes de alocar (o pico de VRAM nunca sobe durante uma ação de recuperação).
4. Um expert que passa para a CPU tem as páginas lidas antes de o routing na placa voltar a ser usado.
5. Qualquer erro HIP numa ação de recuperação segue a política atual: `gpu_fatal()`, processo novo. Não há rollback
   parcial (a [HIP_GPU_FAILURES.md](../HIP_GPU_FAILURES.md) explica porquê).
6. Histerese, limite de frequência e limites mínimos (128 slots, como no motor Qwen) contra oscilação.
7. Sem a opção, nenhum thread, comando ou leitura novos.

### Como verificar

- **Paridade:** greedy com o mesmo prompt: (a) sem redimensionar; (b) encolher a meio da conversa entre dois pedidos,
  continuar; (c) encolher e voltar a crescer. Os tokens têm de ser os mesmos (um expert na CPU dá a mesma escolha?
  a medir: o NVIDIA mediu "token a token" no motor Qwen, o poly HIP nunca foi medido).
- **KV/estado preservados:** depois de um redimensionamento, `RESUME` tem de retomar na mesma posição (sem reler o
  prompt) e a resposta tem de ser a de (a).
- **Falha:** injeção de erro no caminho de redimensionamento com o mock de `tests/poly_gpu_failure_mock/`.
- **Pressão real:** E4.

## 6. Custos esperados (a medir; nada disto foi medido)

- Fase 1: uma chamada `QueryVideoMemoryInfo` por segundo e PDH a cada 2 s, fora do caminho do token.
- Fase 2: igual ao `resize_cache()` de hoje (~4 s medidos antes) mais a leitura das páginas despromovidas.
- Fase 3: um `sync` por ação num limite de token; o crescimento entra pelas trocas, que já competem com a
  descodificação.

## 7. Experiências que faltam (fase 0)

Todas num PC sem outro `strata-poly` ativo (como o `guarded.py` da fase de prefill).

| # | Pergunta | Como |
|---|---|---|
| E1 | `hipFree` devolve a VRAM ao WDDM? | Processo de teste: alocar e tocar 4 GiB, libertar; PDH `Dedicated Usage` antes/depois; noutro processo, `Budget` antes/depois. Também com 8 segmentos de 512 MiB |
| E2 | O défice de ~1,47 GiB é despromoção, KV não tocada ou runtime? | Motor com contexto 8K vs 262K, em repouso; depois de um prompt de 200K; PDH e `D3DKMTQueryStatistics` a cada passo |
| E3 | O evento de orçamento dispara num processo só HIP? | Registar o evento e alocar VRAM noutro processo |
| E4 | O que acontece sob pressão real? | Motor a descodificar; outro processo aloca 2/4/6 GiB; registar orçamento, uso, PDH, tok/s, com e sem `VRAM` |
| E5 | Há VMM no HIP do Windows? | `hipDeviceGetAttribute(hipDeviceAttributeVirtualMemoryManagementSupported)` |
| E6 | Quanto custa `resize_cache()` aqui? | `VRAM 4000` e `VRAM` (sem número) entre pedidos; tempo e tok/s depois, com e sem páginas lidas |
| E7 | `SetVideoMemoryReservation` protege a memória densa/KV do HIP? | Reservar o denso + KV e repetir E4 |
| E8 | Custo de `QueryVideoMemoryInfo` | 10 000 chamadas cronometradas |

## O que não foi feito

- Nenhuma alteração de código e nenhuma compilação.
- Nenhuma alocação na GPU (o motor do servidor estava ativo). Os números de 2 são uma leitura passiva de contadores
  e dos logs que já existiam.
- Não se confirmou o comportamento do evento de orçamento, de `SetVideoMemoryReservation`, do VMM nem do cache de
  memória do runtime HIP no Windows: estão nas experiências E1, E3, E5 e E7.
- As 4 falhas de hoje (`GPU_FATAL` pela espera de 1 s) foram vistas, não investigadas.
