> Historical measurement report. Use the [current setup instructions](../../README.md). Personal paths were generalized for publication; measurements were preserved. Raw campaign links may require the local evidence archive.

# Piso da cache de experts (`--expert-cache-min`): medido

Segue-se a [investigação da residência de VRAM](AMD_HIP_VRAM_RESIDENCY.md) (fase de investigação, sem código
alterado). Aqui está o que foi implementado e o que foi medido.

**O que a opção faz.** `--expert-cache-min N` é o menor número de experts que a cache pode ter. Por omissão é 0 e o
caminho é o de hoje. Quando a margem pedida (`--vram-reserve-mib`) não é alcançável com o piso, **o piso ganha** e o
engine diz quanto ficou a faltar: `vram_reserve_short_mib` na linha `INFO` e no `POST /v1/vram`. O piso é o mínimo, não
um alvo: se a margem couber com mais experts, a cache fica com mais.

## Como foi medido

- RX 9060 XT 16 GB, Windows; orçamento DXGI para este processo: **15 416 de 16 304 MiB** (igual em todos os
  carregamentos medidos aqui).
- Qwen3.6-35B-A3B-UD-Q3_K_M, 10 496 experts encaminhados, 40 camadas MoE com kernels na placa, `--threads 6`.
- `build-hip-poly-win\strata-poly.exe` (engine 0.1.41) e, como controlo do comportamento anterior,
  `build-hip-poly-q8-operational-win\strata-poly-q8.exe`.
- Estado apertado = `VRAM 40000`: pedir uma margem que a placa não pode deixar livre. É a mesma condição que o guarda
  `mind_vram()` vê quando o livre desce abaixo de um quarto da margem (o caso dos logs "little VRAM left").
- Os logs e os scripts das duas medições estão em
  [bench/results/2026-10-18-hip-expert-cache-floor](../../bench/results/2026-10-18-hip-expert-cache-floor/README.md).

## 1. O piso quando a margem não cabe (8K, KV fp16)

`--max-context 8192`, cache cheia depois do carregamento (7 217 experts, 9.35 GiB), 969 MiB livres.

| Depois de `VRAM 40000` | sem a opção (hoje) | `--expert-cache-min 2000` |
|---|---|---|
| cache | 7217 → **0** de 7217 experts (0.00 GiB) | 7217 → **2000** de 7217 experts (2.58 GiB) |
| VRAM livre | 11 660 MiB | 9 021 MiB |
| 8 tokens gerados no estado apertado | **7.4 s** | **2.6 s** |
| 8 tokens gerados com a cache cheia (antes) | 0.6 s | 0.4 s |
| voltar a pedir `VRAM 40000` | volta a encher e a esvaziar | fica em 2000 (não mexe) |

O mesmo par, num pedido depois de `VRAM 8192` com o piso ligado: **0 linhas de corte e 0 de reenchimento** durante o
pedido (o guarda deixa de agir porque o piso já está atingido).

O piso não bloqueia nada do resto: com `--expert-cache-min 2000`, `VRAM 3072` deixou a cache com 5 692 experts e
`VRAM 256` voltou a enchê-la até 7 257 (9.40 GiB) — com o piso em 2000, o corte para `VRAM 8192` parou em 2 625 experts
(é o que cabia com 8 192 MiB livres), não em 2000.

## 2. O que custa uma reconstrução (8K, KV fp16, os tempos que o próprio engine reporta)

| pedido | cache | tempo |
|---|---|---|
| `VRAM 8192` | 7257 → 2625 experts | 2 848 ms |
| `VRAM 3072` | 2625 → 5692 experts | 6 327 ms |
| `VRAM 256` | 5692 → 7257 experts | 7 599 ms |
| `VRAM 8192` | 7257 → 2625 experts | 2 751 ms |
| `VRAM` (por omissão) | 2625 → 7198 experts | 7 686 ms |
| `VRAM 40000` | 7217 → 0 experts | 94 ms |

Encolher é rápido (0.09-2.8 s); encher depois é que custa (6.3-7.7 s), e a primeira leitura das páginas pagas no
pedido seguinte. É a ordem de grandeza do "~4 s por reconstrução" do relatório.

## 3. O piso não custa nada quando a margem cabe

262K com KV `q8_0`, `--expert-cache-max 2000` (a configuração com que a app arranca a 262K), binário anterior e binário
novo com `--expert-cache-min 2000`:

| | anterior | com o piso |
|---|---|---|
| cache | 2000 experts (2 638-2 639 MiB) | 2000 experts (2 638-2 639 MiB) |
| VRAM livre depois do carregamento | 6 201-6 202 MiB | 6 201-6 202 MiB |
| primeira linha `INFO` | sem as duas chaves | `expert_slots_min=2000 vram_reserve_short_mib=0` |
| 3 pedidos de 16 tokens | 0.6 / 0.4 / 0.4 s | 0.6 / 0.4 / 0.4 s |
| cortes, reenchimentos | 0, 0 | 0, 0 |

Sem a opção, a linha `INFO` traz `expert_slots_min=0 vram_reserve_short_mib=0` e os números são os de sempre. A
matemática do corte não muda quando o piso é 0: a função só chega a essa linha quando o livre já está abaixo da margem,
e aí `max(livre - margem, 0)` é o valor de hoje.

A margem curta aparece também ao carregar: com o piso em 2000 no 8K, o carregamento deixou 987 MiB livres para uma
margem de 1 024 e a linha `INFO` disse `vram_reserve_short_mib=24` — informa e não age.

## 4. O caso do relatório: onde estava o meu erro

O caso foi reportado com **KV `q8_0`**. Duas coisas que escrevi antes estão erradas.

**1. "O orçamento nunca desceu".** Li isso da linha `Windows budgets N` dos 17 carregamentos. Essa linha sai ao
arrancar, **antes** de os pesos serem alocados — e é ao alocá-los que o degrau acontece, portanto não pode mostrá-lo. O
degrau mede-se com `hip_poly_probe vram` e mede-se já:

| memória retida pelo processo de teste | o que o Windows retira ao resto |
|---|---|
| até 887 MiB | 1 493 MiB |
| 984 a 10 240 MiB | 887 MiB |
| de 10 752 MiB em diante | ~2 225 MiB |

É o degrau do relatório (orçamento 15 271 → 13 933; uma medição anterior deu 15 416 → 13 937), e aparece **aos ~10.5 GiB
retidos**, o mesmo ponto que o relatório diz.

**O engine já conta o livre dentro desse orçamento.** `src/core/device.cu:188-213`: depois do `hipMemGetInfo`, o engine
tira-lhe o que o orçamento retém (`free -= cartão - QueryVideoMemoryInfo().Budget`) e volta a perguntar em cada
`mem_get_info()` — só a frase `Windows budgets N` sai uma vez, e `STRATA_WDDM_BUDGET=0` desliga a correção. Isto está em
`HEAD` desde `5a7f0860` ("HIP: count the desktop's VRAM on Windows", 2026-10-01) e está também no binário Q8 congelado
(a linha sai no log da corrida A). O degrau chega portanto ao guarda e ao `POST /v1/vram` no pedido seguinte; não é um
ponto cego. Vê-se no grau que o orçamento retinha nas duas corridas de 262K do ponto 5:

| | retido (soma da linha `VRAM MiB`) | livre medido | o que o orçamento retinha | `hip_poly_probe` |
|---|---|---|---|---|
| A e B (com tecto) | 8 860 MiB | 6 201 MiB | ≈1.2 GiB | 887 MiB (grau médio) |
| C (sem tecto) | 12 976 MiB | 1 014 MiB | ≈2.3 GiB | ~2 225 MiB (grau alto) |

(a coluna do meio sai de `16 304 - livre - retido`; a soma `retido` inclui um resto de 1 002 MiB que é o meu resíduo, e
é daí que vem a folga de algumas centenas de MiB, não do degrau.)

**2. O controlo estava na configuração errada.** Comparei com o perfil fp16; o caso é q8_0, e o tecto da app esconde-o:
`tools/hip/poly_launch.py:36` passa `--expert-cache-max 2000` a 262K. É por isso que os **24 logs q8_0 de ≥200K
posições** do repositório mostram todos cache 2 639 MiB, 6 201 MiB livres e 0 reenchimentos — não é o q8_0 que é
seguro, é o tecto que o trava. Sem tecto, o mesmo perfil a 262K fica: dense 1 961 + cache 5 730 + KV 2 992 + palco 16 +
estado 251 + resto 1 002 = **11 952 MiB retidos**, dentro da banda do degrau. Medido agora, binário congelado e binário
novo:

| | sem piso | com `--expert-cache-min 2000` |
|---|---|---|
| cache depois do carregamento | 5 730 MiB (≈4 330 experts a 1.33 MiB) | 5 880 MiB |
| VRAM livre | 1 724 MiB | 1 579 MiB |
| cortes, reenchimentos | 0, 0 | 0, 0 |
| 3 pedidos de 16 tokens (o 1.º a frio) | 9.0 / 1.0 / 0.6 s | 4.8 / 1.2 / 0.4 s |
| pedido de 400 tokens | parou aos 25 (paragem natural), 0.6 s | 400 tokens em 4.2 s |

O binário atual não entra no ciclo descrito no relatório: ao carregar, a cache é encolhida pelo que falta à margem com
a cache no lugar, até a margem estar lá. Isso veio com `473a6476` ("an expert cache that keeps its reserve"), que é
anterior a esta alteração — é esse commit que fecha o ciclo; sem ele o guarda cortava no primeiro pedido e reenchia
~4 s por pedido. O piso não muda nada aqui: o guarda dispara com o livre abaixo de um quarto da margem (384 MiB nos
1 536 passados nesse teste; a app escolhe 1 019 a 262K — ponto 5) e o livre ficou em 1 579-1 724 MiB. A diferença de
150 MiB na cache entre os dois carregamentos está dentro do espalhamento que já se mede entre carregamentos iguais
(923 MiB nos logs fp16) e não a atribuo ao piso.

Do que escrevi mantém-se isto: nos dois casos do repositório com o guarda a disparar (`campaign2-first/G0-r1` e `G4-r1`,
campanha de 2026-10-13 marcada *contaminated*), o gatilho foi o livre a chegar a ≈0 com **outro programa** na placa.

## 5. Pela API, a 262K, sem tecto: a configuração do relatório

A app não usa a linha de comandos: `tools/hip/poly_launch.py` escreve um config e o `serve/server.py` fala com o
engine. Pela API (`logs/poly_server_probe_q8_262k.py`), 262K, KV `q8_0`, o mesmo perfil de experts, três corridas
seguidas — A é a política com que a app arranca a 262K (`--expert-cache-max 2000`, `poly_launch.py:36`), B a mesma com o
piso, C **sem tecto**, que é a configuração do relatório:

| | A. política da app, sem piso | B. com o piso | C. sem tecto, com o piso |
|---|---|---|---|
| binário | congelado | novo | novo |
| conjunto de experts | `a1f73bb14a36142e`, o que o launcher espera a 262K | `3e276751da60c027` | `e634912fe6e7a4cd` |
| cache depois do carregamento | 2 639 MiB (2 000 experts) | 2 639 MiB (2 000) | **6 754 MiB (5 097 experts)** |
| VRAM livre | 6 201 MiB | 6 201 MiB | **1 014 MiB** |
| carregamento | 15.0 s | 28.5 s (12.5 s de páginas frias) | 28.7 s |
| 16 tokens (frio) | 0.9 s | 0.9 s | 0.6 s |
| 400 tokens | 11.9 s (35.4 tok/s) | 12.5 s (33.8 tok/s) | **8.8 s (47.6 tok/s)** |
| prompt de 10 686 tokens | 7.1 s (1 506 tok/s) | 12.8 s (835 tok/s) | 9.6 s (1 118 tok/s) |
| `{"reserve_mib": 12288}`, que a placa não dá | **0 experts** em 2 ms, `status: ok` e nada mais | 2 000 experts ficam, 2.2 s, `vram_reserve_short_mib=6124` | 5 097 → **2 000** (o piso), 2.2 s, `short=6124` |
| de volta à margem do arranque | 7.9 s: 0 → 2 000 experts | 2.1 s: nada a repor | 5.5 s: 2 000 → 5 097 |
| linhas do guarda | 0 | 0 | 0 |

A margem que o engine escolhe sozinho é **1 019 MiB** — é o que `POST /v1/vram` devolve em `reserve_mib`, e é a omissão
de `engine.cpp:903` (um dezasseis avos da placa limitado a 512-1 024 MiB; 16 304/16 = 1 019). É por isso que os logs
fp16 de 262K pousam em 991-1 143 MiB livres.

**O piso faz o que devia (A contra B e C).** A margem impossível deixa de custar a cache inteira: 0 experts e HTTP 200
`ok` sem uma palavra de aviso em A, contra 2 000 experts mantidos em B ou uma perda de 4 115 MiB em vez de 6 754 em C, e
com `vram_reserve_short_mib=6124` mais "- the cache's floor left that reserve short" no stderr do engine.

**O ciclo do relatório não aparece em nenhuma das três.** Zero linhas `little VRAM left`, com um prompt de 10 686 tokens
e — em C — com 6 754 MiB de cache e 1 014 MiB livres, logo acima dos 1 019 da margem. O guarda só age com o livre abaixo
de um quarto da margem (`engine.cpp:1121`: `vram_free_mib() >= reserve_mib_ / 4`), isto é, abaixo de ~255 MiB. De pé
fica o que o ponto 4 diz: o ciclo "esvaziar e encher" precisa do guarda a disparar, e com este binário a cache é
encolhida no carregamento até a margem estar lá.

**O que custa uma reconstrução é o número de experts, não os bytes.** Todos os `kept free` destas medições, mais os do
ponto 2:

| experts na cache depois | tempo | ms por expert |
|---|---|---|
| 0 | 94 ms | — (largar é quase grátis) |
| 2 000 | 2 127-2 276 ms | 1.07-1.14 |
| 2 625 | 2 848 ms | 1.08 |
| 5 097 | 5 391-5 674 ms | 1.06-1.11 |
| 5 692 | 6 327 ms | 1.11 |
| 7 198 | 7 686 ms | 1.07 |
| 7 257 | 7 599 ms | 1.05 |

≈**1.1 ms por expert** — a reposição que o comentário do `mind_vram()` cita como "4 s" é a de uma cache de ~3 700
experts. E o custo não é ler páginas: em C, um `POST /v1/vram` que não muda nada (**5 097 → 5 097 experts**) levou
5 671 ms, porque `resize_cache()` (`engine.cpp:1087`) liberta a alocação e volta a construir a cache toda, sem saída
rápida para o caso de o número não mudar. Fica como está, por decisão registada abaixo.

**As latências entre corridas não são do piso.** Em B o pedido de 10 686 tokens deu 12.8 s (835 tok/s) e o
carregamento 28.5 s, com a linha do próprio engine a dizer que leu as páginas dos experts em 12.5 s
(`STRATA_POLY_WARM`); em A o mesmo pedido deu 7.1 s (1 506 tok/s) e em C 9.6 s (1 118 tok/s), com 2.2 s de leitura. O
carregamento mais lento de todos foi o primeiro de todos (53.5 s, ficheiro de 17 GB frio). Nos dois carregamentos
iguais (A e B) o estado final é idêntico (2 000 experts, 6 201 livres) e o guarda não agiu em nenhum, portanto a
diferença é o estado da cache de páginas do ficheiro, não o piso.

## 6. Estes ensaios escreveram no ficheiro fixado do perfil (corrigido)

`--expert-profile FILE` não é um ficheiro só de leitura: o engine lê as contagens de uso para escolher os experts que
ficam em VRAM e **volta a escrever o ficheiro** no fim (`main.cpp:1100`, "written back on...", `Engine::save_profile()`
chamado em `main.cpp:1438` ao sair). O launcher sabe-o e por isso copia o perfil fixado para a pasta da corrida e passa
a cópia ao engine (`poly_launch.py:179-181`, `expert.write_bytes(PROFILE.read_bytes())`).

As três corridas do ponto 5 foram lançadas por um script meu que passou o **próprio ficheiro fixado**
(`bench/results/2026-10-15-hip-kv-q8-phase4b/provenance/expert-profile-campaign.txt`) ao engine, e o engine reescreveu-o:
as 10 496 linhas mantêm-se, os números é que sobem (é `camada expert contagem`, e cada corrida soma às contagens — a
primeira linha foi de `0 0 204522` para `0 0 204575`). O ficheiro ficou com 149 350 bytes contra 149 330, sha
`36e2a7dc…` em vez de `1bc70f7e…` (mtime 12:57:41, o fim da corrida C), e a partir daí o `START-HIP` recusou com
"SHA-256 inesperado".

Isto mostra que **basta uma corrida** para trocar o sha: a cópia que a corrida de verificação usou estava igual ao
fixado no momento do carregamento (149 330 bytes, `1bc70f7e…`, a linha do log) e saiu do engine com 149 332 bytes,
`188c62f1…`. Não há "escrever o mesmo"; é um acumulador.

Foi corrigido assim:

| | |
|---|---|
| restaurado de | uma cópia do próprio launcher, `tmp/hip-launch-q8/q8-bvgclpci/expert-profile.txt` (10:53:16, anterior à escrita; 149 330 bytes, `1bc70f7e…`) — a pasta `provenance/` não está no git, portanto não havia outra fonte |
| sha agora | `1bc70f7edf706578a24aca5a6e8beee4f308e192142ae88339968f5da1924a9d` — igual ao `PROFILE_SHA` |
| verificado com | `checked_file()` do próprio `poly_launch.py`, para o perfil e para o `.exe` Q8 (ambos passam) |
| a versão escrita pelo engine | guardada ao lado, `…/expert-profile-campaign.txt.engine-wrote-back-2026-10-18` |
| a sonda | passa a copiar o perfil para a pasta da corrida e a recusar o ficheiro fixado (`poly_server_probe_q8_262k.py`) |

**O que a escrita mudou e o que não mudou.** Só as contagens. A corrida A — binário congelado, lançada antes de eu ter
escrito no ficheiro — escolheu o conjunto `a1f73bb14a36142e`, que é exatamente o que `poly_launch.py:36` espera a 262K
com 2 000 experts; B e C, que leram o perfil já com as contagens de A por cima, escolheram `3e276751da60c027` e
`e634912fe6e7a4cd`. O id do conjunto acompanha portanto o **conteúdo** do ficheiro, contagens incluídas: a corrida de
verificação, com o perfil reposto e a mesma política de A (2 000 experts, novo binário, piso ligado), voltou a dar
`a1f73bb14a36142e` — o conjunto que o launcher espera. As contagens em cache nunca dependeram disso (2 000 experts /
2 638 MiB em A e B; 5 097 / 6 754 MiB em C), por isso as medições dos pontos 2 a 5 mantêm-se.

Nota de método: as sondas de 8K (pontos 1 a 3) passaram `C:\models\qwen36-q3-pack\expert-profile.txt`, que é
a cópia de trabalho do próprio pack e que os configs da app já apontam e o engine já escreve em cada corrida — nada está
fixado a ela. Só o ficheiro fixado pelo launcher foi tocado, e está reposto.

## Decisão: `resize_cache()` fica como está (2026-10-18)

O desperdício é real e está medido: em C, dois dos quatro `VRAM` não mudaram nada e custaram 5 671 e 5 391 ms. Fica
mesmo assim como está, por três razões:

1. **Não serve o guarda.** Quando o guarda dispara, o VRAM livre está mais baixo e o que cabe são **menos** experts, logo
   a contagem muda sempre e a saída rápida nunca se aplicaria. O que o guarda precisa é de não ser chamado, e disso trata
   o piso.
2. **A app quase não lá passa.** Num arranque limpo `serve/server.py` não manda `VRAM` nenhum (`vram_reserve = None`,
   `2708`); só o reaplica depois de um reinício do engine (`3034`), uma vez, e aí só se a última margem for igual à do
   arranque é que sai de graça.
3. **A correcção segura não é uma condição a mais.** Uma condição dentro da `resize_cache()` passaria a ter a sua cópia da
   regra de dimensionamento do `fill_expert_cache()`, e duas cópias divergem em silêncio — a classe de defeito que este
   trabalho persegue. A forma segura é tirar a regra para uma função que os dois usem, e isso mexe no caminho de
   carregamento que está medido aqui.

Fica o registo do que faria mudar a decisão: se a margem em falta for **menor que um expert** (1.33 MiB), o guarda
reconstrói a cache toda e acaba com a mesma contagem — dispara outra vez no pedido seguinte, e outra vez, ~4 s por
pedido sem mudar nada. É uma explicação possível para o ciclo do relatório, mas **não está medida**: nas três corridas
do ponto 5 o guarda disparou 0 vezes, e nos dois logs guardados onde disparou as contagens mudaram muito (4 648 → 2 351
e 6 039 → 4 874). Se aparecer num log um par de linhas `little VRAM left` com a **mesma** contagem de experts, esta
decisão muda.

## O que fica por medir

- O degrau **a meio de uma sessão** (outro programa a ocupar a placa depois de carregar): é a experiência E4 do
  relatório. As peças já estão todas no sítio — o engine vê o orçamento a cada chamada e a cache é encolhida no
  carregamento —, mas o caso em que o degrau chega com a cache já no lugar não foi provocado.
- Repetir o ciclo com um binário anterior a minha alteração (o guarda a cortar e a repor por pedido), para medir os
  ~4 s num sítio só.
- O custo por token com a cache no piso durante um pedido longo, não só 8 tokens.
- O que acontece à cache depois de um corte feito pelo **guarda** e não pelo comando `VRAM`: hoje o guarda corta e não há
  caminho que a volte a encher sozinho. O piso evita o corte repetido, mas não repõe o que já foi perdido.
