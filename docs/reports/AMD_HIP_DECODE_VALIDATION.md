> Historical measurement report. Use the [current setup instructions](../../README.md). Personal paths were generalized for publication; measurements were preserved. Raw campaign links may require the local evidence archive.

# PolyStrata HIP: continuação da validação do decode

Estado: esta etapa de validação terminou em **12/10/2026, 19:50 (Europe/Lisbon)**. Os ganhos foram confirmados e os
testes executados não revelaram uma regressão nova de correção. O binário novo continua **experimental**:
a causa inicial do WATCHDOG e a recuperação de um processo que atravessa um TDR não estão resolvidas.

Medido neste PC em 12 de outubro de 2026 (Europe/Lisbon): RX 9060 XT 16 GB, gfx1200, Ryzen 5 5600X,
32 GB de RAM, Windows 11, driver AMD 32.0.31041.1004 de 17/08/2026, ROCm TheRock 10.2.0a20260930.
Modelo: `C:/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`. Não foram alterados driver, TDR ou limites de potência.

## 1. Estado encontrado e preservação

Ramo `hip-poly`, HEAD `7023db0ce1d5a5ee8801e763c8fbcf0a6e02d83f`. A cópia de trabalho já continha alterações
do decode e de experiências com outros modelos. Não havia servidor nem benchmark anterior ativo na inspeção inicial.
Os dois anexos do utilizador foram lidos integralmente; o segundo é um registo resumido da investigação, não uma
captura completa de comandos ou de todos os logs.

| Binário preservado | Data de compilação | SHA-256 |
| --- | --- | --- |
| `build-hip-poly-win/strata-poly-base.exe` | 11/10, 21:05:51 | `d9037b8c9e7cf407693a7a9ddff70256937a9d1a3a802349d950644be899521f` |
| `build-hip-poly-win/strata-poly.exe` | 12/10, 18:39:03 | `7feddc7ef0d1c398f7a8a7a6d68b8bc9fee4282bcf1e1dfb062104096ff2db2b` |

O manifesto, o diff inicial e os resultados desta continuação ficam em
[`bench/results/2026-10-12-hip-decode-validation/`](../../bench/results/2026-10-12-hip-decode-validation).
Os resultados anteriores ficam em [`2026-10-12-hip-decode-opt/`](../../bench/results/2026-10-12-hip-decode-opt).
Os logs temporários ainda existentes foram copiados para a pasta de validação.

Não foi feito commit nem push. Nenhuma otimização nova foi acrescentada nesta continuação.

Sequência reconstruída dos anexos, nomes e timestamps dos artefactos:

| Fase | Evidência preservada |
| --- | --- |
| Referência da migração Q3, 11/10 | Executável base de 21:05 e relatório AMD_QWEN36_Q3_MIGRATION.md |
| Profiling, routing de janelas e atenção, 12/10 | trace.json; parity-w1-*.json e parity-default-ids*.json escritos entre 17:07 e 17:26; attn-decode-before/after.txt |
| Routing do draft | ab-draftcard.json, 17:40 |
| Vocabulário reduzido do draft e controlos PT/código | trace-ptcode.json, 17:43; trace-noswap.json, 17:47; ab-dvocab.json, 17:59 |
| A/B completos | bench.json, último resultado às 18:38:21 |
| Compilação e suite | Executável de 18:39:03; ctest-all.log e LastTest com os quatro erros |
| WATCHDOG e corrida incoerente | WER e tmplqkakx30.log, 18:42; route-check.json, 18:42:49 |
| Repetições normais | tmpi9voynft.log e tmp3qhrvu_6.log, 18:44–18:45; repro.json |

Os timestamps dos JSON assinalam a escrita do resultado, não o instante de cada kernel. A separação dos contributos
do draft/vocabulário fica como experiência intermédia; a instabilidade e a identidade do binário final eram as
partes inconclusivas. Nenhuma dessas experiências é tomada como prova de correção por si só.

## 2. Alterações do Opus presentes no binário

- Routing HIP por token nas janelas de até oito tokens, com slots, pesos, produtos down e somas CPU separados.
  Os kernels MMVQ com slots em memória da GPU passaram a aceitar até 64 slots.
- Publicação do routing da janela depois de todos os blocos terminarem; consumo da soma CPU com sequência e
  espera limitada. O timeout de um segundo, a proteção durante o carregamento das páginas e a sequência base
  já existiam antes desta fase. O novo código aplica o caminho também ao bloco draft.
- Atenção Qwen com scores reutilizados e soma dos valores em partes, mantendo o softmax causal. O caminho antigo
  continua selecionável por `STRATA_POLY_ATTN=0` e é usado quando a geometria ou o espaço de scratch não permitem
  o novo. O caminho de KV dividida mantém a atenção anterior.
- Cabeça de saída do draft limitada por omissão aos primeiros 65 536 IDs. `STRATA_POLY_DRAFT_VOCAB=0` usa todo o
  vocabulário. A verificação e a geração do modelo principal usam o vocabulário completo.
- `STRATA_POLY_TRACE=1` mede as rondas no host; `=2` usa eventos na GPU. Estão desligados por omissão.

A correspondência foi verificada pelos diffs, configuração Ninja HIP/gfx1200, timestamps dos objetos posteriores
às fontes, mensagens do executável e execução dos caminhos de routing e atenção. Os hashes permitem identificar
exatamente os binários testados. Esta continuação não recompilou nem substituiu os binários.

## 3. WATCHDOG e geração B-31K

### Evidências

- Windows Error Reporting registou `LiveKernelEvent 141` às **18:42:39**, com
  `WATCHDOG-20261012-1842.dmp`, e voltou a reportar o mesmo dump às 18:42:41.
- O log da corrida incoerente `tmplqkakx30.log` terminou às **18:42:47**; `route-check.json` foi escrito às 18:42:49.
  O B-31K produziu 256 tokens, **55,08 tok/s**, 3/26 palpites corretos, 92,71% de hits de especialistas na VRAM
  e prefill 1 182,4 tok/s. Os **36 tok/s** do pedido inicial não são confirmados pelo artefacto preservado.
- O primeiro ID foi **225721**, em vez de **8160**. A divergência já existia no primeiro token amostrado do prefill,
  antes de se pedir um palpite MTP para essa resposta. Isto delimita o problema para além da aceitação do draft.
- A seleção de especialistas passou 35 309 decisões, sem divergência; a maior diferença de pesos foi 1,19e-7.
  Esse controlo compara o routing dos mesmos scores, não verifica que ativações, pesos ou KV estejam corretos.
  Assim, routing correto e geração incoerente podem coexistir.
- Dois processos novos logo depois geraram texto normal: 90,07 e 90,61 tok/s, 140/194 e 140/195 palpites corretos.
- O evento de 18:42 também reenviou relatórios antigos: minidumps de 5, 6 e 10 de outubro, e outros dumps de fevereiro
  e setembro. Não são crashes novos a contar nesta campanha. Há histórico de timeouts anterior a estas otimizações.
- O log não contém timeout do `wait_add`, erro HIP ou mensagem de OOM. Tinha cerca de 1 GiB de reserva de VRAM.
  Estes factos não excluem uma falha de acesso nem pressão de memória sob WDDM.

### Conclusão técnica provisória

O candidato mais forte é uma falha de execução ou perda de estado durante o reset da GPU, contemporânea do prefill
B-31K. É uma inferência temporal, não uma causa raiz demonstrada. Um erro de sincronização, um acesso inválido,
um kernel bloqueado ou um problema do driver podem ter desencadeado o WATCHDOG; os dados preservados não os distinguem.
Não há fundamento para atribuir o incidente só ao hardware ou só às alterações.
Não há timestamp do primeiro token ou do início do prefill dessa corrida nem dump de logits. Assim, a ordem causal
exata entre corrupção e reset não está estabelecida: a corrupção pode ter ocorrido durante a recuperação, ou a
falha que a produziu pode ter desencadeado o WATCHDOG. A ausência de repetição não resolve esta distinção.

O código foi auditado quanto a dimensões, slots por token, sequência de publicação CPU/GPU, reset do contador dos
blocos, atualizações da tabela de cache e sincronização antes de ler logits. Não foi identificado um defeito
determinístico que explique o incidente. As repetições ficam na secção 4.

O código 0x141 corresponde a timeout de um motor de vídeo, segundo a
[Microsoft](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x141---video-engine-timeout-detected).
A documentação [TDR](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery)
explica a perda de recursos no reset para Direct3D; isto dá contexto, mas não prova o comportamento específico do HIP.
O dump recente tem ACL de administrador e não pôde ser lido nesta sessão; não foram alteradas as permissões do sistema.

Há relatos upstream relacionados, em outros motores/configurações:
[#613](https://github.com/Niko1221/Strata/issues/613),
[#1233](https://github.com/Niko1221/Strata/issues/1233) e
[#1785](https://github.com/Niko1221/Strata/issues/1785). Não demonstram a causa neste PC. A pesquisa não encontrou
um tópico aberto HIP no fork PolyStrata. Não foi publicado nenhum comentário ou issue.

## 4. Correção e estabilidade

O teste `--no-draft` anterior não foi executado: `argparse` rejeitou `--engine-args "--no-draft"`.
Foi corrigido para `--engine-args="--no-draft"`, e a cadeia passa a parar quando um comando falha.
O validador de paridade também passou a rejeitar saídas vazias, prefixos truncados e probabilidades não finitas.
Quatro testes de regressão do validador passaram.

Paridade sem draft, janela 1, seis threads, contra as referências llama.cpp HIP da migração:

| Prompt | Tokens de entrada | Prefixo igual / 64 | Maior diferença de probabilidade |
| --- | ---: | ---: | ---: |
| short | 22 | 64 | 0,043 |
| code | 870 | 28 | 0,090 |
| doc-2.5k | 3 069 | 64 | 0,110 |
| doc-8k | 11 481 | 39 | 0,062 |
| doc-30k | 35 575 | 64 | 0,061 |

As duas divergências ocorreram em escolhas quase empatadas. Todos passaram o critério do projeto: diferença
máxima de probabilidade até 0,15, divergência apenas com gap até 0,1. Isto não é identidade bit a bit com llama.cpp.

`poly_kernel_bench attn-decode 30`: janelas de 2/3/4/8 tokens contra tokens isolados, a 2K/8K/31K/45K, sem
divergências bit a bit. A comparação atenção nova/antiga de um token teve diferença absoluta máxima **1,12e-8**.
`cols`: Q8_0, Q6_K, router F32 e delta net, de dois a oito tokens, todos iguais bit a bit.

A suite HIP serial terminou com **91 testes aprovados, nove saltados e três falhas**, em 139,20 s:

| Falha | Evidência e classificação |
| --- | --- |
| expert_cache_segmented_test | Recusa explícita: cache segmentada é CUDA-only. Já falhava na suite de 11/10; também reportado em [#1475](https://github.com/Niko1221/Strata/issues/1475) |
| hip_handoff | Limitação Windows/HIP da cópia para memória mapeada, documentada em docs/AMD_HIP.md; já falhava em 11/10 |
| ple_parity | Fixture/tabela Q2_0 externa em falta; já falhava em 11/10 |
| ple_reader_selftest, na suite anterior | A falha foi 6 leituras keep-alive em 300 ms, mínimo esperado 8. Passou cinco vezes isoladamente (112,73 s) e na suite serial. Compatível com pressão/agendamento; não prova absoluta da causa |

Os testes saltados não contam como validação bem sucedida. CUDA e SYCL não foram compilados ou executados neste PC.

Campanha B-31K (`validate_runs.py stability`): cinco processos, 40 respostas de 256 tokens, **10 240 tokens**.
Cada processo leu A-2K, B-2K, B-31K duas vezes, A-8K, B-31K, B-45K e B-31K. A repetição imediata reutilizou
31 295 tokens; as outras B-31K releram o prompt inteiro. Foram 20 B-31K, 15 com prefill completo e cinco com reutilização.

| Caminho | B-31K, tok/s (quatro corridas) | Routing verificado |
| --- | --- | ---: |
| Novo, processo 1 | 90,42 / 95,20 / 94,04 / 95,77 | 98 078 |
| Referência | 63,94 / 66,99 / 67,15 / 65,61 | não instrumentado nesta campanha |
| Novo, processo 2 | 92,92 / 93,85 / 97,25 / 100,58 | 94 220 |
| Novo, janelas no host | 80,10 / 81,68 / 80,86 / 83,33 | 12 826, tokens individuais/draft |
| Novo, sem draft e janela 1 | 65,13 / 65,89 / 66,20 / 65,83 | 81 920 |

Total: **287 044 decisões**, zero divergências de seleção, diferença de pesos máxima 1,19e-7. Todas as B-31K
começaram com os 12 IDs esperados. Os textos foram descodificados e revistos: nenhuma repetiu a mistura incoerente
de alfabetos da corrida de 18:42. Há variações de redação entre repetições, inclusive no binário base; não se
afirma geração bit a bit determinística entre estados diferentes da cache. Os prefixos coincidentes com a primeira
B-31K base variaram de 39 a 256 tokens. Dados completos em `stability.json`, `stability-decoded.json` e
`stability-summary.json`. Estas velocidades com verificação de routing não se misturam com o benchmark final.

Comparação direta base/novo, sem draft e janela 1, seis threads e expert profile, 64 passos por prompt:

| Prompt | Tokens iguais | Todos os logits finitos | Maior diferença de probabilidade | Maior diferença absoluta de logits |
| --- | ---: | --- | ---: | ---: |
| short, 22 tokens | 64/64 | sim | 0,0524 | 3,0619 |
| code, 870 tokens | 64/64 | sim | 0,0467 | 1,0184 |
| doc-30k, 35 575 tokens | 64/64 | sim | 0,0372 | 1,7536 |

É paridade pelo critério do projeto, não igualdade numérica. A comparação usa a configuração real com cache adaptativa;
o número e a seleção de especialistas em VRAM podem variar entre processos. Não se atribuem todas as diferenças
de logits ao kernel de atenção. Os logits completos de **95 354 880 valores** nos seis processos foram verificados;
ficam hashes, top-40 e estatísticas em `direct-parity.json`, e os binários de logits foram removidos depois.

## 5. Benchmarks A/B

Os três A/B anteriores já estavam completos antes do WATCHDOG. Recalculados diretamente do JSON, sem combinar
traços com benchmarks: B-2K 92,43 → 129,21 (+39,79%); B-8K 84,72 → 110,57 (+30,51%);
B-31K 69,88 → 102,79 (+47,10%); B-45K 63,62 → 94,62 (+48,73%).
O fim desses benchmarks foi às 18:38:21, antes da última compilação de `strata-poly.exe` às 18:39:03. Não há hash
do executável em cada corrida histórica. Portanto, a identidade dos resultados históricos com o binário final
não é presumida; a campanha abaixo valida o executável cujo SHA-256 foi preservado nesta sessão.

A campanha final usa de novo `tools/hip/bench_compare.py`: mesmos IDs A/B, contexto 49 152, seis threads,
expert profile, 128 tokens greedy, aquecimento separado de cada binário e três pares alternados base/novo.
Uma cópia congelada do expert profile é reposta antes de **cada processo**, incluindo os aquecimentos: o motor
grava contadores novos no ficheiro ao sair, pelo que reutilizar o mesmo perfil mutável não fixa a seleção inicial.
Esta condição está explicitada e os resultados novos são separados dos A/B históricos. Dentro de cada processo,
a cache adapta-se normalmente. Os testes anteriores de estabilidade e logits usaram o perfil operacional, que
foi atualizado pelo motor da forma habitual; nenhum contador foi descartado para repor uma versão anterior.
Prefill, TTFT, latência, MTP e picos de memória ficam no JSON. A comparação de 256 tokens com routing verificado
serve para correção e fica separada da medição de velocidade.

Medianas e intervalos das **três corridas por binário**, fase B a quente, com todo o prompt relido:

| Contexto real | Base tok/s (mín.–máx.) | Novo tok/s (mín.–máx.) | Ganho |
| --- | ---: | ---: | ---: |
| 2 068 | 92,39 (91,04–94,60) | 127,29 (123,60–127,48) | +37,77% |
| 8 235 | 87,86 (85,77–89,22) | 126,91 (118,90–127,69) | +44,45% |
| 31 296 | 68,40 (67,90–70,81) | 103,85 (101,40–106,02) | +51,83% |
| 45 355 | 64,17 (62,08–65,06) | 94,63 (94,38–95,43) | +47,47% |

| Contexto | Prefill base → novo, tok/s | TTFT base → novo, s | Latência total base → novo, s | Aceitação MTP base → novo |
| --- | ---: | ---: | ---: | ---: |
| 2K | 1 131,6 → 1 121,3 | 1,831 → 1,847 | 3,184 → 2,858 | 86,36% → 85,26% |
| 8K | 1 433,5 → 1 433,6 | 5,746 → 5,746 | 7,209 → 6,757 | 80,85% → 83,00% |
| 31K | 1 329,3 → 1 335,3 | 23,549 → 23,443 | 25,433 → 24,700 | 83,72% → 83,16% |
| 45K | 1 210,3 → 1 215,1 | 37,480 → 37,336 | 39,477 → 38,692 | 91,46% → 84,21% |

Cada taxa de aceitação é a mediana das três taxas, não a razão de totais. O novo pode aceitar uma proporção
menor mas aceitar mais tokens por executar mais palpites: a 45K, foram 238/282 contra 224/247 no conjunto das três
corridas. O ganho não depende de afirmar que a aceitação sobe em todos os contextos.

Picos por processo, medianas das três corridas, MiB:

| Memória | Base (mín.–máx.) | Novo (mín.–máx.) |
| --- | ---: | ---: |
| VRAM dedicada | 13 072 (13 069–13 122) | 13 072 (13 049–13 135) |
| Memória GPU partilhada | 2 042 (2 042–2 042) | 2 042 (2 042–2 042) |
| RAM privada | 11 821 (11 814–11 871) | 11 805 (11 786–11 873) |
| Working set | 11 495 (11 440–11 504) | 11 556 (11 478–11 602) |

Prefill a quente: −0,91% a +0,45%; TTFT a quente: diferença absoluta mediana até 0,144 s.
O primeiro prompt A-2K, num processo novo, teve mediana menos favorável: TTFT **2,503 → 2,757 s** e prefill
**823,8 → 747,9 tok/s**. Os intervalos TTFT foram 2,496–3,699 e 2,750–2,972 s. A primeira resposta é sensível
ao carregamento das páginas e ao primeiro uso das bibliotecas; esta amostra não demonstra a causa nem uma regressão
consistente, mas o resultado desfavorável é preservado. Não se afirma que o TTFT inicial melhorou.

Os 60 pedidos medidos produziram 128 tokens cada e terminaram normalmente. `bench-final.json` guarda as corridas,
`bench-final-summary.md` as fases A/B/cache com intervalos e `bench-derived.json` as estatísticas da tabela,
incluindo intervalos de prefill, TTFT, latência e memória. Não houve amostras removidas. Os ganhos históricos
31–49% foram portanto reproduzidos no binário atual, com **37,8–51,8%** nesta campanha de condições fixadas.

## 6. Perfis e integrações

O estado inicial do menu guardava Empero como última escolha. Foi reposto apenas
`build-hip-poly-win/last-model.txt` para `qwen3.6-35b-a3b-ud-q3_k_m-hip`, conforme o pedido do utilizador.
Os quatro JSON Qwen e os JSON Empero foram preservados. A configuração global OpenCode foi apenas lida para hash.
Os testes API usam uma cópia de configuração e servidor temporário em `127.0.0.1:8091`.

Os quatro perfis Qwen (48K, 128K, 262K e 262K streaming) arrancaram. Em cada um, o prompt curto da referência
gerou os mesmos 64 tokens, com MTP ligado. Também passaram os quatro `START-POLY-HIP.bat check <perfil>`;
o modelo escolhido por omissão é agora Qwen3.6. São testes de arranque e geração curta, não validação de uma
conversa até ao limite de 262K.

Contexto longo atual, `tools/hip/kv_needles.py`, notas a 5/50/95% de um prompt de **65 519 tokens**, sem pensamento:

| Configuração | Notas individuais + juntas | Prefill inicial | TTFT inicial | Pico VRAM dedicada / partilhada | Pico RAM privada / working set |
| --- | ---: | ---: | ---: | ---: | ---: |
| contexto 131 072, KV toda em VRAM | 6/6 | 1 083,6 tok/s | 60,5 s | 13 069 / 1 786 MiB | 9 778 / 12 499 MiB |
| contexto 262 144, KV residente 49 152 | 6/6 | 1 028,0 tok/s | 63,8 s | 13 078 / 6 364 MiB | 15 685 / 15 828 MiB |

As perguntas seguintes releram cerca de 2K tokens, TTFT 3,0–3,2 s. Encontraram `4817-PLUM`, `HERON-62` e `3B-917`,
sozinhos e juntos. Na KV dividida, a reserva de RAM fixada explica parte da memória partilhada; o contador de memória
partilhada por si só não prova paginação da VRAM. Os testes até ~261K da migração são históricos e não foram repetidos
nesta continuação: seriam cerca de nove minutos de prefill por modo, e o caminho de KV dividida não foi alterado.

API: **15/15** em `tools/hip/poly_api_check.py`. Inclui OpenAI e Anthropic, ferramentas em streaming e sem streaming,
ciclo de resultado de ferramenta, cancelamento seguido de nova resposta, raciocínio separado, `preserve_thinking`,
título e sessão de 12 turnos. A resposta longa produziu 2 094 tokens e terminou normalmente. O servidor escutava
apenas `127.0.0.1:8091`; foi terminado depois do teste.

OpenCode **v2.0.18**, o executável instalado pelo OpenChamber: configuração de projeto isolada, fornecedor apontado
para esse servidor temporário. A ferramenta `read` leu `notes.yaml` e respondeu **417**; `--continue` respondeu **418**
sem reler o ficheiro. Logs JSONL e resultado em `integration.json`. A configuração global foi comparada por SHA-256
antes e depois: **igual**. Não foi testada a interface gráfica do OpenChamber nem a versão npm global do OpenCode.

Empero, com janela 1 e os seus pesos/configuração preservados, contra as referências llama.cpp HIP GPU já existentes:
**42/42 tokens** no prompt curto, diferença máxima de probabilidade 0,031; **64/64** no prompt de código, máximo 0,072.
Ambos passaram. O teste usou uma cópia dos seus contadores; não alterou o expert profile operacional de Empero.
É uma verificação breve de regressão do motor partilhado, não uma nova campanha de desempenho desse modelo.

## 7. Limitações e recomendação

A causa inicial do WATCHDOG e a recuperação HIP de um processo que atravessa um reset continuam sem prova.
Não se provoca um TDR deliberadamente no PC de trabalho. Após um timeout do driver, deve ser descartado o processo
de inferência afetado e iniciado um novo; o sucesso de `/health` por si só não valida o estado dos pesos ou da KV.

Não houve novos `LiveKernelEvent`, `BlueScreen` ou erros `strata-poly` na consulta aos eventos de aplicação entre
**18:53 e 19:50**. O resultado da consulta, incluindo `NoMatchingEventsFound`, fica em `events-during-validation.json`.
Os processos novos foram iniciados depois do incidente, sem reiniciar o PC. Isto valida uma sessão de cerca de uma
hora e os workloads descritos, não operação durante dias nem a sobrevivência de um processo ao reset do driver.

**Recomendação: conservar o novo binário como experimental**, com a referência disponível para comparar ou regressar
temporariamente. Não há uma regressão séria isolada que justifique reverter as otimizações; não se perdeu o ganho
de decode. Antes de o promover a produção, falta analisar o dump do TDR ou obter instrumentação que identifique
o kernel/falha e validar que um processo afetado recusa servir resultados após perda de estado. Voltar ao binário
base não é uma cura demonstrada para o driver, que já tinha timeouts antes desta fase.

O B-31K está delimitado pelos artefactos e pelas 20 repetições, mas não é apresentado como bug corrigido. O TTFT
inicial menos favorável fica documentado. Não se declara a versão pronta para produção nem se altera o driver
ou o registo para esconder timeouts.

Alterações desta continuação: correção e reforço do validador/cadeia `--no-draft`, última escolha local Qwen3.6,
relatórios, scripts e dados. **Nenhum ficheiro C++/CUDA/HIP do motor foi modificado nesta continuação.** Os executáveis,
quatro perfis Qwen, configurações Empero, launcher e configuração do fornecedor OpenCode mantiveram os hashes
iniciais, exceto a última escolha do menu. O expert profile Qwen foi atualizado pelo motor nos testes operacionais,
como descrito na secção 5. `preservation-check.json` confirma essas distinções. Não ficaram processos de teste ativos.

## 8. Reprodução

Na raiz do repositório, com acesso normal à GPU e nenhuma outra inferência simultânea:

```powershell
.venv/Scripts/python.exe bench/results/2026-10-12-hip-decode-validation/audit_existing.py
.venv/Scripts/python.exe bench/results/2026-10-12-hip-decode-validation/test_validation.py
.venv/Scripts/python.exe bench/results/2026-10-12-hip-decode-validation/validate_runs.py stability
.venv/Scripts/python.exe bench/results/2026-10-12-hip-decode-validation/direct_parity.py
.venv/Scripts/python.exe bench/results/2026-10-12-hip-decode-validation/validate_runs.py profiles
.venv/Scripts/python.exe bench/results/2026-10-12-hip-decode-validation/validate_runs.py bench
.venv/Scripts/python.exe bench/results/2026-10-12-hip-decode-validation/api_integration.py
.venv/Scripts/python.exe bench/results/2026-10-12-hip-decode-validation/summarize.py
.venv/Scripts/python.exe bench/results/2026-10-12-hip-decode-validation/check_preserved.py
```

Contexto longo, usar `tools/hip/kv_needles.py --engine build-hip-poly-win/strata-poly.exe --model
C:/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --tokenizer C:/models/qwen36-q3-pack/tokenizer
--lengths 65536`, com `--ctx 131072 --engine-args="--threads 6 --expert-profile ..."`, ou `--ctx 262144`
e `--engine-args="--threads 6 --kv-resident 49152 --expert-profile ..."`. Os argumentos completos estão nos scripts,
JSON e logs; os valores medidos nessa ferramenta usam a sua contagem própria de decode e não entram nos A/B.

O benchmark recusa misturar corridas se `bench-final.json` já existir. Para repetir, usar uma cópia da pasta para
outra campanha e ajustar `HERE`. Os logits completos são comparados em ficheiros temporários e removidos depois;
ficam hashes, estatísticas, tokens e top-40 por passo. Os dumps e modelos não foram adicionados ao Git.
