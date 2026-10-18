> Historical measurement report. Use the [current setup instructions](../../README.md). Personal paths were generalized for publication; measurements were preserved. Raw campaign links may require the local evidence archive.

# Port HIP do `strata-poly` — fase 2

Ramo `hip-poly`, commits `d308977a..16a97015` sobre o fim da fase 1 (`ba706504`). Não foi feito push nem aberto PR.
Fase anterior: [AMD_PORT_IMPLEMENTATION.md](AMD_PORT_IMPLEMENTATION.md).

**Onde foi medido:** este PC. RX 9060 XT 16 GB (gfx1200, Adrenalin 32.0.31041.1004), Ryzen 5 5600X, 31,9 GB de RAM,
Windows 11 Pro, ROCm 10.2.0a20260930 (TheRock). O PC estava em uso normal (Firefox, ChatGPT, Bionic e outros
abertos): antes de cada corrida do benchmark estavam livres 20,9–24,0 GB de RAM e outros processos ocupavam
2,4–3,0 GB de VRAM (registado por corrida). Modelo: `Qwen3.5-35B-A3B-Q3_K_M.gguf` (Unsloth), o mesmo da fase 1.

---

## 1. Resumo executivo

- **O PolyStrata HIP é o mais rápido dos três motores neste PC, com o mesmo GGUF.** Com prompts de 2K a 45K
  tokens (mediana de 3 corridas alternadas, configuração de cada motor escolhida por varrimento):
  - prefill de 1 235–1 561 tok/s contra 1 097–1 243 (llama.cpp HIP) e 870–958 (llama.cpp Vulkan);
  - decode de 54,9–80,1 tok/s contra 16,2–16,7 e 34,6–36,5.
  - Usa menos RAM (12,5 GB privados contra ~20 GB) e demora mais a carregar (16 s contra 8–9 s).
- **Otimização do decode mantida:** o routing de um token passou para a placa e o token é entregue inteiro. Decode
  59,0 → 75,4 tok/s (+28 %) a 2K, +20 % a 31K e 45K, +18 % com janelas de tokens adivinhados, +18 % com pouca VRAM.
  Os especialistas escolhidos são os do host em 10 240 de 10 240 routings comparados. Fallback:
  `STRATA_POLY_CARD_ROUTE=0`.
- **Precisão: sem problema real a corrigir.** O PolyStrata dá os mesmos 64 tokens que o llama.cpp em todos os
  prompts testados (até 35 575 tokens), e as diferenças de probabilidade (máx. 0,042–0,107 conforme a referência)
  estão dentro das que o próprio llama.cpp tem entre as suas configurações (até 0,122). A diferença de 0,141 da
  fase 1 vem sobretudo de onde se calculam os especialistas (CPU ou GPU), não de um erro.
- **Integração:** a API passa as 12 verificações (streaming, multi-turno, ferramentas com ciclo completo, Anthropic,
  cancelamento, resposta longa, sessão de 40 turnos). O OpenChamber (com o OpenCode que traz) usa o PolyStrata
  com uma configuração separada. Corrigidos dois problemas reais encontrados nos testes:
  - a reutilização de contexto em conversas (`--turn-token`);
  - um título de sessão que pensava até 133 s (configuração do agente `title`).
  - O Bionic não foi integrado: não encontrei nele um fornecedor externo configurável.
- **Correção da comparação da fase 1:** a configuração do llama.cpp usada então (`--n-cpu-moe 8`) transbordava para
  memória partilhada com o resto do PC a ocupar VRAM. Com `--n-cpu-moe 12 -ub 2048` (sem transbordo) o Vulkan sobe de
  13 para 36 tok/s de decode a 31K. As comparações abaixo usam as configurações sem transbordo.
- **Testes:** ctest 103 testes, 91 passam, 9 saltados (hardware/tabelas), 3 falham: os 2 já documentados e
  `expert_cache_segmented_test`, uma limitação anterior ao port (funcionalidade só CUDA), fora do código alterado.

## 2. Estado inicial e preservação

- Ramo `hip-poly` em `ba706504` (fim da fase 1), árvore limpa salvo o trabalho desta fase. Os números da fase 1
  ficaram guardados primeiro: `bench/results/2026-10-10-hip-poly-rx9060xt-baseline/` (commit `d308977a`).
- Nenhum modelo foi descarregado, alterado ou apagado. O caminho CUDA não foi alterado salvo a opção opt-in
  `--turn-token` (`src/poly/main.cpp`), que sem a opção não muda nada; todo o resto está sob `STRATA_USE_HIP` /
  `__HIPCC__` (ver secção 9).
- **Dependência que desapareceu:** `C:\strata` (de onde a fase 1 tirava o llama.cpp `3cf0325` para compilar)
  já não existe nem está na Reciclagem. Com a tua autorização (pergunta feita durante a sessão), foi descarregado só
  esse commit (arquivo do GitHub, 39,6 MB, SHA-256 `cbe23c594282…`) para `third_party/llama.cpp-3cf0325`, excluído
  do git por `.git/info/exclude`. A compilação usa-o com `STRATA_GGML_DIR`.

## 3. Benchmark: PolyStrata HIP vs llama.cpp HIP vs llama.cpp Vulkan

### 3.1 Metodologia

- **Motores:**
  - `strata-poly` deste ramo (com o routing na placa, ligado por omissão no Windows);
  - llama.cpp Vulkan do runtime do LM Studio (`llama.cpp-win-x86_64-vulkan-avx2-2.55.0`, b11511 `fc9ce6b`), já
    instalado neste PC, não foi preciso compilar;
  - llama.cpp HIP do Unsloth (b11160 `a3c12db9d`).
- **Mesmos ids de tokens:** template de chat e tokenização feitos pelo `llama-server`. Greedy, 128 tokens por
  resposta, contexto 49 152, 6 threads em todos.
  - Prompts: dois conjuntos de 4 (A e B, ~2K, 8K, 31K e 45K tokens) de documentos e código deste repositório.
- **Uma corrida:**
  - um processo, com o tempo de arranque até "pronto" medido à parte (carregamento);
  - depois, no mesmo processo, o conjunto A ("frio": os primeiros pedidos do processo), o B ("morno": pedidos
    consecutivos) e B-2k e B-8k outra vez ("cache": o que o motor guardou).
  - TTFT é do pedido ao 1.º token (só processamento; o carregamento fica fora).
  - VRAM dedicada/partilhada e RAM privada são os picos do processo (contadores do Windows).
  - 3 corridas de cada configuração, alternadas (poly, Vulkan, poly `--window 1`, llama HIP, e de novo).
  - Ferramenta: `tools/hip/bench_compare.py` (`prepare`, `run`, `summary`). Dados:
    `bench/results/2026-10-10-hip-poly-vs-llama-vulkan/`.
- **Configuração de cada llama.cpp:** a melhor sem transbordar a VRAM, por varrimento de `--n-cpu-moe` e `-ub`
  (secção 3.3). Do PolyStrata: a por omissão e `--window 1`.
- **Limites da comparação:**
  - Arquitetura: o PolyStrata decide por pedido que especialistas ficam na VRAM (cache que segue o uso) e o
    llama.cpp põe camadas inteiras de especialistas na CPU.
  - Especulação: o modo por omissão do PolyStrata adivinha tokens (repetições da conversa) e confirma-os em
    janelas; o llama.cpp não. `--window 1` é o PolyStrata sem isso.
  - Cache de prompts: o `llama-server` guarda prompts inteiros na RAM (`--cache-ram`, por omissão), o que explica
    o TTFT de 0,2–0,7 s na fase "cache". O PolyStrata guarda só a conversa em curso, salvo
    `--conversation-cache-mib`.
  - O 1.º pedido de cada processo inclui aquecimentos (compilação de shaders no Vulkan, primeiras chamadas na HIP),
    por isso A-2k "frio" é mais lento para todos.

### 3.2 Resultados (mediana de 3 corridas, entre parênteses o intervalo)

Pedidos consecutivos no mesmo processo carregado (conjunto B, "morno"):

| Motor | Prompt | TTFT s | Prefill tok/s | Decode tok/s |
| --- | --- | ---: | ---: | ---: |
| PolyStrata HIP (por omissão) | B-2k (2 068) | 1,63 (1,55–1,64) | 1 271 (1 260–1 334) | 80,1 (79,7–80,8) |
| | B-8k (8 235) | 5,28 (5,22–5,34) | 1 561 (1 542–1 579) | 76,1 (75,8–76,6) |
| | B-31k (31 296) | 22,8 (22,6–22,9) | 1 375 (1 368–1 388) | 61,3 (61,2–61,7) |
| | B-45k (45 355) | 36,7 (36,4–36,8) | 1 235 (1 232–1 245) | 54,9 (54,7–55,4) |
| PolyStrata HIP `--window 1` | B-2k | 1,54 (1,52–1,59) | 1 345 (1 299–1 360) | 80,4 (79,5–80,4) |
| | B-8k | 5,17 (5,17–5,19) | 1 592 (1 588–1 594) | 76,3 (76,3–76,5) |
| | B-31k | 22,6 (22,5–22,6) | 1 385 (1 384–1 390) | 61,5 (61,2–61,8) |
| | B-45k | 36,2 (36,1–36,6) | 1 252 (1 240–1 256) | 54,9 (53,2–55,6) |
| llama.cpp HIP `--n-cpu-moe 12 -ub 2048` | B-2k | 2,11 (2,07–2,23) | 1 097 (1 059–1 121) | 16,7 (16,1–17,0) |
| | B-8k | 6,96 (6,90–7,00) | 1 195 (1 186–1 202) | 16,7 (15,9–17,1) |
| | B-31k | 25,3 (25,2–25,3) | 1 243 (1 243–1 249) | 16,6 (16,0–16,6) |
| | B-45k | 38,4 (38,2–38,4) | 1 191 (1 187–1 192) | 16,2 (16,0–16,3) |
| llama.cpp Vulkan `--n-cpu-moe 12 -ub 2048` | B-2k | 2,48 (2,43–3,48) | 940 (678–963) | 34,9 (34,7–35,6) |
| | B-8k | 8,68 (8,43–8,78) | 958 (947–982) | 36,5 (36,0–36,6) |
| | B-31k | 34,5 (34,4–34,5) | 911 (910–912) | 35,8 (35,5–36,0) |
| | B-45k | 52,4 (52,4–52,4) | 870 (869–870) | 34,6 (32,4–34,8) |

Primeiros pedidos do processo (conjunto A, "frio"), prefill / decode:

| Motor | A-2k | A-8k | A-31k | A-45k |
| --- | --- | --- | --- | --- |
| PolyStrata (por omissão) | 981 / 69,7 | 1 594 / 74,3 | 1 394 / 58,2 | 1 246 / 54,6 |
| PolyStrata `--window 1` | 988 / 73,7 | 1 635 / 75,2 | 1 406 / 61,1 | 1 256 / 55,0 |
| llama.cpp HIP | 298 / 16,6 | 1 233 / 16,8 | 1 252 / 15,8 | 1 188 / 16,1 |
| llama.cpp Vulkan | 638 / 35,0 | 948 / 37,1 | 898 / 35,7 | 863 / 34,4 |

O mesmo prompt outra vez (fase "cache", B-2k / B-8k):

| Motor | Tokens lidos | TTFT s |
| --- | --- | --- |
| PolyStrata (por omissão) | todos (2 068 / 8 235) | 1,59 / 5,30 |
| PolyStrata `--conversation-cache-mib 6144` | B-2k: 1 token em 2 das 3 corridas; B-8k: 1 token em 1 das 3 (nas outras, todos) | 0,05 quando reaproveitou |
| llama.cpp HIP (`--cache-ram` por omissão) | 4 / 4 | 0,71 / 0,33 |
| llama.cpp Vulkan (idem) | 4 / 4 | 0,58 / 0,23 |

Com `--conversation-cache-mib 6144` o PolyStrata não guardou algumas conversas porque isso deixaria menos de 2,5 GB
de RAM física disponível (o seu limite de segurança, `conversation_available_memory()`): o processo tinha ~21 GB de
working set (inclui as páginas mapeadas do GGUF) e o resto do PC ~10 GB.

Arranque e memória (mediana):

| Motor | Carregamento s | VRAM dedicada MiB | VRAM partilhada MiB | RAM privada MiB |
| --- | ---: | ---: | ---: | ---: |
| PolyStrata (por omissão) | 16,2 | 13 451 | 2 042 | 12 524 |
| PolyStrata `--window 1` | 16,0 | 13 521 | 2 034 | 12 767 |
| llama.cpp HIP | 8,4 | 13 803 | 354 | 19 905 |
| llama.cpp Vulkan | 8,8 | 13 026 | 438 | 20 861 |

A "VRAM partilhada" do PolyStrata é a sua memória fixada no host (buffers de staging e de upload dos
especialistas), que o Windows conta como memória partilhada da GPU; não é transbordo (o PolyStrata dimensiona a
cache dentro do orçamento do Windows). Nas configurações escolhidas do llama.cpp a partilhada ficou em 354–438 MiB.

Pelo servidor (`polystrata.py benchmark`, o mesmo da fase 1, com `--turn-token`), decode:
- curto 58,7 → 80,5 tok/s;
- 2 743 tokens 62,7 → 76,4;
- 10 615 tokens 60,5 → 74,2;
- 31 352 tokens 52,7 → 62,7;
- 44 419 tokens 50,7 → 55,8.

Prefill e TTFT ficaram iguais aos da fase 1 (`server-benchmark-48k-phase2.json`).

### 3.3 Escolha da configuração do llama.cpp (varrimentos)

A-8k e A-31k, contexto 49 152 (`sweep*.json`). Prefill / decode tok/s, VRAM dedicada / partilhada MiB:

| Configuração | A-8k | A-31k | VRAM | VRAM de outros processos |
| --- | --- | --- | --- | --- |
| Vulkan `--n-cpu-moe 8 -ub 2048` (1.º varrimento) | 878 / 29,0 | 700 / 13,4 | 13 400 / 1 362 | não registado |
| Vulkan `--n-cpu-moe 10 -ub 2048` | 161 / 21,4 | 161 / 20,1 | 13 006 / 1 090 | 2 334 |
| **Vulkan `--n-cpu-moe 12 -ub 2048`** | **827 / 37,3** | **920 / 36,3** | 13 026 / 402 | 2 860 |
| Vulkan `--n-cpu-moe 14 -ub 2048` | 808 / 31,1 | 875 / 29,5 | 12 358 / 402 | 2 859 |
| Vulkan `--n-cpu-moe 16 -ub 2048` | 766 / 32,8 | 835 / 31,8 | 11 693 / 402 | 2 855 |
| HIP `--n-cpu-moe 8` (1.º varrimento) | 420 / 16,0 | 680 / 10,5 | 13 663 / 1 200 | não registado |
| **HIP `--n-cpu-moe 12 -ub 2048`** | **678 / 17,1** | **1 238 / 16,0** | 13 803 / 354 | 2 855 |
| HIP `--n-cpu-moe 14 -ub 2048` | 635 / 14,9 | 1 162 / 12,3 | 13 202 / 360 | 2 628 |
| HIP `--n-cpu-moe 16 -ub 2048` | 549 / 13,3 | 1 112 / 13,0 | 12 574 / 363 | 2 368 |
| HIP `--n-cpu-moe 14` (`-ub` por omissão) | 396 / 14,6 | 542 / 14,7 | 12 730 / 180 | 2 797 |

Com menos camadas na CPU (`--n-cpu-moe` 4–10) o llama.cpp ocupa mais VRAM do que o Windows lhe dá com o resto do
PC aberto e transborda para memória partilhada (≥ 1 GB): prefill cai para ~160 tok/s e o decode a 31K para
~11–14 tok/s. A comparação da fase 1 (llama.cpp HIP `--n-cpu-moe 8`, contexto 16K, ~24 tok/s) e o 1.º varrimento
estavam nessa zona; a escolha acima é a mais rápida das que não transbordaram nas condições medidas. O PolyStrata
ajusta a cache ao orçamento do Windows sozinho; o llama.cpp precisa que se escolha `--n-cpu-moe` à mão, e o valor
certo muda com o que o resto do PC ocupa na VRAM.

## 4. Precisão numérica

### 4.1 O que o harness passou a medir

`tools/hip/poly_llama_parity.py` (commits `442d6cb3`, `afccd89f`):
- tokens iguais até à primeira diferença e, aí, se é um quase-empate;
- por passo, a maior diferença de probabilidade entre os 5 tokens mais prováveis da referência e as do motor, com
  o máximo, a média e o percentil 95 sobre os passos comparados, e o passo do máximo;
- um prompt de ~30K tokens (`--long`);
- a comparação de duas referências entre si (`--against-refs`): é isso que dá uma tolerância justificada.

Resultados em `bench/results/2026-10-10-hip-poly-parity/` (4 prompts de 22 a 11 481 tokens, 64 tokens cada).

### 4.2 Resultados

Referências do llama.cpp:
- A: HIP, todos os especialistas na CPU (`--cpu-moe`), a da fase 1;
- B: HIP, a maioria na GPU (`--n-cpu-moe 10`);
- C: Vulkan, `--n-cpu-moe 10`;
- A30: como A, prompt de 35 575 tokens.

| Comparação | Tokens iguais | Maior diferença | Média por prompt |
| --- | --- | ---: | --- |
| **llama.cpp A contra llama.cpp B** | 64/64 nos 4 | 0,061 | 0,0021–0,0034 |
| **llama.cpp A contra llama.cpp C** | 64/64 nos 4 | 0,110 | 0,0022–0,0083 |
| **llama.cpp B contra llama.cpp C** | 64/64 nos 4 | 0,122 | 0,0016–0,0082 |
| PolyStrata `--window 1` (routing na placa) contra A | 64/64 nos 4 | 0,079 | 0,0019–0,0072 |
| PolyStrata `--window 1` contra B | 64/64 nos 4 | 0,042 | 0,0014–0,0049 |
| PolyStrata `--window 1`, `STRATA_POLY_CARD_ROUTE=0`, contra A | 64/64 nos 4 | 0,092 | 0,0020–0,0070 |
| PolyStrata `--no-gpu-experts` contra A | 64/64 nos 4 | 0,107 | 0,0022–0,0079 |
| PolyStrata por omissão (com especulação), só tokens, contra A | 64/64 em 3; o curto separa-se no token 19 num quase-empate (0,535 / 0,465 na referência) | – | – |
| PolyStrata `--window 1` contra B e contra C, só tokens | 64/64 nos 4, nos dois | – | – |
| PolyStrata `--window 1` contra A30 (35 575 tokens) | 64/64 | 0,079 (num quase-empate da referência: 0,419 / 0,412) | 0,0052 |
| PolyStrata `--no-gpu-experts` contra A30 | 64/64 | 0,067 | 0,0069 |

### 4.3 Origem da diferença de 0,141 da fase 1 e tolerância

- **De onde vem:** a referência da fase 1 calculava todos os especialistas na CPU (ggml, ativações quantizadas em
  q8_K). O PolyStrata calcula 92–99,6 % deles na GPU, conforme o prompt (MMVQ, ativações em q8_1).
  - Contra uma referência do llama.cpp com os especialistas na GPU (B), a maior diferença desce para 0,042.
  - O próprio llama.cpp, entre CPU e GPU, difere até 0,061; entre o seu backend HIP e o Vulkan, até 0,122.
  - O valor exato do máximo varia de corrida para corrida, porque a cache de especialistas muda (trocas
    assíncronas e o perfil de uso), e com ela quais especialistas vão para a CPU. Numa repetição desta fase (não guardada nos
    resultados) o mesmo prompt deu 0,138 com o routing do host e 0,085 com o da placa.
- **Tolerância justificada:** a que o harness já aplica:
  - todos os tokens iguais, ou a primeira diferença num quase-empate (as duas mais prováveis a menos de 0,1 de
    distância);
  - e maior diferença ≤ 0,15.
  - Está acima da dispersão do próprio llama.cpp (0,122) e mantém margem para a variação da cache.
  - A média por prompt (≤ 0,0083 em todas as comparações, do llama.cpp e do PolyStrata) é um indicador mais estável
    que o máximo.
- **Nada a corrigir:** nenhuma comparação mostrou um problema real. A especulação (janelas de tokens adivinhados)
  dá os mesmos tokens que `--window 1` salvo no quase-empate já conhecido da fase 1.
- **Quantizações:** só há neste PC um GGUF qwen35moe (Q3_K_M), e não descarreguei outros. Os kernels de 10 formatos
  continuam bit a bit iguais entre as variantes de lançamento (`poly_kernel_bench`, 20 de 20).
## 5. Integração Bionic / OpenCode / OpenChamber

### 5.1 O que foi inspecionado (só leitura)

- **OpenCode** (`C:\Users\<user>\.config\opencode\opencode.json`, sem alterações): fornecedores `lmstudio` (3 modelos),
  `strata` (`providers`, `http://127.0.0.1:8080/v1`, modelo `strata`, variantes de esforço), compaction, MCP
  playwright. Há ainda `C:\Users\<user>\Documents\opencode.json` (outro formato, `lmstudio/ternary-bonsai-27b`), que o
  OpenCode junta como configuração de projeto para tudo o que está dentro de `Documents`. Nenhum foi alterado.
- **CLI do OpenCode instalado por npm:** o lançador `opencode.ps1/.cmd` aponta para
  `node_modules\@opencode\cli\bin\opencode.exe`, que não existe; o executável (v2.0.3) ficou numa pasta temporária
  de uma atualização interrompida (`node_modules\@opencode\.cli-FEI3vopc\...`). Não foi reparado (é a instalação do
  utilizador). Nota: ao correr `opencode debug config` com esse executável, o OpenCode arrancou o seu serviço de
  fundo (processo `opencode`, iniciado às 21:25); a tentativa de o terminar foi recusada pelas permissões desta
  sessão, por isso pode continuar aberto (`opencode service stop`, ou terminar o processo, resolve).
- **OpenChamber** (`%LOCALAPPDATA%\Programs\@openchamberelectron`, dados em `~\.config\openchamber`): é uma interface
  sobre o OpenCode e traz o seu próprio `opencode.exe` (v2.0.18, `resources\opencode-cli`). Lê a mesma configuração
  do OpenCode (global + `opencode.json` do projeto aberto). O modelo por omissão nas definições é `strata/strata`.
  Nenhum ficheiro do OpenChamber foi alterado.
- **Bionic** (`%LOCALAPPDATA%\Programs\Bionic`, v1.1.7, "productName Bionic", pacote `lm-studio`): é um derivado do
  LM Studio. As definições estão em `~\.lmstudio\apps\bionic\settings.json` e `.internal\settings.json`; os modelos
  são GGUF locais que ele próprio corre (`localModels`, `lastSelectedModelSpecifier` do tipo `indexed`) ou modelos
  da nuvem (`cloudInference`). Não encontrei nas definições, nem nas strings da interface, um campo para um
  fornecedor externo compatível com OpenAI (há "Base URL" do servidor local que ele próprio expõe). **Não foi
  testado com o PolyStrata** e não foi alterado. Os testes de API abaixo cobrem o que um cliente OpenAI/Anthropic
  envia; se o Bionic ganhar (ou tiver numa versão que não vi) um fornecedor personalizado, a URL é
  `http://127.0.0.1:8090/v1`.

### 5.2 Configuração separada

- `tools/hip/opencode-polystrata-hip.json` (no repositório): um fornecedor `polystrata-hip`
  (`aisdk:@ai-sdk/openai-compatible`, `http://127.0.0.1:8090/v1`, modelo `qwen3.5-35b-a3b-q3_k_m-hip`, contexto
  49152, variantes `none`/`low`/`medium`) e o agente `title` nesse modelo com a variante `none`.
- Para o teste foi copiado para `opencode.json` na raiz do repositório (configuração de projeto do OpenCode, só
  vale com a pasta do PolyStrata aberta; excluído do git por `.git/info/exclude`). A configuração global não foi
  tocada. Para o usar noutro projeto: copiar o ficheiro para a raiz desse projeto como `opencode.json`, ou juntar a
  entrada `providers.polystrata-hip` à configuração global.
- O servidor escuta só em `127.0.0.1:8090` (verificado com `netstat` pelo script de teste); sem `--api-key` não deve
  ser exposto a outros endereços.

### 5.3 Testes da API (`tools/hip/poly_api_check.py`)

Contra o servidor (`serve/server.py --engine strata --config strata-qwen3.5-35b-a3b-q3_k_m-hip.json --port 8090`),
com o pensamento desligado salvo onde dito:

| Verificação | Resultado |
| --- | --- |
| Ligação, `/v1/models` | ok |
| System prompt obedecido ("começa por Arr") | ok |
| Streaming SSE com `[DONE]` e `usage` (`stream_options.include_usage`) | ok, 69 pedaços, 1.º token 0,18 s |
| Multi-turno (lembra nome e número, soma 417+1) | ok |
| Ferramentas, ciclo completo (chamada `get_weather` → resultado → resposta), sem e com streaming | ok |
| Resposta longa (ensaio, `max_tokens` 2500) | ok, 2 379 tokens a 62 tok/s |
| Cancelamento (fechar o stream ao 10.º evento) e pedido seguinte | ok, seguinte respondido em 0,42 s |
| Anthropic `/v1/messages`, com e sem streaming, e ciclo de ferramentas `tool_use`/`tool_result` | ok |
| Pensamento (`reasoning_effort: low`) | ok, `reasoning_content` + resposta |
| Sessão longa: 40 turnos, até 14 049 tokens, e pergunta sobre o 1.º turno | ok (com `--turn-token`, ver abaixo) |

**Problema encontrado e corrigido — reutilização de contexto em conversas.** Sem alterações, `cached_tokens` era 0
em todos os turnos até um múltiplo de 2 048: cada turno relia até 2 047 tokens (turno 7: 2 348 tokens, 1,74 s até
ao 1.º token). Causa (não é do HIP; vale para qualquer backend): o template do Qwen3.5 reescreve o turno do
assistente sem o bloco `<think>` quando ele passa a histórico, e o `strata-poly` só tinha checkpoints a cada 2 048
tokens e antes do último token do prompt. Correção: `--turn-token ID` (opt-in, commit `75616df8`), um checkpoint
à frente do último turno do prompt. Com `--turn-token 248045` (`<|im_start|>`): cada turno reutiliza o prompt
anterior menos ~7 tokens; 1.º token a 0,69–0,75 s do turno 1 ao 12 e ainda 0,72 s no turno 40 (14 049 tokens).

### 5.4 OpenChamber (o OpenCode que ele traz, `opencode run --standalone`, na pasta do repositório)

| Passo | Resultado |
| --- | --- |
| Ligação e streaming ("Reply with exactly PONG") | ok; prompt do agente de 15 320 tokens lido em ~11 s |
| Agente com ferramenta (ler um ficheiro e responder) | ok: chamada `read` concluída, resposta certa |
| Segundo turno na mesma sessão (`--continue`) | ok: 18 523 de 21 047 tokens reutilizados, 3,8 s |

**Problema encontrado e corrigido na configuração — o título.** O OpenCode pede um título da sessão ao mesmo modelo,
sem esforço de raciocínio: o Qwen3.5 pensou 3 542 e 9 345 tokens (53 s e 133 s) antes de responder, e o turno do
utilizador esperava atrás disso. `small_model` com `#none` é rejeitado pelo OpenCode v2 ("omitted unsupported
legacy model reference": a referência não aceita variante); a forma certa é o agente `title` com
`"variant": "none"`. Com isso: título em 1 s (11 tokens) e o turno completo com ferramenta em 23 s (era 155 s).

Não testado: a interface gráfica do OpenChamber (o teste usou o executável do OpenCode que ela traz, com a mesma
configuração); o Bionic.
## 6. Otimização do decode: a sincronização CPU↔GPU por camada MoE

### 6.1 Medição antes de mudar código

No caminho de um token por chamada, cada uma das 40 camadas MoE fazia: router na GPU → cópia das pontuações e da
entrada dos especialistas para a memória fixada → o host espera pelo evento → o host escolhe os 8 especialistas
→ lança os que estão em VRAM e calcula na CPU os outros → soma. Medido no motor (`--window 1`, prompt B-2k,
256 tokens, `STRATA_POLY_PROFILE=1` para as contagens):

| Medida | Valor |
| --- | --- |
| Decode | 60,7–61,6 tok/s (16,4 ms/token) |
| Espera do host pelo router (`router wait`) | 1,7 ms/token |
| Especialistas na CPU | 0,1 ms/token |
| Especialistas usados que vinham da VRAM | 96,7 % |
| Camadas MoE com os 8 especialistas em VRAM | 75,3 % (7 713 de 10 240) |

`hip_poly_probe` (uma "volta" de camada: kernel → 9 KiB para o host → espera → kernel seguinte), mediana de 2 000:

| Forma | µs |
| --- | ---: |
| `cudaMemcpyAsync` + espera do evento (antes do port) | 195,5 |
| cópia por kernel + espera do evento (o motor no Windows desde a fase 1) | 61,6 |
| cópia por kernel + evento consultado em ciclo | 61,0 |
| flag em memória fixada coerente, consultada em ciclo | 60,9 (só depois de um `hipStreamQuery`: sem ele o runtime não entrega os lançamentos à placa e a espera não acaba) |

Conclusão da medição: trocar a espera do evento por uma flag não ganha nada; os ~60 µs são o tempo de entregar
lançamentos e voltar. **Limite superior** do que há a ganhar: uma experiência temporária (não guardada) que não
esperava pelo host em nenhuma camada — respostas erradas, porque ignora os especialistas da CPU — deu 82,7 tok/s
contra 63,6 nas mesmas condições (sem trocas de especialistas): 3,6 ms de 15,7 ms por token vão para estas esperas.

### 6.2 Primeira tentativa (descartada): routing na GPU, lançamentos antes da espera

O routing passou a ser um kernel e os especialistas em cache eram lançados antes de o host esperar. Ganho medido:
+1,2 % (mediana de 3 alternadas: 59,3 → 60,0 tok/s), dentro do ruído. O host continuava a ter de ver cada camada
antes de entregar a seguinte. Substituída pela versão abaixo.

### 6.3 Implementação mantida: o token inteiro entregue à placa (commits `0a6b29aa`, `6687a925`)

- Um kernel (`route_topk_softmax`) faz o routing do Qwen3.5 na placa (softmax, os 8 mais prováveis com o
  desempate do host, pesos renormalizados com o mesmo piso), escreve os ponteiros dos especialistas escolhidos nos
  argumentos dos lançamentos (em memória da placa, a partir de uma tabela de slots que segue a cache do host) e
  copia para memória fixada coerente os ids, pesos, a entrada dos especialistas e um número de sequência.
- Os especialistas em cache são lançados logo (variantes dos kernels que leem os slots da memória da placa; um
  especialista fora da VRAM é um slot sem matriz: zeros, peso 0). Um kernel `wait_add` só espera se houver
  especialista fora da VRAM nessa camada: lê uma flag em memória coerente posta pelo host depois de calcular esses
  especialistas, e soma o resultado. Desiste ao fim de 1 s (erro reportado, sem bloquear a placa).
- O host enfileira o token inteiro, entrega-o com `cudaStreamQuery` e percorre as camadas à medida que a placa as
  encaminha (contagens, trocas da cache, especialistas da CPU).
- Ordem das somas igual à do caminho antigo; os produtos de cada especialista são os mesmos kernels (bit a bit:
  20 de 20 formas no `poly_kernel_bench`). O `exp()` do routing é o da placa: pesos podem diferir no último bit.
- Fallback: `STRATA_POLY_CARD_ROUTE=0` (caminho antigo). Ativo por omissão só no Windows/HIP; noutros sistemas
  `=1`. Desligado automaticamente com `STRATA_POLY_PROFILE=1`, com `--dump-layers`, nas janelas de tokens
  adivinhados em metades e no bloco de rascunho. `STRATA_POLY_CARD_ROUTE_CHECK=1` refaz cada routing no host e
  compara.

### 6.4 Resultados (antes/depois, mesmo binário, alternados)

| Caso (256 tokens) | `STRATA_POLY_CARD_ROUTE=0` | ligado | Δ |
| --- | ---: | ---: | ---: |
| B-2k, `--window 1`, 3 corridas alternadas (mediana) | 59,0 (57,9–61,0) | 75,4 (75,0–76,3) | +28 % |
| A-31k, `--window 1` | 47,9 | 57,5 | +20 % |
| A-45k, `--window 1` | 43,7 | 52,4 | +20 % |
| B-2k, modo por omissão (janelas de tokens adivinhados) | 60,4 | 71,1 | +18 % |
| A-31k, modo por omissão | 46,4 | 56,3 | +21 % |
| B-2k, `--window 1 --no-graphs` | 56,0 | 76,4 | +36 % |
| B-2k, `--window 1`, `STRATA_POLY_COPY_KERNELS=0` | 41,0 | 73,5 | +79 % |

Verificação do routing (`STRATA_POLY_CARD_ROUTE_CHECK=1`, 256 tokens): 10 240 routings, **0 especialistas
diferentes** dos do host, pesos que não são bit a bit iguais em 26 113 de 81 920, diferença máxima 1,19e-7.
Paridade com o llama.cpp: secção 4.

Pelo servidor com pouca VRAM (`--vram-reserve-mib 8000`, 64 % dos especialistas da VRAM, `--window 1`, B-2k):
46,7 tok/s com `STRATA_POLY_CARD_ROUTE=0`, 54,9 com o routing na placa (+18 %); verificação do routing: 0
especialistas diferentes em 10 240.

### 6.5 Perfil antes/depois

`STRATA_POLY_PROFILE=1` sincroniza depois de cada parte, por isso desliga o routing na placa (o host esperaria pela
placa que espera por ele) e só mede o caminho antigo:
- até ao router e especialista partilhado: 14,9 ms/token;
- especialistas encaminhados: 6,5 ms;
- resto das camadas: 2,8 ms;
- logits: 1,5 ms.

Do caminho novo há as medições de ponta a ponta acima:
- a espera do host pelo router (`router wait`) desce de 1,7 para 1,3 ms/token;
- e o tempo por token de 16,9 para 13,3 ms (59,0 → 75,4 tok/s).

O limite medido sem nenhuma espera (respostas erradas) era 12,1 ms/token, nas condições dessa medição.

## 7. Problemas encontrados e correções

| Problema | Causa | Correção |
| --- | --- | --- |
| Sem fonte do llama.cpp para compilar | `C:\strata` foi removido | `3cf0325` descarregado (autorizado) para `third_party/llama.cpp-3cf0325`, fora do git |
| llama.cpp lento na comparação da fase 1 e no 1.º varrimento | `--n-cpu-moe` baixo transborda a VRAM com o resto do PC aberto | Varrimento com a VRAM de outros processos registada; configuração sem transbordo (`--n-cpu-moe 12 -ub 2048`) |
| Uma corrida do Vulkan a ~160 tok/s de prefill | Primeiro RAM (1,8 GB livres), depois VRAM ocupada por outros programas | Corridas abortadas e repetidas; `bench_compare.py` regista RAM livre e VRAM de outros processos |
| Conversas reliam até 2 047 tokens por turno | Template do Qwen3.5 reescreve o turno anterior; checkpoints só a cada 2 048 | `--turn-token` (opt-in; `poly_config.py` põe-no quando o tokenizer tem `<\|im_start\|>`) |
| OpenCode/OpenChamber: título pensava até 133 s | Pedido de título ao mesmo modelo, sem esforço | Agente `title` com a variante `none` na configuração separada |
| `small_model` com `#none` rejeitado | O OpenCode v2 não aceita variante nessa chave | Usado o agente `title` |
| 2.ª tentativa de otimização com ganho de 1 % | O host ainda tinha de ver cada camada | Substituída pela entrega do token inteiro (+28 %) |
| Comparar duas referências pedia `--model` | Argumento obrigatório | Opcional com `--against-refs` |
| Sonda da volta por flag ficava bloqueada | O runtime HIP não entrega lançamentos sem um `hipStreamQuery` | Timeout na sonda; o motor faz `cudaStreamQuery` depois de enfileirar o token |

## 8. Testes executados

| Teste | Resultado |
| --- | --- |
| Compilação HIP (`tools\hip\build_windows.bat poly` e `poly tests`) | sem erros |
| `hip_poly_probe` | todos os passos correm; tempos na secção 6.1 |
| `poly_kernel_bench`, variantes de lançamento bit a bit | 20 de 20 formas iguais |
| ctest, 103 testes (build `poly tests`) | 91 passam, 9 saltados (hardware, tabelas ou fixtures), 3 falham: `hip_handoff` e `ple_parity` (já documentados como esperados no Windows) e `expert_cache_segmented_test` |
| Paridade com o llama.cpp (secção 4) | todas passam |
| Routing na placa contra o do host | 0 escolhas diferentes em 10 240 (normal e com pouca VRAM) |
| Contexto ~48K: A-45k (45 236) e servidor com 44 419 tokens | sem erros; decode 52,4–55,8 tok/s |
| Pouca VRAM (`--vram-reserve-mib 8000`) | sem erros, 64 % da VRAM |
| Pedidos consecutivos no mesmo processo sem reconstruir a cache | 10 pedidos por corrida × 9 corridas do PolyStrata, TTFT sem saltos; nenhuma redução da cache nos registos do servidor desta fase nem no registo do motor guardado (na fase 1, antes da correção, eram 4 s por pedido) |
| Grafos desligados (`--no-graphs`) | sem erros, 76,4 tok/s |
| Fallback de cópias (`STRATA_POLY_COPY_KERNELS=0`) | sem erros, com e sem o routing na placa |
| Fallback do routing (`STRATA_POLY_CARD_ROUTE=0`) | sem erros; mesmo desempenho que antes da mudança |
| API (`tools/hip/poly_api_check.py`, secção 5.3) | 12 de 12 |
| `tools/test_polystrata_names.py` | OK |

`expert_cache_segmented_test` falha porque o cache segmentado do motor principal (`--vram-elastic`) é só CUDA e o
teste não trata isso como "saltado". O teste e o código são do repositório principal (3 de outubro) e nenhum commit
deste ramo lhes toca; não o alterei. A suite completa da fase 1 tinha 73 testes; esta compilação (`poly tests`, com
`STRATA_BUILD_TESTS`) regista 103, e não consegui confirmar se aquele já estava entre os 73.

## 9. O que não foi verificado

- **Compilação CUDA:** não há toolchain CUDA neste PC. O que muda no código partilhado está sob `__HIPCC__` /
  `STRATA_USE_HIP`, salvo `--turn-token` (`src/poly/main.cpp`, opt-in); o CUDA não foi compilado.
- **Linux/HIP:** o routing na placa e as cópias por kernel ficam desligados por omissão fora do Windows (não medidos).
- **Interfaces e clientes:**
  - Bionic: sem fornecedor externo configurável encontrado; não integrado.
  - Interface gráfica do OpenChamber: testado o OpenCode que ele traz, com a mesma configuração.
  - CLI do OpenCode instalado por npm: o lançador está partido (atualização interrompida); não reparado.
- **Partes do motor por exercitar:**
  - O timeout de 1 s do `wait_add` (o caminho de erro) não foi provocado.
  - As janelas de tokens adivinhados e o bloco de rascunho continuam a encaminhar no host.
- **Fora do âmbito do que foi testado:**
  - Outras quantizações e outros modelos (gemma4, glm5-next).
  - Várias GPUs.
  - Sessões de agente reais com dezenas de turnos e ferramentas (testada uma sessão de API de 40 turnos e 3 turnos
    do OpenCode).
- **llama.cpp HIP:** não compilei uma versão mais recente (usei a b11160 do Unsloth, já instalada); uma mais nova
  pode ter outro desempenho.

## 10. Commits desta fase

| Commit | O quê |
| --- | --- |
| `d308977a` | números da fase 1 guardados |
| `75616df8` | `--turn-token` |
| `9334b786` | sonda da volta por camada; contagem de camadas com tudo em VRAM no perfil |
| `442d6cb3`, `afccd89f` | harness de paridade: estatísticas, prompt de 30K, referência contra referência |
| `b4a96542` | `poly_api_check.py` e configuração OpenCode separada |
| `0a6b29aa` | kernels HIP: produtos com slots em memória da placa, routing na placa |
| `6687a925` | motor: routing na placa e token entregue inteiro (Windows por omissão) |
| `3e466ad1` | `docs/AMD_HIP.md` |
| `85c80bc0`, `90e54676`, `07235424` | `bench_compare.py` e resultados contra o llama.cpp |
| `017fb81d` | resultados de precisão |
| `16a97015` | `poly_config.py` com `--turn-token` |

## 11. Instruções

O mais simples: `START-POLY-HIP.bat` (duplo clique). Compila o motor e cria o `.venv` se faltarem, arranca o
servidor em `127.0.0.1:8090` e abre a página. Na primeira vez, `START-POLY-HIP.bat C:\caminho\modelo.gguf`
(ou arrastar o `.gguf` para cima do script) escreve a configuração; `START-POLY-HIP.bat chat` abre um chat na
janela; `START-POLY-HIP.bat check` diz o que arrancaria. Os passos à mão:

Compilar (a fonte do llama.cpp está agora no repositório, fora do git):

```bat
set STRATA_GGML_DIR=<repo>\third_party\llama.cpp-3cf0325
set STRATA_HIP_ARCHS=gfx1200
tools\hip\build_windows.bat poly
```

Configuração de execução (escreve `strata-<nome>.json`, com `--turn-token`):

```bat
.venv\Scripts\python tools\hip\poly_config.py --model C:\models\Qwen3.5-35B-A3B-Q3_K_M.gguf --context 49152 --threads 6 --data C:\models\qwen35-pack --name qwen3.5-35b-a3b-q3_k_m-hip
```

Servidor (só `127.0.0.1`; para outros endereços, `--api-key`):

```bat
.venv\Scripts\python serve\server.py --engine strata --config strata-qwen3.5-35b-a3b-q3_k_m-hip.json --port 8090
```

OpenCode / OpenChamber: copiar `tools\hip\opencode-polystrata-hip.json` para `opencode.json` na raiz do projeto que
abres (ou juntar a entrada `providers.polystrata-hip` e o agente `title` à tua configuração global) e escolher
"PolyStrata (HIP, this PC)".

Variáveis:
- `STRATA_POLY_CARD_ROUTE=0`: routing no host, o caminho antigo.
- `STRATA_POLY_CARD_ROUTE_CHECK=1`: compara cada routing com o do host.
- `STRATA_POLY_COPY_KERNELS=0`: cópias pelo motor de cópia.
- `STRATA_POLY_PROFILE=1`: perfil por camada; desliga o routing na placa.

Verificações:

```bat
.venv\Scripts\python tools\hip\poly_api_check.py --base http://127.0.0.1:8090
.venv\Scripts\python tools\hip\poly_llama_parity.py --engine <repo>\build-hip-poly-win\strata-poly.exe --model C:\models\Qwen3.5-35B-A3B-Q3_K_M.gguf --refs bench\results\2026-10-10-hip-poly-rx9060xt-baseline\llama-hip-refs.json --engine-args "--window 1 --threads 6"
.venv\Scripts\python tools\hip\bench_compare.py summary --json bench\results\2026-10-10-hip-poly-vs-llama-vulkan\full.json
```

## 12. Próximos gargalos

1. **Janelas de tokens adivinhados ainda encaminham no host.** No modo por omissão, as chamadas de vários tokens
   (e as metades) mantêm a espera por camada. O routing na placa para n > 1 tokens é o passo seguinte natural.
2. **Camadas com especialistas na CPU** (cerca de 1 em 4 com cache cheia) param a placa à espera do host. O limite
   medido sem esperas era ~12 ms/token contra 13,3.
3. **Decode com contexto longo:** 80 → 55 tok/s de 2K para 45K, o custo da atenção sobre a cache KV nas 10 camadas
   de atenção completa.
4. **Arranque:** 16 s contra 8–9 s do llama.cpp, incluindo o enchimento da cache de especialistas.
5. **Cache de conversas em RAM** recusa guardar quando a RAM física disponível desce dos 2,5 GB. Num PC de 32 GB
   com o browser aberto, isso acontece com contextos longos.
