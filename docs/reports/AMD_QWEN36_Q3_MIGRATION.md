> Historical measurement report. Use the [current setup instructions](../../README.md). Personal paths were generalized for publication; measurements were preserved. Raw campaign links may require the local evidence archive.

# PolyStrata HIP: migração para o Qwen3.6-35B-A3B UD-Q3_K_M

Ramo `hip-poly`, sobre `00ded250` (fim da fase 4, [AMD_QWEN36_IMPLEMENTATION.md](AMD_QWEN36_IMPLEMENTATION.md)). Não
foi feito push nem aberto PR. O motor não mudou: o binário é o da fase 4.

**Onde foi medido:** este PC. RX 9060 XT 16 GB (gfx1200), Ryzen 5 5600X (6 núcleos), 31,9 GB de RAM, Windows 11 Pro,
ROCm 10.2.0a20260930 (TheRock), em uso normal: outros programas ocupavam ~4,8 GB de VRAM e deixavam ~18 GB de RAM
livre antes de cada corrida (registado por corrida nos JSON). Dados: `bench/results/2026-10-11-hip-qwen36-q3/`.

---

## 1. Resumo

- **Ficou um só modelo:** o Qwen3.6-35B-A3B **UD-Q3_K_M** (Unsloth, 17,1 GB, SHA-256 verificado). Os GGUF do Qwen3.5
  Q3_K_M e do Qwen3.6 UD-Q4_K_M foram apagados: **39,0 GB libertados**, 21,9 GB a menos no disco depois do novo.
- **Funciona sem alterações ao motor.** Os especialistas do modelo principal são IQ3_XXS/IQ4_XS/Q6_K, os formatos do
  Qwen3.5 Q3_K_M. Os do bloco MTP são Q3_K/Q4_K.
- **Precisão** (greedy, contra o llama.cpp HIP b11160 no mesmo GGUF, os 5 prompts da fase 4):
  - **as 9 configurações passam**, incluindo as duas que falhavam no Q4: o routing na placa contra a referência B
    (agora 64/64 nos 5 prompts) e todos os especialistas na CPU (o passo 45 que ficara por explicar não se repete);
  - o produto em lote (MMQ) dos formatos do modelo principal tem o mesmo erro do produto de um token, contra FP64.
    O problema do Q4_K/Q5_K não existe no modelo principal; só o Q4_K do bloco MTP o mantém;
  - **resta um resíduo:** o passo 29 do `doc-2.5k` (0,125–0,134, limite 0,15) só aparece com o MMQ **e** lotes de
    prefill de 1 024. Com lotes de 256 desaparece, o que no Q4 não acontecia (secção 5.4).
- **Desempenho** (mediana de 3 corridas alternadas, contexto 49 152, pedidos consecutivos):
  - decode **93 / 86 / 71 / 62 tok/s** a 2K / 8K / 31K / 45K, contra 65 / 68 / 59 / 53 no Q4 e 81 / 77 / 62 / 56 no
    Qwen3.5;
  - prefill 1 123–1 433 tok/s;
  - working set 11,4 GB (17,2 no Q4).
- **O MTP passa a render:**
  - acerta 87,5 % dos palpites;
  - a quente, +12 a +17 % sobre um token por chamada (no Q4, +1 a +7 %);
  - o routing das janelas continua no host: o ganho vem de haver mais especialistas na VRAM (97 % dos usados).
- **Contexto longo:** as notas a 5, 50 e 95 % foram encontradas em todos os comprimentos (8K a 261K, KV na VRAM e em
  streaming). A pergunta pelas três de uma vez falhou a 32K e a 131K no modo por omissão: um **quase-empate**
  (`,` 0,346 contra o fim 0,298) que a janela do MTP desempata para o outro lado (secção 7.2).
- **Integração:**
  - `START-POLY-HIP.bat` só com o Q3 e os 4 contextos;
  - OpenCode/OpenChamber atualizados (global com cópia de segurança, só as entradas PolyStrata);
  - API 15/15;
  - OpenCode com ferramenta e continuação de sessão a funcionar.
- **GLM-4.7-Flash:** viável no mesmo motor como **classe de arquitetura nova** (`deepseek2`, atenção MLA), não como
  backend novo. As peças existem no `Glm5Next`, que nunca correu em HIP. As variantes Q3 cabem com folga a 48K. A
  KV latente (54 KB por posição, 2,4× a do Qwen3.6) é o limite em contextos longos. Os GGUF da Unsloth não trazem o
  bloco MTP (secção 10).

## 2. Modelos eliminados e espaço libertado

Os GGUF foram localizados por uma procura de `*.gguf` em `C:\` e `D:\`, e identificados pelos metadados
(`tools/hip/gguf_head.py`) e pelas configurações de execução do PolyStrata que os usavam.

| Ficheiro | Bytes | Metadados | Usado por |
| --- | ---: | --- | --- |
| `C:\models\Qwen3.5-35B-A3B-Q3_K_M.gguf` | 16 356 375 168 | `Qwen3.5-35B-A3B`, Unsloth, `file_type` 12 (Q3_K_M), 40 blocos | 4 perfis `strata-qwen3.5-35b-a3b-q3_k_m-hip*.json` |
| `C:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22 663 387 424 | `Qwen3.6-35B-A3B`, Unsloth, `file_type` 15 (Q4_K_M), 41 blocos (MTP) | 4 perfis `strata-qwen3.6-35b-a3b-ud-q4_k_m-hip*.json` |

**Total libertado: 39 019 762 592 bytes (36,3 GiB).** Com o modelo novo (17 104 402 720), o disco tem 21,9 GB a
menos ocupados. O `C:` tinha 82,5 GB livres antes e 114,1 GB no fim: a diferença maior vem de outras variações
do sistema durante as ~3 h de testes.

Passos:
- o servidor PolyStrata estava a correr com o Q4 (perfil 128K, lançado pelo `START-POLY-HIP.bat`); o `server.py` e
  o `strata-poly.exe` foram parados antes de apagar;
- só os dois ficheiros foram apagados; nenhum diretório;
- não havia downloads incompletos (`.part`) nem ficheiros temporários associados (procurados em `models`, no
  repositório, em `Downloads` e na cache do Hugging Face).

**Não foi tocado:**
- `C:\Users\<user>\.lmstudio\models\Manojb\Qwen_Qwen3.5-35B-A3B-Q3_K_M.gguf\Qwen_Qwen3.5-35B-A3B-Q3_K_M.gguf` (LM
  Studio): é outro ficheiro (16 248 175 136 bytes), não usado pelo PolyStrata;
- os outros modelos do LM Studio, do Bionic e do Unsloth;
- os relatórios, benchmarks e código.

**Configurações históricas preservadas** em `C:\models\archive\2026-10-11-polystrata-hip-configs\`:
- as 8 configurações de execução antigas e os seus logs, que estavam na raiz do repositório (excluídas do git);
- cópias do `opencode.json` global e do do repositório de antes da alteração.

Ficam também `C:\models\qwen35-pack` e `qwen36-pack` (tokenizers e perfis de especialistas, ~10 MB cada).

## 3. Modelo novo e verificação

| | |
| --- | --- |
| Origem | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, revisão `5bc3e238` (a fixada no `setup.py`) |
| Ficheiro | `C:\models\Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` |
| Bytes | 17 104 402 720 (o valor fixado no `setup.py`) |
| SHA-256 | `8966dd0cd8c543c4228490a2a8b0e0814fc4f1e6a8e199ceed4de6754ae7b8e1`: igual ao fixado; confirmado pelo `verify_sha256()` do `setup.py` e de novo pelo `Get-FileHash` |
| Download | `setup.download()` (com retoma por `Range` e ficheiro `.part`), 251 s; marca `.done` com o SHA-256 |

**Quantização realmente presente** (`gguf-headers/q36-ud-q3_k_m.txt`, `general.file_type` 12):

| Tensores | Formato |
| --- | --- |
| Especialistas gate/up, camadas 0–38 | **IQ3_XXS** |
| Especialistas gate/up, camada 39 | IQ4_XS |
| Especialistas down, 37 camadas | **IQ4_XS** |
| Especialistas down, camadas 34, 38, 39 | Q6_K |
| Bloco MTP (`blk.40`): especialistas gate/up · down | **Q3_K · Q4_K** |
| Atenção, delta net, especialista partilhado | Q8_0 (normas e `ssm_alpha`/`ssm_beta` em F32) |
| `token_embd`, `output` | Q6_K |

São os formatos de especialistas do Qwen3.5 Q3_K_M, com as mesmas três camadas em Q6_K. 13,3 GiB de especialistas
no modelo principal, contra 18,2 GiB no UD-Q4_K_M. O tokenizer escrito a partir deste GGUF é igual byte a byte ao
do UD-Q4_K_M (vocabulário, merges, tipos e o template com `preserve_thinking`).

## 4. Configurações atualizadas

| Onde | O quê |
| --- | --- |
| Raiz do repositório (excluídas do git) | `strata-qwen3.6-35b-a3b-ud-q3_k_m-hip.json` (48K), `-128k`, `-262k`, `-262k-stream`: escritas por `tools/hip/poly_config.py --name qwen3.6-35b-a3b-ud-q3_k_m-hip` e `poly_profile.py`. Porta 8090, `switch_on_request`, `--threads 6`, `--turn-token 248045`. Os 8 perfis antigos foram arquivados (secção 2) |
| `C:\models\qwen36-q3-pack\` | tokenizer novo; perfil de especialistas começado com uma cópia do do Q4 (o mesmo modelo, os mesmos routers), atualizado pelo uso |
| `START-POLY-HIP.bat` | só um modelo configurado: não mostra o menu de modelo e usa-o. O menu de contexto mostra as velocidades medidas do UD-Q3_K_M (~93 / ~27 / ~14 tok/s); para outro modelo diz "velocidade nao medida". O exemplo da mensagem de primeira utilização aponta para o GGUF novo |
| `tools/hip/opencode-polystrata-hip.json` e `opencode.json` (cópia local, excluída do git) | fornecedor `polystrata-hip`: os 4 perfis do Qwen3.5 removidos, os 4 do Qwen3.6 com os `modelID` do UD-Q3_K_M; o modelo `current` e o agente `title` mantêm-se |
| `~/.config/opencode/opencode.json` (global, lido pelo OpenChamber) | **cópia de segurança antes** (`opencode.json.backup-20261011-q3-migration`, e no arquivo da secção 2). Só no fornecedor `polystrata-hip`: os 4 modelos do Qwen3.5 removidos; nos 4 do Qwen3.6, `modelID` e nome (Q4 → Q3). Chaves, limites e variantes (incluindo `high`) iguais. `lmstudio`, `strata`, `mcp`, `compaction` e o BOM do ficheiro intactos (comparados por programa depois da edição) |

**Os IDs vêm da API**, não foram inventados: com o servidor a correr, `GET /v1/models` devolve
`qwen3.6-35b-a3b-ud-q3_k_m-hip` e `GET /api/models` os 4 perfis (`…-hip`, `…-hip-128k`, `…-hip-262k`,
`…-hip-262k-stream`), que são os `modelID` escritos. Os modelos antigos já não aparecem.

**As chaves do OpenCode ficaram iguais** (`qwen3.6-35b-a3b`, `-128k`, `-262k`, `-262k-stream`). Assim os
"recentes" e os esforços guardados pelo OpenChamber (`~/.config/openchamber/settings.json`) continuam a apontar para
modelos que existem. Esse ficheiro (estado da interface, não configuração) não foi alterado. Ainda tem uma entrada
recente `polystrata-hip/qwen3.5-35b-a3b-128k` que já não corresponde a nenhum modelo; o OpenChamber deve ignorá-la,
mas isso não foi verificado na interface.

**Versões do OpenCode:**
- v2.0.18, o que vem com o OpenChamber (`@openchamberelectron\resources\opencode-cli`): o testado;
- v2.0.3, global do npm: lê a mesma configuração global; não testado;
- as duas usam o formato desta configuração (`providers`, `package: "aisdk:@ai-sdk/openai-compatible"`, `variants`),
  o da fase 4.

## 5. Testes de precisão

### 5.1 Pesos e kernels

- **Integridade:** tamanho e SHA-256 iguais aos fixados (secção 3). O motor carrega o ficheiro inteiro:
  - 41 blocos, MTP incluído;
  - 40 de 40 camadas MoE com routing na placa;
  - 6 329 de 10 496 especialistas na VRAM (8,2 GiB) a 48K.
- **Kernels contra FP64** (`poly_expert_parity` no GGUF real, `logs/expert-parity.log`), erro relativo:

  | Formato e forma | Um token (MMVQ) | Lote (MMQ) |
  | --- | ---: | ---: |
  | IQ3_XXS 2 048 → 512 (gate/up) | 0,0117 | 0,0117 |
  | IQ4_XS 512 → 2 048 (down) | 0,0121 | 0,0121 |
  | IQ4_XS 2 048 → 512 (camada 39) | 0,0117 | 0,0117 |
  | Q6_K 512 → 2 048 (down, 3 camadas) | 0,0115 | 0,0115 |
  | Q8_0 (atenção, delta net, partilhado) | 0,0114–0,0120 | 0,0114–0,0120 |
  | Q3_K 2 048 → 512 (MTP) | 0,0114 | 0,0114 |
  | **Q4_K 512 → 2 048 (MTP)** | 0,0111 | **0,0221** |
  | Especialista inteiro, CPU / GPU | 0,032–0,049 / 0,014–0,020 | — |

  No Q4 os lotes Q4_K/Q5_K davam 0,0233/0,0355. Aqui, todos os formatos do modelo principal dão o mesmo erro em
  lote e num token. O Q4_K do bloco MTP mantém o dobro, mas o bloco só faz palpites, que o modelo verifica.
- **Routing na placa contra o do host** (`STRATA_POLY_CARD_ROUTE_CHECK=1`, 256 tokens): 10 240 routings, **0
  escolhas diferentes**, pesos com diferença ≤ 8,9e-8 (`card-route-check.err`).

### 5.2 Paridade com o llama.cpp

Método da fase 4 (`tools/hip/poly_llama_parity.py`):
- 5 prompts: curto (22 tokens), código (870), documentos de 3 069, 11 481 e 35 575;
- greedy, 64 tokens por prompt;
- critério: os mesmos tokens, ou a primeira diferença num quase-empate (as duas mais prováveis a menos de 0,1), e
  diferença máxima ≤ 0,15.

Referências novas no mesmo GGUF, llama.cpp HIP b11160 (Unsloth), que ignora o bloco MTP:
- **A:** `--cpu-moe`;
- **B:** `--n-cpu-moe 22`.

| Configuração | Tokens | Diferença máxima | Média por prompt | Resultado | No Q4 |
| --- | --- | ---: | ---: | --- | --- |
| llama.cpp B contra A | 64/64 em 4; `doc-8k` separa-se no 39 num quase-empate (0,366/0,355 na B) | 0,054 | 0,0015–0,0023 | PASS | PASS (0,033) |
| `--window 1`, routing na placa, contra A | 64/64 em 3; `code` (28) e `doc-8k` (39) em quase-empates | 0,148 (`doc-8k`, no empate) | 0,0028–0,0066 | PASS | PASS (0,146; média até 0,0125) |
| idem, contra B | **64/64 nos 5** | 0,134 | 0,0047–0,0060 | **PASS** | **FAIL** (0,153–0,158) |
| `--window 1`, routing no host, contra A | 64/64 em 3; os mesmos dois quase-empates | 0,127 | 0,0020–0,0062 | PASS | PASS |
| `--window 1`, todos os especialistas na CPU, contra A | 64/64 em 4; `doc-8k` no quase-empate | 0,081 | 0,0022–0,0055 | **PASS** | **FAIL** (passo 45, 0,161) |
| `--window 1`, `STRATA_POLY_MMQ=0`, contra A | 64/64 em 3; `doc-30k` (37) e `doc-8k` (42) em quase-empates | 0,095 | 0,0038–0,0076 | PASS | PASS (0,085) |
| idem, contra B | 64/64 em 4; `code` num quase-empate | 0,124 | 0,0047–0,0062 | PASS | PASS (0,099) |
| Por omissão (MTP), só ids, contra A | os mesmos pontos de separação do `--window 1` | — | — | PASS | PASS |
| `--no-draft`, só ids, contra A | idem | — | — | PASS | PASS |

Notas:
- As referências do Q3 e as do Q4 são de ficheiros diferentes, por isso os quase-empates estão noutros passos. A
  comparação justa é a de cada configuração contra o próprio llama.cpp: no Q3 a média por prompt fica abaixo de
  0,0066 com o MMQ (até 0,0125 no Q4).
- Sem MMQ, a média não melhora no Q3 (0,0038–0,0076). O `short` contra B chega a 0,124 sem MMQ e a 0,085 com ele:
  a esta escala há variação nos dois caminhos.
- O bloco MTP acertou 184 de 231 palpites nestes prompts (80 %).

### 5.3 O passo sensível: `doc-2.5k`, passo 29

É o mesmo passo da fase 4. Nos dois modelos as referências concordam e o PolyStrata afasta-se. Probabilidade do
token das referências (`diag-step29.json`, `--window 1`):

| Configuração | Q3 | Q4 (fase 4) |
| --- | ---: | ---: |
| llama.cpp A / B | 0,933 / 0,941 | 0,927 / 0,931 |
| Prefill em lotes de 1 024 (omissão) | **0,815** | 0,751 |
| Lotes de 256 | **0,914** | 0,793 |
| Lotes de 16 | 0,873 | 0,880 |
| `STRATA_POLY_MMQ=0` | 0,867 | 0,867 |
| Todos os especialistas na CPU | 0,881 | 0,904 |

Leitura:
- no Q4 o desvio aparecia com o MMQ em qualquer lote grande (0,751–0,793), coerente com o arredondamento das
  escalas Q4_K/Q5_K;
- no Q3 o modelo principal não tem esses formatos e o desvio **só aparece com o MMQ em lotes de 1 024**: com
  lotes de 256, 0,914, o mais perto das referências de todas as configurações;
- falta uma causa no código: o MMQ do llama.cpp escolhe tamanhos de tile e de divisão do trabalho conforme o número
  de linhas, e é a hipótese a verificar. **Não foi investigado mais.**

O efeito é um passo de um prompt; a média por prompt é baixa (secção 5.2). Não se alterou o tamanho de lote por
omissão: isso muda a velocidade do prefill e teria de ser medido.

### 5.4 Conclusão sobre a precisão

| Problema do Q4 | No Q3 |
| --- | --- |
| MMQ Q4_K/Q5_K no prefill (2–3× o erro de um token) | **eliminado no modelo principal** (os mesmos erros em lote e num token). Só o Q4_K do bloco MTP o mantém, sem efeito nos tokens escritos |
| Diferença máxima > 0,15 contra B (2 de 5 prompts) | eliminado: 64/64 nos 5, máximo 0,134 |
| Separação fora de um quase-empate só com especialistas na CPU (passo 45) | não se repete: PASS, máximo 0,081 |
| Passo 29 do `doc-2.5k` | **persiste, menor e com outra assinatura**: só MMQ com lotes de 1 024 (0,815 contra 0,93) |

O Q3 não é "mais preciso" por ser Q3: os pesos são mais grosseiros. O que melhorou foi a concordância do motor com
o llama.cpp no mesmo ficheiro, porque os formatos IQ3_XXS/IQ4_XS/Q6_K não passam pelo caminho de escalas FP16 do
MMQ. A qualidade das respostas do Q3 contra o Q4 não foi medida (secção 11).

## 6. Benchmarks

Método da fase 4 (`tools/hip/bench_compare.py`):
- os mesmos prompts de ~2K, 8K, 31K e 45K (`bench/results/2026-10-10-hip-poly-vs-llama-vulkan/prompts.json`);
- contexto 49 152, 6 threads, 128 tokens por resposta, greedy;
- 3 corridas de cada modo, alternadas (por omissão com MTP, `--window 1`), depois de um aquecimento não contado.

Mediana, pedidos consecutivos no mesmo processo (conjunto B), prefill / decode tok/s:

| Prompt | Qwen3.6 **UD-Q3_K_M** (MTP) | Q3 `--window 1` | Qwen3.6 UD-Q4_K_M (MTP) | Qwen3.5 Q3_K_M |
| --- | --- | --- | --- | --- |
| B-2k (2 068) | 1 123 / **93,4** | 1 183 / 79,6 | 696 / 65,2 | 1 213 / 80,6 |
| B-8k (8 235) | 1 433 / **85,8** | 1 464 / 75,0 | 1 210 / 68,0 | 1 502 / 77,0 |
| B-31k (31 296) | 1 326 / **70,7** | 1 340 / 61,1 | 1 273 / 59,1 | 1 367 / 62,0 |
| B-45k (45 355) | 1 210 / **61,9** | 1 223 / 55,2 | 1 190 / 53,4 | 1 244 / 56,3 |

| | Q3 (MTP) | Q3 `--window 1` | Q4 (MTP) | Qwen3.5 |
| --- | --- | --- | --- | --- |
| TTFT B-2k / 8k / 31k / 45k (s) | 1,84 / 5,75 / 23,6 / 37,5 | 1,75 / 5,63 / 23,4 / 37,1 | 2,97 / 6,81 / 24,6 / 38,1 | 1,71 / 5,49 / 22,9 / 36,5 |
| Primeiro pedido do processo (A-2k): prefill / decode | 483 / 65,6 | 531 / 68,7 | 161 / 48,3 | 272 / 70,2 |
| Carga (até READY) | 27,8 s | 29,4 s | 31,6 s | 34,6 s |
| Especialistas na VRAM | 6 329 de 10 496 (8,2 GiB, 60 %) | idem | 4 503 de 10 496 (8,0 GiB, 43 %) | 6 570 de 10 240 (8,4 GiB) |
| Especialistas usados vindos da VRAM (pedidos B) | 96,6–97,4 % | 98,3–99,3 % | 91,9–95,6 % | 97,8–99,0 % |
| Palpites do MTP certos | 87,5 % (2 114 de 2 417) | — | 87,2 % | — |
| Pico de VRAM do processo (dedicada / partilhada) | 13 162 / 2 042 MiB | 13 158 / 2 034 MiB | 12 986 / 2 041 MiB | 12 986 / 2 042 MiB |
| RAM: working set / "privada" (commit) | **11,4** / 11,9 GB | 10,5 / 12,2 GB | 17,2 / 11,7 GB | 10,5 / 12,1 GB |

**São quantizações e versões diferentes:**
- Qwen3.6 Q3 e Q4: o mesmo modelo, especialistas de 3–4 bits contra 4–5 bits;
- Qwen3.5: outro modelo, com os mesmos formatos de especialistas do Qwen3.6 Q3.

As medições do Q4 e do Qwen3.5 são as da fase 4 (mesmo PC, mesmo binário, mesmos prompts, 11/10/2026), não
repetidas, porque os ficheiros foram apagados.

O que mudou:
- **Decode:**
  - o Q3 escreve +43 % a 2K e +16 % a 45K sobre o Q4;
  - com o mesmo modo (`--window 1`), fica igual ao Qwen3.5 (79,6 contra 80,9 a 2K, 55,2 contra 55,7 a 45K): os
    mesmos formatos, quase a mesma fração na VRAM;
  - a diferença para o Q4 vem do tamanho dos especialistas (13,3 contra 18,2 GiB): 60 % cabem na VRAM em vez de
    43 %, e 97 % dos usados vêm de lá.
- **MTP:** +17 / +14 / +16 / +12 % a quente sobre o `--window 1`, contra +1 a +7 % no Q4. As janelas de palpites
  continuam a encaminhar no host; com quase todos os especialistas na VRAM, isso pesa menos.
- **Prefill:** ao nível do Qwen3.5. O prefill de 2K, que era o ponto fraco do Q4 (~700 tok/s), passa a 1 123.
- **Memória:** 5,8 GB a menos de working set do que o Q4. Com os outros programas abertos, a RAM disponível depois
  da carga fica em ~14,5 GB.

Ficheiros: `bench.json`, `bench-summary.md`, `logs/bench-*.log`.

## 7. Contexto longo e memória

### 7.1 Notas a 5, 50 e 95 %

`tools/hip/kv_needles.py`, como na fase 4:
- três notas em documentos do repositório;
- greedy, sem pensamento;
- cada nota perguntada sozinha, depois as três de uma vez, e um resumo de 128 tokens para o decode;
- modo por omissão (MTP ligado).

| Configuração (perfil do launcher) | Prompt | Notas | Prefill (1.º pedido) | Decode no resumo | Q4 (fase 4): prefill / decode |
| --- | ---: | ---: | ---: | ---: | --- |
| 48K | 8 175 | 6/6 | 1 405 | 66,0 | 689 / 38,0 |
| 48K | 32 751 | **4/6** (as 3 sozinhas ✓; as 3 juntas: 1) | 1 347 | 55,4 | 1 192 / 47,0 |
| 48K | 44 983 | 6/6 | 1 246 | 56,7 | 1 190 / 46,0 |
| 128K, KV na VRAM | 65 519 | 6/6 | 1 133 | 37,9 | 1 037 / 24,8 |
| 128K, KV na VRAM | 130 883 | **4/6** (idem) | 780 | 27,4 | 768 / 24,2 |
| 262K, KV na VRAM | 260 983 | 6/6 | 506 (8,6 min) | 14,1 | 497 / 13,8 |
| 262K com streaming a partir de 49 152 | 65 519 | 6/6 | 1 041 | 28,1 | — |
| 262K com streaming a partir de 49 152 | 260 983 | 6/6 | 457 (9,5 min) | 3,6 | 460 / 3,7 |

Cada nota perguntada sozinha foi encontrada **em todos os comprimentos e configurações**: 24 de 24, KV na VRAM e em
streaming, com o bloco MTP a ler a KV dividida. A atenção leu sempre todas as posições.

### 7.2 A pergunta pelas três notas de uma vez

A 32K e a 131K o modelo respondeu só "4817-PLUM" e terminou. Repetido a 32K (`longctx/ctx32k-*.json`,
`ctx32k-combined.json`):

| Modo | Resposta |
| --- | --- |
| Por omissão (MTP), duas corridas pelo `--tokens` | `4817-PLUM` (o mesmo nas duas: determinístico) |
| `--window 1` | `4817-PLUM, HERON-62, 3B-917` |
| `--no-draft` | idem, 3/3 |
| `STRATA_POLY_MMQ=0` | idem, 3/3 |
| Todos os especialistas na CPU | idem, 3/3 |

Com `--window 1 --top-probs`, o passo a seguir a "4817-PLUM" dá `,` com 0,346, o fim da resposta (`<|im_end|>`) com
0,298 e ` HER` com 0,250.

É um **quase-empate** (0,048 de diferença, abaixo do critério de 0,1 da paridade). A janela de verificação do MTP
calcula vários tokens de uma vez e desempata para o outro lado. Não é um defeito do MTP nem da KV: é a sensibilidade
normal de duas formas de cálculo num quase-empate, a mesma que a paridade aceita. Para o utilizador, a consequência
é que, nesta pergunta, o modo por omissão termina mais cedo do que o `--window 1`. A fase 4 não registou o mesmo
com o Q4, mas eram outros pesos e outros empates.

### 7.3 VRAM (relatório de carga)

| MiB | 48K | 128K | 262K | 262K streaming | Q4 48K / 128K / 262K / stream |
| --- | ---: | ---: | ---: | ---: | --- |
| Pesos densos | 1 961 | 1 961 | 1 961 | 1 961 | 1 961 (iguais) |
| Cache de especialistas | **8 396** | **6 635** | **3 818** | **7 905** | 8 204 / 6 442 / 3 628 / 7 988 |
| Especialistas na VRAM | 6 329 (60 %) | 5 004 (48 %) | 2 883 (27 %) | 5 961 (57 %) | 4 503 (43 %) a 48K |
| KV na VRAM | 1 056 | 2 816 | 5 632 | 1 056 (+ 458 de staging) | iguais |
| Estado recorrente | 251 | 251 | 251 | 251 | iguais |
| Scratch e o resto | 1 001 | 1 002 | 1 002 | 1 002 | ~1 017 |
| Livre (reserva) | 1 006 | 1 006 | 1 006 | 1 023 | ~1 006 |
| KV na RAM fixada | 0 | 0 | 0 | 4 576 | 0 / 0 / 0 / 1 760 (stream a 128K) |

A cache de especialistas tem o mesmo tamanho em MiB do que no Q4. Como cada especialista é menor, cabem mais.

### 7.4 RAM

| Configuração | Working set máximo do processo | "Privada" (commit) | RAM disponível antes / depois |
| --- | ---: | ---: | --- |
| 48K (notas até 45K) | 11,3 GB | 11,8 GB | 18,7 / 18,5 GB |
| 128K | 13,1 GB | 10,1 GB | 18,5 / 18,1 GB |
| 262K | 15,4 GB | 13,0 GB | 18,2 / 19,0 GB |
| 262K streaming | 16,4 GB | 16,0 GB | 19,0 / 18,9 GB |
| Q4, 48K / 262K (fase 4) | 17,1 / 19,3 GB | 11,6 / 12,8 GB | — |

O ficheiro de paginação ficou estável (variação ≤ 440 MiB). Como na fase 4, a "RAM privada" é compromisso que
acompanha as alocações na VRAM, não RAM ocupada.

## 8. API, OpenCode e OpenChamber

### 8.1 API (`tools/hip/poly_api_check.py`, servidor no perfil de 48K, `api-check.json`)

**15 de 15:**
- modelos; system prompt;
- streaming (1.º token a 0,25 s); multi-turno;
- ferramentas com ciclo completo, com e sem streaming; ferramentas Anthropic;
- resposta longa (2 100 tokens a **79,7 tok/s**; Q4: 2 326 a 55,5);
- cancelamento (pedido seguinte em 0,37 s); Anthropic com e sem streaming;
- raciocínio e raciocínio em streaming (`reasoning_content` a partir de 0,28 s);
- `preserve_thinking` (2.º turno com 429 tokens reutilizados);
- título (0,57 s);
- **sessão de 40 turnos**: 40 de 40 certas, cada turno reutiliza o anterior (no 40.º, 12 906 de 13 255 tokens da
  cache), 1.º token a 0,79–1,26 s, e a pergunta sobre o 1.º turno respondida.

O servidor escuta só em `127.0.0.1`.

**Troca de perfil por pedido** (`switch_on_request`, o mesmo servidor):

| Pedido com `model` | Respondido por | Tempo |
| --- | --- | ---: |
| `qwen3.6-35b-a3b-ud-q3_k_m-hip` | o carregado (48K) | 0,3 s |
| `polystrata` (o do título) | o carregado, sem troca | 0,2 s |
| `qwen3.6-35b-a3b-ud-q3_k_m-hip-128k` | 128K, depois de trocar | 31,7 s |
| `polystrata` | 128K, sem troca | 0,2 s |
| `qwen3.6-35b-a3b-ud-q3_k_m-hip` | 48K, depois de trocar | 31,6 s |

Com um só ficheiro, ele fica na cache de ficheiros entre perfis. A troca cai para ~32 s, contra 40–55 s entre o Q4
e o Qwen3.5.

### 8.2 OpenCode (o v2.0.18 do OpenChamber, `run --standalone`)

| Passo | Configuração | Resultado |
| --- | --- | --- |
| "Reply with exactly PONG", `polystrata-hip/qwen3.6-35b-a3b#none` | **global** (a que o OpenChamber lê), numa pasta sem configuração de projeto | ok, 28 s; o prompt do agente (14 630 tokens) lido em 9 s |
| Agente com ferramenta: ler `notes.yaml` e responder | global, `--auto` | ok: chamou `read`, resposta certa, 18 s |
| Continuação (`--continue`) | global | ok: resposta certa a partir do turno anterior, 6 s; só a parte nova lida |
| PONG no perfil 128K (`qwen3.6-35b-a3b-128k`), com o servidor em 48K | do repositório (`opencode.json`, com `current` e o agente `title`) | ok: uma troca para 128K e a resposta, 59 s no total |

No conjunto destes pedidos, 94,8 % dos tokens dos prompts não foram lidos outra vez (cache de conversas).

**Não testado:** a interface gráfica do OpenChamber. Foi usado o executável do OpenCode que ela traz, com a mesma
configuração global.

## 9. Problemas encontrados

| Problema | Como apareceu | Estado |
| --- | --- | --- |
| Duas cópias diferentes de "Qwen3.5-35B-A3B Q3_K_M" no disco | procura de `*.gguf` | Resolvido: apagada só a do PolyStrata (identificada pelo caminho nas configurações, pelo tamanho e pelos metadados); a do LM Studio (`Manojb`, 16 248 175 136 bytes) ficou |
| Servidor a correr com o Q4 | lista de processos | Parado antes de apagar |
| `poly_config.py` sem `--name` escreve `…-q3-k-m-hip` (troca `_` por `-`) | primeira configuração | Refeito com `--name qwen3.6-35b-a3b-ud-q3_k_m-hip`, o padrão das configurações anteriores |
| O `opencode.json` global tem BOM UTF-8 | edição | Lido e escrito com `utf-8-sig`; BOM mantido |
| Com um só modelo, o `START-POLY-HIP.bat` não sabia qual era e mostrava as velocidades do Qwen3.5 | leitura do script | Com um modelo configurado, é esse; velocidades do UD-Q3_K_M, ou "nao medida" para outro |
| `sed -i` no Git Bash deixou o `.bat` com fins de linha LF na cópia de trabalho | `file` | Reposto o CRLF (o git guarda LF e converte; `.gitattributes`: `eol=crlf`) |
| Passo 29 do `doc-2.5k` (MMQ, lotes de 1 024) | paridade | Medido e documentado (secção 5.3); não corrigido |
| Pergunta combinada com MTP a 32K e 131K | notas | Quase-empate (secção 7.2); não é defeito |
| `opencode models` não escreve nada sem terminal | verificação dos modelos | Verificado por `opencode run` com cada perfil |

Nenhum código do motor ou do servidor foi alterado. Por isso não se repetiram o ctest, os testes do servidor nem o
`poly_kernel_bench`: o binário é o da fase 4, onde passaram (ctest com as mesmas 3 falhas conhecidas).

## 10. Viabilidade preliminar do GLM-4.7-Flash

Só análise: o GLM não foi descarregado e o port não começou. Fontes:
- o `config.json` oficial (`zai-org/GLM-4.7-Flash`, revisão `7dd20894`);
- os cabeçalhos de três GGUF da Unsloth (`unsloth/GLM-4.7-Flash-GGUF`, revisão `0d32489e`): metadados e tabela de
  tensores lidos com `tools/hip/gguf_head.py` por pedidos HTTP parciais (~9,5 MB por ficheiro, sem os pesos).

Ficheiros em `bench/results/2026-10-11-hip-qwen36-q3/glm-4.7-flash-headers/`. **Tudo o que é desempenho nesta
secção é estimativa, não medição.**

### 10.1 Arquitetura

| | GLM-4.7-Flash | Qwen3.6-35B-A3B (para comparar) |
| --- | --- | --- |
| Classe HF / arquitetura no GGUF | `Glm4MoeLiteForCausalLM` / **`deepseek2`** | `qwen3_5_moe` / `qwen35moe` |
| Camadas | 47: a 1.ª com FFN densa (10 240), 46 com MoE | 40: 30 delta net + 10 atenção, todas MoE |
| Atenção | **MLA** (latente, como o DeepSeek-V2/V3) em todas as 47: 20 cabeças, `q_lora_rank` 768, `kv_lora_rank` 512, chave 192 + 64 de RoPE, valor 256; no GGUF, `attn_k_b`/`attn_v_b` já separados (forma absorvida) | atenção completa com GQA em 10 camadas, delta net nas outras 30 |
| Especialistas | 64 por camada, **4 usados**, 1 partilhado; largura 1 536 | 256 por camada, 8 usados, 1 partilhado; largura 512 |
| Routing | **sigmóide + bias** (`noaux_tc`, um só grupo), pesos normalizados × 1,8 | softmax, os 8 maiores, renormalizados |
| RoPE | θ = 10⁶ nas 64 dimensões de RoPE, sem YaRN | parcial, θ = 10⁷ |
| Contexto nativo | 202 752 | 262 144 |
| Vocabulário / pré-tokenizador | 154 880 / `glm4` | 248 320 / `qwen35` |
| MTP | o `config.json` declara 1 camada (`num_nextn_predict_layers`), **mas os GGUF da Unsloth não a trazem** (47 blocos, nenhum tensor `nextn`) | bloco MTP no ficheiro (`blk.40`) |

**Compatibilidade com o motor:** não precisa de um backend novo, mas precisa de uma **classe de arquitetura nova**,
como `Qwen35`, `Gemma4` e `Glm5Next` são classes do mesmo `strata-poly`. Hoje o motor reconhece `glm5-next`,
`qwen35moe` e `gemma4` (`src/poly/engine.cpp`), não `deepseek2`. Tudo o que é comum já serve: a cache de
especialistas, a execução híbrida CPU/GPU, os produtos com slots, os graphs, o orçamento de VRAM, a gestão das
páginas no Windows e o servidor.

O `Glm5Next` (GLM 5.3 Flash) já tem quase todas as peças, mas **só correu em Linux com NVIDIA**. Compila no binário
HIP; nunca foi executado em HIP. As peças:
- atenção latente absorvida: `q_a` → norma → `q_b` → `k_b` absorvido na query, cache latente FP16, `v_b`; o ramo
  denso de `dsa()` é atenção causal normal;
- routing sigmóide + bias com pesos normalizados e escalados (`Glm5Next::route`);
- FFN densa nas primeiras camadas e especialista partilhado.

O servidor já trata o pré-tokenizador `glm4` e o formato de ferramentas do GLM (`<arg_key>`/`<arg_value>`,
`serve/test_glm.py`).

### 10.2 Componentes a adaptar

1. **Classe `deepseek2`** (estimativa: ~400–600 linhas, ao nível de `glm5next.cpp`):
   - nomes de tensores e chaves `deepseek2.*`;
   - a camada sem as hyper-connections nem a KDA do GLM 5.3: resíduo normal, MLA, FFN densa ou MoE;
   - RoPE nas 64 dimensões da chave partilhada, conferida contra o grafo `deepseek2` do llama.cpp;
   - `row_bytes`/`save_rows`/`load_rows` para a cache de conversas.
2. **Atenção MLA em HIP:**
   - o `dsa_attend` (hipBLAS sobre a cache latente) nunca correu em HIP e precisa de paridade contra o llama.cpp;
   - no prefill de prompts longos vale medir a forma descomprimida (chaves e valores de 256 por cabeça) contra a
     absorvida (576/512): a absorvida faz ~2× as contas por posição.
3. **KV:**
   - 576 valores FP16 por camada e posição × 47 camadas = **54 144 bytes por posição**, 2,4× os 22 528 do Qwen3.6
     (que só tem 11 camadas com atenção);
   - 2,5 GiB a 48K, 6,6 GiB a 128K, 10,2 GiB a 202 752;
   - o KV streaming da fase 3 foi feito para K/V por cabeça e teria de aprender a linha latente.
4. **Routing na placa:**
   - o kernel atual é só `route_topk_softmax`; falta uma variante sigmóide + bias + normalização × 1,8 (64
     especialistas, 4 usados);
   - os produtos com slots já servem linhas de 1 536 e de 2 048 (o kernel geral de `e33aa738`) e os formatos dos
     ficheiros Q3: Q3_K, Q4_K, Q5_K, IQ3_XXS, IQ3_S, IQ4_XS. O Q3_K foi exercitado nesta fase no bloco MTP do
     Qwen3.6 (secção 5.1).
5. **Graph capture:**
   - a cabeça de camada nova tem de ser gravável (sem sincronizar com o host), como as atuais;
   - os fallbacks por variável (`STRATA_POLY_CARD_ROUTE=0`, `STRATA_POLY_COPY_KERNELS=0`, ...) servem sem
     alteração.
6. **Servidor e conversas:**
   - template (`[gMASK]<sop>`, `<|user|>`, `<think>`);
   - o `--turn-token` da cache de conversas (o id de `<|user|>`);
   - conferir o formato das ferramentas do 4.7 com o do 5.3, que o servidor já trata.

### 10.3 Memória (16 GB de VRAM, 32 GB de RAM)

| GGUF (Unsloth) | Ficheiro | Especialistas | Formatos dos especialistas | Termo de mínimo (o caminho do MMQ do Q4, secção 5) |
| --- | ---: | ---: | --- | --- |
| Q3_K_M | 14,6 GB | 12,29 GiB | Q3_K, **Q4_K** (45 tensores), Q5_K (1) | sim, em 1/3 dos tensores |
| UD-Q3_K_XL | 13,8 GB | 11,34 GiB | Q3_K, Q4_K (9) | pouco |
| UD-IQ3_XXS | 12,9 GB | 10,54 GiB | IQ3_XXS, IQ3_S, IQ4_XS | **não** (os formatos do Qwen3.6 Q3 validado aqui) |

Estimativa a partir do relatório de carga do Qwen3.6 UD-Q3_K_M (secção 7.3), com o mesmo orçamento:
- o Qwen3.6 tem 1,96 GiB de pesos densos e 1,06 GiB de KV a 48K, e a cache de especialistas fica com 8,2 GiB;
- no GLM os pesos densos são ~1,3 GiB e a KV a 48K 2,5 GiB. Ficariam **~7,5 GiB para especialistas**: ~60 % dos do
  Q3_K_M, ~70 % dos do UD-IQ3_XXS;
- a 128K, a KV (6,6 GiB) deixaria ~2–3 GiB para especialistas, com os ~4,8 GB que os outros programas ocupam aqui.
  Os 202K exigem KV streaming.

**RAM:**
- o ficheiro inteiro (13–15 GB) cabe na cache de ficheiros ao lado dos outros programas;
- os especialistas fora da VRAM seriam ~4–5 GiB, menos do que os ~5 GiB do Qwen3.6 Q3, que dá 11,4 GB de working
  set (secção 6).

**Conclusão:** as variantes Q3 cabem confortavelmente a 48K. A 128K e acima, o limite é a KV latente, não os
especialistas.

### 10.4 Desempenho a investigar (estimativas)

Bytes lidos por token:
- **especialistas:** 4 × 46 × 4,27 MiB = **~790 MiB** no Q3_K_M (~680 MiB no UD-IQ3_XXS), contra ~425 MiB no
  Qwen3.6 Q3 (8 × 40 × 1,33 MiB);
- **pesos densos:** ~1,2 GiB contra ~1,9 GiB no Qwen3.6, que tem a cabeça de saída maior (248K de vocabulário) e
  a delta net.

O lado da placa fica parecido. Pesam:
- a parte dos especialistas que fica na CPU: ~2,5× maior por token (menos especialistas na VRAM, cada um 3×
  maior);
- a leitura da KV latente em contextos longos: a 32K, ~1,7 GB por token contra ~0,7 GB;
- a falta de MTP no GGUF.

| | Estimativa para o GLM-4.7-Flash Q3, sem MTP | Medido no Qwen3.6 UD-Q3_K_M |
| --- | --- | --- |
| Decode, conversa curta | 50–70 tok/s | 79,6 (`--window 1`), 93,4 (MTP) |
| Decode a ~32K | 30–45 tok/s | 61,1 (`--window 1`), 70,7 (MTP) |
| Prefill a 2–8K | 800–1 200 tok/s | 1 123–1 433 |
| Prefill a ~32K | 300–600 tok/s: a atenção MLA de 47 camadas passa a dominar as contas | 1 326 |

São ordens de grandeza para decidir o que medir primeiro, não promessas. A investigar, por esta ordem:
1. a fração dos especialistas na VRAM e o hit rate com 64 especialistas e 4 usados;
2. o decode a 8K–45K, limitado pela leitura da KV latente;
3. o prefill longo, com a MLA absorvida contra a descomprimida.

### 10.5 GPU routing e speculative decoding

- **Routing na placa:** o trabalho principal é o kernel sigmóide + bias e a verificação contra o host
  (`STRATA_POLY_CARD_ROUTE_CHECK=1`). O resto da infraestrutura é genérico: cabeças gravadas por forma de routing,
  espera por camada, desistência verificada no fim de cada token.
- **Speculative decoding:**
  - os palpites por repetição (o modo `--no-draft`) funcionam sem nada de novo;
  - para o MTP seria preciso um GGUF com a camada `nextn` (convertido dos safetensors oficiais, que a declaram) e
    o `block_attend`/`block_rows` dessa camada MLA, a partir dos do `Glm5Next`;
  - convém antes fazer o routing na placa para as janelas de palpites (secção 12 de
    `AMD_QWEN36_IMPLEMENTATION.md`): no Qwen3.6 Q3, o MTP rende +12 a +17 % mesmo com essas janelas a encaminhar
    no host.

**Recomendação para a próxima fase:**
1. descarregar o **UD-IQ3_XXS** (12,9 GB; formatos sem termo de mínimo, os que o Qwen3.6 Q3 validou aqui) ou o
   UD-Q3_K_XL;
2. escrever a classe `deepseek2` a partir do `Glm5Next`;
3. validar primeiro `--window 1` com routing no host contra o llama.cpp HIP b11160 (que corre `deepseek2`) no mesmo
   GGUF;
4. só depois o routing na placa e o KV streaming.

## 11. O que não foi testado

- **Qualidade das respostas do Q3 contra o Q4** (benchmarks de tarefas): só a concordância com o llama.cpp no mesmo
  ficheiro e a recuperação de notas. Q3 tem pesos mais grosseiros do que Q4; não se mediu quanto isso custa.
- A interface gráfica do OpenChamber, e o OpenCode v2.0.3 global do npm.
- A causa do passo 29 com lotes de 1 024 (secção 5.3).
- Imagens (`mmproj`, não descarregado), como na fase 4.
- ctest, testes do servidor e `poly_kernel_bench`, porque nada no código mudou desde a fase 4.
- Uso de horas seguidas.
- CUDA e Linux: nada mudou no código.

## 12. Como usar

`START-POLY-HIP.bat` (duplo clique): como só há um modelo, mostra só o menu de contexto (48K, 128K, 262K, 262K com
KV streaming), com as velocidades medidas deste modelo, e a última escolha fica como omissão. Sem menus:

```bat
START-POLY-HIP.bat 48k
START-POLY-HIP.bat 128k
START-POLY-HIP.bat chat
START-POLY-HIP.bat check 262k-stream
```

Para voltar a configurar outro modelo no futuro, `START-POLY-HIP.bat C:\caminho\Outro.gguf` (ou
`tools\hip\poly_config.py`); o menu de modelo volta a aparecer quando houver mais do que um. As configurações
antigas estão em `C:\models\archive\2026-10-11-polystrata-hip-configs\`: para reativar uma, basta
devolvê-la à raiz do repositório com o seu GGUF no sítio.

No OpenCode/OpenChamber: fornecedor "PolyStrata (HIP, RX 9060 XT)" (global) ou "PolyStrata (HIP, this PC)"
(`tools/hip/opencode-polystrata-hip.json`), modelos "Qwen3.6 35B A3B Q3 - 48K / 128K / 262K / 262K Streaming".
