> Historical measurement report. Use the [current setup instructions](../../README.md). Personal paths were generalized for publication; measurements were preserved. Raw campaign links may require the local evidence archive.

# Port HIP do `strata-poly` — fase 4: Qwen3.6-35B-A3B

Ramo `hip-poly`, commits `e33aa738..` sobre o fim da fase 3 (`3f0e79aa`). Não foi feito push nem aberto PR.
Fases anteriores: [AMD_PORT_IMPLEMENTATION.md](AMD_PORT_IMPLEMENTATION.md), [AMD_PORT_PHASE2.md](AMD_PORT_PHASE2.md),
[AMD_KV_STREAMING.md](AMD_KV_STREAMING.md).

**Onde foi medido:** este PC. RX 9060 XT 16 GB (gfx1200), Ryzen 5 5600X (6 núcleos), 31,9 GB de RAM, Windows 11 Pro,
ROCm 10.2.0a20260930 (TheRock), em uso normal: antes de cada corrida, outros programas ocupavam 4,7–4,9 GB de VRAM
e deixavam ~20 GB de RAM livre (registado por corrida nos JSON). Modelos:

- **Qwen3.6-35B-A3B UD-Q4_K_M** (Unsloth, `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` revisão `5bc3e238`), 22 663 387 424
  bytes, SHA-256 `0b21525e…7dac58b` (o fixado no `setup.py`). Descarregado nesta fase, com autorização, para
  `C:\models`.
- **Qwen3.5-35B-A3B Q3_K_M** (Unsloth), o mesmo das fases 1 a 3: a referência.

Dados: `bench/results/2026-10-11-hip-qwen36/`.

---

## 1. Resumo executivo

- **O Qwen3.6-35B-A3B corre no PolyStrata HIP, com as otimizações das fases anteriores.** É a mesma arquitetura
  GGUF do Qwen3.5 (`qwen35moe`), por isso a classe `Qwen35` serve os dois sem uma segunda implementação. O
  ficheiro exercitou pela primeira vez em HIP o bloco MTP, especialistas K-quant e um ficheiro maior do que cabe na
  RAM ao lado dos outros programas. Isso revelou quatro problemas, todos corrigidos (secção 9):
  - o routing na placa ficava desligado com especialistas Q4_K/Q5_K/Q6_K;
  - o motor guardava páginas do ficheiro de que não precisava, e o Windows despejava as que a CPU usa;
  - uma desistência da espera na placa podia dar um token errado **sem erro** (defeito que já existia);
  - as gravações das cabeças das camadas confundiam as duas formas de routing.
- **Correção numérica:** greedy, os mesmos ids, contra o llama.cpp no mesmo GGUF:
  - o modo por omissão (MTP), `--no-draft` e `--window 1` com routing no host passam nos 5 prompts (22 a 35 575
    tokens);
  - o `--window 1` com routing na placa passa contra a referência A (llama.cpp, especialistas na CPU); contra a B
    (especialistas na GPU) dois prompts de documento passam o limite de 0,15 por 0,003–0,008;
  - com todos os especialistas na CPU, o `doc-2.5k` separa-se no passo 45 fora de um quase-empate (0,161).
  - **Causa medida:** o MMQ do llama.cpp na RDNA4 (o que o PolyStrata usa no prefill, revisão `3cf0325`) guarda em
    FP16 o produto das escalas Q4_K/Q5_K e tem 2–3× o erro do produto de um token (secção 5.3). Sem ele
    (`STRATA_POLY_MMQ=0`) os 5 prompts passam contra A e B (64/64 contra B) com diferenças de 0,018–0,099, a
    dispersão entre os próprios backends HIP e Vulkan do llama.cpp. A falha só-CPU não foi explicada por esta
    causa (sem MMQ esse prompt passa).
- **Desempenho** (mediana de 3 corridas alternadas, contexto 49 152, pedidos consecutivos):
  - Qwen3.6 UD-Q4_K_M: decode 65,2 / 68,0 / 59,1 / 53,4 tok/s a 2K / 8K / 31K / 45K; prefill 696–1 273 tok/s;
  - Qwen3.5 Q3_K_M: 80,6 / 77,0 / 62,0 / 56,3 e 1 213–1 502.
  - A diferença vem do tamanho dos especialistas (18,2 contra 13,0 GiB; cabem ~44 % contra ~64 % na VRAM) e encolhe
    com o contexto.
  - O MTP acerta ~85 % dos palpites mas rende só +1–7 % sobre um token por chamada, porque as janelas de palpites
    ainda encaminham no host.
- **Contexto longo:** notas a 5, 50 e 95 % encontradas todas (48 de 48), aos 48K, 65K, 131K e 261K, com a KV toda
  na VRAM e com KV streaming, incluindo o bloco MTP com a KV dividida (por testar desde a fase 3).
- **Sem regressão no Qwen3.5:** binário de antes contra o de agora, 3 corridas alternadas: decode e prefill iguais
  dentro da dispersão; o primeiro pedido melhora; o working set do processo baixa de 18,5 para 10,6 GB. A paridade
  com as referências da fase 2 continua a dar 64/64. ctest: as mesmas 3 falhas conhecidas.
- **Integração:**
  - `START-POLY-HIP.bat` com menu de modelo; **os dois modelos na mesma porta (8090)**: um pedido que nomeia o
    outro modelo troca o servidor para ele (`switch_on_request`, secção 8.4);
  - um só fornecedor OpenCode com os 8 perfis dos dois modelos, sem alterar a configuração global nem os modelos
    que já estavam;
  - API 15/15 verificações;
  - OpenCode (o do OpenChamber): agente com ferramenta, continuação de sessão e títulos em ~2 s sem raciocínio;
  - o `preserve_thinking` do template do Qwen3.6 chega ao modelo, e com ele cada turno lê só o que é novo.

## 2. Diferenças arquiteturais entre Qwen3.5 e Qwen3.6

### 2.1 Fontes

- **Modelo oficial:** `Qwen/Qwen3.6-35B-A3B` (Hugging Face, revisão `995ad96e`, 24/04/2026), `config.json` e
  `chat_template.jinja`, comparados com os de `Qwen/Qwen3.5-35B-A3B` (`59d61f3c`).
- **GGUF:** `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, revisão `5bc3e238`, a que o `setup.py` já fixa. Os metadados e a
  tabela de tensores de 10 variantes foram lidos por pedidos HTTP parciais (só o cabeçalho, ~11 MB cada), sem
  descarregar os ficheiros.
- **Implementação de referência:** o grafo `qwen35moe` do llama.cpp (o `strata-poly` segue-o, `src/poly/qwen35.cpp`)
  e o llama.cpp HIP b11160 do Unsloth, que carrega o ficheiro e ignora o bloco MTP ("unused tensor blk.40.nextn.*").

### 2.2 Modelo

O `config.json` oficial é **o mesmo** nos dois modelos em tudo o que define a rede: `model_type` `qwen3_5_moe`,
40 camadas (30 gated delta net + 10 de atenção completa, `full_attention_interval` 4), 2 048 de largura, 256
especialistas com 8 usados, especialista partilhado de 512, atenção 16/2 cabeças de 256 dimensões com porta de
saída, RoPE parcial (64 de 256 dimensões, `mrope_section` [11, 11, 10] intercalado, θ = 10⁷), delta net 16/32
cabeças de 128, convolução de 4, `rms_norm_eps` 1e-6, vocabulário de 248 320, contexto nativo 262 144, **1 camada
MTP** (`mtp_num_hidden_layers`) e o mesmo codificador de imagens. Diferem só campos sem efeito na rede
(`bos_token_id`, `pad_token_id`, `output_router_logits`, `tie_word_embeddings` explícito como `False`, a versão do
transformers).

Nos GGUF, todas as chaves `qwen35moe.*` são iguais, salvo:

| Chave | Qwen3.5 Q3_K_M (Unsloth) | Qwen3.6 UD-Q4_K_M (Unsloth) |
| --- | --- | --- |
| `block_count` | 40 | 41 (a 41.ª é o bloco MTP) |
| `nextn_predict_layers` | — | 1 |
| `tokenizer.ggml.bos_token_id` / `add_bos_token` | — | 248 044 / `false` |

### 2.3 Componente a componente

| Componente | Diferença | Consequência no motor |
| --- | --- | --- |
| Tensores da rede principal | Os mesmos nomes e formas. `ssm_alpha` e `ssm_beta` (2 048 × 32) vêm em **F32** no 3.6 e em Q8_0 no 3.5 | `upload()` aceita F32 (`kAnyMat`); o produto F32 já existia. Sem alteração |
| Atenção, delta net, normalização, RoPE | Iguais (mesmos metadados, mesmo grafo) | Reutilizados sem alteração |
| MoE e routing | Iguais: softmax sobre 256, os 8 maiores, pesos renormalizados com o mesmo piso | O routing na placa (`route_topk_softmax`) serve tal como está |
| Formatos dos especialistas | No UD-Q4_K_M: gate/up **Q4_K**, down **Q5_K**, e down **Q6_K** nas camadas 34, 38 e 39 (no Qwen3.5 Q3_K_M: IQ3_XXS / IQ4_XS) | Os produtos com slots na memória da placa só existiam para as linhas que cabem em meio warp: **o routing na placa ficava desligado no modelo inteiro.** Corrigido (secção 3) |
| Bloco MTP (`blk.40`) | Novo no ficheiro: uma camada de atenção com MoE (especialistas Q4_K/Q5_K, router **BF16**), `nextn.eh_proj` Q8_0, `enorm`/`hnorm`/`shared_head_norm` F32 | O `strata-poly` já o carregava e corria em CUDA (`Engine::load_drafter`, `Qwen35::block_attend`); em HIP nunca tinha corrido. Validado (secção 5) |
| Tokenizer | Mesmo vocabulário, merges e tipos de tokens (ficheiros iguais byte a byte), mesmo `pre` (`qwen35`) | Os mesmos ids; o servidor trata o raciocínio e as ferramentas da mesma forma |
| Template de chat | O do GGUF (Unsloth) junta dois system/developer seguidos, aceita o papel `developer` em qualquer posição, não levanta erros por papéis inesperados, e tem `preserve_thinking`: com ele, o raciocínio dos turnos anteriores fica no histórico | O tokenizer é escrito a partir do GGUF (template novo). O servidor passa a deixar chegar `chat_template_kwargs.preserve_thinking` ao template (secção 3) |
| BOS | Declarado (248 044 = `<\|endoftext\|>`) mas `add_bos_token = false` e o template não o usa | Nada é acrescentado: os mesmos ids que no Qwen3.5 |
| Speculative decoding | O ficheiro traz o bloco MTP (no Qwen3.5 Q3_K_M da Unsloth não vinha) | Janelas de 8 com 3 palpites do bloco, além dos palpites por repetição |
| Imagens | Mesmo codificador (`mmproj`) | Não testado nesta fase (não descarregado) |

**Conclusão:** é a mesma arquitetura GGUF, por isso não há uma segunda implementação. A classe `Qwen35` serve os
dois modelos (já servia em CUDA). O trabalho foi fazer funcionar em HIP o que o ficheiro do 3.6 exercita pela
primeira vez: especialistas K-quant no routing na placa, o bloco MTP, um ficheiro maior do que cabe na RAM ao lado
dos outros programas, e as partes do servidor e da integração.

### 2.4 Quantização escolhida

Variantes da revisão fixada (tamanho dos dados; especialistas gate/up · down):

| Ficheiro | GB | Especialistas | Notas |
| --- | ---: | --- | --- |
| UD-Q3_K_M | 17,1 | IQ3_XXS · IQ4_XS (Q6_K em 3 camadas) | os mesmos formatos do Qwen3.5 Q3_K_M |
| UD-Q3_K_XL | 17,2 | idem; embedding Q8_0 | |
| UD-IQ4_XS | 18,2 | **IQ3_S** · IQ4_XS | "Q4" só no nome |
| UD-Q4_K_S | 21,4 | Q4_K · Q4_K | |
| **UD-Q4_K_M** | **22,7** | **Q4_K · Q5_K** (Q6_K em 3 camadas) | escolhida; SHA-256 fixado no `setup.py` |
| MXFP4_MOE | 22,2 | MXFP4 · Q5_K | o MXFP4 não tem kernel nativo no motor |

A proposta inicial era o UD-Q3_K_M (o mais comparável e o mais rápido). Por tua escolha ficou o **UD-Q4_K_M**:
melhor qualidade, e o primeiro modelo real com especialistas K-quant em HIP. Os seus especialistas ocupam 18,2 GiB
(contra 13,0 GiB do Qwen3.5 Q3_K_M), dos quais ~9 GiB cabem na VRAM. As comparações com o Qwen3.5 (secção 6) são
por isso entre quantizações diferentes, e é assim que estão documentadas.

## 3. Ficheiros e funções alterados

Commits sobre `3f0e79aa` (fim da fase 3):

| Commit | O quê |
| --- | --- |
| `e33aa738` | kernels HIP: produtos com slots na memória da placa para linhas maiores do que meio warp |
| `e959480d` | páginas dos especialistas que a CPU lê ficam na RAM |
| `0bdd9f94` | desistência da placa nunca passa em silêncio; routing no host durante a leitura das páginas; gravações separadas |
| `db07852e` | `STRATA_POLY_WARM=1` |
| `0d2dbb69` | servidor: `preserve_thinking` chega ao template |
| `4021967b` | menu de modelos no `START-POLY-HIP.bat`, `poly_profile.py --list/--base`, fornecedor OpenCode do Qwen3.6 |
| `500278a9` | `STRATA_POLY_MMQ=0` |
| `73df2989` | `poly_api_check.py`: raciocínio em streaming, `preserve_thinking`, título |
| `4d77fa82` | menu de contexto com as velocidades do modelo escolhido |
| (este) | este relatório, `docs/AMD_HIP.md`, resultados, `tools/hip/gguf_head.py` |

| Ficheiro | Função / parte | Alteração | Só HIP? |
| --- | --- | --- | --- |
| `src/kernels/cuda/native_mmvq.cu` | `native_mmvq_slots_dev_kernel<F, ROWS>`, `launch_slots_dev()`, `native_mmvq_slots_dev()` | Produtos de especialistas com os slots na memória da placa para linhas maiores do que meio warp: o corpo de `native_mmvq_slots_kernel` linha a linha, lançado onde `launch_slots()` lança o gémeo; K-quants e Q4_0, Q5_0, Q8_0, IQ4_NL. Sem isto, o routing na placa ficava desligado com especialistas Q4_K/Q5_K/Q6_K | sim (`__HIPCC__`) |
| `src/poly/kernel_bench.cu` | teste "slots" | Compara também os slots na memória da placa (e um slot sem matriz, que dá zeros) com um lançamento por matriz, a 256, 512 e 2 048 entradas | ferramenta HIP |
| `src/poly/engine.cpp` | `expert_pages()`, `drop_pages()`, `map_pages()`, `fill_expert_cache()`, `follow_loop()`, `load()` | Páginas do ficheiro: um especialista copiado para a VRAM sai do working set (`VirtualUnlock`); `map_pages()` lê só os especialistas fora da cache, pedidos em blocos grandes (`PrefetchVirtualMemory`); o enchimento da cache pede os especialistas de uma camada enquanto copia a anterior. `STRATA_POLY_PAGES=0` volta ao de antes | sim (`STRATA_USE_HIP && _WIN32`) |
| `src/poly/engine.cpp`, `engine.hpp` | `card_host()`, `run()`, `reset()`, `Layer::card_heads`, `warming_` | Routing na placa: uma desistência da placa nunca passa em silêncio (verificada no fim de cada token; erro e routing no host daí em diante); routing no host enquanto as páginas são lidas depois da carga; gravações das cabeças separadas por forma de routing; `reset()` limpa a flag | sim |
| `src/poly/engine.cpp` | `load()` | `STRATA_POLY_WARM=1`: a carga espera pelas páginas (benchmarks, verificação do routing na placa) | sim |
| `src/poly/engine.cpp` | `alloc()` | `STRATA_POLY_MMQ=0`: o prefill sem MMQ (verificação de precisão) | sim |
| `src/poly/kernels.cu` | `wait_add_kernel` | Desiste logo quando uma camada anterior do token já desistiu | sim (kernel só HIP) |
| `serve/frontend.py` | `openai_to_messages()` | `chat_template_kwargs.preserve_thinking` (booleano) chega ao template | não: comum, opt-in por pedido |
| `serve/test_server.py` | `test_preserve_thinking_reaches_the_template` | teste novo | — |
| `tools/hip/poly_api_check.py` | `c_thinking_stream`, `c_preserve_thinking`, `c_title` | três verificações novas | ferramenta |
| `tools/hip/poly_profile.py` | `--list`, `--base`, porta herdada | vários modelos configurados | ferramenta |
| `START-POLY-HIP.bat` | menu de modelo | escolhe entre os modelos configurados; `START-POLY-HIP.bat qwen3.6 128k`; velocidades do modelo escolhido no menu de contexto | Windows |
| `tools/hip/gguf_head.py` | novo | lê os metadados e a tabela de tensores de um GGUF local ou por URL (pedidos parciais), sem os dados | ferramenta |
| `tools/hip/opencode-polystrata-hip.json` | fornecedor `polystrata-hip` | os 4 perfis do Qwen3.5 e os 4 do Qwen3.6 na porta 8090, e o modelo `current` (o carregado, sem troca) para o agente `title` | ferramenta |
| `serve/server.py` | `Service.switch_for()` | `"switch_on_request": true` na configuração: um pedido que nomeia outro modelo instalado carrega-o primeiro | não: comum, opt-in |
| `serve/test_models.py` | dois testes novos | a troca por pedido, direta e pela API | — |
| `tools/hip/poly_config.py` | `switch_on_request` | escrito por omissão (`--no-switch-on-request` tira-o) | ferramenta |

**Não foi preciso alterar** `src/poly/qwen35.cpp` / `qwen35.hpp`: a classe já lia a forma do ficheiro, o bloco MTP
(`block_attend`, `block_rows`, `check_draft_block`) e os tensores F32/BF16. O carregamento, a atenção, a delta net,
o routing, o prefill (MMQ do llama.cpp para K-quants) e os especialistas na CPU (traits do ggml) servem o Qwen3.6
tal como estão.

**Caminho CUDA:** tudo o que muda no motor está sob `STRATA_USE_HIP` (ou `__HIPCC__`, ou `_WIN32` dentro disso).
O build CUDA não foi compilado (este PC não tem toolchain CUDA). A alteração comum é a do servidor, que sem
`preserve_thinking` no pedido não muda nada.

## 4. Funcionalidades implementadas e estado

| Funcionalidade | Qwen3.6 em HIP | Como foi verificado |
| --- | --- | --- |
| Metadados GGUF, tensores, F32/BF16, bloco MTP | ✅ sem alterações ao código da arquitetura | carga; paridade |
| Atenção, delta net, MoE, prefill (MMQ), decode | ✅ | paridade com o llama.cpp (secção 5) |
| Especialistas K-quant na GPU e na CPU | ✅ | `poly_expert_parity` contra FP64; paridade |
| Routing na placa (fase 2) | ✅ depois da correção dos slots (`e33aa738`) | 10 240 routings, 0 escolhas diferentes do host |
| Cache de especialistas, trocas, CPU/GPU em paralelo | ✅ | benchmark; 1 500–2 300 trocas por resposta |
| Cópias por kernel, graphs e fallbacks | ✅ | todas as medições com graphs; fallbacks por variável |
| Speculative decoding | ✅ bloco MTP (janelas de 8, 3 palpites) e palpites por repetição | paridade (só ids): os mesmos tokens que `--window 1` |
| Cache de conversas e `--turn-token` | ✅ (`poly_config.py` põe-no) | sessão de 40 turnos: cada turno reutiliza o anterior |
| `preserve_thinking` (novo no template do 3.6) | ✅ opt-in por pedido | `poly_api_check.py preserve-thinking` |
| Orçamento de VRAM do Windows | ✅ | relatório de carga: 1 005 MiB de reserva em todas as cargas |
| Páginas do ficheiro na RAM | ✅ novo (`e959480d`) | RAM disponível depois da carga 4–6 → ~15 GB |
| KV na VRAM e KV streaming | ✅ (também o bloco MTP com a KV dividida) | notas a 48K–261K |
| API OpenAI/Anthropic, OpenCode/OpenChamber | ✅ | secção 8 |
| Imagens (`mmproj`) | não testado | o codificador não foi descarregado |

Limitação específica com fallback: o prefill dos especialistas Q4_K/Q5_K usa o MMQ do llama.cpp, menos preciso
neste formato na RDNA4. `STRATA_POLY_MMQ=0` dá o caminho preciso, mas é ~8× mais lento no prefill (secção 5.3).

## 5. Resultados dos testes de paridade

### 5.1 Método

`tools/hip/poly_llama_parity.py`, o método de `docs/BACKENDS.md` e das fases 1–3:
- os prompts são postos no template de chat e tokenizados pelo `llama-server`, por isso os dois motores leem os
  mesmos ids;
- greedy, 64 tokens por prompt; em cada passo, as 5 probabilidades mais altas da referência procuradas nas do
  motor;
- 5 prompts: curto (22 tokens), código (870), documentos de 3 069, 11 481 e 35 575 tokens;
- critério: os mesmos tokens, ou a primeira diferença num quase-empate (as duas mais prováveis a menos de 0,1), e
  diferença máxima ≤ 0,15.

Referências do llama.cpp no mesmo GGUF (que ignora o bloco MTP):
- **A:** HIP b11160 (Unsloth), `--cpu-moe`;
- **B:** HIP b11160, `--n-cpu-moe 22`;
- **C:** Vulkan b11511 (LM Studio), `--n-cpu-moe 22`.

### 5.2 Resultados

| Comparação | Tokens | Diferença máxima | Média por prompt | Resultado |
| --- | --- | ---: | ---: | --- |
| llama.cpp B contra A | 64/64 em 4; `doc-2.5k` separa-se no 43 num empate (0,514/0,486) | 0,033 | 0,0012–0,0030 | PASS |
| llama.cpp C (Vulkan) contra A | 64/64 em 3; dois separam-se em quase-empates | 0,086 | 0,0015–0,0079 | PASS |
| llama.cpp C contra B | 64/64 em 4; `doc-8k` separa-se num quase-empate | 0,082 | 0,0017–0,0081 | PASS |
| PolyStrata `--window 1`, routing na placa, contra A | 64/64 em 3; 2 separam-se em quase-empates | 0,146 | 0,0017–0,0125 | PASS |
| idem, contra B | 64/64 em 4; `doc-2.5k` separa-se num quase-empate | 0,158 (`doc-8k`), 0,153 (`doc-2.5k`) | 0,0017–0,0107 | **FAIL** (2 por diferença máxima) |
| PolyStrata `--window 1`, routing no host, contra A | — | ≤ 0,15 | — | PASS |
| PolyStrata, todos os especialistas na CPU, contra A | 64/64 em 4 | 0,161 | 0,0022–0,0112 | **FAIL**: `doc-2.5k` separa-se no passo 45 (0,518/0,318 contra 0,357/0,465) |
| PolyStrata por omissão (MTP), só ids, contra A | 64/64 em 4; `doc-2.5k` no 36 num quase-empate da referência | — | — | PASS |
| PolyStrata `--no-draft`, só ids, contra A | 64/64 em 2; 3 em quase-empates da referência | — | — | PASS |
| **PolyStrata `--window 1`, `STRATA_POLY_MMQ=0`, contra A** | 64/64 em 4; `doc-2.5k` no 43 num empate (0,5006/0,4994) | 0,085 | 0,0010–0,0075 | PASS |
| **idem, contra B** | **64/64 nos 5** | 0,099 | 0,0022–0,0080 | PASS |
| Routing na placa contra o do host (`STRATA_POLY_CARD_ROUTE_CHECK=1`, 256 tokens) | 0 escolhas diferentes em 10 240 | pesos ≤ 8,9e-8 | — | — |

Ficheiros: `parity-*.json`, `refs-*.json`, `logs/parity*.log`.

### 5.3 Onde está a diferença: o MMQ Q4_K/Q5_K no prefill

**Isolamento.** Os desvios concentram-se em passos sensíveis dos prompts de documento (passo 29 do `doc-2.5k`,
passo 59 do `doc-8k`), sempre os mesmos, em todas as configurações e contra todas as referências. Nesses passos as
referências concordam entre si. Probabilidade do token mais provável no passo 29 do `doc-2.5k`:

| | Probabilidade |
| --- | ---: |
| llama.cpp A / B / C | 0,927 / 0,931 / 0,870 |
| PolyStrata, prefill em lotes de 1 024 (omissão) | 0,751 |
| PolyStrata, lotes de 256 | 0,793 |
| PolyStrata, lotes de 16 (sem os caminhos de lote) | 0,880 |
| PolyStrata, `STRATA_POLY_MMQ=0` | 0,867 |
| PolyStrata, todos os especialistas na CPU (sem MMQ) | 0,904 |

**Kernels contra FP64** (`poly_expert_parity` no GGUF real, erro relativo):

| Formato e forma | Um token (MMVQ do PolyStrata) | Lote de 40 tokens (MMQ do llama.cpp) |
| --- | ---: | ---: |
| Q8_0 (vários) | 0,0113–0,0120 | 0,0113–0,0120 |
| Q6_K 512 → 2 048 (down, 3 camadas) | 0,0113 | 0,0113 |
| **Q4_K 2 048 → 512 (gate/up)** | 0,0128 | **0,0233** |
| **Q5_K 512 → 2 048 (down)** | 0,0108 | **0,0355** |
| Especialista inteiro, CPU / GPU | 0,028–0,048 / 0,012–0,019 | — |

**Causa no código.** No caminho WMMA (RDNA3/4) do MMQ do llama.cpp, `load_tiles_q4_K`/`q5_K` guarda
`dm * (sc, m)` como um `half2` por sub-bloco (`mmq-load-tiles.cuh`). O produto da escala do bloco pela do
sub-bloco é arredondado a FP16. O termo do mínimo (`dmin·m·Σy`) quase cancela o outro e amplifica esse
arredondamento. O caminho dp4a e o MMVQ guardam as escalas em separado e multiplicam em float. O Q6_K, o Q8_0 e os
i-quants do Qwen3.5 não têm termo de mínimo, e por isso não são afetados. A revisão b11160 tem o mesmo código.
Neste modelo, todas as 40 camadas fazem o prefill dos especialistas por este caminho; na referência B só 18.

**O que não está explicado:** a falha só-CPU no passo 45 do `doc-2.5k`, que não usa MMQ. Sem MMQ mas com os
especialistas na GPU, esse prompt passa. Pode ser só a sensibilidade daquele prompt, que já se separa em
quase-empates nos passos 36, 43 e 53 conforme a configuração, também entre referências do llama.cpp. Não foi
investigado mais.

**Decisão.**
- A omissão continua a usar o MMQ, porque sem ele o prefill cai de ~1 200 para ~145 tok/s.
- `STRATA_POLY_MMQ=0` fica como modo preciso, opt-in.
- A tolerância do teste não foi alterada.
- Tudo o que se separa da referência fá-lo em quase-empates; as falhas são da diferença máxima (0,153–0,161,
  limite 0,15) e a de um prompt só-CPU.
- A correção (guardar as escalas em separado no MMQ, ou um kernel próprio para K-quants em lote) está nas
  próximas otimizações.

### 5.4 Qwen3.5 com o binário novo

Contra as referências guardadas na fase 2, `--window 1`: 64/64 nos 4 prompts contra A (máximo 0,045–0,090) e
contra B (0,029–0,081). Fase 2: 0,079 e 0,042. É a mesma variação entre execuções que a fase 3 mediu (secção 8.2):
a cache de especialistas muda de corrida para corrida.

## 6. Benchmarks comparativos

### 6.1 Qwen3.5 Q3_K_M contra Qwen3.6 UD-Q4_K_M

Método:
- `tools/hip/bench_compare.py`, os prompts da fase 2 (dois conjuntos de ~2K, 8K, 31K e 45K tokens; as mesmas ids
  servem aos dois modelos, porque para uma mensagem de utilizador os templates dão os mesmos tokens);
- contexto 49 152, 6 threads, o perfil de especialistas de cada modelo, 128 tokens por resposta, greedy;
- 3 corridas de cada configuração, alternadas (Qwen3.5, Qwen3.6, Qwen3.5 `--window 1`, Qwen3.6 `--window 1`).

Mediana, pedidos consecutivos no mesmo processo (conjunto B), prefill / decode tok/s:

| Prompt | Qwen3.5 omissão | Qwen3.6 omissão (MTP) | Qwen3.5 `--window 1` | Qwen3.6 `--window 1` |
| --- | --- | --- | --- | --- |
| B-2k (2 068) | 1 213 / 80,6 | 696 / 65,2 | 1 258 / 80,9 | 724 / 64,6 |
| B-8k (8 235) | 1 502 / 77,0 | 1 210 / 68,0 | 1 523 / 76,8 | 1 231 / 66,4 |
| B-31k (31 296) | 1 367 / 62,0 | 1 273 / 59,1 | 1 368 / 62,1 | 1 285 / 55,6 |
| B-45k (45 355) | 1 244 / 56,3 | 1 190 / 53,4 | 1 249 / 55,7 | 1 196 / 50,4 |

| | Qwen3.5 omissão | Qwen3.6 omissão |
| --- | --- | --- |
| TTFT B-2k / B-8k / B-31k / B-45k (s) | 1,71 / 5,49 / 22,9 / 36,5 | 2,97 / 6,81 / 24,6 / 38,1 |
| Primeiro pedido do processo (A-2k): prefill / decode | 272 / 70,2 | 161 / 48,3 |
| Carga (até READY) | 34,6 s | 31,6 s |
| Especialistas na VRAM depois da carga | 6 570 de 10 240 (8,4 GiB) | 4 503 de 10 496 (8,0 GiB) |
| Especialistas usados vindos da VRAM (pedidos B) | — | 92–96 % |
| Palpites do MTP certos | — | ~85 % (71 de 86 a 2K) |

Notas:
- **Carga:** ~35 s nos dois porque os dois ficheiros (39 GB) não cabem juntos na cache de ficheiros quando se
  alternam. Com o mesmo modelo de seguida, o Qwen3.5 carrega em ~21 s (teste de regressão, secção 9).
- **Prefill de 2K do Qwen3.6:** ~700 tok/s; num lote curto, os especialistas fora da VRAM passam pelo staging, e os
  do Q4 são maiores.
- **MTP:** +1 a +7 % sobre `--window 1`. Cada janela produz ~2,3 tokens, mas encaminha no host (sem o routing na
  placa) e tem os especialistas da CPU de vários tokens.
- **Sessões prolongadas:** pela API, uma sessão de 40 turnos até 14 000 tokens com o tempo até ao 1.º token estável,
  e uma resposta de 2 326 tokens a 55,5 tok/s.

### 6.2 Contextos longos

`tools/hip/kv_needles.py` (notas a 5, 50 e 95 % de documentos do repositório, greedy, sem pensamento, um resumo de
128 tokens para o decode), modo por omissão (MTP ligado):

| Configuração | Prompt | Notas | Prefill (1.º pedido) | Decode no resumo | Especialistas da VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6, contexto 49 152 | 8 175 / 32 751 / 44 983 | 18/18 | 689 / 1 192 / 1 190 | 38,0 / 47,0 / 46,0 | 85–94 % |
| Qwen3.6, 131 072, KV na VRAM | 65 519 / 130 883 | 12/12 | 1 037 / 768 | 24,8 / 24,2 | 76–84 % |
| Qwen3.6, 131 072, streaming a partir de 49 152 | 65 519 / 130 883 | 12/12 | 1 005 / 712 | 16,9 / 9,1 | 86–88 % |
| idem, `--no-draft` | 65 519 / 130 883 | 12/12 | 994 / 713 | 15,8 / 8,0 | 86–91 % |
| **Qwen3.5**, 131 072, KV na VRAM (mesmo dia) | 65 519 / 130 883 | 12/12 | 1 109 / 784 | 40,6 / 30,1 | 89–94 % |
| Qwen3.6, 262 144, KV na VRAM | 260 983 | 6/6 | 497 (8,8 min) | 13,8 | 57 % |
| Qwen3.6, 262 144, streaming a partir de 49 152 | 260 983 | 6/6 | 460 (9,4 min) | 3,7 | 85 % |

Comparação com a fase 3 (Qwen3.5): 262K na VRAM 477 tok/s de prefill e 16,7 de decode; streaming 2,8.

- Não houve descarte de histórico: a atenção lê sempre todas as posições.
- As notas do início, do meio e do fim foram encontradas também perguntando pelas três de uma vez.
- O 1.º resumo a 8K (38 tok/s) ainda apanhou a leitura das páginas a seguir à carga.
- Não foram feitas extensões de contexto (sem YaRN): 262 144 é o contexto nativo.

## 7. Consumo de memória

### 7.1 VRAM (relatório de carga)

| MiB | Qwen3.5, 48K | Qwen3.6, 48K | Qwen3.6, 128K | Qwen3.6, 128K streaming | Qwen3.6, 262K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pesos densos | 1 910 | 1 961 | 1 961 | 1 961 | 1 961 |
| Cache de especialistas | 8 559 | 8 204 | 6 442 | 7 988 | 3 628 |
| KV na VRAM | 960 | 1 056 | 2 816 | 1 056 | 5 632 |
| Staging da KV | 0 | 0 | 0 | 202 | 0 |
| Estado recorrente | 251 | 251 | 251 | 251 | 251 |
| Scratch, gravações, resto | 809 | 1 017 | 1 018 | 1 018 | 1 018 |
| Livre (reserva) | 1 006 | 1 005 | 1 006 | 1 006 | 1 006 |
| KV na RAM fixada | 0 | 0 | 0 | 1 760 | 0 |
| Outra RAM fixada | 1 078 | 1 088 | 1 088 | 1 088 | 1 088 |

O Qwen3.6 tem **11 camadas de atenção**, contando a do bloco MTP: 22 KiB de KV por posição contra 20 KiB, 5,5 GiB a
262 144. Com streaming a 262K ficam 4,47 GiB fixados na RAM.

### 7.2 RAM

- **RAM física = working set do processo.** São ~3–4 GB do motor (1,1 GB fixados, scratch, checkpoints), mais as
  páginas do GGUF dos especialistas fora da VRAM. No Qwen3.6 UD-Q4_K_M são ~10 GiB de especialistas fora da VRAM,
  o que dá um working set de 16–19 GB durante o uso (17,7 GB depois de duas sessões do OpenCode). Com os ~12 GB dos
  outros programas, a máquina fica com ~1,6 GB disponíveis. O Windows pode libertar as páginas do ficheiro, mas não
  há folga.
- **"RAM privada" (~10–12 GB, a mesma do Qwen3.5 na fase 2) não é RAM ocupada.** É compromisso (commit charge) que
  acompanha as alocações na VRAM. Medido no Qwen3.6 logo depois da carga:

  | Cache de especialistas | Privada | Working set |
  | ---: | ---: | ---: |
  | 8 341 MiB | 10,3 GiB | 3,9 GiB |
  | 2 777 MiB | 2,1 GiB | 3,8 GiB |
  | 0 | 1,0 GiB | 2,8 GiB |

  Conta para o limite de commit (65 GB aqui; usado 29–35 GB), não para a RAM.
- **Gestão das páginas** (`e959480d`): antes, o motor mantinha no working set as páginas dos especialistas copiados
  para a VRAM e lia outra vez todos os especialistas. Medido no Qwen3.5:

  | | Antes | Depois |
  | --- | ---: | ---: |
  | RAM disponível depois da carga | 4,1–6,3 GB | 14,7–15,3 GB |
  | Working set do processo (benchmark) | 18,5 GB | 10,6 GB |

  Com os dois ficheiros no disco e a RAM partilhada com os outros programas, a CPU lia especialistas do disco:
  26,7 tok/s antes, 64–70 depois.
- **Ficheiro de paginação:** estável em todas as corridas (variação de −230 a +31 MiB), salvo no 262K com streaming
  (+927 MiB, com 4,5 GiB fixados).

## 8. Integração OpenChamber/OpenCode

### 8.1 Configuração

- `tools/hip/opencode-polystrata-hip.json`: **um só fornecedor**, `polystrata-hip` (`http://127.0.0.1:8090/v1`),
  com:
  - os 4 perfis do Qwen3.5 e os 4 do Qwen3.6 (48K, 128K, 262K, 262K stream), cada um com as variantes
    `none`/`low`/`medium`;
  - o modelo `current` (`modelID` `polystrata`): um nome que nenhum modelo instalado tem, respondido por quem
    estiver carregado, para o agente `title` (variante `none`).
- **Uma só porta para os dois modelos:** o servidor só tem um modelo na memória de cada vez (os dois não cabem
  juntos na VRAM). Sem mais nada, responderia com o carregado seja qual for o nome do pedido, ou seja, uma resposta
  do Qwen3.5 a quem escolheu o Qwen3.6, sem aviso. Por isso as configurações do PolyStrata HIP têm
  `"switch_on_request": true` (secção 8.4). A primeira versão desta fase usava portas diferentes (8091 para o
  Qwen3.6); passou a uma só porta para simplificar a configuração do OpenCode.
- O agente `title` não pode nomear um dos modelos: numa sessão do outro, cada título trocava o modelo (~40 s) e
  voltava. Daí o modelo `current`.
- Para os testes, o ficheiro foi copiado para `opencode.json` na raiz do repositório (configuração de projeto,
  excluída do git). A configuração global do OpenCode não foi alterada, e nenhum modelo foi removido.

### 8.2 API (`tools/hip/poly_api_check.py`, servidor Qwen3.6 48K; medido quando ainda estava na porta 8091)

**15 de 15**:
- modelos; system prompt;
- streaming (1.º token a 0,36 s); multi-turno;
- ferramentas com ciclo completo, com e sem streaming; ferramentas Anthropic;
- resposta longa (2 326 tokens a 55,5 tok/s);
- cancelamento (pedido seguinte em 0,81 s); Anthropic com e sem streaming;
- raciocínio; **raciocínio em streaming** (`reasoning_content` a partir de 0,58 s, a resposta depois, sem tags);
- **`preserve_thinking`** (1 139 tokens no 2.º turno, 1 116 reutilizados; sem a opção, 59);
- **título** (0,97 s, 7 tokens);
- sessão de 40 turnos (cada turno reutiliza o anterior, a pergunta sobre o 1.º turno é respondida).

O servidor escuta só em `127.0.0.1`.

A primeira versão da verificação do `preserve_thinking` falhou por um defeito do teste, não do motor: pedia o
"número secreto" escolhido só no raciocínio, mas o modelo pondera vários sem se fixar. A verificação passou a medir
o mecanismo.

### 8.3 OpenCode (o `opencode.exe` v2.0.18 que vem com o OpenChamber, `run --standalone`)

| Passo | Resultado |
| --- | --- |
| Ligação e streaming ("Reply with exactly PONG") | ok, 29 s; o prompt do agente (15 328 tokens) lido em 12,4 s |
| Agente com ferramenta: ler `tools/hip/poly_profile.py` e responder | ok: resposta certa, 25 s |
| Continuação da sessão (`--continue`) | ok: resposta certa em 12 s; 15 340 de 21 549 tokens reutilizados |
| Títulos | gerados em 1,8–2,0 s, 3–7 tokens, sem raciocínio |

- **Títulos:** com o agente `title` a apontar para o Qwen3.5 (servidor parado), os pedidos de título chegaram ao
  servidor do Qwen3.6, ou seja, o OpenCode usou o modelo da sessão. Um agente `title` sem `model` (só
  `"variant": "none"`) bloqueou o OpenCode antes de qualquer pedido; ficou a configuração que funciona.
- **Não testado:** a interface gráfica do OpenChamber (usei o executável do OpenCode que ela traz, com a mesma
  configuração) e sessões de agente reais com dezenas de turnos e ferramentas.

### 8.4 Os dois modelos na mesma porta (`switch_on_request`)

Com `"switch_on_request": true` na configuração do servidor (que o `poly_config.py` passa a escrever), um pedido
(`/v1/chat/completions`, `/v1/messages`, `/v1/responses`) cujo `model` é exatamente o nome de outro modelo instalado
(`GET /api/models`) carrega esse modelo antes de responder, como `POST /api/models/load`:
- qualquer outro nome (o do modelo carregado, um alias, um nome desconhecido ou parcial) é respondido pelo modelo
  carregado, como antes;
- um pedido enquanto outro corre recebe 409 (`model_busy`);
- um modelo que não arranca dá 503, com o anterior carregado de novo.

Medido no servidor real (Qwen3.5 48K a arrancar, porta 8090):

| Pedido com `model` | Respondido por | Tempo |
| --- | --- | ---: |
| `qwen3.5-35b-a3b-q3_k_m-hip` | Qwen3.5 | 6,7 s |
| `polystrata` (o do título) | Qwen3.5 (sem troca) | 0,2 s |
| `qwen3.6-35b-a3b-ud-q4_k_m-hip` | Qwen3.6, depois de trocar | 54,4 s |
| `polystrata`, `default` | Qwen3.6 (sem troca) | 0,4 s |
| `qwen3.5-35b-a3b-q3_k_m-hip` | Qwen3.5, depois de trocar | 50,7 s |

- **Tempo de troca:** ~40–55 s. Os dois ficheiros não cabem juntos na cache de ficheiros, por isso o modelo que
  entra é lido do disco.
- **`START-POLY-HIP.bat qwen3.6`** com o servidor do Qwen3.5 a correr troca-o (41 s), em vez de recusar.
- **OpenCode** (configuração nova, o Qwen3.6 carregado, uma sessão pedida ao perfil do Qwen3.5): uma só troca, o
  título (562 tokens, 3,3 s) sem troca, a resposta certa ao fim de 71 s no total.
- **Testes:** `serve/test_models.py` (direto e pela API) e `serve/test_server.py` (276) passam.

## 9. Problemas encontrados e resolvidos, e testes de regressão

### 9.1 Problemas

| Problema | Como apareceu | Causa | Correção |
| --- | --- | --- | --- |
| Routing na placa desligado no modelo inteiro (−18 a −28 % no decode, fase 2) | `STRATA_POLY_CARD_ROUTE` ligava 40/40 camadas no Qwen3.5 e não ligava com os especialistas Q4_K/Q5_K/Q6_K | `native_mmvq_slots_dev()` só tinha o kernel de meio warp (linhas de ≤ 16 threads: o down IQ4_XS de 512) | `e33aa738`: o kernel geral com os slots na memória da placa; 30/30 formas com os mesmos bits; 0 escolhas diferentes em 10 240 routings |
| Qwen3.5 a 26,7 tok/s (era 75) e Qwen3.6 parado por timeout, depois de descarregar o Qwen3.6 | primeira corrida | As páginas dos especialistas já copiados para a VRAM ficavam no working set, e `map_pages()` lia todos os especialistas outra vez; com 22,7 GB de ficheiro e ~12 GB de outros programas, o Windows despejava as páginas da CPU | `e959480d`: `VirtualUnlock` depois de copiar para a VRAM; `map_pages()` só fora da cache, com `PrefetchVirtualMemory`; leitura antecipada por camada no enchimento. Fallback `STRATA_POLY_PAGES=0` |
| **Token errado sem erro** quando a placa desistia de esperar pelos especialistas da CPU | Qwen3.6, primeiro token depois da carga (7 s a ler o disco) | O host só via a flag de desistência enquanto esperava por uma camada; se estivesse a calcular os especialistas, o token passava sem a soma deles. Já existia; o Qwen3.5 nunca o disparara. A flag também nunca voltava a zero | `0bdd9f94`: a flag é vista no fim de cada token (erro, routing no host daí em diante); `reset()` limpa-a; as esperas seguintes desistem logo |
| Primeiro token depois da carga com especialistas a vir do disco | idem | A leitura das páginas corre em fundo depois da carga | Routing no host até ela acabar (sem limite de tempo); `STRATA_POLY_WARM=1` espera por ela na carga |
| "The card did not route a token in 30 s" | depois da correção anterior | Uma cabeça de camada gravada (graph) com o routing do host era repetida num token com o routing na placa (e vice-versa) | `0bdd9f94`: gravações separadas por forma de routing (`Layer::card_heads`) |
| Primeiro token da carga lento com a leitura em fundo | medição | Os pedidos grandes da leitura antecipada enchiam a fila do disco | Testada a prioridade de I/O de fundo: 1.º token 0,4–0,9 s em vez de 7 s, mas decode 20–26 em vez de 47 tok/s. **Revertida**, documentado no código |
| Diferenças de probabilidade de 0,15 nos prompts de documento | paridade | MMQ Q4_K/Q5_K do llama.cpp na RDNA4 (secção 5.3) | `500278a9`: `STRATA_POLY_MMQ=0` (opt-in); a correção fica para depois |
| Template do Qwen3.6 com `preserve_thinking` sem efeito | leitura do template | O servidor só passava `enable_thinking`/`reasoning_effort` | `0d2dbb69`: passa `preserve_thinking` (booleano) |
| Dois modelos, uma configuração por omissão | `poly_profile.py` | Escolhia a primeira configuração por ordem alfabética | `4021967b`: `--list`, `--base`, porta herdada, menu no `START-POLY-HIP.bat` |
| Agente `title` do OpenCode sem `model` | teste | Bloqueia o OpenCode v2.0.18 | Não usado; ficou a configuração anterior |
| `bench_compare.py` sem `psutil` | primeira tentativa do benchmark | O Python do sistema não o tem | Corrido com o `.venv` |
| O `build_windows.bat` devolve 0 com erros de compilação | uma compilação | O `.bat` não propaga o erro do ninja | Verificado sempre o `error:` na saída (secção 11) |

### 9.2 Testes de regressão

| Teste | Resultado |
| --- | --- |
| Compilação HIP (`build_windows.bat poly` e `poly tests`) | sem erros |
| `poly_kernel_bench` (bit a bit) | slots por valor: 30/30; slots na memória da placa: 30/30 (10 formatos × 256, 512, 2 048 entradas) |
| `poly_expert_parity` no GGUF do Qwen3.6 | todas as matrizes < 0,05 contra FP64 (secção 5.3) |
| ctest, 103 testes | 91 passam, 9 saltados, 3 falham: os mesmos 3 das fases 2 e 3 (`expert_cache_segmented_test`, `hip_handoff`, `ple_parity`), fora do código alterado (`ctest.txt`) |
| `serve.test_server` (testes novos e vizinhos) | ok |
| `tools/test_setup_*.py` (30) | todos passam |
| `tools/test_polystrata_names.py` | falha por causa do `AMD_KV_STREAMING.md` (fase 3), também no commit anterior às alterações desta fase; o relatório novo não é apontado. Não alterado |
| Qwen3.5: binário de antes (`3f0e79aa`) contra o novo, 3 corridas alternadas | tabela abaixo |
| Qwen3.5: paridade com as referências da fase 2 | 64/64 nos 4 prompts, contra A e B |
| Routing na placa (Qwen3.6) contra o do host | 0 diferenças em 10 240 |
| KV streaming (128K e 262K, com o bloco MTP) | todas as notas encontradas |
| `START-POLY-HIP.bat check` com os dois modelos | escolhe a configuração e a porta certas |

Qwen3.5 Q3_K_M, antes / depois (mediana de 3, `regression/summary.md`):

| | Antes | Depois |
| --- | ---: | ---: |
| Decode B-2k / B-8k / B-31k / B-45k (tok/s) | 79,1 / 76,1 / 61,5 / 54,9 | 78,4 / 76,1 / 61,2 / 55,9 |
| Prefill B-8k / B-45k (tok/s) | 1 506 / 1 225 | 1 504 / 1 222 |
| Primeiro pedido A-2k: prefill (tok/s) | 741 | 825 |
| Carga | 21,1 s | 21,9 s |
| Working set do processo | 18,5 GB | 10,6 GB |

Sem regressão. As diferenças estão dentro da dispersão entre corridas, salvo a memória e o primeiro pedido, que
melhoram.

## 10. Limitações conhecidas

1. **Precisão do prefill com especialistas Q4_K/Q5_K** (secção 5.3). As maiores diferenças de probabilidade para o
   llama.cpp passam o limite de 0,15 em 2 de 5 prompts contra a referência B (0,153–0,158). Os tokens só se separam
   em quase-empates, mas é uma diferença real e com causa conhecida. `STRATA_POLY_MMQ=0` tira-a, com o prefill ~8×
   mais lento. Com o UD-Q3_K_M (IQ3_XXS/IQ4_XS, sem termo de mínimo) não deverá acontecer; não foi descarregado.
2. **Uma separação fora de um quase-empate com todos os especialistas na CPU** (`doc-2.5k`, passo 45), não
   explicada pela causa anterior.
3. **O MTP rende pouco** (+1 a +7 %): as janelas de palpites encaminham no host.
4. **Qwen3.6 UD-Q4_K_M é mais lento do que o Qwen3.5 Q3_K_M** neste PC: −19 % de decode a 2K, −5 % a 45K, −39 % a
   65K com a KV de 128K, −17 % a 261K (contra a fase 3). A comparação é entre quantizações diferentes; a qualidade relativa não foi
   medida.
5. **RAM no limite:** com o Qwen3.6 UD-Q4_K_M carregado e ~12 GB de outros programas, ficam ~1,6 GB disponíveis.
   Programas grandes abertos ao mesmo tempo empurram páginas do modelo para fora, e o decode baixa.
6. **O primeiro pedido depois da carga é mais lento** (161 tok/s de prefill a 2K): as páginas dos especialistas
   ainda estão a ser lidas, e o routing fica no host entretanto.
7. **262K com streaming:** 3,7 tok/s de decode, e o ficheiro de paginação cresceu 0,9 GB nessa corrida.
8. **Não testado:**
   - imagens (sem o `mmproj`);
   - a interface gráfica do OpenChamber;
   - o UD-Q3_K_M e as outras quantizações;
   - o timeout de 1 s da espera na placa provocado de propósito (o caminho de erro só foi exercitado
     acidentalmente, antes da correção);
   - uso de horas seguidas.
9. **CUDA e Linux** (as limitações conhecidas mantêm-se):
   - o build CUDA não foi compilado (não há toolchain neste PC); o que muda no motor está sob `STRATA_USE_HIP`, e a
     alteração comum ao servidor é opt-in por pedido;
   - Linux com HIP não foi testado; a gestão de páginas e o routing na placa estão desligados por omissão fora do
     Windows.
10. **ROCm por nightlies do TheRock:** as medições são desta versão (10.2.0a20260930).

## 11. Instruções para executar o Qwen3.6

**O mais simples:** `START-POLY-HIP.bat` (duplo clique). Com os dois modelos configurados mostra primeiro um menu
de modelo (Qwen3.5 / Qwen3.6) e depois o de contexto (48K, 128K, 262K, 262K com streaming); as últimas escolhas
ficam como omissão. Os dois modelos usam `http://127.0.0.1:8090/`; com o servidor já a correr, o script pede-lhe a
troca de modelo. Sem menus:

```bat
START-POLY-HIP.bat qwen3.6 48k
START-POLY-HIP.bat qwen3.6 128k
START-POLY-HIP.bat chat qwen3.6
START-POLY-HIP.bat check qwen3.6 262k
```

### Compilar

```bat
set STRATA_GGML_DIR=<repo>\third_party\llama.cpp-3cf0325
set STRATA_HIP_ARCHS=gfx1200
tools\hip\build_windows.bat poly
tools\hip\build_windows.bat poly tests
```

O `.bat` devolve 0 mesmo quando um ficheiro não compila: confirmar no fim da saída que não há `error:` e que a
data do `build-hip-poly-win\strata-poly.exe` mudou.

### Configurar (uma vez; já feito neste PC)

```bat
.venv\Scripts\python tools\hip\poly_config.py --model C:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --name qwen3.6-35b-a3b-ud-q4_k_m-hip --context 49152 --threads 6 --data C:\models\qwen36-pack
.venv\Scripts\python tools\hip\poly_profile.py 128k --base qwen3.6
.venv\Scripts\python tools\hip\poly_profile.py --list
```

O primeiro escreve `strata-qwen3.6-35b-a3b-ud-q4_k_m-hip.json` e o tokenizer (com o template do Qwen3.6) a partir do
GGUF. O segundo cria um perfil (`48k`, `128k`, `262k`, `262k-stream`) com a mesma porta, tokenizer e perfil de
especialistas.

### Servidor e chat

```bat
.venv\Scripts\python serve\server.py --engine strata --config strata-qwen3.6-35b-a3b-ud-q4_k_m-hip.json --port 8090
.venv\Scripts\python polystrata.py --model qwen3.6-35b-a3b-ud-q4_k_m-hip
```

Só em `127.0.0.1`; para outros endereços, `--api-key`. Um pedido pode pedir o raciocínio dos turnos anteriores no
histórico com `"chat_template_kwargs": {"preserve_thinking": true}`.

### OpenCode / OpenChamber

Copiar `tools\hip\opencode-polystrata-hip.json` para `opencode.json` na raiz do projeto que se abre (ou juntar o
fornecedor `polystrata-hip` e o agente `title` à configuração global). No fornecedor "PolyStrata (HIP, this PC)"
escolhe-se o modelo e o contexto ("Qwen3.6-35B-A3B UD-Q4_K_M 48K", "Qwen3.5-35B-A3B Q3_K_M 128K", ...). Se o
servidor tiver outro carregado, o primeiro pedido troca-o (~40–55 s) e os seguintes são normais. Não usar dois
modelos ao mesmo tempo (por exemplo, duas sessões com modelos diferentes): cada pedido trocaria o modelo.

### O motor diretamente

```bat
tools\hip\poly_windows.bat run --model C:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --tokens 9707,11,1879 --max-new 128 --threads 6 --max-context 16384
```

Por omissão: janelas de 8 tokens com 3 palpites do bloco MTP. `--window 1`: um token por chamada (para comparar
probabilidades); `--no-draft`: só palpites por repetição.

Variáveis (todas opcionais):

| Variável | Efeito |
| --- | --- |
| `STRATA_POLY_PAGES=0` | as páginas do ficheiro como antes desta fase (todas lidas e mantidas) |
| `STRATA_POLY_WARM=1` | a carga espera pela leitura das páginas dos especialistas fora da VRAM |
| `STRATA_POLY_CARD_ROUTE=0` | routing no host (sem o routing na placa) |
| `STRATA_POLY_CARD_ROUTE_CHECK=1` | cada routing na placa refeito no host e comparado |
| `STRATA_POLY_COPY_KERNELS=0` | cópias pelo copy engine em vez de por kernel |
| `STRATA_POLY_KV_AHEAD=0` | KV streaming sem pré-carregamento |
| `STRATA_POLY_PROFILE=1` | perfil por camada (desliga o routing na placa) |

## 12. Próximas otimizações recomendadas

Por ordem de ganho esperado, com o número que o justifica:

1. **Routing na placa para as janelas de palpites (MTP).**
   - Medido: o MTP acerta ~85 % dos palpites e cada chamada produz ~2,3 tokens, mas o decode só sobe 1–7 %, porque
     as janelas encaminham no host (a espera por camada que a fase 2 tirou aos tokens sozinhos: 3,6 ms de 15,7).
   - Com o routing na placa para n > 1 tokens, o decode do Qwen3.6 deverá aproximar-se de 2× o de um token por
     chamada nas respostas previsíveis. Estimativa, não medida.
2. **Prefill K-quant preciso** (secção 5.3). Duas formas:
   - corrigir o MMQ do llama.cpp para guardar as escalas `d·sc` e `dmin·m` em float (ou as escalas inteiras à
     parte) no caminho WMMA; seria também uma correção a propor ao llama.cpp;
   - ou um kernel próprio de lote para os especialistas K-quant.
   - Objetivo: a precisão de `STRATA_POLY_MMQ=0` (máximo 0,085–0,099) com o prefill de ~1 200 tok/s.
3. **Prefill de prompts curtos do Qwen3.6:** 2K a ~700 tok/s contra ~1 200 no Qwen3.5. Os especialistas fora da VRAM
   passam pelo staging a cada lote; vale medir o perfil do lote (`stage-mib`, `stage-ahead`) com especialistas Q4.
4. **Atenção de decode a contextos longos** (da fase 3): a ~125 GB/s de 300 possíveis. No Qwen3.6, a KV de 128K/262K
   tira à cache de especialistas 1,8–4,6 GB, e o decode cai para 24 e 14 tok/s.
5. **Bloco MTP com a KV dividida:** a sua atenção lê as posições na RAM pelo PCIe diretamente (zero-copy), sem a
   cópia antecipada das outras camadas. A 128K ainda ganha ao `--no-draft` (16,9 contra 15,8 tok/s), mas a 262K
   pode pesar mais. Juntá-lo às cópias antecipadas.
6. **Memória neste PC:** o commit de ~10 GB que acompanha a VRAM não ocupa RAM, mas o working set de 16–19 GB do
   Qwen3.6 UD-Q4_K_M deixa pouca folga. Um UD-Q3_K_M (17,1 GB, os formatos do Qwen3.5) deixaria ~3,8 GB a mais de
   especialistas na VRAM e na RAM, e evitaria a limitação do MMQ. Vale descarregá-lo para comparar velocidade e
   qualidade.
7. **Integração no `setup.py`:** aceitar AMD para o `strata-poly` (a família `qwen36` já existe) e escrever
   `--threads`, a porta e o perfil de contexto.
