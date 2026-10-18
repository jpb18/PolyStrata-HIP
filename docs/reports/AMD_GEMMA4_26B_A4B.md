> Historical measurement report. Use the [current setup instructions](../../README.md). Personal paths were generalized for publication; measurements were preserved. Raw campaign links may require the local evidence archive.

# PolyStrata HIP: avaliação do Google Gemma 4 26B-A4B (não integrado)

Ramo `hip-poly`, sobre `7023db0c` e as alterações por publicar do Empero ([AMD_EMPERO_QWEN38.md](AMD_EMPERO_QWEN38.md)).
Segue a metodologia das avaliações do North Mini Code ([AMD_NORTH_MINI_CODE.md](AMD_NORTH_MINI_CODE.md)) e do
GLM-4.7-Flash ([AMD_GLM47_FLASH.md](AMD_GLM47_FLASH.md)): primeiro a investigação, e só depois, se houver uma
melhoria plausível, um teste.

**Nada no motor, no servidor, nos perfis ou no OpenCode mudou.** Nenhum modelo foi descarregado nem apagado. Só se
leram os cabeçalhos GGUF por HTTP parcial (`tools/hip/gguf_head.py`), guardados em
`bench/results/2026-10-12-hip-gemma4-26b-a4b/gguf-headers/`. Sem commit nem push.

Data: 2026-10-12. Máquina alvo: RX 9060 XT 16 GB (gfx1200), Ryzen 5 5600X, 31,9 GB de RAM, Windows 11 Pro.

Neste relatório, cada número diz de onde vem: **publicado** (cartão do modelo ou terceiros), **medido** (por este
projeto, com a máquina indicada) ou **estimado** (contas, sem medição).

---

## 1. Resumo e recomendação

- **A avaliação parou antes do teste.** O Gemma 4 26B-A4B não tem uma vantagem plausível sobre o Qwen3.6-35B-A3B
  naquilo que se quer melhorar (raciocínio, programação, ferramentas no OpenCode):
  1. **Os números da própria Google** já o põem abaixo do Qwen3.6 em todos os testes que os dois publicam com as
     mesmas condições: GPQA 82,3 contra 86,0, LiveCodeBench v6 77,1 contra 80,4, HLE 8,7 contra 21,4, AIME 26 88,3
     contra 92,7, MMLU-Pro 82,6 contra 85,2 (secção 3.1).
  2. **Programação agêntica:** a Google não publica SWE-bench nem Terminal-Bench para este modelo. A tabela da Qwen,
     que mediu os dois, dá ao Qwen3.6 vantagem em todos os testes de agente de programação, por exemplo
     Terminal-Bench 2.0 51,5 contra 34,2, e em 6 de 7 de uso de ferramentas (secção 3.2). É a tabela de um
     concorrente, mas a direção é a mesma dos números da Google.
  3. **Medições independentes:** não encontrei nenhuma, com metodologia descrita, que ponha o Gemma 4 26B-A4B à frente.
     O SWE-rebench não o lista. A Artificial Analysis ainda marca o índice dele como estimativa (17 contra 18).
  4. **Velocidade neste PC, estimada:** sem ganho. Em CUDA, numa placa de 8 GB, o motor já mediu os dois com a
     mesma quantização: decode igual, prefill ~1,35× mais rápido no Gemma. Aqui, em HIP, o Gemma ficaria sem o
     routing na placa, que deu +18–28 % ao Qwen3.6 (secção 5).
- **O suporte técnico existe quase todo, e isso é a única coisa a favor.** Ao contrário do North e do GLM-4.7-Flash, o
  `strata-poly` já tem a classe `Gemma4`, feita para este modelo e estes GGUF:
  - o setup tem a família `gemma4`, com revisão e SHA-256 fixados;
  - o servidor tem o dialeto de raciocínio e ferramentas, com testes (`serve/test_gemma.py`);
  - a paridade com o llama.cpp foi medida em CUDA (64/64 tokens em 4 de 5 prompts);
  - o executável HIP deste PC (`build-hip-poly-win/strata-poly.exe`) já inclui a classe.

  **Nunca correu em HIP.** Um teste funcional não exige código novo (secção 4).
- **Recomendação: descartar** como substituto do Qwen3.6. O **Qwen3.6-35B-A3B UD-Q3_K_M continua o modelo
  principal.** O teste da secção 7 é barato (~13 GB e ~3 horas, sem tocar no motor), mas só se justifica para
  verificar o caminho `gemma4` em HIP, não para substituir o Qwen3.6.
- **Estado:** o Qwen3.6 e as suas 4 configurações, o Empero, o `START-POLY-HIP.bat` e o OpenCode ficaram como estavam.

## 2. O modelo

Fontes: [cartão do modelo](https://huggingface.co/google/gemma-4-26B-A4B-it) e os cabeçalhos dos GGUF do
[Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF), revisão `c099eb48` (a que o `setup.py` fixa).

| | Gemma 4 26B-A4B | Qwen3.6-35B-A3B (atual) |
| --- | --- | --- |
| Licença | Apache 2.0 | Apache 2.0 |
| Parâmetros | 25,2B total, **3,8B ativos** | 35B total, 3B ativos |
| Arquitetura no GGUF | `gemma4` | `qwen35moe` |
| Camadas | 30, todas com atenção e MoE | 40 (+1 MTP): 30 delta net + 10 atenção |
| Atenção | 25 camadas por janela de **1 024** (8 KV × 256, RoPE θ 10⁴) e 5 globais (camadas 5, 11, …, 29; 2 KV × 512, **o valor é a própria chave**, RoPE proporcional θ 10⁶) | 10 camadas de atenção total, 2 KV × 256 |
| Cabeças de consulta | 16 | 16 |
| FFN | um MLP denso GELU (2 112) **ao lado** dos especialistas, cada um com a sua norma | especialista partilhado SwiGLU |
| Especialistas | 128, 8 ativos, FFN 704, GELU; escala própria por especialista | 256, 8 ativos, FFN 512 |
| Router | lê o resíduo (não a entrada dos especialistas); softmax de todos, top-8, normalizado, × escala | softmax, top-8, normalizado |
| Saída | tabela de tokens (partilhada), `softcap` 30 nos logits | tabela própria |
| Vocabulário | 262 144 | 248 320 |
| Contexto | 262 144 | 262 144 |
| Adivinhar à frente | modelo de rascunho separado (`gemma4-assistant`, 0,46 GB Q8_0), da Google | bloco MTP no ficheiro |
| Raciocínio | `<\|channel>thought` … `<channel\|>`; o template só pensa com `enable_thinking` (falso por omissão); **sem níveis de esforço** | `<think>`, `enable_thinking`, níveis |
| Ferramentas | `<\|tool_call>call:NOME{chave:valor,…}<tool_call\|>`, respostas em `<\|tool_response>` | XML Qwen |
| Amostragem recomendada | `temperature 1.0, top_p 0.95, top_k 64` (também nos metadados do GGUF) | |
| Imagens | sim (codificador de ~550M) | sim |

Leitura da arquitetura:
- **Contexto longo assenta em 5 camadas.** As outras 25 só veem os últimos 1 024 tokens. As 5 globais têm 2 cabeças
  KV de 512. É menos atenção global do que as 10 camadas do Qwen3.6 (que tem ainda o estado das 30 delta net), com os
  mesmos bytes de KV por posição (secção 5).
- **Mais parâmetros ativos (3,8B contra 3B)** não se traduzem em melhores resultados publicados (secção 3).
- **Uma limitação de quantização própria:** a largura dos especialistas, 704, não é múltipla de 256. Por isso o `down`
  dos especialistas não pode usar os formatos K nem os IQ de blocos de 256, só os de blocos de 32. O Unsloth usa
  IQ4_NL (UD-Q3_K_M e UD-Q2_K_XL) e Q5_1 (UD-Q4_K_M). Os dois têm kernels no motor (secção 4.2).

## 3. Qualidade publicada

Cada fabricante usa o seu arnês e as suas condições. A comparação orienta, não mede.

### 3.1 Números da Google (cartão do Gemma 4) contra os da Qwen (cartão do Qwen3.6)

| Teste | Gemma 4 26B-A4B | Qwen3.6-35B-A3B | Diferença |
| --- | ---: | ---: | ---: |
| MMLU-Pro | 82,6 | **85,2** | −2,6 |
| GPQA Diamond | 82,3 | **86,0** | −3,7 |
| HLE (sem ferramentas) | 8,7 | **21,4** | −12,7 |
| LiveCodeBench v6 | 77,1 | **80,4** | −3,3 |
| AIME 2026 | 88,3 | **92,7** | −4,4 |
| τ²-Bench (média de 3) | 68,2 | (publica o TAU3-Bench) | não comparável |
| MRCR v2, 8 agulhas, 128K | 44,1 | não publicado | |
| Codeforces ELO | 1 718 | não publicado | |

- Nos 5 testes que os dois publicam, o Qwen3.6 está à frente nos 5. O HLE é a maior diferença: o Gemma 4 26B-A4B fica
  em menos de metade do seu irmão denso de 31B (19,5).
- O MRCR (44,1 % a 128K) é o único número de contexto longo da Google. O 31B denso tem 66,4 %.

### 3.2 A tabela da Qwen, que mediu os dois

O cartão do Qwen3.6 tem uma coluna "Gemma4-26BA4B". Nos testes de conhecimento e STEM os valores são os da Google
(iguais à tabela acima); nos de agente, são medidos pela Qwen:

| Teste | Gemma 4 26B-A4B | Qwen3.6-35B-A3B |
| --- | ---: | ---: |
| SWE-bench Verified | 17,4 | **73,4** |
| SWE-bench Multilingual | 17,3 | **67,2** |
| SWE-bench Pro | 13,8 | **49,5** |
| Terminal-Bench 2.0 | 34,2 | **51,5** |
| Claw-Eval (média / pass³) | 58,8 / 28,0 | **68,7 / 50,0** |
| SkillsBench | 12,3 | **28,7** |
| NL2Repo | 11,6 | **29,4** |
| TAU3-Bench | 59,0 | **67,2** |
| VITA-Bench | **36,9** | 35,6 |
| DeepPlanning | 16,2 | **25,9** |
| Tool Decathlon | 12,0 | **26,9** |
| MCPMark | 14,2 | **37,0** |
| MCP-Atlas | 50,0 | **62,8** |

- **O SWE-bench (17,4) parece anómalo**: o Gemma 4 31B denso tem 52,0 na mesma tabela, e um modelo com 77 no
  LiveCodeBench não costuma ficar em 17. É provável que reflita o arnês da Qwen (formato de ferramentas) mais do que o
  modelo. **Não o uso como prova.**
- Mesmo sem o SWE-bench, a direção é a mesma no Terminal-Bench 2.0 (−17,3), no MCPMark, no Tool Decathlon e no
  MCP-Atlas. O Gemma só ganha no VITA-Bench, por 1,3 pontos.
- Um agregador ([BenchLM](https://benchlm.ai/compare/gemma-4-26b-a4b-vs-qwen3-5-35b-a3b)) dá ao Gemma 69,2 no
  SWE-bench Verified e 40,5 no Terminal-Bench 2.0, sem fonte primária que eu tenha encontrado. Mesmo esses valores
  ficam abaixo do Qwen3.6 (73,4 e 51,5).

### 3.3 Medições independentes

- **[Artificial Analysis](https://artificialanalysis.ai/models/comparisons/qwen3-6-35b-a3b-vs-gemma-4-26b-a4b)**, página
  lida a 2026-10-12, os dois com raciocínio:

  | | Qwen3.6-35B-A3B | Gemma 4 26B-A4B |
  | --- | ---: | ---: |
  | Intelligence Index | 18 | 17 (**estimativa**, "independent evaluation forthcoming") |
  | AA-LCR v1.1 (raciocínio em contexto longo) | 72 % | 66 % |
  | SciCode | 37 % | 40 % |
  | AA-Omniscience | −22 | −51 |
  | GDPval-AA / AA-Briefcase (tarefas de trabalho) | 892 / 713 | 565 / 315 |

  **Cuidado:** a página do próprio Gemma 4 26B-A4B diz que os testes "não estão disponíveis publicamente", e o HLE que
  a comparação lhe dá (19 %) é o do 31B denso, não o do cartão do 26B (8,7). Os valores do Gemma nesta tabela podem
  estar misturados com os do 31B. Uso só a direção: mesmo assim, o Qwen3.6 fica à frente em quase tudo.
- **[SWE-rebench](https://swe-rebench.com/)** (janela 15/05–01/07/2026): o Qwen3.6-35B-A3B resolve 24,7 % ± 0,8. O Gemma
  4 26B-A4B não está listado; o 31B aparece sem valores.
- A única vantagem relatada com consistência é a preferência humana em conversa (Arena), que não é o nosso uso.

### 3.4 Fiabilidade relatada com ferramentas

Relatos públicos, **não verificados por mim**:
- no OpenCode, ciclos de chamadas e edições falhadas em tarefas mais longas
  ([dotnetperls](https://dotnetperls.com/2026_4_6_gemma-4-vibe-coding-opencode)); noutro caso, planos em Markdown
  sem chamar nenhuma ferramenta, com o modelo a gerar chamadas certas por `curl`;
- no Ollama 0.20.0/0.20.1, chamadas perdidas pelo analisador (corrigido depois); no llama.cpp, recomenda-se um build
  recente com `--jinja`.

A maior parte aponta para analisadores e templates, não para o modelo. O analisador do PolyStrata
(`serve/frontend.py`, dialeto `gemma`) segue o do llama.cpp (`common/parsers/gemma4.cpp`) e tem testes.

## 4. Compatibilidade com o motor `strata-poly` HIP

### 4.1 O que já existe

| Peça | Estado |
| --- | --- |
| Classe do motor | `src/poly/gemma4.{hpp,cpp}` (584 linhas), registada em `Engine::open` (`src/poly/engine.cpp:588`) para `general.architecture = gemma4`. O cabeçalho diz que é para o 26B-A4B |
| Geometria do ficheiro | verificada contra o cabeçalho real: 30 camadas, padrão de janelas `[5×janela, 1×global]`, `head_count_kv` por camada `[8,…,2]`, chaves 256/512, `shared_kv_layers = 0`, `embedding_length_per_layer_input = 0` (as duas que a classe recusa se não forem 0), `ffn_gate_up_exps` fundido, `ffn_down_exps.scale`, `ffn_gate_inp.scale`, `layer_output_scale`, globais sem `attn_v` |
| Modelo de rascunho | `load_drafter()`/`draft_step()`; o ficheiro `mtp-gemma-4-26B-A4B-it-Q8_0.gguf` tem a arquitetura `gemma4-assistant` que a classe exige |
| Compilação HIP | `src/poly/gemma4.cpp` está na lista do `strata-poly` (`CMakeLists.txt:1593`) para CUDA e HIP; o executável HIP deste PC contém a classe |
| Setup | família `gemma4`, 3 tamanhos, revisão e SHA-256 fixados (`setup.py:424`) |
| Servidor | dialeto `gemma` (raciocínio por canal, chamadas `call:NOME{…}`, respostas por ordem), `<bos>` e `enable_thinking`; o servidor liga o raciocínio a não ser que o pedido o desligue (`serve/server.py:3309`); 21 testes em `serve/test_gemma.py` |
| Paridade (CUDA, RTX 5090) | UD-Q3_K_M: 64, 64, 64, 64, 43 tokens iguais ao llama.cpp, diferença máxima 0,12; no portátil de 8 GB, 64 ×5 ([docs/BACKENDS.md](../BACKENDS.md#how-each-backend-was-checked)) |
| Imagens | `sees()`, `whole()`, `mmproj-F16.gguf` |

### 4.2 Formatos dos pesos (cabeçalhos lidos, `gguf-headers/summary.txt`)

| GGUF | Ficheiro | Especialistas | Formatos dos especialistas | Densos | 1 especialista | **Especialistas lidos por token** |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| UD-Q2_K_XL | 10,55 GB | 8,13 GiB | gate_up IQ2_XS (29) / IQ3_XXS (1); down IQ4_NL | 1,68 GiB (Q5_K, Q6_K, Q8_0) | 2,17 MiB | 520 MiB |
| **UD-Q3_K_M** (o primeiro do setup) | 12,73 GB | 9,60 GiB | gate_up IQ3_XXS (29) / IQ4_XS (1); down IQ4_NL (29) / Q8_0 (1) | 2,24 GiB (Q8_0, tabela Q6_K) | 2,56 MiB | **614 MiB** |
| UD-Q4_K_M | 16,95 GB | 13,37 GiB | gate_up Q4_K; down Q5_1 (29) / Q8_0 (1) | 2,40 GiB (Q8_0) | 3,57 MiB | 856 MiB |
| Modelo de rascunho Q8_0 | 0,46 GB | | | 0,42 GiB (Q8_0) | | |
| **Qwen3.6 UD-Q3_K_M (atual)** | 17,10 GB | 13,26 GiB | IQ3_XXS / IQ4_XS / Q6_K | 2,26 GiB (Q8_0) | 1,33 MiB | **424 MiB** |

"Lidos por token" = 8 especialistas × 30 camadas (Gemma) ou 8 × 40 (Qwen3.6).

- Todos os formatos têm kernels no motor HIP: IQ3_XXS, IQ4_XS, IQ4_NL, Q8_0, Q6_K, Q5_K, Q4_K, Q5_1 e IQ2_XS estão no
  `native_mmvq` (um token), nas instâncias MMQ (lote, `CMakeLists.txt:1568`) e no `vec_dot` do ggml (CPU).
- O UD-Q3_K_M tem os densos em Q8_0, o formato com kernels de várias colunas rápidos (o que faltou ao Empero para o
  MTP compensar). Seria o ficheiro a testar.
- O erro em lote do IQ4_NL **não foi medido em HIP** (`poly_expert_parity` precisa do ficheiro).

### 4.3 O que falta, ou é diferente, em HIP

| Peça | Situação | Efeito | Trabalho |
| --- | --- | --- | --- |
| Correr em HIP | compila, nunca correu | desconhecido até correr; os kernels de atenção são genéricos na largura da cabeça (múltiplo de 4), por isso 512 é aceite | nenhum código; validação (paridade) |
| Routing na placa | `card_route_floor()` só é definido pelo `Qwen35`; o Gemma faz routing no host | o Qwen3.6 ganhou **+18 a +28 %** de decode com ele (medido, [AMD_PORT_PHASE2.md](AMD_PORT_PHASE2.md) 6.4) | o kernel `route_topk_softmax` serve quase: falta a escala 1/√2816 nos logits e a escala por especialista depois de normalizar. ~50–100 linhas em `kernels.cu` (partilhado com CUDA: compilar os dois) |
| Streaming da KV (262K) | só o `qwen35` o tem | 262K sem streaming deixa ~2,5 GiB para especialistas | não fazer; 48K e 128K cabem sem ele |
| Níveis de raciocínio | o template só tem ligado/desligado | as variantes `low`/`medium`/`high` do OpenCode dão o mesmo resultado | nenhum; documentar |
| Amostragem | o OpenCode não envia nenhuma; sem bloco `sampling`, o servidor gera por greedy | o Empero entrou em ciclos em greedy | bloco `sampling` no perfil (`1.0 / 0.95 / 64`), como no Empero |

**Esforço para um teste funcional: zero linhas de código.** Para uso diário seria preciso o routing na placa (meio
dia, com compilação CUDA e HIP e a verificação `STRATA_POLY_CARD_ROUTE_CHECK=1`) e um perfil com amostragem e
menu, como o do Empero (meio dia). Risco: baixo do lado do motor (a classe já passou paridade em CUDA); a incerteza
está no comportamento do modelo com ferramentas.

## 5. Memória e velocidade estimada

Contas para a RX 9060 XT a 48K (49 152 posições, KV em FP16), **estimadas, não medidas**:

| | Gemma 4 UD-Q3_K_M | Qwen3.6 UD-Q3_K_M |
| --- | ---: | ---: |
| KV por posição (camadas globais) | 5 × 2 × 512 × 2 × 2 B = 20 KiB | 10 × 2 × 256 × 2 × 2 B = 20 KiB |
| KV global a 48K | 0,94 GiB | 0,94 GiB (medido: 1 056 MiB com o estado) |
| Anéis das janelas (1 024 + 1 024 do lote) | 25 × 2 048 × 8 KiB = 0,39 GiB | — (estado recorrente, 251 MiB medido) |
| Densos na VRAM (a tabela de tokens é a saída e fica na placa) | ~2,24 GiB | 1,92 GiB (medido) |
| Rascunho | 0,42 GiB | 0,39 GiB (no ficheiro) |
| Especialistas na VRAM | ~7,3 GiB ≈ 2 850 de 3 840 (**~74 %**) | ~8 GiB ≈ 6 140–6 330 de 10 240 (~60 %, medido) |
| Especialistas lidos por token | 614 MiB | 424 MiB |
| A 128K | KV 2,5 + 0,4 GiB: cabe, como o perfil 128K do Qwen3.6 | 2,5 GiB |
| RAM | o ficheiro, 12,7 GB, cabe na cache de ficheiros | 17,1 GB |

Velocidade:
- **Medido em CUDA** (RTX 4070 Laptop 8 GB, [docs/BACKENDS.md](../BACKENDS.md#on-a-laptop-with-8-gb-of-vram)), mesmo
  motor e mesma quantização UD-Q3_K_M, com palpites: decode **70–72 tok/s (Gemma) contra 69–74 (Qwen3.6)**; prefill de
  31K a 1 907 contra 1 391 tok/s. Sem palpites, 50–51 contra 55–56.
- **Estimado em HIP:** a mesma relação, menos o routing na placa que só o Qwen3.6 tem. Com a perda que o Qwen3.6 tinha
  sem ele (−18 a −28 %), o decode do Gemma ficaria ~20 % abaixo do Qwen3.6 (que faz 91 / 62 tok/s a 2K / 45K) até
  se escrever esse kernel. O prefill pode ficar à frente. Só se sabe a correr.
- Para comparar: a Artificial Analysis mede 129 tok/s para o Qwen3.6 e 73 para o Gemma nas APIs (não é deste PC).

**Conclusão da secção:** cabe com folga e não há razão para esperar ganho de velocidade. No melhor caso, empata.

## 6. Decisão

O pedido diz para parar se não houver vantagem plausível. Não há:
- os números da Google e da Qwen apontam para o mesmo lado, e as diferenças em raciocínio (HLE −12,7) e uso de
  terminal (Terminal-Bench 2.0 −17,3) são grandes;
- a quantização não as desfaz: os dois correriam com especialistas IQ3_XXS e densos Q8_0;
- a velocidade não compensa: empate em CUDA, provável desvantagem em HIP sem trabalho extra.

**Não descarreguei os pesos (13,2 GB) nem corri testes.** O facto de o teste não precisar de código é a única razão
para o considerar, e não chega para justificar 13 GB e ~3 horas de máquina contra estas evidências.

## 7. Teste rápido possível (não feito)

Se se quiser um dado nosso, ou validar o caminho `gemma4` em HIP para uso futuro (por exemplo, para imagens), é o
teste mais barato de todas as avaliações até agora, porque usa o próprio PolyStrata e não o llama.cpp. Sem tocar nos
perfis de produção:

1. **Download** (13,19 GB; o C: tem 86 GB livres, nada a remover): `gemma-4-26B-A4B-it-UD-Q3_K_M.gguf` e
   `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf`, revisão `c099eb48`, com o `setup.download()` e o `verify_sha256()` do
   `setup.py` (os SHA-256 já lá estão). ~20 min.
2. **Tokenizer:** `tools/strata_tokenizer.py` para uma pasta própria.
3. **Configuração fora da raiz**, por exemplo no `bench/results/…`: o servidor só lista as `strata-*.json` da raiz
   (`serve/models.py:154`), por isso não entra no menu, na troca por pedido nem no OpenCode global. Arrancar com
   `serve/server.py --config <ficheiro> --port 8091`, `--max-context 49152`, `--draft-model`, amostragem
   `1.0 / 0.95 / 64` e uma corrida em greedy para a paridade.
4. **Paridade** com o llama.cpp HIP b11160 que já está no PC (`tools/hip/poly_llama_parity.py`, os 5 prompts, as
   referências A `--cpu-moe` e B `--n-cpu-moe`) e `poly_expert_parity` para o IQ4_NL em lote. ~45 min.
5. **Comparação com o Qwen3.6**, no mesmo dia:

   | Teste | Como | Critério |
   | --- | --- | --- |
   | Os 4 erros (`bench/results/2026-10-12-hip-empero-qwen38/codetask`) | OpenCode v2.0.18, `--standalone --auto`, 4 corridas com raciocínio e 2 sem, testes corridos à parte | 4/4 e 2/2 como o Qwen3.6, sem ciclos |
   | Contexto longo | `tools/hip/kv_needles.py`, 8K/32K/45K, cada nota sozinha e as 3 juntas | 6/6 nos três, como o Qwen3.6 |
   | Operação Atlas | **o enunciado não está no repositório nem nos dados do OpenCode deste PC**; é preciso o texto para o reproduzir. Verificar cada restrição e cada decisão sob incerteza, às cegas, contra o Qwen3.6 | melhor que o Qwen3.6 em restrições cumpridas |
   | Ferramentas com vários passos | uma tarefa de 5–10 chamadas (ler, procurar, editar, correr, corrigir) no OpenCode, 3 corridas | sem chamadas mal formadas, sem planos sem ação |
   | Prefill, decode, VRAM, estabilidade | `tools/hip/bench_compare.py` (2K/8K/31K/45K, 3 rondas alternadas) e `bench/results/2026-10-12-hip-empero-qwen38/scripts/soak.py` 20 min | registar; não é critério de troca |

   ~2 horas.

**Critério para avançar:** o Gemma teria de ganhar claramente na tarefa dos 4 erros, nas ferramentas com vários passos
e na Operação Atlas, e não perder nas agulhas. Contra os números da secção 3, não espero que ganhe.

## 8. Alternativas

- **Gemma 4 31B (denso):** melhor que o 26B-A4B em tudo (HLE 19,5, LiveCodeBench 80,0, MRCR 66,4), mas mesmo assim
  abaixo do Qwen3.6 nos agentes de programação da tabela da Qwen (Terminal-Bench 2.0 42,9, SWE-bench Verified 52,0). E
  um modelo denso de 31B lê todos os pesos por token: a ~3,5 bits são ~13 GB, que não cabem na placa ao lado da KV e
  dos ~3,4 GB que os outros programas ocupam. Não serve aqui.
- **Qwen3.6-27B (denso):** a tabela da Qwen e um agregador dão-lhe números de agente acima do 35B-A3B (Terminal-Bench
  2.0 59,3 no [BenchLM](https://benchlm.ai/md/compare/gemma-4-26b-a4b-vs-qwen3-6-27b.md), **não verificado**). Tem o
  mesmo problema de qualquer denso de ~27B nesta placa: ~12 GB lidos por token e sem espaço para a KV. Só a menciono;
  não a investiguei. Neste PC, os MoE de ~3B ativos continuam a ser o formato que o `strata-poly` aproveita.

## 9. Estado dos perfis e do OpenCode

Sem alterações nesta avaliação:

| | Estado |
| --- | --- |
| Qwen3.6 UD-Q3_K_M, 4 configurações (`strata-qwen3.6-35b-a3b-ud-q3_k_m-hip{,-128k,-262k,-262k-stream}.json`) | iguais; modelo por omissão |
| Empero IQ3_M, configurações e ficheiro | iguais; mantido |
| `START-POLY-HIP.bat`, `tools/hip/opencode-polystrata-hip.json`, `tools/hip/poly_profile.py` | com as alterações por publicar do Empero, não tocadas |
| OpenCode global | não tocado |
| `C:\models` | só o Qwen3.6 e o Empero, como estavam; nada descarregado nem apagado |
| Novo | este relatório e `bench/results/2026-10-12-hip-gemma4-26b-a4b/gguf-headers/` (4 cabeçalhos JSON, 4 listagens e `summary.txt`) |
