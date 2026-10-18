> Historical measurement report. Use the [current setup instructions](../../README.md). Personal paths were generalized for publication; measurements were preserved. Raw campaign links may require the local evidence archive.

# PolyStrata HIP: avaliação do GLM-4.7-Flash (não integrado)

Ramo `hip-poly`, sobre `7023db0c` e as alterações por publicar do Empero ([AMD_EMPERO_QWEN38.md](AMD_EMPERO_QWEN38.md)).
Segue a metodologia da avaliação do North Mini Code ([AMD_NORTH_MINI_CODE.md](AMD_NORTH_MINI_CODE.md)): primeiro a
investigação, e a implementação só se houver viabilidade técnica e uma melhoria plausível.

**Nada no motor, no servidor, nos perfis ou no OpenCode mudou.** Nenhum modelo foi descarregado nem apagado. Só se
leram os cabeçalhos GGUF por HTTP parcial (`tools/hip/gguf_head.py`), guardados em
`bench/results/2026-10-12-hip-glm47-flash/gguf-headers/`. Sem commit nem push.

Data: 2026-10-12. Máquina alvo: RX 9060 XT 16 GB (gfx1200), Ryzen 5 5600X, 31,9 GB de RAM, Windows 11 Pro.

---

## 1. Resumo e recomendação

- **A integração parou na fase de avaliação.** O GLM-4.7-Flash não tem uma vantagem plausível sobre o
  Qwen3.6-35B-A3B em nenhuma das três coisas que interessam ao agente local:
  1. **Qualidade publicada (números dos fabricantes):** o Qwen3.6 está à frente em todos os testes que os dois
     publicam: SWE-bench Verified 73,4 contra 59,2, LiveCodeBench v6 80,4 contra 64,0, GPQA 86,0 contra 75,2 e HLE
     21,4 contra 14,4 (secção 3).
  2. **Medições independentes (Artificial Analysis):** o Qwen3.6 está à frente em tudo o que a página mostra para os
     dois. Em contexto longo (AA-LCR), 72 % contra 42 %. O índice do GLM-4.7-Flash está marcado como estimativa.
  3. **Velocidade estimada neste PC:** abaixo do Qwen3.6, e com a diferença a crescer com o contexto (secção 5):
     - cada token lê ~1,7× mais bytes de especialistas;
     - a cache de atenção (MLA) ocupa ~2,6× mais por posição;
     - o GGUF não traz o bloco MTP.
- **É tecnicamente viável, mas não é pequeno.** No GGUF, a arquitetura é `deepseek2` (MLA da DeepSeek), não
  `glm5-next`. O `Glm5Next` existente tem peças reutilizáveis: as projeções MLA, o router sigmoid com viés, o
  especialista partilhado, a camada densa, o tokenizador e o dialeto de ferramentas. Faltam:
  - uma classe de motor nova;
  - RoPE na parte posicional da chave;
  - um kernel de atenção MLA densa para contexto longo (o `dsa_attend` atual foi feito para uma seleção esparsa);
  - adaptar o streaming da KV.

  Estimativa: **3 a 5 dias** (secção 6).
- **Recomendação:** o **Qwen3.6-35B-A3B UD-Q3_K_M continua o modelo principal.** Não integrar o GLM-4.7-Flash.
- **Estado:** o Qwen3.6 e as suas 4 configurações, o Empero, o `START-POLY-HIP.bat` e o OpenCode ficaram como estavam.

## 2. O modelo

Fontes: [cartão do modelo](https://huggingface.co/zai-org/GLM-4.7-Flash) e o seu `config.json`, os cabeçalhos dos
GGUF do [Unsloth](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF), e o
[guia do Unsloth](https://unsloth.ai/docs/models/glm-4.7-flash).

| | GLM-4.7-Flash | Qwen3.6-35B-A3B (atual) |
| --- | --- | --- |
| Lançamento, licença | jan. 2026 (as correções do llama.cpp são de 21–22 jan.), MIT | Apache 2.0 |
| Parâmetros | 30B total (31B no safetensors), ~3B ativos | 35B total, 3B ativos |
| Classe HF / arquitetura no GGUF | `Glm4MoeLiteForCausalLM` / **`deepseek2`** | `qwen35moe` |
| Camadas | 47: 1 densa (FFN 10 240) + 46 MoE | 40 (+1 MTP): 30 delta net + 10 atenção |
| Especialistas | 64, **4 ativos**, FFN 1 536, 1 partilhado | 256, 8 ativos, FFN 512, 1 partilhado |
| Router | sigmoid + viés (`exp_probs_b`), top-4, normalizado, escala 1,8, 1 grupo | softmax, top-8, normalizado |
| Atenção | **MLA em todas as 47 camadas**: 20 cabeças, latente KV 512 + 64 com RoPE, q_lora 768, chave 192+64, valor 256 | 10 camadas de atenção total, 2 KV × 256 |
| RoPE | θ 10⁶, sem escalamento | parcial 64, θ 10⁷ |
| Vocabulário | 154 880, tabela de saída separada | 248 320 |
| Contexto | 202 752 | 262 144 |
| Bloco MTP | o `config.json` tem 1; **os GGUF não** (`block_count` 47, 844 tensores, nenhum `nextn`) | sim |
| Raciocínio | `<think>` aberto pelo template; `enable_thinking`; `clear_thinking` | `enable_thinking` |
| Ferramentas | `<tool_call>nome<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`, resultados em `<|observation|><tool_response>` | XML Qwen |
| Amostragem recomendada | geral `temperature 1.0, top_p 0.95`; SWE-bench/Terminal-Bench `0.7, top_p 1.0`; τ² `0` | |

## 3. Qualidade

### 3.1 Publicada pelos fabricantes

Números dos cartões dos modelos. Cada fabricante usa o seu próprio arnês e as suas condições, por isso a comparação
serve de orientação, não de medição.

| Teste | GLM-4.7-Flash | Qwen3.6-35B-A3B | Diferença |
| --- | ---: | ---: | ---: |
| SWE-bench Verified | 59,2 | **73,4** | −14,2 |
| LiveCodeBench v6 | 64,0 | **80,4** | −16,4 |
| GPQA | 75,2 | **86,0** | −10,8 |
| HLE | 14,4 | **21,4** | −7,0 |
| Terminal-Bench 2.0 | não publicado | 51,5 | |
| τ²-Bench | 79,5 | (publica o TAU3-Bench, 67,2: teste diferente) | não comparável |
| AIME | 91,6 (AIME 25) | 92,7 (AIME 26) | anos diferentes |
| BrowseComp | 42,8 | não publicado | |

- O cartão do GLM-4.7-Flash compara-se com o Qwen3-30B-A3B-Thinking-2507 e o GPT-OSS-20B, dois modelos anteriores ao
  Qwen3.5/3.6.
- O único teste agêntico de programação que os dois publicam é o SWE-bench Verified, e ali a diferença é grande:
  14 pontos.

### 3.2 Medições independentes

- **[Artificial Analysis](https://artificialanalysis.ai/models/comparisons/qwen3-6-35b-a3b-vs-glm-4-7-flash)**, página
  lida a 2026-10-12, ambos com raciocínio:

  | | Qwen3.6-35B-A3B | GLM-4.7-Flash |
  | --- | ---: | ---: |
  | Intelligence Index | 18 | 15 (**estimativa**, avaliação independente "a seguir") |
  | Humanity's Last Exam | 22 % | 8 % |
  | AA-LCR (raciocínio em contexto longo) | **72 %** | **42 %** |
  | AA-Omniscience | −22 | −63 |
  | Velocidade na API (não é deste PC) | 129 tok/s | 69 tok/s |

  O resumo da pesquisa mostrou outra versão do índice (32 contra 23), com a mesma ordem. Os valores da tabela são os
  da página tal como a li.
- **[SWE-rebench](https://swe-rebench.com/)** (tarefas novas, sem contaminação):
  - na janela atual (15/05–01/07/2026), o Qwen3.6-35B-A3B resolve 24,7 % ± 0,8 (pass@5 43,2 %). O GLM-4.7 Flash
    aparece sem valor nessa janela;
  - um espelho de terceiros ([benchmarklist](https://benchmarklist.com/benchmarks/swe_rebench/)) dá-lhe 34 % numa
    janela anterior, com outras tarefas. As duas janelas não são comparáveis, e não encontrei uma janela com os dois.
- Não encontrei medições independentes com metodologia descrita que ponham o GLM-4.7-Flash à frente do Qwen3.6.

### 3.3 Fiabilidade relatada

- **Ciclos de repetição:** o llama.cpp punha `softmax` no lugar de `sigmoid` no router. Foi corrigido a 21 jan. 2026 e
  os GGUF foram refeitos. Os ficheiros lidos aqui têm `expert_gating_func = 2` (sigmoid). Mesmo depois disso, o guia
  do Unsloth recomenda `--dry-multiplier 1.1` a `1.5` se o modelo entrar em ciclo.
- **Quantização:** o Unsloth recomenda pelo menos 4 bits (UD-Q4_K_XL). Num comentário público, diz que retirou os
  UD-Q2_K_XL por não funcionarem. Isto não foi verificado por mim.
- Para comparar: o Empero entrou em ciclo em greedy, e o Qwen3.6 não entrou em nenhum dos testes do relatório do
  Empero.

## 4. Compatibilidade com o motor `strata-poly` HIP

O motor escolhe a classe pela `general.architecture` (`src/poly/engine.cpp:586`): hoje `glm5-next`, `qwen35moe` e
`gemma4`. **`deepseek2` não corre sem código novo.** Apesar do nome, o GLM-4.7-Flash não é da família do GLM 5.3 Flash
(`glm5-next`): não tem hyper-connections, nem camadas KDA, nem indexador DSA. A atenção de todas as camadas é a MLA
densa da DeepSeek.

| Peça | O que o GLM-4.7-Flash pede | No motor hoje | Trabalho |
| --- | --- | --- | --- |
| Projeções MLA | `attn_q_a`, `q_a_norm`, `q_b`, `kv_a_mqa`, `kv_a_norm`, `k_b`, `v_b` (mesmos nomes) | `Glm5Next::dsa` carrega e absorve estes tensores | adaptar: aqui o `kv_a_mqa` dá 512 + **64 posicionais**, e o `k_b` absorve só os 192 sem posição |
| RoPE | só nas 64 dimensões posicionais de q e k, θ 10⁶ | o DSA do GLM 5.3 não usa RoPE; o RoPE existe no motor | pequeno, com o tipo de RoPE a confirmar na paridade |
| Atenção | causal densa sobre **todas** as posições: chave 576, valor 512, 20 cabeças, 1 KV | `dsa_attend` copia as linhas escolhidas para FP32 e faz um GEMM: foi feito para um top-k esparso, não para 48K–200K linhas | **kernel novo** (flash-decode MLA para HIP); é a peça maior |
| Cache KV | 576 FP16 por camada e posição, 47 camadas | a do DSA guarda 512 | adaptar; o streaming da KV (fase de 128K/262K) também |
| Router | sigmoid + viés, normalizado, × 1,8 | `Glm5Next::route` faz exatamente isto, no host | reutiliza; o routing na placa do HIP só existe para softmax (kernel novo, opcional) |
| Camada 0 densa, especialista partilhado | SwiGLU 10 240; partilhado de 1 536 | `dense_ffn`, `moe_shared` | reutiliza (sem o `swiglu_clamp`) |
| Bloco | um só fluxo residual, sequencial | o `layer_head`/`layer_tail` do `Qwen35` | reutiliza o padrão |
| Formatos dos pesos | Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL; `q_b` em F16 | todos em `native_mmvq_supported`; o F16 passa a BF16 (perde 3 bits de mantissa) | reutiliza; o MXFP4 não tem kernel |
| Adivinhar à frente | o GGUF não tem bloco MTP | | `--window 1`; o lookup por n-gramas continua possível |
| Tokenizador | `gpt2` com pre `glm4` | `tools/strata_tokenizer.py` já trata `glm4` (GLM 5.3) | reutiliza |
| Servidor: ferramentas | `<tool_call>nome<arg_key>…` | dialeto `glm` do `OutputParser` (`serve/frontend.py:733`) | reutiliza, com testes |
| Servidor: raciocínio | `<think>` aberto pelo template; o raciocínio depois da última mensagem do utilizador volta ao modelo ("Preserved Thinking", que a Z.ai recomenda para tarefas agênticas) | `enable_thinking` existe | verificar se o OpenCode devolve o `reasoning_content`; se não devolver, o modelo perde o raciocínio entre chamadas |

**Referência para paridade:** o llama.cpp HIP b11160 do Unsloth (`~/.unsloth/llama.cpp`, commit `a3c12db`) tem
`src/models/deepseek2.cpp` e carrega o modelo.
- O `fattn.cu` tem flash attention para chave 576 / valor 512.
- No RDNA4 o caminho WMMA só aceita cabeças até 256, por isso cai no kernel genérico `tile` (`fattn-tile.cuh`, casos
  576/512). Funciona, mas a velocidade do llama.cpp neste PC não diria o que um kernel próprio daria.
- O analisador de ferramentas do GLM está no `common/chat.cpp`.

## 5. Ficheiros, memória e velocidade estimada

Quantizações do Unsloth (não há GGUF oficial da Z.ai). Cabeçalhos lidos sem descarregar os pesos (`summary.txt` na
pasta acima):

| GGUF | Ficheiro | Especialistas | Formatos dos especialistas | Densos | 1 especialista | **Especialistas lidos por token** |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| UD-IQ3_XXS | 12,91 GB | 10,54 GiB | gate/up IQ3_XXS (30) / IQ3_S (16); down IQ3_XXS (25) / IQ3_S (12) / IQ4_XS (9) | 1,47 GiB | 3,67 MiB | 675 MiB |
| UD-Q3_K_XL | 13,78 GB | 11,34 GiB | gate/up Q3_K; down Q3_K (37) / Q4_K (9) | 1,48 GiB | 3,94 MiB | 726 MiB |
| Q3_K_M | 14,61 GB | 12,29 GiB | gate/up Q3_K; down Q4_K (45) / Q5_K (1) | 1,31 GiB | 4,27 MiB | 786 MiB |
| IQ4_XS | 16,27 GB | 13,75 GiB | IQ4_XS | 1,40 GiB | 4,78 MiB | 880 MiB |
| UD-Q4_K_XL (o que o Unsloth recomenda) | 17,52 GB | 14,81 GiB | gate/up Q4_K; down Q4_K (35) / Q5_K (11) | 1,50 GiB | 5,15 MiB | 948 MiB |
| MXFP4_MOE | 16,97 GB | 13,75 GiB | MXFP4: sem kernel no motor | 2,05 GiB | | |
| **Qwen3.6 UD-Q3_K_M (atual)** | 17,10 GB | 13,26 GiB | IQ3_XXS / IQ4_XS / Q6_K | 2,26 GiB | 1,33 MiB | **424 MiB** |

"Lidos por token" = 4 especialistas × 46 camadas (GLM) ou 8 × 40 (Qwen3.6). O especialista partilhado está à parte e
fica sempre na VRAM.

Se se integrasse, a escolha seria entre dois ficheiros:
- o **UD-IQ3_XXS**: tem os formatos de especialistas do Qwen3.6 atual, cujo produto em lote tem o mesmo erro que o de
  um token;
- o **UD-Q4_K_XL**, pela recomendação do Unsloth: tem os `Q4_K` cujo lote (MMQ) tem 2–3× o erro de um token (fase 4 e
  relatório do Empero), e lê 2,2× mais bytes por token do que o Qwen3.6.

Contas para a RX 9060 XT, KV em FP16, **calculadas, não medidas**:

| | GLM-4.7-Flash UD-Q3_K_XL | Qwen3.6 UD-Q3_K_M |
| --- | ---: | ---: |
| KV por posição | 47 × 576 × 2 B = 52,9 KiB | 10 × 2 × 256 × 2 × 2 B = 20 KiB |
| KV a 48K (49 152) | **2,48 GiB** | 0,94 GiB |
| KV a 128K | **6,61 GiB** | 2,50 GiB |
| KV a 200K | **10,2 GiB**: não deixa espaço para especialistas sem streaming da KV | 3,9 GiB |
| Densos na VRAM | ~1,7 GiB (1,48 + `k_b`/`v_b` em BF16) | 1,92 GiB (medido) |
| Especialistas na VRAM a 48K | ~7,0 GiB ≈ 1 800 de 2 944 (~62 %) | 6 140–6 330 de 10 240 (~60 %, medido) |
| Especialistas na VRAM a 128K | ~2,9 GiB ≈ 750 de 2 944 (~25 %) | ~6,4 GiB ≈ 4 900 (~48 %) |
| KV lida por token a 45K | 2,27 GiB | 0,86 GiB |

Velocidade de decode, **estimativa grosseira** a partir das medições do Qwen3.6 neste PC (91,0 tok/s a 2K e 62,4 a
45K):
- **2K:** o custo dominante são os especialistas lidos fora da VRAM. Com uma fração na VRAM parecida e 1,7× mais
  bytes, a estimativa é de **~50–60 tok/s**.
- **45K:** soma-se a leitura de 1,4 GiB a mais de KV por token (~5–8 ms a ~320 GB/s). A estimativa é de **~30–35
  tok/s**.
- **128K:** com só ~25 % dos especialistas na VRAM, a estimativa fica abaixo disso. Os relatos públicos do llama.cpp
  mostram o mesmo padrão: ~10 tok/s a 64K e ~7,6 a 100K num
  [DGX Spark](https://forums.developer.nvidia.com/t/llama-cpp-glm-4-7-flash-benchmark/358749), e de 88,5 a 16,8 tok/s
  entre 4K e 128K em [2× RTX 3090](https://hackernoon.com/glm-47-flash-on-2x-rtx-3090-my-hands-on-experience).
- Com 64 especialistas por camada e um viés que equilibra a carga, o uso tende a ser mais uniforme do que no Qwen3.6.
  A escolha por frequência pode ganhar menos aqui. Só se mede a correr.

A RAM chega: qualquer dos ficheiros até 17,5 GB cabe na cache de ficheiros ao lado dos outros programas.

## 6. Custo estimado da integração

| Parte | Conteúdo | Tamanho estimado |
| --- | --- | --- |
| Motor | `src/poly/deepseek2.{hpp,cpp}`: geometria (com verificação de que é este modelo, porque `deepseek2` também cobre os DeepSeek V2/V3 e o Kimi), MLA com a parte RoPE, KV de 576, router (do `Glm5Next`), camada densa, linhas da conversa; registo no `Engine::open` e no CMake | 450–650 linhas C++ |
| Kernel de atenção | flash-decode MLA para HIP: uma KV partilhada por 20 cabeças, chave 576 / valor 512, divisão por blocos do contexto; e o prefill em lote | 300–500 linhas HIP; é o maior risco de desempenho |
| Streaming da KV | adaptar a fase 128K/262K à cache MLA | médio |
| Routing na placa (opcional) | sigmoid + viés, top-4 | 50–100 linhas HIP |
| Servidor | o dialeto `glm` existe; falta confirmar `<think>`/`enable_thinking`, o raciocínio preservado entre chamadas e o fim do turno (`eot` 154 827, `eom` 154 829); testes | 50–150 linhas Python |
| Perfis | `runconfig`, perfil 48K, `START-POLY-HIP.bat`, OpenCode | pequeno, como o do Empero |
| Validação | compilar HIP e CUDA (`kernels.cu` é partilhado); paridade com o llama.cpp b11160; benchmark 2K/8K/31K/45K; estabilidade; contexto longo; tarefas no OpenCode | ~1 dia de máquina |

Total: **3 a 5 dias**. É mais do que o North, por causa do kernel de atenção, mas o servidor custa menos. Riscos:
- o RoPE só nas 64 dimensões e a absorção de 192 dimensões são fáceis de errar em silêncio (a paridade apanha-os);
- o kernel MLA decide a velocidade em contexto longo, que é onde o modelo já parte em desvantagem.

## 7. Testar antes de portar (opcional, não feito)

Os testes de velocidade, VRAM, raciocínio, ferramentas e contexto longo no PolyStrata **não foram corridos**, porque
exigem o motor integrado. Há um caminho que **não toca no motor** e mede a qualidade, que é a questão principal:

1. Descarregar `GLM-4.7-Flash-UD-Q4_K_XL.gguf` (17,52 GB) ou `UD-IQ3_XXS` (12,91 GB). O C: tem 86 GB livres, e não é
   preciso remover nada.
2. Servi-lo com o `llama-server` HIP b11160 que já está no PC (`--jinja`, `temperature 0.7, top_p 1.0, min_p 0.01`
   para ferramentas, sem penalização de repetição).
3. No OpenCode, correr a tarefa dos 4 erros do relatório do Empero
   (`bench/results/2026-10-12-hip-empero-qwen38/codetask`) e as agulhas de contexto longo (8K/32K/45K, 3 notas de
   uma vez). Comparar com o Qwen3.6 servido da mesma forma.

Leva 1 a 2 horas. A velocidade medida ali não vale para o PolyStrata: no RDNA4, o llama.cpp usa o kernel `tile`
genérico para esta atenção. **Critério:** só se justifica integrar se o GLM-4.7-Flash ganhar claramente nas duas
tarefas, contra os números publicados pelos fabricantes e contra o AA-LCR. Com 30 pontos de diferença no AA-LCR e
14 no SWE-bench Verified, não espero que ganhe.

## 8. Estado dos perfis e do OpenCode

Sem alterações nesta avaliação:

| | Estado |
| --- | --- |
| Qwen3.6 UD-Q3_K_M, 4 configurações (`strata-qwen3.6-35b-a3b-ud-q3_k_m-hip{,-128k,-262k,-262k-stream}.json`) | iguais; modelo por omissão |
| Empero IQ3_M, 2 configurações | iguais |
| `START-POLY-HIP.bat`, `tools/hip/opencode-polystrata-hip.json`, `tools/hip/poly_profile.py` | com as alterações por publicar do Empero, não tocadas |
| OpenCode global | não tocado |
| `C:\models` | só o Qwen3.6 e o Empero, como estavam; nada descarregado nem apagado |
| Novo | este relatório e `bench/results/2026-10-12-hip-glm47-flash/gguf-headers/` (6 cabeçalhos JSON e `summary.txt`) |
