> Historical measurement report. Use the [current setup instructions](../../README.md). Personal paths were generalized for publication; measurements were preserved. Raw campaign links may require the local evidence archive.

# PolyStrata HIP: avaliação do Cohere North Mini Code 1.0 (não integrado)

Ramo `hip-poly`, sobre `7023db0c` e as alterações por publicar do Empero ([AMD_EMPERO_QWEN38.md](AMD_EMPERO_QWEN38.md)).
**Nada no motor, no servidor, nos perfis ou no OpenCode mudou.** Nenhum ficheiro foi descarregado nem apagado; só se
leram os cabeçalhos GGUF por HTTP parcial (`tools/hip/gguf_head.py`). Sem commit nem push.

Data: 2026-10-12. Máquina alvo: RX 9060 XT 16 GB (gfx1200), Ryzen 5 5600X, 31,9 GB de RAM, Windows 11 Pro.

---

## 1. Resumo e recomendação

- **A integração parou na fase de avaliação**, pelas duas condições do pedido:
  1. **Não há vantagem plausível sobre o Qwen3.6-35B-A3B.** Na tabela do próprio fabricante (cartão do modelo), o
     Qwen3.6 está à frente em 5 de 6 testes, e nos 4 agênticos por margens largas (secção 3). O North só ganha em
     SciCode, por 2,4 pontos.
  2. **A integração não é pequena.** É uma arquitetura nova para o `strata-poly` (`cohere2moe`): uma classe de motor
     nova, um dialeto novo de raciocínio e ferramentas no servidor, e a validação completa (secção 6). Custo estimado:
     2 a 4 dias de trabalho, com risco médio.
- **É tecnicamente viável.** Nada no modelo impede a integração:
  - os formatos dos pesos do ficheiro indicado já têm kernels HIP;
  - cabe na VRAM e na RAM como o Qwen3.6;
  - o llama.cpp HIP b11160 que já está neste PC carrega a arquitetura, e serve de referência de paridade.
- **Recomendação:** o **Qwen3.6-35B-A3B UD-Q3_K_M continua o modelo principal.** Não integrar o North Mini Code.
- **Espaço:** há 86 GB livres no C:. O Empero IQ3_M não precisa de ser removido e **não foi tocado**.
- **Estado:** o Qwen3.6 e as suas 4 configurações, o Empero, o `START-POLY-HIP.bat` e o OpenCode ficaram como estavam.
- Os benchmarks pedidos (velocidade, VRAM, tarefas no OpenCode) **não foram corridos**, porque exigem o motor
  integrado. A secção 8 propõe uma verificação independente barata, sem tocar no motor, caso se queira um dado nosso
  antes de fechar o assunto.

## 2. O modelo

Fontes: [cartão do modelo](https://huggingface.co/CohereLabs/North-Mini-Code-1.0) (`config.json`, `chat_template.jinja`),
[publicação da Cohere](https://huggingface.co/blog/CohereLabs/introducing-north-mini-code),
[documentação](https://docs.cohere.com/docs/north-mini-code-1.0).

| | North Mini Code 1.0 | Qwen3.6-35B-A3B (atual) |
| --- | --- | --- |
| Lançamento, licença | 9 jun. 2026, Apache 2.0 ("open weights research release") | Apache 2.0 |
| Parâmetros | 30B total, ~3B ativos | 35B total, 3B ativos |
| Camadas | 49: 1 densa (FFN 3 072) + 48 MoE | 40 (+1 MTP): 30 delta net + 10 atenção |
| Especialistas | 128, 8 ativos, FFN 768, sem partilhado | 256, 8 ativos, FFN 512, 1 partilhado |
| Router | sigmoid, top-8, pesos **sem** normalização, sem viés | softmax, top-8, normalizado |
| Atenção | 3 em 4 por janela de 4 096 com RoPE; 1 em 4 global **sem posição** (NoPE); a camada 0 é global com RoPE | 10 camadas de atenção total |
| Cabeças | 32 de consulta, 4 KV, dimensão 128 | 16 de consulta, 2 KV, dimensão 256 |
| Bloco | paralelo: `x + atenção(n(x)) + MoE(n(x))`, uma só norma RMS | sequencial |
| Vocabulário | 262 144, tabela de saída = tabela de tokens | 248 320 |
| Contexto | 256K (treino: 64K no SFT 1, 128K no SFT 2 e no RL) | 262K |
| Bloco MTP | não (o GGUF tem 442 tensores, nenhum `nextn`) | sim |
| Raciocínio | ligado por omissão (`reasoning`/`reasoning_effort` no template) | `enable_thinking` |
| Amostragem recomendada | `temperature 1.0, top_p 0.95` | |

Formato da conversa (template Cohere, diferente de todos os que o servidor já lê):

```
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|>raciocínio<|END_THINKING|>
  <|START_TEXT|>resposta<|END_TEXT|>                                    (ou)
  <|START_ACTION|>[{"tool_call_id": "0", "tool_name": "bash", "parameters": {...}}]<|END_ACTION|>
<|END_OF_TURN_TOKEN|>
```

Os resultados das ferramentas voltam como turno de sistema, `<|START_TOOL_RESULT|>[...]<|END_TOOL_RESULT|>`.

## 3. Qualidade publicada

Tabela do cartão do modelo (imagem da Cohere, lida diretamente; `*` = medido pela Cohere):

| Teste | North Mini Code | Qwen3.6-35B-A3B | Diferença |
| --- | ---: | ---: | ---: |
| Terminal-Bench v2 | 36,0 | **51,5** | −15,5 |
| Terminal-Bench Hard | 31,1 | **35,0** | −3,9 |
| SWE-Bench Verified | 67,6 | **73,4** | −5,8 |
| SWE-Bench Pro | 40,2 | **49,5** | −9,3 |
| SciCode | **38,2** | 35,8 | +2,4 |
| LiveCodeBench v6 | 70,3 | **80,4** | −10,1 |

- É a tabela do próprio fabricante: as condições escolhidas por ele, o North com 3 sementes a `temperature 1.0`, e os
  valores do Qwen3.6 retirados da publicação da Qwen.
- Mesmo assim, o North perde nos 4 testes de programação agêntica e com ferramentas, que são exatamente o que se quer
  no OpenCode. A maior diferença é no uso autónomo do terminal (Terminal-Bench v2).
- Os números mais altos que aparecem em artigos (80,2 % no SWE-Bench Verified, 55,1 % no Terminal-Bench v2) são
  **pass@10** de um ponto intermédio do treino, não comparáveis com pass@1.
- Índice de programação da Artificial Analysis, citado por terceiros e **não verificado por mim**: Qwen3.6 35,2,
  North 33,4.
- Não encontrei medições independentes que contrariem isto com metodologia descrita.

Os dois modelos correriam com quantizações de ~3,5 bits por peso (secção 5), por isso a quantização não desfaz a
diferença a favor do North.

## 4. Compatibilidade com o motor `strata-poly` HIP

O motor escolhe a classe pela `general.architecture` do GGUF (`Engine::open`, `src/poly/engine.cpp:581`). Hoje conhece
`glm5-next`, `qwen35moe` e `gemma4`. **`cohere2moe` não corre sem código novo.** Ser MoE não basta.

| Peça | O que o North pede | No motor hoje | Trabalho |
| --- | --- | --- | --- |
| Carregar o GGUF, especialistas em cache VRAM/CPU, lote MMQ | tensores `ffn_{gate,up,down}_exps` separados, 128 especialistas | `Engine::load_moe` já lê gate/up separados | reutiliza |
| Formatos dos pesos (UD-Q3_K_M) | especialistas IQ3_XXS / IQ4_XS / Q6_K; densos Q8_0 e Q6_K | os mesmos formatos do Qwen3.6 UD-Q3_K_M, com kernels de um token, lote e CPU | reutiliza |
| Norma | RMS (o GGUF tem `layer_norm_rms_epsilon`, por isso o llama.cpp usa RMS) | `rms_norm` | reutiliza |
| Bloco paralelo | atenção e MoE leem a mesma entrada normada; saída `x + atenção + MoE` | o motor separa `layer_head`/`layer_tail`; o router lê `xn_` | novo, na classe |
| Atenção por janela + global | anéis de 4 096 + camadas globais; mesma dimensão de cabeça em todas | o Gemma 4 já tem anéis e camadas globais (dimensões diferentes) | adaptar do Gemma 4 |
| RoPE só nas janelas e na camada 0; globais sem posição | RoPE "normal" (não NEOX), base 50 000 | RoPE existe; ligar/desligar por camada é novo | pequeno |
| Camada 0 densa | SwiGLU de 3 072 | o Gemma 4 tem MLP denso (GELU) | pequeno |
| Router | sigmoid, sem normalização nem viés | sigmoid no host existe (GLM); **o routing na placa do HIP só existe para softmax** (`card_route_floor`) | host: reutiliza; na placa: kernel novo, opcional |
| Saída | tabela de tokens Q6_K, 262 144 linhas | o Gemma 4 já usa a tabela como saída | reutiliza |
| Estado da conversa (guardar/repor linhas) | anéis + globais | o do Gemma 4 | adaptar |
| Adivinhar à frente (MTP) | o ficheiro não tem bloco | | não se aplica (`--window 1`) |
| Servidor: raciocínio e ferramentas | `<|END_THINKING|>`, `<|START_TEXT|>`, `<|START_ACTION|>[JSON]` com várias chamadas por bloco; opção `reasoning` | o `OutputParser` (`serve/frontend.py`) tem dialetos Qwen XML, GLM e Gemma | **dialeto novo** + testes |

Referência para paridade: o llama.cpp HIP b11160 do Unsloth (`~/.unsloth/llama.cpp`) já inclui
`src/models/cohere2moe.cpp` (PR [ggml-org/llama.cpp#24260](https://github.com/ggml-org/llama.cpp/pull/24260), integrado a
13 jun. 2026) e um analisador de ferramentas `common/parsers/cohere2moe.cpp`.

## 5. Ficheiros e memória

Quantizações disponíveis: [Unsloth](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF) (UD-IQ1_M 9,4 GB a Q8 33 GB)
e [bartowski](https://huggingface.co/bartowski/North-Mini-Code-1.0-GGUF) (IQ2_XXS 8,5 GB a Q8 32 GB). Não há GGUF oficial
da Cohere. Cabeçalhos lidos (só metadados e tabela de tensores):

| GGUF | Ficheiro | Especialistas | Formatos dos especialistas | Densos |
| --- | ---: | ---: | --- | --- |
| **Unsloth UD-Q3_K_M** | 14,21 GB | 11,84 GiB | gate/up IQ3_XXS (47 camadas), IQ4_XS (1); down IQ4_XS (46), Q6_K (2) | 1,39 GiB: Q8_0 + tabela Q6_K |
| Unsloth UD-IQ3_S | 12,77 GB | 10,70 GiB | gate/up **IQ2_S** (47), IQ3_S (1); down IQ4_XS / Q6_K | 1,18 GiB, quase tudo Q6_K |
| Qwen3.6 UD-Q3_K_M (atual) | 17,10 GB | 13,26 GiB | IQ3_XXS / IQ4_XS / Q6_K | 2,26 GiB, Q8_0 |

Seria o **UD-Q3_K_M**: os mesmos formatos de especialistas do Qwen3.6 atual, e densos em Q8_0, o formato denso para o
qual o motor tem kernels de várias colunas rápidos. O UD-IQ3_S desce gate/up a 2 bits.

Contas a 48K (49 152 posições, KV em FP16), **calculadas, não medidas**:

| | North UD-Q3_K_M | Qwen3.6 UD-Q3_K_M |
| --- | ---: | ---: |
| Pesos densos em VRAM | 1,39 GiB | 2,26 GiB |
| KV das camadas globais | 13 × 49 152 × 2 KiB = 1,22 GiB | 10 × 49 152 × 2 KiB = 0,94 GiB |
| KV das janelas | 36 anéis de ~5 120 linhas ≈ 0,35 GiB | — (estado recorrente, pequeno) |
| Sobra para especialistas, comparada com o Qwen3.6 | ≈ +0,2 GiB | referência: 6 140–6 330 especialistas, ~8 GiB |
| Um especialista | 1,97 MiB | 1,33 MiB |
| Especialistas em VRAM (estimativa) | ~4 250–4 400 de 6 144 (~70 %) | 6 140–6 330 de 10 240 (~60 %, medido) |
| **Bytes de especialistas lidos por token** | 8 × 48 × 1,97 ≈ **758 MiB** | 8 × 40 × 1,33 ≈ **424 MiB** |
| RAM | o ficheiro, 14,2 GB, cabe na cache de ficheiros | 17,1 GB |

- **Cabe:** a VRAM chega para os densos, a KV de 48K e uma cache de especialistas maior em proporção do que a do
  Qwen3.6. A RAM chega com folga.
- **Velocidade, por medir:** cada token lê 1,8× mais bytes de especialistas do que o Qwen3.6, e a atenção global a 45K
  lê ~30 % mais KV (13 camadas contra 10). O decode deve ficar abaixo do Qwen3.6 neste PC. Quanto abaixo depende da
  concentração do uso dos especialistas, que só se mede a correr.
- O router sigmoid ficaria no host, sem o routing na placa que o Qwen3.6 usa, a não ser que se escreva o kernel.

## 6. Custo estimado da integração

| Parte | Conteúdo | Tamanho estimado |
| --- | --- | --- |
| Motor | `src/poly/cohere2moe.{hpp,cpp}` a partir do `gemma4.cpp`: geometria, camadas, bloco paralelo, RoPE por camada, router sigmoid, linhas da conversa; registo em `Engine::open` e no CMake | 450–600 linhas C++ |
| Routing na placa (opcional) | variante sigmoid sem normalização de `route_topk_softmax` | 50–100 linhas HIP |
| Servidor | dialeto `cohere` no `OutputParser`: fim do raciocínio, `<|START_TEXT|>`, blocos `<|START_ACTION|>` com várias chamadas, streaming dos argumentos, `reasoning` no template, ids das chamadas; testes como `test_gemma.py` | 250–400 linhas Python |
| Instalação e perfis | `runconfig`, perfil 48K, `START-POLY-HIP.bat`, OpenCode | pequeno, reutiliza o do Empero |
| Validação | compilar HIP (e CUDA, por tocar em ficheiros partilhados); paridade com o llama.cpp b11160; benchmark 2K/8K/31K/45K; estabilidade; contexto longo; tarefas no OpenCode | como no relatório do Empero, ~1 dia de máquina |

Total: **2 a 4 dias**. Riscos: a mistura NoPE/RoPE e o bloco paralelo são fáceis de errar em silêncio (a paridade
apanha-os). O dialeto de ferramentas novo é a parte com mais casos: várias chamadas por bloco, texto depois do
raciocínio, chamadas dentro do raciocínio.

## 7. Espaço em disco

- C: tem 86 GB livres (931 GB, 91 % usado). D: tem 43 GB livres.
- O North UD-Q3_K_M ocupa 14,2 GB. Cabe sem remover nada.
- O ficheiro `C:\models\Qwen3.8-35B-A3B-Distill-IQ3_M.gguf` (16 339 529 600 bytes, Empero) **foi mantido**,
  como o pedido manda quando há espaço. Não foi apagado nada.

## 8. O que não foi feito, e uma verificação barata possível

Os testes pedidos na parte 4 (velocidade, VRAM/RAM, raciocínio, OpenCode, ferramentas, contexto longo) não foram
corridos: exigem o modelo a correr no PolyStrata, e a avaliação parou antes disso.

Se se quiser um dado nosso antes de fechar o assunto, há um caminho que **não toca no motor**:
- descarregar o `North-Mini-Code-1.0-UD-Q3_K_M.gguf` (Unsloth, 14,21 GB);
- servi-lo com o `llama-server` HIP b11160 que já está no PC, que tem a arquitetura e o analisador de ferramentas;
- correr no OpenCode a mesma tarefa dos 4 erros do relatório do Empero (`bench/results/2026-10-12-hip-empero-qwen38/codetask`)
  e os testes de agulhas de contexto longo, contra o Qwen3.6 servido da mesma forma.

Leva 1 a 2 horas. Mediria qualidade e uso de ferramentas, não a velocidade que o PolyStrata teria. Só se justifica
integrar se o North ganhar claramente ali, contra a tabela do próprio fabricante.

## 9. Estado dos perfis e do OpenCode

Sem alterações nesta avaliação:

| | Estado |
| --- | --- |
| Qwen3.6 UD-Q3_K_M, 4 configurações (`strata-qwen3.6-35b-a3b-ud-q3_k_m-hip{,-128k,-262k,-262k-stream}.json`) | iguais; modelo por omissão |
| Empero IQ3_M, 2 configurações | iguais |
| `START-POLY-HIP.bat`, `tools/hip/opencode-polystrata-hip.json`, `tools/hip/poly_profile.py` | com as alterações por publicar do Empero, não tocadas |
| OpenCode global | não tocado |
