> Historical measurement report. Use the [current setup instructions](../../README.md). Personal paths were generalized for publication; measurements were preserved. Raw campaign links may require the local evidence archive.

# PolyStrata HIP: Empero Qwen3.8-35B-A3B-Distill IQ3_M ao lado do Qwen3.6 UD-Q3_K_M

Ramo `hip-poly`, sobre `7023db0c` (fim da migração para o Qwen3.6 Q3, [AMD_QWEN36_Q3_MIGRATION.md](AMD_QWEN36_Q3_MIGRATION.md)).
Não foi feito push nem aberto PR. **O motor não mudou**: o binário é o da fase 4.

**Onde foi medido:** este PC. RX 9060 XT 16 GB (gfx1200), Ryzen 5 5600X (6 núcleos), 31,9 GB de RAM, Windows 11 Pro,
ROCm 10.2.0a20260930 (TheRock), em uso normal: outros programas ocupavam ~3,4 GB de VRAM e deixavam ~18 GB de RAM
livre antes de cada corrida do benchmark (registado por corrida nos JSON). Dados:
`bench/results/2026-10-12-hip-empero-qwen38/`.

---

## 1. Resumo

- **Funciona sem alterações ao motor.**
  - É a arquitetura `qwen35moe` do Qwen3.6: 41 blocos com MTP, 256 especialistas, 8 usados, 30 camadas delta net e
    10 de atenção.
  - Todos os formatos do ficheiro têm kernels em HIP: um token, lote (MMQ) e CPU.
- **Ficheiro escolhido:** o **IQ3_M** (16,3 GB, SHA-256 verificado).
  - Especialistas: 13,06 GiB (13,26 no Qwen3.6 UD-Q3_K_M).
  - Pesos densos: 1,31 GiB (2,26 no Qwen3.6). Por isso cabem mais especialistas na VRAM a 48K (6 680–6 850 de 10 496,
    contra 6 140–6 330).
- **Precisão:** paridade com o llama.cpp HIP b11160 no mesmo GGUF:
  - **PASS** na configuração usada (`--window 1`, routing na placa), contra as duas referências;
  - routing na placa igual ao do host em 10 240 de 10 240 routings;
  - um passo (`code`, 30) em que as próprias referências A e B do llama.cpp escolhem tokens diferentes; ali, duas
    configurações não usadas por omissão (sem MMQ, especialistas na CPU) dão FAIL contra A e PASS contra B (secção 5).
- **Desempenho** (mediana de 3 corridas alternadas no mesmo dia, contexto 49 152, a quente):

  | | 2K | 8K | 31K | 45K |
  | --- | ---: | ---: | ---: | ---: |
  | Qwen3.6 Q3 (MTP), decode tok/s | 91,0 | 86,2 | 70,6 | 62,4 |
  | Empero, perfil (`--window 1`), decode tok/s | **92,1** | **87,0** | 67,8 | 60,6 |
  | Qwen3.6 / Empero, prefill tok/s | 1 116 / 1 271 | 1 423 / 1 488 | 1 324 / 1 326 | 1 221 / 1 209 |

- **O MTP do Empero não compensa neste motor** e o perfil usa `--window 1`:
  - o bloco acerta 76–88 % dos palpites, mas as janelas de verificação custam mais do que poupam. Os pesos densos
    deste ficheiro são IQ3_S/Q4_K, e o motor só tem kernels de várias colunas rápidos para Q8_0 (o formato denso do
    Qwen3.6 Unsloth);
  - com MTP, 2–8 % mais lento a quente e 35 % mais lento no primeiro pedido (secção 6.2).
- **Ferramentas e tarefa de programação** (OpenCode, 4 erros num módulo Python, testes corridos à parte):
  - com raciocínio `medium` e a amostragem do perfil, Empero 4/4 e Qwen3.6 4/4, em tempos parecidos (49–106 s contra
    63–143 s);
  - sem raciocínio, Empero 1/2 e Qwen3.6 2/2.
- **Problema encontrado e causa identificada:** em greedy, o Empero entrou duas vezes num ciclo de repetição no
  raciocínio:
  - 27 853 tokens de frases repetidas, e um colapso em `^^^^`;
  - o mesmo pedido, repetido no **llama.cpp, degenera da mesma maneira** (19 360 `^` seguidos). É do modelo, não do
    motor;
  - mitigação: o perfil do Empero usa a amostragem do cartão do modelo (`temperature 0.6, top_p 0.95, top_k 20`) para
    os pedidos que não a indicam (o OpenCode não indica nenhuma). Reduz o problema mas não o elimina (secção 8.3).
- **Estabilidade:** 727 pedidos em 20 min, 0 erros, VRAM constante (12 535 MiB), RAM privada +130 MiB.
- **Contexto longo:**
  - cada nota (5/50/95 %) é encontrada sozinha a 8K, 32K e 45K;
  - a pergunta pelas três de uma vez falha a 32K e 45K (o modelo termina depois da primeira). O Qwen3.6 acerta as
    três;
  - o cartão do modelo avisa que foi treinado com exemplos de 8 192 tokens.
- **Integração:**
  - o Qwen3.6 continua o modelo por omissão e as suas 4 configurações ficaram iguais byte a byte;
  - o Empero entra no menu do `START-POLY-HIP.bat`, na mesma porta (troca por pedido) e no OpenCode/OpenChamber
    (global, com cópia de segurança).

## 2. Ficheiros disponíveis e escolha

Repositório `empero-ai/Qwen3.8-35B-A3B-Distill-GGUF`, revisão `b1f9d1dc`. Cabeçalhos lidos por HTTP parcial
(`tools/hip/gguf_head.py`, `gguf-headers/`), sem descarregar os pesos:

| GGUF | Ficheiro | Especialistas (modelo principal) | Formatos dos especialistas | Pesos densos | Bloco MTP |
| --- | ---: | ---: | --- | --- | --- |
| **IQ3_M** | 16,34 GB | 13,06 GiB | gate/up **IQ3_S**; down IQ3_S (35 camadas), Q4_K (camadas 0–4) | IQ3_S e Q4_K, 1,31 GiB | Q8_0, 0,84 GiB |
| Q3_K_M | 17,66 GB | 14,28 GiB | gate/up Q3_K; down **Q4_K** (38), Q5_K (2) | Q3_K/Q4_K/Q5_K, 1,32 GiB | Q8_0 |
| Q2_K | 13,84 GB | 10,86 GiB | gate/up Q2_K; down Q3_K | Q2_K/Q3_K/Q4_K, 1,18 GiB | Q8_0 |
| Qwen3.6 UD-Q3_K_M (atual, para comparar) | 17,10 GB | 13,26 GiB | IQ3_XXS / IQ4_XS / Q6_K | Q8_0, 2,26 GiB | Q3_K/Q4_K, 0,39 GiB |

Porquê o **IQ3_M**, com 16 GB de VRAM e 32 GB de RAM:
- **VRAM:** o motor não precisa que o modelo caiba na VRAM; enche-a de especialistas pela frequência de uso e deixa o
  resto na RAM. O que conta é a fração dos especialistas usados que vem da VRAM:
  - os especialistas do IQ3_M ocupam quase o mesmo que os do Qwen3.6 atual;
  - os pesos densos são 0,9 GiB menores, por isso cabem **mais** especialistas do que hoje;
  - o Q3_K_M tem 1,2 GiB a mais de especialistas (menos na VRAM, decode mais lento).
- **RAM:** o IQ3_M cabe na cache de ficheiros ao lado dos outros programas (~18 GB livres). O Q3_K_M também, com
  menos folga.
- **Precisão do motor:**
  - o Q3_K_M tem os `down` em Q4_K/Q5_K, os formatos cujo produto em lote (MMQ) tem 2–3× o erro do produto de um token
    (fase 4 e secção 5.1);
  - no IQ3_M esse caminho só toca no `down` de 5 camadas e em pesos densos.
- **Q2_K** não foi necessário: só se justificaria se o IQ3_M não coubesse ou fosse lento, e tem pesos mais grosseiros.
- **Offload parcial do Q3_K_M**: é o que o motor faz sempre com qualquer ficheiro; não traz vantagem sobre o IQ3_M.

## 3. Download e integridade

| | |
| --- | --- |
| Origem | `empero-ai/Qwen3.8-35B-A3B-Distill-GGUF`, revisão `b1f9d1dcc3de8aa867669b0ab919384aeeb9b8d5`, `Qwen3.8-35B-A3B-IQ3_M.gguf` |
| Ficheiro | `C:\models\Qwen3.8-35B-A3B-Distill-IQ3_M.gguf` (com "Distill" no nome, para não se confundir com um modelo da Qwen) |
| Bytes | 16 339 529 600: igual ao da API do Hugging Face |
| SHA-256 | `0ad7b253ecee6de7b38ecaf82ea5553d302c3f924961b7091b7088cc99a30f95`: igual ao `SHA256SUMS` do repositório e ao `lfs.oid` da API. Confirmado pelo `verify_sha256()` do `setup.py` e de novo pelo `Get-FileHash` |
| Download | `setup.download()` (retoma por `Range`, `.part`), 1 304 s; marca `.done` com o SHA-256 |

Nenhum ficheiro existente foi apagado ou alterado. O disco tinha 112 GB livres antes.

## 4. Compatibilidade

### 4.1 Arquitetura e loader

- Metadados iguais aos do Qwen3.6 em tudo o que o motor lê:
  - `qwen35moe`, 41 blocos, `nextn_predict_layers` 1, 16 cabeças, 2 KV, `key_length` 256;
  - RoPE parcial 64 com secções [11, 11, 10], θ 10⁷;
  - delta net 16/32 cabeças, estado 128, conv 4;
  - 256 especialistas, 8 usados, largura 512.
- **Chave nova:** `qwen35moe.attention.recurrent_layers` (o padrão 3 delta net + 1 atenção). O motor não a lê: decide
  cada camada pela presença do tensor `attn_qkv` (`Qwen35::load_layer`), o que dá o mesmo padrão.
- **Carga:** 40/40 camadas MoE com routing na placa, 41 blocos.
  - VRAM (servidor, 48K): pesos densos 1 177 MiB (1 961 no Qwen3.6), cache de especialistas 9 033 MiB, KV 1 056 MiB,
    estado recorrente 251 MiB.
  - Especialistas na VRAM a 48K: 6 680–6 850 de 10 496 (8,8 GiB), conforme a VRAM livre.

### 4.2 Quantização e kernels

`upload()` (`src/poly/engine.cpp`) envia cada peso denso tal como está quando o `native_mmvq` tem esse tipo. IQ3_S,
Q4_K, Q3_K, Q2_K, Q6_K e Q8_0 estão todos na lista; nada é convertido para BF16. O MMQ (prefill) tem os mesmos
tipos, e a CPU usa o `vec_dot` do ggml.

`poly_expert_parity` no ficheiro real (`logs/expert-parity.log`), erro relativo contra FP64, **PASS**:

| Formato e forma | Um token (MMVQ) | Lote (MMQ) |
| --- | ---: | ---: |
| IQ3_S 2 048 → 512 (gate/up), 512 → 2 048 (down) | 0,0121 / 0,0110 | 0,0121 / 0,0110 |
| IQ3_S 2 048 → 32 (`ssm_alpha`/`ssm_beta`, 32 linhas) | 0,0110 | 0,0110 |
| IQ3_S 2 048 → 4 096 / 8 192, 4 096 → 2 048 (atenção, delta net) | 0,0111–0,0116 | 0,0111–0,0116 |
| IQ3_S 2 048 → 248 320 (`token_embd`) | 0,0114 | 0,0114 |
| **Q4_K** 512 → 2 048 (down, camadas 0–4) | 0,0116 | **0,0347** |
| **Q4_K** 2 048 → 512 / 8 192, 4 096 → 2 048 (`attn_v`, `attn_qkv`, `attn_output`) | 0,0114–0,0117 | **0,0255–0,0268** |
| Q6_K (`output`) | 0,0110 | 0,0110 |
| Q8_0 (bloco MTP) | 0,0110–0,0117 | 0,0110–0,0117 |
| Especialista inteiro, CPU / GPU | 0,031–0,046 / 0,013–0,019 | — |

O Q4_K em lote tem 2–3× o erro de um token, como na fase 4: é o termo de mínimo com escalas FP16 do MMQ do llama.cpp.
Aqui toca no `attn_qkv` das 30 camadas delta net e no `attn_v`/`attn_output` das 10 de atenção. Efeito medido na
paridade (secção 5): pequeno.

### 4.3 Router MoE e atenção híbrida

- **Router:** `ffn_gate_inp` em F32, softmax e os 8 maiores, como no Qwen3.6.
  - `STRATA_POLY_CARD_ROUTE_CHECK=1` (prompt B-2k, 256 tokens, `card-route-check.err`): **10 240 routings, 0
    escolhas diferentes do host**, pesos com diferença ≤ 8,94e-08.
  - Na paridade, routing na placa e no host dão o mesmo resultado (secção 5).
- **Atenção híbrida:** as 30 camadas delta net (`ssm_*`, estado recorrente) e as 10 de atenção com GQA e porta de saída
  correm os mesmos caminhos do Qwen3.6.
  - A 35 575 tokens, os 64 tokens greedy são iguais aos do llama.cpp (secção 5).
  - As notas a 5/50/95 % de 45K tokens são encontradas (secção 7).

### 4.4 Template de chat e tokenizer

- **Tokenizer** (escrito do GGUF por `tools/strata_tokenizer.py`): `vocab.json`, `merges.txt` e `token_type.json`
  iguais byte a byte aos do Qwen3.6; `add_bos_token` false nos dois.
  - Muda o `padding_token_id` (248044 `<|endoftext|>`, contra 248055), que o servidor não usa para gerar.
- **Template:** é o original da Qwen, não a versão corrigida da Unsloth que o Qwen3.6 traz. Diferenças:
  - o papel `developer` e um segundo `system` dão erro;
  - `tool_call.arguments|items` exige um objeto;
  - a falta de mensagem de utilizador dá erro (`raise_exception`).

  O servidor já normaliza tudo isto antes do template (`serve/frontend.py`): `developer` passa a `system`, um
  `system` tardio passa a `user`, os argumentos passam a objeto, e o erro do template é um 400. Para os casos que o
  servidor envia (mensagem simples, sem pensamento, system + ferramentas, ciclo de ferramenta, `preserve_thinking`), os
  dois templates dão **o mesmo texto** e as mesmas capacidades.
- **Raciocínio:** `<think>` separado em `reasoning_content` (streaming e não), `enable_thinking: false` respeitado.

## 5. Paridade com o llama.cpp

Método das fases anteriores (`tools/hip/poly_llama_parity.py`, `parity-*.json`, `logs/parity.log`):
- 5 prompts (22, 870, 3 069, 11 481 e 35 575 tokens), greedy, 64 tokens;
- critério: os mesmos tokens, ou a separação num quase-empate (< 0,1), e diferença máxima ≤ 0,15;
- referências llama.cpp HIP b11160 no mesmo GGUF: **A** `--cpu-moe`, **B** `--n-cpu-moe 22`.

| Configuração | Tokens iguais (short / code / 2.5k / 8k / 30k) | Diferença máxima | Média por prompt | Resultado |
| --- | --- | ---: | ---: | --- |
| llama.cpp B contra A | 26/42 · 30 · 64 · 64 · 64 | 0,068 | 0,0047–0,0082 | PASS |
| **`--window 1`, routing na placa (o perfil), contra A** | 26/42 · 30 · 33 · 3 · 64 | 0,083 | 0,0122–0,0219 | **PASS** |
| idem, contra B | 42/42 · 64 · 48 · 64 · 64 | 0,103 | 0,0083–0,0177 | **PASS** |
| `--window 1`, routing no host, contra A | 26/42 · 64 · 33 · 3 · 64 | 0,080 | 0,0102–0,0231 | PASS |
| `--window 1`, `STRATA_POLY_MMQ=0`, contra A | 26/42 · 30 · 14 · 3 · 64 | 0,167 | 0,0075–0,0264 | FAIL (`code`, passo 30) |
| idem, contra B | 42/42 · 64 · 31 · 3 · 64 | 0,101 | 0,0100–0,0206 | PASS |
| `--window 1`, todos os especialistas na CPU, contra A | 26/42 · 30 · 14 · 64 · 64 | 0,133 | 0,0069–0,0219 | FAIL (`code`, passo 30) |
| Por omissão (MTP), só ids, contra A | 26/42 · 30 · 33 · 64 · 64 | — | — | FAIL (`code`, passo 30) |
| `--no-draft`, só ids, contra A | 26/42 · 30 · 33 · 47 · 64 | — | — | FAIL (`code`, passo 30) |

Todas as outras separações são quase-empates. Leitura:
- **O passo 30 do `code`:**

  | | Escolhe | Primeiro / segundo |
  | --- | --- | --- |
  | llama.cpp A (especialistas na CPU) | `15992` | 0,545 / 0,414 |
  | llama.cpp B (especialistas na GPU) | `3431` | 0,482 / 0,477 (quase-empate) |
  | PolyStrata, perfil | `3431` | 0,493 / 0,462 |
  | PolyStrata sem MMQ | `3431` | 0,581 / 0,380 |
  | PolyStrata com especialistas na CPU | `3431` | 0,547 / 0,414 |

  As duas referências do llama.cpp não concordam neste passo, e o PolyStrata escolhe sempre o token de B. As
  configurações que falham contra A passam contra B. É um passo sensível do modelo, do tipo do passo 29 da fase
  anterior, e não um defeito.
- **Médias mais altas do que no Qwen3.6** (0,012–0,022 contra 0,003–0,007): o próprio llama.cpp, entre B e A, também
  difere 3–4× mais neste ficheiro do que no Qwen3.6 (0,005–0,008 contra 0,0015–0,0023). Com pesos densos a 3–4 bits,
  o modelo é mais sensível a qualquer diferença de cálculo. A razão para o ruído do llama.cpp (~2×) é a mesma do
  Qwen3.6.
- **Sem MMQ** a média quase não muda (0,0127 contra 0,0138 no `doc-30k`) e o prefill cai para 139 tok/s. O erro maior do
  Q4_K em lote não pesa nos tokens.
- **MTP:** o bloco acertou 126 de 166 palpites (76 %) na paridade e escreve os mesmos tokens do `--window 1`.

## 6. Benchmarks

### 6.1 Os dois modelos no mesmo dia

Método das fases anteriores (`tools/hip/bench_compare.py`):
- os mesmos prompts de ~2K, 8K, 31K e 45K (`2026-10-10-hip-poly-vs-llama-vulkan/prompts.json`). O texto que o template
  do Empero dá para uma mensagem de utilizador é igual ao do Qwen3.6, por isso os ids servem aos dois;
- contexto 49 152, 6 threads, 128 tokens greedy;
- aquecimento não contado de cada modelo, depois 3 rondas alternadas: Qwen3.6 por omissão, Empero por omissão,
  Empero `--window 1`.

Mediana (intervalo em `bench-summary.md`), prefill / decode tok/s:

| Prompt | Qwen3.6 UD-Q3_K_M (MTP) | Empero por omissão (MTP) | **Empero `--window 1` (perfil)** |
| --- | --- | --- | --- |
| Primeiro pedido (A-2k) | 215 / 62,0 | 317 / 53,8 | 310 / **83,0** |
| A-8k (a frio) | 1 389 / 77,8 | 1 450 / 77,4 | 1 496 / 86,1 |
| B-2k (a quente) | 1 116 / 91,0 | 1 218 / 90,4 | 1 271 / **92,1** |
| B-8k | 1 423 / 86,2 | 1 448 / 84,2 | 1 488 / **87,0** |
| B-31k | 1 324 / **70,6** | 1 304 / 66,8 | 1 326 / 67,8 |
| B-45k | 1 221 / **62,4** | 1 179 / 56,0 | 1 209 / 60,6 |

| | Qwen3.6 (MTP) | Empero (MTP) | Empero `--window 1` |
| --- | --- | --- | --- |
| TTFT B-2k / 8k / 31k / 45k (s) | 1,86 / 5,79 / 23,6 / 37,2 | 1,70 / 5,69 / 24,0 / 38,5 | 1,63 / 5,54 / 23,6 / 37,5 |
| Carga (até READY) | 29,9 s | 28,9 s | 26,2 s |
| Pico de VRAM do processo (dedicada / partilhada) | 12 906 / 2 042 MiB | 12 935 / 2 106 MiB | 12 853 / 2 034 MiB |
| RAM: working set / "privada" | 11,7 / 11,7 GB | 10,1 / 12,6 GB | **8,8** / 12,8 GB |

Leitura:
- **Conversas e código até ~8K:** o Empero (perfil) é igual ou um pouco mais rápido do que o Qwen3.6, sem MTP.
  - Tem mais especialistas na VRAM, e os pesos densos são menores (1,2 contra 2,0 GB lidos por token).
  - O primeiro pedido de cada processo é muito mais rápido: 83 contra 62 tok/s.
- **Prompts longos (31K–45K):** o Qwen3.6 fica 3–4 % à frente, graças ao MTP, que nele compensa.
- **Prefill:** igual (diferenças ≤ 10 %, a favor do Empero nos curtos).
- **Memória:** working set menor (8,8 contra 11,7 GB); VRAM igual (o motor ocupa o que há).

A tabela da migração (11/10) dava ao Qwen3.6 93,4 / 85,8 / 70,7 / 61,9 a quente: as medidas de hoje (91,0 / 86,2 / 70,6
/ 62,4) repetem-nas.

### 6.2 O MTP do Empero

Palpites por pedido (`bench.json`): o bloco acertou 1 170 de 1 418 palpites (82 %; 88 % a quente), mas o decode com
palpites nunca passa o `--window 1`. Nas rondas 1 e 3, o motor quase deixou de pedir palpites (4–8 por pedido até 31K). É a política `Gate`
(`src/poly/guesses.hpp`): mede tokens por milissegundo com e sem o bloco e desliga-o quando não compensa.

Para separar as causas (`bench-mtp-diag.json`):

| Corrida | A-2k (frio) | B-2k | B-8k | B-31k | B-45k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Empero `--window 1` (mediana de 3) | 83,0 | 92,1 | 87,0 | 67,8 | 60,6 |
| Empero, MTP do próprio ficheiro (Q8_0) | 53,8 | 90,4 | 84,2 | 66,8 | 56,0 |
| Empero, MTP do GGUF do Qwen3.6 (`--draft-model`, Q3_K/Q4_K) | 71,7 | 91,2 | 83,5 | 65,5 | 60,1 |
| Empero `--no-draft` (só palpites por repetição) | 79,5 | 94,4 | 88,3 | 67,7 | 60,8 |

- Com o bloco menor do Qwen3.6, o primeiro pedido melhora: o bloco Q8_0 (0,84 GiB, 2× o do Qwen3.6) pesa no arranque.
- **A quente continua sem ganho**, mesmo com 86 % de palpites certos (69 de 80 a 2K). A causa principal são as janelas
  de verificação no modelo principal:
  - com os pesos densos em IQ3_S/Q4_K, verificar vários tokens de uma vez custa quase o mesmo por token do que um token
    sozinho;
  - o motor tem kernels de várias colunas específicos para Q8_0 (`q8_packed`, `native_mmvq_pair`), o formato denso do
    Qwen3.6 Unsloth, e para IQ4_XS/Q4_K–Q6_K (`native_mmvq_il`), mas não para IQ3_S.
- Escrever esses kernels para IQ3_S seria otimização, fora do âmbito desta fase. Não foi feito.

Por isso o perfil do Empero usa **`--window 1`**: foi o mais rápido em todas as medidas e passou a paridade contra as
duas referências. É configuração do perfil novo; o Qwen3.6 continua com MTP.

## 7. Contexto longo (48K)

`tools/hip/kv_needles.py`: três notas a 5, 50 e 95 % de documentos do repositório, greedy, sem pensamento, cada nota
sozinha e depois as três juntas (`longctx/`).

| Modelo | Prompt | Notas | Prefill (1.º pedido) | Decode no resumo |
| --- | ---: | --- | ---: | ---: |
| Empero (perfil) | 8 175 | 6/6 | 1 337 | 78,1 |
| Empero | 32 751 | **4/6** (as 3 sozinhas ✓; juntas: 1) | 1 311 | 63,3 |
| Empero | 44 983 | **4/6** (idem) | 1 210 | 58,1 |
| Qwen3.6 (perfil, MTP) | 8 175 / 32 751 / 44 983 | 6/6 · 6/6 · 6/6 | 1 045 / 1 249 / 1 193 | 49,2 / 52,3 / 50,9 |

Cada nota perguntada sozinha foi encontrada em todos os comprimentos: a atenção e a KV leem todas as posições. Na
pergunta pelas três, o Empero responde "4817-PLUM" e termina. Com os ids exatos do teste a 32K
(`longctx/emp-ctx32k-combined*.err`), o passo a seguir dá:
- fim da resposta 0,386, `,` 0,278, ` HER` 0,144;
- com e sem MTP, o mesmo.

O Qwen3.6 tinha ali `,` 0,346 contra fim 0,298 (fase anterior). É uma preferência do modelo, e o cartão avisa que o
treino usou exemplos de 8 192 tokens.

## 8. API, ferramentas e tarefa de programação

### 8.1 API (`tools/hip/poly_api_check.py`, perfil de 48K, `api-check-empero.json`)

14 de 15, e a 15.ª é um falso negativo da heurística do teste:
- ok: modelos, system prompt, streaming (1.º token a 0,34 s), multi-turno;
- ok: ferramentas com e sem streaming, ferramentas Anthropic, Anthropic com e sem streaming;
- ok: resposta longa (1 797 tokens a **77,0 tok/s**), cancelamento (pedido seguinte em 0,42 s);
- ok: raciocínio e raciocínio em streaming, título (0,65 s);
- ok: sessão de 12 turnos, todos certos e a reutilizar a cache;
- `preserve-thinking` "FAIL", mas o mecanismo funciona:
  - com a opção, o 2.º turno lê 137 tokens (≥ os 115 do 1.º) e reutiliza 114;
  - sem ela, lê 59. A diferença, 78, é exatamente o raciocínio do 1.º turno;
  - o teste exige menos de metade do 1.º turno (57), e o Empero raciocinou pouco.

### 8.2 Tarefa de programação (OpenCode v2.0.18, `codetask/`)

- **Fixture:** `textstats.py` com 4 erros que fazem falhar 5 de 8 testes (minúsculas, ordenação, frases vazias, a
  última linha do `wrap`).
- **Pedido:** corrigir sem mudar os testes e corrê-los.
- **Cada corrida:**
  - uma cópia nova num repositório git próprio;
  - `opencode run --standalone --auto`, com a configuração do fornecedor só na pasta da corrida;
  - no fim, os testes são corridos à parte e é confirmado que o ficheiro de testes não mudou.

| Modelo e raciocínio | Passou | Tempo (s) | Chamadas de ferramentas |
| --- | --- | --- | --- |
| Qwen3.6, `medium` | 4/4 | 143*, 85, 63, 82 | 9–11 |
| Qwen3.6, `none` | 2/2 | 94, 59 | 14–24 |
| **Empero, `medium`, amostragem do perfil** | **4/4** | 106, 73, 49, 75 | 10–11 |
| Empero, `none`, amostragem do perfil | 1/2 | 56, 16 | 12, 0 |
| Empero, `medium`, greedy (antes do bloco `sampling`) | 5/7 | as 5 que passaram: 62–109; as 2 que falharam: 110 e 513 | 8–13 |
| Empero, `none`, greedy | 1/2 | 87; 28 (falhou) | 13, 0 |

\* inclui a troca de modelo (secção 10.3).

- **As falhas sem raciocínio** são sempre iguais: o modelo escreve "Let me start by reading the test file…" e termina o
  turno sem chamar a ferramenta (20 tokens, `stop`). Não há chamada mal formada.
- Com o Empero, o raciocínio `low`/`medium` é o recomendado para uso com ferramentas.

### 8.3 As degenerações em greedy: causa

Duas corridas `medium` em greedy pararam no mesmo ponto lógico. O modelo não viu que o `wrap()` perde a última linha,
convenceu-se de que "devia funcionar" e entrou em ciclo (`codetask/degenerate/`):
- num caso, 106 661 caracteres de frases repetidas (27 853 tokens, até ao limite);
- no outro, depois de ~4 000 caracteres do mesmo ciclo, um colapso em `^^^^`. A proteção `repeat_stop_tokens` do
  servidor cortou-o aos 256 tokens.

Para separar modelo e motor:
1. Um proxy de registo (`scripts/logproxy.py`) mostrou que **o OpenCode não envia `temperature`, `top_p` nem
   `top_k`**. Sem bloco `sampling` na configuração, o servidor gera por greedy. O cartão do modelo avisa que greedy em
   gerações longas é um modo conhecido de ciclos de repetição.
2. O pedido exato que degenerou foi reconstruído da sessão gravada pelo OpenCode e de um pedido gravado pelo proxy
   (`replay/request.json`, 23 610 tokens) e repetido:

   | Motor | Greedy | Amostragem do cartão (sementes) |
   | --- | --- | --- |
   | PolyStrata | 6 000 tokens sem sair do raciocínio | 1 e 3: ferramenta chamada; 2: 6 000 tokens |
   | **llama.cpp b11160, mesmo GGUF** | ferramenta chamada após 2 685 tokens | 1, 3, 4: ferramenta; **2: 19 360 `^` seguidos**; 5: ciclo de frases (57 % de linhas repetidas) |

   O colapso em `^` vem da linha de marcadores (`~~~~^^^^`) que o Python 3.11+ escreve nos tracebacks e que o modelo
   copia. Os dois motores degeneram da mesma maneira, com taxas parecidas, no mesmo pedido. O greedy de cada um segue
   outra trajetória depois do primeiro quase-empate, como na paridade.

**Conclusão:** é comportamento do modelo (destilado, IQ3_M), não um defeito da integração.

**Mitigação aplicada** (só no perfil do Empero): `"sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}`, o que o
cartão recomenda, para os pedidos que não indicam amostragem. Um pedido que a indique ganha sempre (`temperature: 0`
continua a ser greedy). Com ela, 4/4 na tarefa e 0 degenerações nos 727 pedidos da secção 9. As repetições acima
mostram que **não elimina** o problema: num ponto difícil, o modelo pode entrar em ciclo também com amostragem.

## 9. Estabilidade

`scripts/soak.py`, 20 min seguidos no perfil do Empero (`soak-empero.json`). Pedidos em ciclo:
- curto;
- com raciocínio e resposta aritmética verificada;
- chamada de ferramenta com argumento verificado;
- documento de ~8K tokens;
- streaming cancelado a meio.

| | Resultado |
| --- | --- |
| Pedidos | **727, 0 erros, 0 respostas erradas** (182 curtos, 181 com raciocínio, 182 ferramentas, 91 documentos, 91 cancelamentos) |
| Processo | vivo no fim |
| VRAM dedicada / partilhada | 12 535 / 1 362 MiB, iguais do 1.º ao último minuto |
| RAM privada | 11 473 → 11 603 MiB (+130) |
| Working set | 8,5 → 9,5 GB (máximo 9,7): páginas do GGUF mapeado |

## 10. Integração

### 10.1 Configurações

| Onde | O quê |
| --- | --- |
| `strata-qwen3.8-35b-a3b-distill-iq3_m-hip.json` (raiz, excluída do git) | Escrita por `tools/hip/poly_config.py --name qwen3.8-35b-a3b-distill-iq3_m-hip --context 49152 --threads 6`. Depois: `display_name` "Qwen3.8-35B-A3B-Distill IQ3_M (Empero, HIP, experimental)", `--window 1` e o bloco `sampling`. Porta 8090, `switch_on_request`, `--turn-token 248045` |
| `C:\models\qwen3.8-35b-a3b-distill-iq3_m-hip-pack\` | tokenizer do GGUF; perfil de especialistas novo, aprendido pelo uso |
| As 4 configurações do Qwen3.6 | **iguais byte a byte** (SHA-256 comparado antes e depois); o Qwen3.6 continua com MTP e greedy por omissão |

### 10.2 Ficheiros do repositório alterados

| Ficheiro | Alteração |
| --- | --- |
| `START-POLY-HIP.bat` | O comentário do cabeçalho passa a nomear os dois modelos. No menu de contexto, o Empero mostra "~92 tok/s numa conversa curta" no 48K; os outros contextos dele dizem "velocidade nao medida". O menu de modelos já existia e volta a aparecer (há dois); o Qwen3.6 é o 1 e continua a omissão (`last-model.txt`) |
| `tools/hip/poly_profile.py` | Um perfil novo criado a partir da configuração base herda o `--window` e o bloco `sampling` dela. Para o Qwen3.6, que não tem nenhum dos dois, nada muda (testado: as 4 configurações ficaram iguais) |
| `tools/hip/opencode-polystrata-hip.json` | Modelo `qwen3.8-35b-a3b-distill` → `qwen3.8-35b-a3b-distill-iq3_m-hip`, 48K, variantes none/low/medium. As entradas do Qwen3.6 não mudaram |

Sem a alteração ao `poly_profile.py`, escolher 128K para o Empero no menu criaria um perfil com MTP e greedy. Foi o
que aconteceu num `START-POLY-HIP.bat check`: o `poly_profile.py` escreve a configuração mesmo em `check`, e o
ficheiro foi apagado. **Só o perfil de 48K do Empero existe.**

### 10.3 OpenCode e OpenChamber

- **Configurações alteradas** (por `scripts/opencode_add_empero.py`, que só acrescenta uma chave e compara o resto por
  programa):
  - `opencode.json` local (excluído do git);
  - `~/.config/opencode/opencode.json` global, lido pelo OpenChamber, com **cópia de segurança** em
    `opencode.json.backup-20261012-empero`;
  - o resultado: uma entrada nova `qwen3.8-35b-a3b-distill` em `polystrata-hip.models`, com as variantes da entrada de
    48K do Qwen3.6 (incluindo `high`); o resto é idêntico, incluindo a ordem.
- **Teste com a configuração global**, numa pasta sem configuração de projeto, com o servidor arrancado no Qwen3.6
  (`logs/opencode-*.txt`):

  | Passo | Resultado |
  | --- | --- |
  | PONG no Empero (`#none`) | ok, com troca automática Qwen3.6 → Empero, 80 s |
  | Ler `notes.yaml` com a ferramenta (`#medium`, `--auto`) | ok: `read` chamado, "HERON-62", 26 s |
  | Continuação (`--continue`) | ok: "02:00–03:00" a partir do que já tinha lido, 10 s |
  | PONG no Qwen3.6 | ok, com troca de volta, 89 s |

- **As trocas levam o dobro do tempo:** cada uma acontece duas vezes no log (`X → Y`, depois `Y → Y`):
  - causa: `switch_model` (`serve/server.py`) verifica se o alvo já é o modelo atual antes de obter o lock;
  - o OpenCode envia o pedido do título e o da tarefa ao mesmo tempo. O segundo espera pelo lock e recarrega o mesmo
    modelo (~30 s);
  - é anterior a esta fase e afeta qualquer par de modelos. **Não foi corrigido** (fora do âmbito); ficou registado
    como tarefa separada.
  - Entre pedidos do mesmo modelo não há troca.

## 11. Problemas encontrados

| Problema | Causa | Estado |
| --- | --- | --- |
| MTP do Empero mais lento do que sem palpites | janelas de verificação caras com pesos densos IQ3_S/Q4_K (sem kernels de várias colunas para IQ3_S); bloco MTP em Q8_0 | Perfil com `--window 1`; kernels não escritos (otimização) |
| Ciclos de repetição em greedy (raciocínio longo) | o modelo; reproduzido no llama.cpp com o mesmo pedido | Amostragem do cartão no perfil; mitiga, não elimina |
| Turno terminado sem chamar a ferramenta, sem raciocínio | o modelo (2 em 4 corridas `none`) | Recomendado `low`/`medium` |
| Pergunta pelas 3 notas a 32K/45K responde só 1 | o modelo (fim 0,386 contra `,` 0,278) | Documentado |
| `code`, passo 30: FAIL contra A sem MMQ / com especialistas na CPU | as referências A e B do llama.cpp discordam nesse passo | Documentado; o perfil passa contra as duas |
| `preserve-thinking` FAIL no `poly_api_check` | heurística do teste (raciocínio curto) | O mecanismo funciona; teste não alterado |
| Cada troca de modelo carrega duas vezes | verificação antes do lock em `switch_model` | Registado, não corrigido |
| `START-POLY-HIP.bat check` escreve a configuração do perfil | `poly_profile.py` escreve sempre | O ficheiro criado foi apagado; comportamento não alterado |
| A tarefa no OpenCode lia a pasta errada | o OpenCode (Bun) usa a variável `PWD`, que o Git Bash define, e não o `cwd` do processo | `PWD` passado em cada corrida; uma corrida contaminada foi anulada e a fixture reposta (SHA-256 em `codetask/fixture.sha256`) |

## 12. O que não foi testado

- Perfis de 128K, 262K e 262K stream do Empero: não existem (o pedido era 48K).
- Qualidade geral das respostas (benchmarks de tarefas): só uma tarefa de programação, as notas e a paridade.
- O MTP do Empero com um bloco requantizado (por exemplo Q4_K): só o do próprio ficheiro e o do Qwen3.6.
- Imagens (`mmproj`, não descarregado).
- A interface gráfica do OpenChamber; foi usado o executável do OpenCode que ela traz, com a configuração global.
- O OpenCode v2.0.3 global do npm.
- ctest, testes do servidor e `poly_kernel_bench`: nada no motor nem no servidor mudou. Correram
  `tools/test_setup_amd.py` (36 OK), `tools/test_setup_choices.py` (40 OK), `serve/test_runconfig.py` (9 OK) e
  `tools/test_polystrata_names.py`; este falha só no `AMD_KV_STREAMING.md`, como antes desta fase.
- Uso de horas seguidas (20 min no total).

## 13. Como usar

`START-POLY-HIP.bat` (duplo clique):
1. menu de modelo: **1** Qwen3.6 (omissão), **2** Empero;
2. menu de contexto: para o Empero, só o **48K** foi preparado e medido.

Sem menus:

```bat
START-POLY-HIP.bat qwen3.8 48k
START-POLY-HIP.bat qwen3.6 48k
```

Com o servidor a correr, um pedido que nomeie o outro modelo troca-o sozinho (~30 s, ou o dobro com o OpenCode,
secção 10.3).

No OpenCode/OpenChamber, fornecedor "PolyStrata (HIP, RX 9060 XT)":
- "Qwen3.8 35B A3B Distill IQ3_M (Empero) - 48K", com raciocínio `low` ou `medium` para trabalho com ferramentas;
- os modelos do Qwen3.6 continuam lá, iguais.

Para desfazer:
- apagar `strata-qwen3.8-35b-a3b-distill-iq3_m-hip.json` da raiz do repositório;
- repor `~/.config/opencode/opencode.json.backup-20261012-empero`.
