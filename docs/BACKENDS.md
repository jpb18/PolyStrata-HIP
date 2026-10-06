# Models and backends

PolyStrata runs more than one model family. This page says which, how the engine is put together so that it can,
how a model is recognized, how to add one, and how to change the model that answers.

One model is in memory at a time. The measurements below are from one machine: an RTX 5090 (32 GB), 24 threads of
a Xeon Platinum 8470Q, 120 GiB of RAM, Linux, CUDA 12.8. On that card the three new models fit in VRAM whole. A
card with 8 GB was not there: [one table](#with-8-gb-of-vram) holds the 5090 to 8 GB instead.

## What runs

| Model | Architecture | Engine | File it was checked with | Guesses ahead |
| --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next and fine-tunes | `qwen4exp` | `strata` | the packs setup builds | separate MTP model |
| GLM 5.3 Flash | `glm5-next` | `strata-poly` | GSQ-RCO 3.0bit, REAP50-Q3_K_M, UD-IQ2_XXS | draft block |
| Qwen3.6-35B-A3B | `qwen35moe` | `strata-poly` | Unsloth UD-Q3_K_M, 17.1 GB | draft block in the file |
| Ornith 1.5 35B-A3B | `qwen35moe` | `strata-poly` | APEX-MTP-I-Compact, 17.4 GB | draft block in the file |
| Gemma 4 26B-A4B | `gemma4` | `strata-poly` | Unsloth UD-Q3_K_M, 12.7 GB | its draft model, a file of 0.46 GB |

`strata` is the first engine and runs Qwen3.8 only; nothing in it changed. `strata-poly` is the engine that reads a
GGUF in place, and it now has three architectures. It was called `strata-glm`, after its first model; an install
from then keeps its engine and moves to the new name the next time setup updates it.

## Install

Setup installs all five. The three new models are in its list of models, and each has a family name that goes
straight to it:

```bash
./setup.sh --family qwen36
```

```bash
./setup.sh --family ornith
```

```bash
./setup.sh --family gemma4
```

| Family | Model | File | Download | RAM beside an 8 GB card (estimate) |
| --- | --- | --- | --- | --- |
| `qwen36` | Qwen3.6-35B-A3B | Unsloth UD-Q3_K_M | 17.1 GB | about 21 GB |
| `ornith` | Ornith 1.5 35B-A3B | APEX-MTP-I-Compact | 17.4 GB | about 22 GB |
| `gemma4` | Gemma 4 26B-A4B | Unsloth UD-Q3_K_M and its draft model | 13.2 GB | about 18 GB |

The steps are GLM's ([GLM.md](GLM.md#install)). Setup asks the context and whether to keep several conversations in
RAM, compiles `strata-poly`, downloads the model's file from a pinned revision (for Gemma 4 also its draft model,
0.46 GB), checks the SHA-256, writes the tokenizer directory, the run config and a start script, and starts the
model. `--gguf-dir /folder/with/the/file` takes a file you already have; a draft model that is not in that folder is
downloaded to setup's own. The engine is compiled once for all of them. On the test machine, with the files on the
disk, the first install took a minute, most of it the compile, and the other two 17 seconds each.

Each install is a model of its own: `--list-models`, `/model` and the web page's list show all of them, and
`./setup.sh --check` has a line for each with what this PC's card and RAM make of it.

The three new models read pictures. Say yes to setup's question about images, or pass `--vision yes`:

```bash
./setup.sh --family gemma4 --vision yes
```

Setup then downloads the model's image encoder too (0.9 GB, Gemma 4's 1.2 GB), compiles `strata-vision` beside
the engine and writes both into the run config. On the GPU the encoder holds 1.6 GB of VRAM (Gemma 4's 2.4 GB),
which a small card needs for the model, so on a card with less than 16 GB it runs on the CPU and a picture takes
a few seconds. `--vision gpu` and `--vision cpu` choose the place. GLM 5.3 Flash reads no pictures.

The RAM column is the experts an 8 GB card does not hold, plus 10 GB for the rest. It is setup's estimate and was
not checked on such a PC. Setup does not stop a PC below the 32 GB that Qwen3.8 needs when one of these families is
named.

## How a backend works

There are two levels.

**The process.** The server starts an engine as a child process and talks to it over lines of text: `GEN` with
token ids goes in, `T` lines with tokens and a `DONE` line with timings come out. Any program that speaks those
lines is a backend to the server. Changing the model stops the child and starts another, so its VRAM, its pinned
memory and its file mappings are given back by the operating system when the process ends, not by cleanup code
that could miss something.

A picture goes through a second child, `strata-vision`: llama.cpp's image code with the model's encoder file. It
turns the picture into rows as wide as a token's embedding. The server puts one placeholder token a row into the
prompt and sends `GENI` with the rows' file and `image_pad=`, the placeholder's id; the engine reads a row where
a placeholder stands. The template says where a picture goes: Qwen's writes the tokens around it, Gemma 4's
writes one token and the server adds the two that llama.cpp puts around the rows.

**The engine.** Inside `strata-poly`, `Engine` (src/poly/engine.hpp) is what every architecture shares:

- the GGUF, read in place, in one file or several
- the dense weights in VRAM in the type the file has them
- the routed experts: a VRAM cache filled by how much each is used, experts the cache lacks computed on the CPU
  while the card does its share, and the cache following what an answer uses
- a prompt batch: every token's experts on the card with llama.cpp's batched kernels, through a staging buffer
  for the experts that are not cached
- the windows of guessed tokens and their check, the draft block that guesses, and guesses from repeats
- the conversation state: checkpoints, parked conversations
- CUDA graphs for a token's fixed launches
- the serve protocol, sampling, statistics

An architecture is a class derived from `Engine` (`Glm5Next`, `Qwen35`, `Gemma4`). It supplies:

| Hook | What it does |
| --- | --- |
| `load_geometry()` | the model's shape from the metadata; `load_shape()` reads what all have |
| `load_layer(il)` | a layer's weights to VRAM; `load_moe()` loads a MoE layer's router and finds its experts |
| `recurrent_floats()`, `alloc_caches()`, `assign_scratch()`, `widest()` | its state, its attention caches, its scratch |
| `layer_head()`, `layer_tail()` | a layer up to its router, and what happens to the experts' sum |
| `route()`, `moe_shared()`, `router_scores()` | the routing rule, the shared expert, the router's input |
| `output_rows()`, `finish_logits()`, `embed_scale()` | the output norm, a cap on the logits, a scale on embeddings |
| `block_attend()`, `block_rows()` | the draft block's attention |
| `sees()`, `whole()` | whether it reads pictures, and how many of a prompt's rows the next call takes |
| `row_bytes()`, `save_rows()`, `load_rows()`, `extra_bytes()`, `save_extra()`, `load_extra()` | what a conversation keeps |

A layer's own weights live in the class's `Block`, one a layer, beside the shared `Layer`. The engine calls a hook
once a layer or once a call, never inside a kernel, so the calls cost nothing that can be measured: GLM writes at
the same speed before and after the split (51.8, 50.6 and 46.1 tokens/s against 50.7, 49.5 and 45.8).

Nothing in the shared code asks which model it runs. What differs between architectures is a hook or a value the
architecture sets (`gelu_` for the experts' activation, the distance between experts in a fused tensor).

Pictures show it. The shared code puts a picture's rows where token embeddings go and gives the conversation
cache a number for each row made from the picture, so that it tells one picture from another of the same size.
The rest is the architecture's. Qwen3.6's rotary embedding has three axes: text has one position on all of them,
a picture's rows share a position on the first and add their line and their column on the other two, and the
text after a picture goes on as many positions later as the picture has lines or columns, whichever is more.
Gemma 4 counts a picture's rows like text, lets them see each other in its window layers and keeps its full
layers causal, so it takes a picture in a call of its own (`whole()`). Both are what llama.cpp does.

## How a model is recognized

A GGUF names its architecture in `general.architecture`. `Engine::open()` reads it and makes the class for it; a
file of an architecture the engine does not have is refused with its name. The server's registry reads the same
key, so a model the engine cannot run is listed as unavailable with the reason instead of failing at a switch.

| `general.architecture` | Class |
| --- | --- |
| `glm5-next`, `glm5next` | `Glm5Next` |
| `qwen35moe` | `Qwen35` |
| `gemma4` | `Gemma4` |

Within an architecture the shape comes from the file: layer counts, widths, head counts, expert counts, which
layers have which attention, whether there is a draft block. Qwen3.6 and Ornith 1.5 are one class because their
files describe the same layout with other weights.

The server takes how a model talks from its tokenizer directory: the chat template, the tokens that end a reply,
and, by the tokenizer's `pre` name, how the model writes its reasoning and its tool calls (`serve/frontend.py`).

| Model | Reasoning | A tool call |
| --- | --- | --- |
| Qwen3.8, Qwen3.6, Ornith | the prompt opens it, `</think>` ends it | `<tool_call><function=NAME><parameter=P>...` |
| GLM | the same | `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...` |
| Gemma 4 | `<\|channel>thought` to `<channel\|>`, opened by the model | `<\|tool_call>call:NAME{key:value,...}<tool_call\|>` |

Gemma 4's template thinks only when it is told to, reads a tool in OpenAI's shape and starts with the tokenizer's
`<bos>`; the server gives it all three. A tool's answers are put in the order of their calls for every model,
because a model reads them by position.

## The registry

`serve/models.py` lists the installation's models. A model is a run config, `strata-<name>.json`, beside
`setup.py`: the engine, its arguments, the tokenizer directory. An entry has:

| Field | From |
| --- | --- |
| `id` | the config's `model_name` |
| `display_name` | the config's `display_name` (setup writes it), else the GGUF's `general.name` |
| `backend` | the GGUF's architecture (`qwen4exp` for a pack) |
| `path` | the config's `--model` |
| `quantization` | the file's name (`UD-Q3_K_M`, `3.0bit`), else its most common tensor type |
| `params_total`, `params_active` | the GGUF's size label (`35B-A3B`) |
| `context` | the config's `--max-context` |
| `multimodal` | the config has a vision encoder |
| `mtp` | the file has a draft block, or the config names one |
| `vram_gb` | `least`: the dense weights and 1 GB; `all`: with every expert too |
| `ram_gb` | the file's size: it is read in place, so that much file cache at most |
| `status` | `available`, `unavailable` with `problem`, and for the model in use `loaded` or `unloaded` |

`models.json` in the same folder may add to an entry or correct it:

```json
{"qwen3.6-35b-a3b-ud-q3_k_m": {"display_name": "Qwen3.6 35B-A3B", "notes": "the daily one"}}
```

The estimates are rough. `least` leaves out the attention caches, which grow with the context.

## Changing the model

From a terminal:

```bash
python polystrata.py --list-models
```

```bash
python polystrata.py --model qwen3.6
```

`--model` chats with a model. If a server is already running on the port it is switched to the model; if none is,
one is started and stopped again when the chat ends. In the chat:

| Command | |
| --- | --- |
| `/model` | the model that answers |
| `/model list` | the installed models |
| `/model switch <name>` | change to another; a new conversation starts |
| `/model info [<name>]` | what a model is |
| `/model unload` | give its memory back until the next message |

A part of a name is enough when only one name has it.

In the web page the header has a list of the models when more than one is installed.

Over HTTP:

| | |
| --- | --- |
| `GET /api/models` | every model, the one in use marked `current` |
| `GET /api/models/current` | the one in use, and whether it is loaded |
| `POST /api/models/load` | `{"model": "<name>"}` |
| `POST /api/models/unload` | stop the engine; the next request starts it again |

All three ways go through `Service.switch_model()` in the server. A switch:

1. finds the model and checks that its engine, tokenizer and file are there
2. waits for a running request, up to two minutes, and is refused after that
3. stops the engine that is loaded, so the two models are never in memory together
4. starts the new engine and waits until it is ready
5. on failure starts the earlier model again and says which one is loaded
6. on success takes over the new model's tokenizer, chat template, stop tokens, sampling defaults and aliases

Measured: 8 to 16 s a switch between the four models on the machine above, with their files in the page cache.

With `"memory_debug": true` in a run config, or `STRATA_MEMORY_DEBUG=1`, a switch prints the free VRAM before it,
after the old engine ended and with the new one:

```
[strata] memory: 18095 MiB of VRAM free with gemma-4-26b-a4b, 32109 MiB after it stopped, 12197 MiB with qwen3.6-35b-a3b
```

## Benchmark

```bash
python polystrata.py benchmark --model qwen3.6 --json qwen36.json
```

It sends a short prompt and two long ones (2000 and 8000 tokens, roughly; `--prompt-tokens` changes them), 256
tokens of answer each, greedy, and prints for each: the time to the first token, the prefill and decode speeds,
the expert cache's hit rate, the share of guessed tokens that were right, and the machine's highest VRAM and RAM
use and mean GPU and CPU load while it ran. The speeds and the cache figures are the engine's own clock and
counters.

Tokens per second on the machine above, context 32768, every expert in VRAM for the three new models, as setup
installs them. The first build of each backend read the longest prompt at 636, 632 and 213 tokens/s and wrote
after it at 68, 54 and 55; [What was tuned](#what-was-tuned) has the steps between.

| Model | Prompt, tokens | First token, s | Reads, tokens/s | Writes, tokens/s | Experts from VRAM | Guesses right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 21 | 0.11 | 204 | 253 | 100% | 85% |
| | 2,568 | 0.76 | 3,468 | 192 | 100% | 73% |
| | 10,039 | 2.4 | 3,442 | 161 | 100% | 85% |
| Ornith 1.5 | 21 | 0.09 | 248 | 194 | 100% | 91% |
| | 2,568 | 0.72 | 3,704 | 167 | 100% | 80% |
| | 10,039 | 2.3 | 3,579 | 134 | 100% | 56% |
| Gemma 4 26B-A4B | 26 | 0.22 | 133 | 206 | 100% | 67% |
| | 2,656 | 0.89 | 3,156 | 185 | 100% | 75% |
| | 10,393 | 2.9 | 3,006 | 127 | 100% | 62% |
| GLM 5.3 Flash 3.0bit | 23 | 0.85 | 27 | 43 | 49% | 85% |
| | 2,424 | 7.3 | 331 | 43 | 49% | 77% |
| | 9,446 | 23.1 | 321 | 46 | 58% | 82% |

A short prompt is read a few tokens at a time, which is why its reading speed is low. Gemma 4 thinks in this
table, as the others do, and guesses with its draft model. Before the last three changes to writing in
[What was tuned](#what-was-tuned), the three new models wrote 125, 102 and 122 tokens/s after the short prompt
and 95, 78 and 93 after the longest.

GLM's speeds are where they were before the changes listed there, when it wrote 43, 46 and 46 tokens/s in the
same test: half its experts are on the CPU, and they bound its tokens. A second run of the benchmark gave 43, 46
and 48. GLM runs here from a config written by hand, with its draft block and an expert profile.

GLM is sensitive to what the machine did before. After the other models had run, its first request came out at
28 to 35 tokens/s until its file was back in the file cache, and its starts were ended by a kill signal while the
pod's memory was at its limit with other files' pages, until those pages were dropped. Setup's build of the engine
wrote 38 to 43 tokens/s where the development build wrote 41 to 44, from the same sources; the two take ggml from
different checkouts, and the difference was not looked into.

llama.cpp (Unsloth's fork at 86ebfef) with every layer on the same card writes about 180 tokens/s with Qwen3.6
and Ornith and 165 with Gemma 4, and reads prompts at 6000 to 9000. It cannot do that on a card the model does
not fit, which is the case this engine is for.

### With 8 GB of VRAM

The same card with 24.4 GB kept free (`--vram-reserve-mib 24400`), so the engine has about 8 GB: the dense
weights, the caches and as many experts as fit. The rest of the experts are computed on the CPU, and the cache
follows what the answer uses. This is not an 8 GB PC: the CPU, the RAM and the card's speed are the machine's
above. 300-token answers, tokens per second; the two numbers are without and with guessed tokens.

| Model | Experts from VRAM | Writes after a short prompt | after about 2,100 tokens | after about 7,700 | Reads a long prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 43% to 61% | 97 to 106, 117 to 121 | 104, 121 | 82, 91 | 1,710 to 2,590 |
| Ornith 1.5 | 47% to 60% | 106 to 124, 101 to 109 | 127, 124 | 92, 90 | 1,890 to 2,690 |
| Gemma 4 26B-A4B | 48% to 61% | 101 to 102, 117 to 120 | 89, 104 | 78, 83 | 1,930 to 3,000 |

The short prompts are two of three: a model's first run can start with its file out of the file cache, and
Qwen3.6's wrote 77 tokens/s. With guesses a smaller share of the experts comes from VRAM (48% against 60% for
Gemma 4). Ornith writes a little slower with guesses than without: each of these runs is a process of its own,
and the engine asks the block in its first rounds, before it has measured that asking does not pay.

Here the changes to writing gave less than with everything in VRAM, because the experts on the CPU take most of
a token's time. Before them the three wrote 77 to 109 tokens/s without guesses after a short prompt and 87 to
118 with.

With part of the experts on the CPU, two runs of a prompt no longer write the same tokens: which experts are
cached changes while the answer is written, and an expert on the CPU differs from the same one on the card in the
last bits. In an earlier round of these runs one was ended by a kill signal while it read its prompt (Gemma 4,
2,212 tokens); four runs of the same command afterwards finished. The pod's memory was at its limit then (file
cache), as it was when a GLM start was killed later; the pod's own counter of memory kills stayed at zero, and the
cause was not found.

## How each backend was checked

Against llama.cpp on the same file: greedy, 64 tokens after five chat prompts of about 25 to 8000 tokens, the
token ids compared and the probabilities of the reference's five likeliest tokens at every step.

| Model | Same tokens as llama.cpp (of 64) | Largest probability difference |
| --- | --- | --- |
| Qwen3.6-35B-A3B | 64, 64, 64, 64, 64 | 0.07 |
| Ornith 1.5 | 60 of 60, 7, 39, 40, 64 | 0.48 |
| Gemma 4 26B-A4B | 64, 64, 32, 64, 64 | 0.07 |

Two correct implementations do not write the same tokens for long. Activations are quantized to 8 bits at every
quantized matrix, in a layout that depends on the kernel, so the two differ in the last bits and a near tie
between two tokens goes either way. Ornith's file has 3-bit experts and 4-bit attention and is the most sensitive:
llama.cpp's own GPU and CPU runs of it agree for 17 of 50, 7 of 64 and 15 of 64 tokens on the three short prompts
and differ by up to 0.19, which is as far apart as this engine is from either.

The numbers move with every change to a kernel that changes the order of a sum: before the last ones Ornith's
were 60, 7, 56, 2, 64 and Gemma 4's fourth was 44. Where a change kept the arithmetic (the first rewrite of the
delta net's step, the threads of the routing, the sum of a call's experts), the numbers and the tokens stayed as
they were, and that was the check of the change.

Also checked, for each model: with windows of guessed tokens and without, 300-token answers are the same tokens;
with every expert on the CPU the short prompts still follow the reference to a near tie. GLM writes the same
tokens as before the engine was split, with every expert on the CPU.

Gemma 4 with its draft model writes the same tokens as without it on nine prompts of 26 to 8,001 tokens, with and
without thinking and tools; on the prompts with thinking the engine's tokens are llama.cpp's for 96 of 96 and 82
of 82 and up to a near tie on two more. Its tool calls were run through the server with the real template: a
call and its answer with thinking on and off, a streamed call, two calls answered in the other order, the
Anthropic route, a follow-up after an answer with text. The same script passes with Qwen3.6, and with GLM but for
one check that expected reasoning where GLM gave none.

The tokenizers give llama.cpp's ids on the same prompts (27 to 8001 tokens) for Qwen3.6, Ornith and Gemma 4.

Pictures were checked the same way, greedy and up to 64 tokens, on three prompts: one picture; two pictures of
different shapes with text between and after them; a picture, an answer and a second question. PolyStrata's
server code rendered each prompt. llama.cpp's server got it as text with its media marker where a picture goes
and encoded the pictures itself; the engine got the ids and `strata-vision`'s rows. Both read the same number of
prompt tokens.

| Model | Picture rows in the prompt | Same tokens as llama.cpp | Largest probability difference |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | 280, 440, 160 | 11 of 11, 11 of 64, 64 of 64 | 0.05 |
| Ornith 1.5 | 280, 440, 160 | 17 of 17, 35 of 35, 59 of 59 | 0.37 |
| Gemma 4 26B-A4B | 117, 194, 77 | 14 of 14, 19 of 19, 55 of 55 | 0.10 |

Qwen3.6's second answer leaves llama.cpp's at a near tie (0.504 against 0.493 here, 0.518 against 0.480 there).
The differences are what the same models show on text.

A fourth prompt has the picture after a list of 180 lines, about 3,600 tokens: past a prompt batch, a checkpoint
and the rows Gemma 4's window layers keep. Qwen3.6 writes llama.cpp's 64 tokens and Gemma 4 its 31, with prompt
batches of 1,024 and of 4,096 tokens; each names the shapes and reads the right line of the list. Ornith leaves
at its fourth token, at a near tie. A fifth has 43 lines ahead of the picture, so that its rows start in one
prompt batch of Qwen3.6's and end in the next: 64 of 64 tokens.

For Gemma 4 the other ways a picture's rows could see each other were run against the same reference. With no
row seeing a later one, the second prompt's answer leaves llama.cpp's at its 14th token with probabilities 0.63
apart. With the rows seeing each other in the full layers too, the tokens stay the same and the average
difference on that prompt is twice what it is with the window layers alone (0.0077 against 0.0038).

Over the engine's protocol, with Qwen3.6 and Gemma 4: a conversation with a picture goes on from its last token
(332 of 342 and 166 of 176 tokens are not read again) and writes what a fresh engine writes; a mirrored copy of
the picture in the same place, which gives the same ids, is read from the start and answered for what it shows
(the shape on the left is the other one); the first conversation then comes back from the parked ones with its
tokens. Through the server each of the three models answered the three prompts, the first one with the other
picture in its place, and a prompt without pictures after them.

## What was tuned

| Change | Model | Before | After |
| --- | --- | --- | --- |
| The batched kernels also take matrices whose rows end inside one of their steps | Gemma 4, reading 8001 tokens | 229 tokens/s | 1216 |
| A prompt batch's attention through cuBLAS | Gemma 4, reading 8001 tokens | 1216 | 3200 |
| | Qwen3.6, reading 7709 tokens | 750 | 931 |
| The attention's sum over values in eight shares | Qwen3.6, writing after 7709 tokens | 57 | 78 |
| | Gemma 4, writing after 8001 tokens | 53 | 68 |
| A layer's cached experts in a few launches, by how many rows each has | Qwen3.6, reading 7709 tokens | 932 | 1,114 |
| The delta net's state update a warp a column, its norms outside the walk over the tokens | Qwen3.6, reading 7709 tokens | 1,114 | 3,189 |
| A prompt batch's routing on every host thread | Qwen3.6, reading 7709 tokens | 3,189 | 4,241 |
| | Gemma 4, reading 8001 tokens | 3,254 | 4,663 |
| One launch adds up a call's cached experts | Qwen3.6, writing after a short prompt | 122 to 129 | 132 to 138 |
| Gemma 4's draft model | Gemma 4, writing after a short prompt | 94 | 126 to 141 |
| | Gemma 4, writing after 8001 tokens | 73 | 99 |
| A call's cached experts eight at a time, a launch a stage for all of them | Qwen3.6, writing after a short prompt | 132 to 138 | 241 to 269 |
| | Gemma 4, writing after a short prompt | 126 to 141 | 220 to 232 |
| | Gemma 4, writing after 8001 tokens | 99 | 134 |
| Attention layers and windows of guessed tokens recorded, the position read on the card | Qwen3.6, writing after a short prompt | 241 to 269 | 255 to 281 |
| | Gemma 4, writing after a short prompt | 220 to 232 | 236 to 248 |
| | Gemma 4, writing after 8001 tokens | 134 | 148 |
| The draft block asked only while asking it writes faster | Ornith, writing after a short prompt | 154 to 183 | 182 to 197 |

All but Gemma 4's draft model are in the shared runtime or its kernels, so every architecture with that kind of
layer has them. Gemma 4's row for the routing has the two changes before it in it as well. The rows for writing
are with guessed tokens, as setup installs the models, and Gemma 4's from its draft model on are with that.

Each of the three steps of reading a prompt came from timing the one before it. Timed on the 7709-token prompt,
the delta net's step was 4.8 of 6.9 seconds, where this page had said the experts' launches were most of it; they
were a sixth. After it the routing, one thread's work, was 0.9 of 2.4 seconds.

Writing is bound by the engine's thread, not by the card. With the default windows of guessed tokens, and before
the change that adds a call's experts up in one launch, 1.75 of the 2.37 seconds Qwen3.6 took for 300 tokens were
spent handing the cached experts' launches to the card, five an expert, and the card was waited for during 0.3
of them. That change took one of the five away. Recording the launches as a CUDA graph and setting each expert's
slot in it for every token was tried and gave 1 to 2%: setting a node's arguments costs about what a launch
does.

What helped was fewer launches. A kernel that multiplies eight matrices of one format in one launch, each by its
own input, takes a call's cached experts eight at a time: a layer's experts are four launches and one for their
sum, where they were 32 and one. The products are the ones a single launch computes, so no token changed. Timed
after it, a token of Qwen3.6 took the thread 5.2 ms, of which it waited 3.3 for the card at the routers; the
experts' launches were 0.5 and the layers' own, up to the router, 0.7. Gemma 4, none of whose layers was
recorded as a graph then, spent 3.0 of its 7.6 ms on those.

A layer with attention over the cache could not be recorded, because its launches took the position as an
argument. Now they read it on the card, where the engine writes it before each token, and an attention is
recorded again each time the rows it reads double. A window of guessed tokens is recorded the same way, a graph
for each number of tokens. What is left of a token is mostly the card's own time: some 35 small launches a
layer, one after the other, and the thread waits for them at every router.

Ornith wrote slower with guesses than without, 183 tokens/s against 202 to 205, and the policy that sizes a
window could not see why: the block had been run before there was anything to choose. The engine now compares
the tokens a millisecond of the rounds that ask the block with those of the rounds that do not, and asks while
asking is ahead; the side that is behind gets a round now and then, so the choice follows the answer. Ornith's
block is then asked in about one round in eight.

## Adding a model of an architecture that is there

Setup knows the files in the table above. Another file of one of these architectures runs after these steps:

1. Get its GGUF. Check `general.architecture` is one of the table above.
2. Write its tokenizer directory:

   ```bash
   python tools/strata_tokenizer.py --gguf model.gguf --out packs/<name>
   ```

   It writes `packs/<name>/tokenizer/` with the vocabulary, the merges and the model's chat template. Byte-level
   BPE (`gpt2`: Qwen, GLM) and Gemma 4's `gemma4` are read.
3. Write `strata-<name>.json` beside `setup.py`:

   ```json
   {"exe": "engine/strata-poly", "args": ["--model", "/path/model.gguf", "--max-context", "32768"],
    "cwd": ".", "tokenizer": "packs/<name>/tokenizer", "model_name": "<name>", "port": 8080}
   ```
4. `python polystrata.py --list-models` should list it as available, and `python polystrata.py --model <name>`
   should answer.
5. Check it against llama.cpp before trusting it: the same prompt's token ids into both, greedy, and compare the
   tokens that come out.
6. To have setup install it, give it a family in `FAMILIES` in `setup.py`, next to `qwen36`: the repository at a
   pinned revision, the file's name, size and SHA-256, the size of its routed experts and the VRAM it takes before
   any expert. Steps 2 and 3 are then setup's.

## Adding an architecture

1. Read llama.cpp's graph for it (`src/models/<arch>.cpp`) and the ggml ops it builds. That is the reference.
2. Derive a class from `Engine`. `qwen35.cpp` is the shortest example with both a recurrent layer and attention;
   `gemma4.cpp` shows the hooks for a model that departs further.
3. Put what the layers need that no kernel does yet in `src/poly/kernels.cu`, next to the others. Keep a kernel's
   result for a token independent of the tokens beside it when it runs in a window of guessed tokens.
4. Name the class in `Engine::open()` and the file in `CMakeLists.txt`.
5. Check it against llama.cpp on the same file before any tuning: tokens and probabilities on short and long
   prompts, with every expert on the CPU, and with and without guessed tokens.

## Limits

- Linux and one NVIDIA card. Built and run on one machine.
- The three new models have run with all their experts in VRAM and with the card held to 8 GB, not on a card
  that has 8 GB, nor on a PC with 32 GB of RAM. The target of 8 GB of VRAM with 32 GB of RAM is not measured.
- Setup installed the three new models from files that were already on the disk. Of the new files its own
  download fetched Gemma 4's draft model (0.46 GB, through a mirror) and found the pinned SHA-256; the three
  addresses of the model files were asked for their headers only, and each states the pinned size.
- Setup's RAM figures for them are estimates.
- Pictures were read with the encoder on the GPU of the one machine. Setup puts it on the CPU of a card with
  less than 16 GB: that was run through the server with Qwen3.6 (a new picture took 3 to 5 s with 12 threads),
  and not on such a card.
- Setup's install with images ran on the test machine for the three models, with the encoder files already
  beside the models' files: it compiled the encoder and wrote the run configs, and the server read pictures
  with them. The files had been fetched before with setup's download and checked against the pinned SHA-256.
- A picture of Gemma 4's goes into the model in one prompt batch, 1,024 rows unless `--prompt-batch` says more;
  one with more rows is refused. Setup holds pictures to 1,024 rows (`--vision-tokens` changes it).
- Video and audio are not read.
- With everything in VRAM, llama.cpp reads prompts about twice as fast on the same card, 6,000 to 9,000
  tokens/s against 3,000 to 3,700. It writes 165 to 180 tokens/s; this engine writes 150 to 205 a token at a
  time and more with guesses ([Benchmark](#benchmark)). The router's scores still go to the host and back in
  every layer, which a model that fits does not need.
- Ornith's guessed tokens pay less than Qwen3.6's, though the two are one architecture and its guesses are
  right as often, nine in ten: its answers came out slower with the block than without. Their matrices are of
  other formats. Why was not looked into; the engine asks Ornith's block less instead.
- GLM's recurrent layers still use the step the delta net had before it was rewritten for Qwen3.6. GLM's prompts
  are bound by the experts that are not in VRAM, so it was left.
- No test runs the engine without a GPU and the model files: what is said of it here was measured by hand on the
  one machine.
- The benchmark's VRAM and RAM figures are the machine's, not the engine's alone.
