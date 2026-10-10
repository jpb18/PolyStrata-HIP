# Models and backends

PolyStrata runs more than one model family. This page says which, how the engine is put together so that it can,
how a model is recognized, how to add one, and how to change the model that answers.

One model is in memory at a time. The measurements below are from one machine: an RTX 5090 (32 GB), 24 threads of
a Xeon Platinum 8470Q, 120 GiB of RAM, Linux, CUDA 12.8. On that card the three new models fit in VRAM whole.
[One table](#with-8-gb-of-vram) holds the 5090 to 8 GB, and [one section](#on-a-laptop-with-8-gb-of-vram) is from
a second machine, a Windows laptop whose card has 8 GB.

## What runs

| Model | Architecture | Engine | File it was checked with | Guesses ahead |
| --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next and fine-tunes | `qwen4exp` | `strata` | the packs setup builds | separate MTP model |
| GLM 5.3 Flash | `glm5-next` | `strata-poly` | GSQ-RCO 3.0bit, REAP50-Q3_K_M, UD-IQ2_XXS | draft block |
| Qwen3.6-35B-A3B | `qwen35moe` | `strata-poly` | Unsloth UD-Q3_K_M, UD-Q2_K_XL and UD-Q4_K_M, 12.6 to 22.7 GB | draft block in the file |
| Ornith 1.5 35B-A3B | `qwen35moe` | `strata-poly` | APEX-MTP-I-Compact, I-Mini and I-Quality, 14.4 to 23.7 GB | draft block in the file |
| Gemma 4 26B-A4B | `gemma4` | `strata-poly` | Unsloth UD-Q3_K_M, UD-Q2_K_XL and UD-Q4_K_M, 10.5 to 16.9 GB | its draft model, a file of 0.46 GB |

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
| | | Unsloth UD-Q2_K_XL | 12.6 GB | about 17 GB |
| | | Unsloth UD-Q4_K_M | 22.7 GB | about 27 GB |
| `ornith` | Ornith 1.5 35B-A3B | APEX-MTP-I-Compact | 17.4 GB | about 22 GB |
| | | APEX-MTP-I-Mini | 14.4 GB | about 18 GB |
| | | APEX-MTP-I-Quality | 23.7 GB | about 28 GB |
| `gemma4` | Gemma 4 26B-A4B | Unsloth UD-Q3_K_M and its draft model | 13.2 GB | about 18 GB |
| | | Unsloth UD-Q2_K_XL and the draft model | 11.0 GB | about 15 GB |
| | | Unsloth UD-Q4_K_M and the draft model | 17.4 GB | about 21 GB |

A family's first size is the one setup takes. On a PC with less RAM than that wants it takes the second, the
smallest, and the third is by name: `--model UD-Q4_K_M`, `--model APEX-I-QUALITY`. The sizes differ in their
routed experts. UD-Q2_K_XL has them as IQ2_XS and IQ3_XXS, for Gemma 4 as IQ2_XS and IQ4_NL; UD-Q4_K_M as Q4_K
with Q5_K and Q6_K where they write, for Gemma 4 Q4_K and Q5_1; I-Mini as Q3_K in half its layers and IQ2_S in
the other half; I-Quality as IQ4_XS, Q5_K and Q6_K. The i-quants are slower on the CPU than the K-quants, which
shows when the card is small: held to 8 GB, Ornith's I-Mini writes more slowly than its I-Compact with more of
its experts on the card.

The steps are GLM's ([GLM.md](GLM.md#install)). Setup asks the context and whether to keep several conversations in
RAM, compiles `strata-poly`, downloads the model's file from a pinned revision (for Gemma 4 also its draft model,
0.46 GB), checks the SHA-256, writes the tokenizer directory, the run config and a start script, and starts the
model. `--gguf-dir /folder/with/the/file` takes a file you already have; a draft model that is not in that folder is
downloaded to setup's own. The engine is compiled once for all of them. On the test machine, with the files on the
disk, the first install took a minute, most of it the compile, and the other two 17 seconds each.

On Windows the three are installed the same way, with `START-HERE.bat` or `python setup.py` in place of
`./setup.sh`. The engine is compiled there with the CUDA Toolkit and Visual Studio's 2022 compiler, which is the
newest that CUDA 13 takes. A newer Visual Studio serves when it has the 2022 toolset (v143) among its
components; setup offers to install what is missing. GLM 5.3 Flash was not run on Windows, and setup says so
before it installs it there.

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

The RAM column is the experts an 8 GB card does not hold, plus 10 GB for the rest. It is setup's estimate. The
nine sizes ran on the [laptop](#on-a-laptop-with-8-gb-of-vram), which has 32 GB. The machine's RAM in use, 9 GiB
with no model, was at most 22.5, 26.6 and 31.3 GiB with Qwen3.6's three sizes in the benchmark there, 23.9, 26.7
and 31.4 with Ornith's and 22.6, 24.6 and 28.6 with Gemma 4's, of its 31.8. Setup does not stop a PC below the
32 GB that Qwen3.8 needs when one of these families is named.

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
counters. A long prompt is so many characters of this repository's documents, four a token, so its length in
tokens moves with the tokenizer and with those documents. One that the server refuses as longer than the
context is sent again at 95%, 90% and 85% of its text.

Tokens per second on the machine above, context 32768, every expert in VRAM for the three new models, as setup
installs them. The first build of each backend read the longest prompt at 636, 632 and 213 tokens/s and wrote
after it at 68, 54 and 55; [What was tuned](#what-was-tuned) has the steps between.

| Model | Prompt, tokens | First token, s | Reads, tokens/s | Writes, tokens/s | Experts from VRAM | Guesses right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 21 | 0.10 | 231 | 323 | 100% | 84% |
| | 2,568 | 0.63 | 4,239 | 268 | 100% | 86% |
| | 10,039 | 1.4 | 5,789 | 179 | 100% | 84% |
| Ornith 1.5 | 21 | 0.09 | 241 | 264 | 100% | 78% |
| | 2,568 | 0.60 | 4,407 | 223 | 100% | 77% |
| | 10,039 | 1.5 | 5,699 | 158 | 100% | 78% |
| Gemma 4 26B-A4B | 26 | 0.16 | 185 | 276 | 100% | 72% |
| | 2,656 | 0.63 | 4,498 | 233 | 100% | 84% |
| | 10,393 | 1.6 | 5,645 | 166 | 100% | 77% |
| GLM 5.3 Flash 3.0bit | 23 | 0.79 | 29 | 57 | 50% | 80% |
| | 2,424 | 4.5 | 546 | 55 | 49% | 78% |
| | 9,446 | 9.7 | 767 | 58 | 59% | 90% |

A short prompt is read a few tokens at a time, which is why its reading speed is low. Gemma 4 thinks in this
table, as the others do, and guesses with its draft model. GLM's second run read at 520 and 754 and wrote at
60, 55 and 55. GLM's longest prompt starts like the one before it, so its first 2,048 tokens are not read
again: the 9.7 s are for 7,398.

The build of 2026-10-10, the first with the laptop's changes on Linux, wrote after the three prompts at 335,
302 and 271 tokens/s with Qwen3.6, 270, 244 and 223 with Ornith and 304, 287 and 272 with Gemma 4, and read the
two long ones, then 2,743 to 2,847 and 10,609 to 10,988 tokens, at 4,606 to 4,953 and 5,454 to 5,946. GLM wrote
at 59, 56 and 62 and read at 555 and 769.

The table has each new model's first size. The other sizes by the same benchmark, one run each, every expert
in VRAM:

| Model and size | Reads its two long prompts, tokens/s | Writes after the three prompts, tokens/s |
| --- | ---: | ---: |
| Qwen3.6-35B-A3B UD-Q2_K_XL | 4,088 and 5,670 | 345, 284, 190 |
| Qwen3.6-35B-A3B UD-Q4_K_M | 4,205 and 5,572 | 324, 263, 180 |
| Ornith 1.5 APEX-I-Mini | 4,296 and 5,380 | 270, 228, 154 |
| Ornith 1.5 APEX-I-Quality | 4,224 and 5,389 | 264, 224, 152 |
| Gemma 4 26B-A4B UD-Q2_K_XL | 4,302 and 5,476 | 276, 224, 166 |
| Gemma 4 26B-A4B UD-Q4_K_M | 4,511 and 5,600 | 272, 212, 164 |

The last change in [What was tuned](#what-was-tuned) is to what a guess costs. The build before it wrote 312,
230 and 273 tokens/s after the three new models' short prompt, 249, 192 and 225 after the middle one and 177,
158 and 160 after the longest, and read as the rows above do. GLM's rows are that build's: a 400-token answer
of GLM's came at 60 to 66 tokens/s from it and at 60 to 67 from the one after, three runs each in turn.

The four changes before that one are to reading a prompt: one to the attention of a
prompt batch in the three new models, which now runs on the card's tensor cores, and three to GLM's layers,
whose attention is its own. The build before the four, that morning, read the three's long prompts at 4,127
to 4,272 and 4,583 to 4,841 tokens/s, with the first token after 0.62 to 0.67 and 1.7 to 1.9 s, and GLM's at
475 and 579, with the first token after 5.1 and 12.8 s. It wrote within 7 tokens/s of the build after it. The
two changes before those are to the same attention of the three new models: the build before them read their long
prompts at 3,500 to 3,972 and 3,406 to 3,848 tokens/s, with the first token after 0.67 to 0.80 and 2.1 to
2.5 s. The four changes before those are to reading a prompt with experts outside the cache, which only GLM
does in this table. The last of them lets a small call in a read copy ahead with the others, and GLM's longest
prompt ends in one. The build before it read that prompt at 534 to 539 tokens/s, with the first token after
13.8 to 13.9 s. The build before a round of a layer's experts stayed in the staging buffer, when it
was measured,
read GLM's two long prompts at 459 to 497 and 511 to 522, with the first token after 4.9 to 5.3 and 14.2 to
14.5 s. The build before the change that lets a read go past checkpoints, when it was measured: 422 to 424 and
466 to 472, with the first token after 5.7 and 15.7 to 15.9 s. The build before the first of them, a prompt
batch's float matrices as cuBLAS products, when it was measured: GLM's at 336 and 325, first token after 7.2
and 22.8 s, and the three new models' at 3,512 and 3,375, 3,525 and 3,461, and 3,215 and 3,137. It wrote 58 to
59, 55 and 55 to 59 for GLM, and 302, 228 and 275 after the short prompt, 236, 199 and 230 after the middle one
and 185, 133 and 160 after the longest for the three. Ornith's speed after its longest prompt moves from run to
run: 133, 151, 148 and 141 in four of them.

The build before the two changes before those, the swaps' one pass and the picks, when it was measured: GLM 55
and 57, 53 and 54, 55 and 52 tokens/s in its two runs, and the three new models 267, 214 and 232, 222, 189 and
202, and 167, 130 and 146. Three changes further back: GLM 52 to 53, 48 and 51 to 55, and the three new models
259, 201 and 233, 191, 176 and 197, and 162, 125 and 136.

Further back, on the same test: before the changes to writing that begin with the experts eight at a time, the
three new models wrote 125, 102 and 122 tokens/s after the short prompt and 95, 78 and 93 after the longest.
Before the changes for GLM, from the experts' pages on in that table, they wrote 253, 194 and 206, and 161, 134
and 127, and GLM 43, 43 to 46 and 46 to 48, with its prompts read at 331 and 321. Ornith's answer after its longest prompt is
another one since then, and slower.

Half of GLM's experts are on the CPU. One token at a time it takes 19 ms: 11 for the CPU's experts, which come
at the speed the machine's memory is read, and 6 for the card's work on a layer before its router has its
scores. With a guessed token beside it the card's part runs while the CPU computes the other token's experts,
and the CPU's experts are nearly all of the time that is left: 13 of the 15 ms a token of a window takes.
GLM runs here from a config written by hand, with its draft block and an expert profile.

GLM is sensitive to what the machine did before. After the other models had run, its first request came out at
28 to 35 tokens/s until its file was back in the file cache, and its starts were ended by a kill signal while the
pod's memory was at its limit with other files' pages, until those pages were dropped. Setup's build of the engine
wrote 38 to 43 tokens/s where the development build wrote 41 to 44, from the same sources; the two take ggml from
different checkouts, and the difference was not looked into.

It is as sensitive to where its file is cached. The test machine has two NUMA nodes, and the file cache keeps a
page on the node of whatever read it first. The engine keeps its CPU threads on the node that holds most of the
experts' pages and says at load what share that is. After other models had pushed part of GLM's file out of the
cache, a command that was not held to that node read it back, and the pages it read landed on the other node:
a Q2_K expert then took the CPU 99 microseconds where it takes 59, a token's experts 16.4 ms where 10.5, and
every GLM speed measured that hour was a tenth low. The table above is from the file read back on one node.

The draft block is a file of its own, and it was found on the other node later: the line at load is a share
over both files, taken from a sample of their pages, and said 99%. The block's eight experts then took the CPU
127 microseconds each where a layer's take 62, and a guess 1.8 ms where 1.4. GLM's numbers on this page from
before the swaps' one pass and the picks were measured that way; the ones in the table, and the builds they
are compared with from there on, were not.

llama.cpp (Unsloth's fork at 86ebfef) with every layer on the same card writes about 180 tokens/s with Qwen3.6
and Ornith and 165 with Gemma 4, and reads prompts at 6000 to 9000. It cannot do that on a card the model does
not fit, which is the case this engine is for.

### With 8 GB of VRAM

The same card with 24.4 GB kept free (`--vram-reserve-mib 24400`), so the engine has about 8 GB: the dense
weights, the caches and as many experts as fit. The rest of the experts are computed on the CPU, and the cache
follows what the answer uses. This is not an 8 GB PC: the CPU, the RAM and the card's speed are the machine's
above. The table is from before the changes that [a laptop whose card has 8 GB](#on-a-laptop-with-8-gb-of-vram)
brought, which were not built on this machine.
300-token answers, tokens per second; the two numbers are without and with guessed tokens.

| Model | Experts from VRAM | Writes after a short prompt | after about 2,100 tokens | after about 7,700 | Reads about 2,100 tokens | about 7,700 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 45% to 60% | 134, 164 to 166 | 122, 149 | 92, 104 | 2,250 to 2,300 | 4,020 to 4,160 |
| Ornith 1.5 | 44% to 61% | 135 to 154, 137 to 166 | 137, 131 | 101, 110 | 2,270 to 2,310 | 4,050 to 4,120 |
| Gemma 4 26B-A4B | 48% to 61% | 118 to 119, 150 to 152 | 97, 126 | 87, 93 | 2,450 to 2,540 | 4,190 to 4,660 |

The short prompts are the two of three whose answers run to 300 tokens for Qwen3.6 and Gemma 4; one of
Ornith's two stops after 122 to 123, and its answer after the longest prompt after 231 with guesses. With
guesses a smaller share of the experts comes from VRAM (48% to 50% against 58% to 61% for Gemma 4). After
about 2,100 tokens Ornith writes no faster with guesses than without: each of these runs is a process of its
own, and the engine asks the block in its first rounds, before it has measured whether asking pays.

The table is from after the last change in [What was tuned](#what-was-tuned), which is to what a guess costs.
The build before it wrote with guesses at 163 to 173, 137 to 172 and 146 to 155 tokens/s after a short prompt,
152, 135 and 125 after about 2,100 tokens and 105, 90 and 98 after about 7,700. Ornith's answer after its
longest prompt went from slower with guesses than without to faster. The rest are within what one of these
runs differs from the next.

One read takes the longer prompt whole, and a layer's uncached experts are copied to the card once for it. At
these lengths those copies are most of a read: the build before a prompt batch's attention ran on the tensor
cores read the two prompts at 2,170 to 2,360 and 3,770 to 4,100 tokens/s that morning, and wrote within 22
tokens/s of the build after it, faster in some runs and slower in others. The build before that attention took
its rows a block at a time read them at 2,230 to 2,340 and 3,670 to 4,000. The shorter prompt ends in a call of
63 tokens, 163 for Gemma 4, after the checkpoint at 2,048. The build before such a call copied ahead with the
others, when it was measured, read the shorter prompt at 2,060 to 2,080 tokens/s for Qwen3.6, 2,020 to 2,050
for Ornith and 2,050 to 2,130 for Gemma 4. The build before reads went
past checkpoints, when it was measured, read the two at 1,860 to 1,940 and 2,740 to 2,800 tokens/s for Qwen3.6,
1,890 to 1,920 and 2,750 to 2,760 for Ornith and 1,940 to 1,990 and 3,180 to 3,190 for Gemma 4, and wrote within
20 tokens/s of the build after it. The build before the two changes to reading before that one, when it
was measured, read them at 1,840 to 1,890 and 2,630 to 2,640, 1,860 to 1,910 and 2,630 to 2,690, and 1,920 to
1,990 and 2,840 to 2,960. The build before the swaps' one pass and the picks, when it was measured, wrote after
a short prompt: Qwen3.6 125 to 132 without guesses and 144 to 152 with, Ornith 125 to 140 and 117 to 137, Gemma
4 115 to 116 and 128 to 134.

Before the changes for GLM in [What was tuned](#what-was-tuned) the three wrote 97 to 124 tokens/s without
guesses after a short prompt and 101 to 121 with, and before the ones that begin with the experts eight at a
time 77 to 109 and 87 to 118. Each run here is a new process, so most of the gain without guesses is the
experts' pages being mapped before a token needs them: Qwen3.6's experts on the CPU took 5.2 to 5.8 ms a token
and take 3.8 to 3.9. The gain with guesses is the windows' tokens going one behind the other: in the same hour
the build before that wrote 134 to 135, 125 to 141 and 119 to 132 after a short prompt, and 115, 114 and 109
after about 2,100 tokens. Ornith gained nothing from it after a short prompt.

With part of the experts on the CPU, two runs of a prompt no longer write the same tokens: which experts are
cached changes while the answer is written, and an expert on the CPU differs from the same one on the card in the
last bits. In an earlier round of these runs one was ended by a kill signal while it read its prompt (Gemma 4,
2,212 tokens); four runs of the same command afterwards finished. The pod's memory was at its limit then (file
cache), as it was when a GLM start was killed later; the pod's own counter of memory kills stayed at zero, and the
cause was not found.

### A prompt of 30,000 tokens

The README's numbers are this benchmark with one long prompt, `--prompt-tokens 24000`, which comes to 28,000 to
31,000 tokens by the models' tokenizers, twice each. The three new models ran as setup installed them, and held to
8 GB as above but through the server, with the picture encoder on the CPU, where setup puts it for a card under
16 GB, and a profile of the run's own that a first run had started. GLM's three sizes ran from hand-written
configs: 24 threads, the profile the repository ships for the size after a first run's counts, and for 3.0bit
the draft block, which the other two files carry.

| Model | Card | Prompt, tokens | First token, s | Reads, tokens/s | Writes a short chat | Writes after the prompt | Experts from VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B UD-Q3_K_M | whole model on it | 30,126 and 31,369 | 5.7 to 5.8 | 5,523 to 5,580 | 332 to 334 | 186 to 189 | 100% |
| | held to 8 GB | | 7.4 to 7.5 | 4,248 to 4,291 | 162 to 167 | 117 to 122 | 62% to 63% |
| Qwen3.6-35B-A3B UD-Q2_K_XL | whole model on it | 30,126 and 31,369 | 5.7 | 5,535 to 5,584 | 368 to 369 | 190 to 196 | 100% |
| | held to 8 GB | | 7.9 | 3,876 to 3,901 | 178 to 196 | 134 to 140 | 81% |
| Qwen3.6-35B-A3B UD-Q4_K_M | whole model on it | 30,126 and 31,369 | 5.6 | 5,633 to 5,656 | 333 to 337 | 182 to 183 | 100% |
| | held to 8 GB | | 7.0 | 4,522 to 4,525 | 175 to 189 | 125 | 51% to 54% |
| Ornith 1.5 APEX-I-Compact | whole model on it | 30,126 and 31,369 | 5.4 to 5.6 | 5,697 to 5,893 | 267 to 272 | 145 to 149 | 100% |
| | held to 8 GB | | 7.4 to 7.7 | 4,135 to 4,281 | 160 to 161 | 107 | 64% |
| Ornith 1.5 APEX-I-Mini | whole model on it | 30,126 and 31,369 | 5.8 | 5,448 to 5,506 | 265 to 279 | 143 to 144 | 100% |
| | held to 8 GB | | 8.2 | 3,720 to 3,726 | 136 to 159 | 111 to 115 | 78% to 80% |
| Ornith 1.5 APEX-I-Quality | whole model on it | 30,126 and 31,369 | 5.7 | 5,602 to 5,609 | 263 to 265 | 139 to 144 | 100% |
| | held to 8 GB | | 7.2 to 7.4 | 4,284 to 4,366 | 147 to 150 | 102 to 106 | 49% |
| Gemma 4 26B-A4B UD-Q3_K_M | whole model on it | 31,211 and 31,186 | 4.4 | 7,168 to 7,210 | 306 | 198 to 200 | 100% |
| | held to 8 GB | | 5.2 | 6,068 to 6,114 | 173 to 181 | 111 to 114 | 63% |
| Gemma 4 26B-A4B UD-Q2_K_XL | whole model on it | 31,211 and 31,186 | 4.6 | 6,920 to 6,924 | 315 | 208 to 209 | 100% |
| | held to 8 GB | | 5.7 | 5,567 to 5,608 | 187 to 190 | 130 to 136 | 79% to 80% |
| Gemma 4 26B-A4B UD-Q4_K_M | whole model on it | 31,211 and 31,186 | 4.4 | 7,116 to 7,207 | 301 | 199 to 203 | 100% |
| | held to 8 GB | | 5.2 | 6,084 to 6,161 | 180 to 181 | 117 to 119 | 48% |
| GLM 5.3 Flash 3.0bit | 32 GB | 28,354 and 29,156 | 36.2 to 36.3 | 784 to 785 | 58 to 59 | 54 to 57 | 49% to 50% |
| GLM 5.3 Flash UD-IQ2_XXS | 32 GB | 28,354 and 29,156 | 27.8 | 1,027 to 1,029 | 40 to 44 | 37 to 39 | 49% to 52% |
| GLM 5.3 Flash REAP50-Q3_K_M | 32 GB | 28,354 | 22.7 | 1,252 | 56 to 62 | 50 to 51 | 57% to 64% |

Each figure is the better of two passes on this machine, and a row's two prompt lengths are theirs: the prompt's
text is this repository's documents, which changed between them. The first pass, of 2026-10-07, is what the next
paragraphs are about. The second, of 2026-10-10, is of the build with the laptop's changes in
[What was tuned](#what-was-tuned) and that day's two fixes, the first build with them on Linux. With the whole
model on the card it wrote after the long prompt at 139 to 209 tokens/s where the first pass wrote at 77 to 104,
and Gemma 4 read the prompt at 6,920 to 7,210 where 5,516 to 5,683. It was behind in these: the short chat of
Ornith's APEX-I-Mini and APEX-I-Quality, 243 to 256 where 263 to 279, some of the figures held to 8 GB of those
two and of Qwen3.6 UD-Q2_K_XL, and GLM's UD-IQ2_XXS, which wrote at 38 to 40 and 34 to 36 where 40 to 44 and 37
to 39. GLM's REAP50-Q3_K_M is the first pass's.

The nine sizes of the three new models were measured together, a family at a time with its three files in the
file cache, after the last change in [What was tuned](#what-was-tuned), which is to what a guess costs. Their
first 8 GB runs, which started the profiles, wrote the short chat at 130 to 170 tokens/s and read the prompt
at 3,572 to 4,636, like the runs after them. GLM's rows are from the build before that change, which left
GLM's speeds where they were.

The build before that change wrote the short chat at 302 to 335 tokens/s with Qwen3.6's three sizes, 219 to
231 with Ornith's and 246 to 276 with Gemma 4's, and held to 8 GB at 149 to 160, 127 to 155 and 132 to 156.
In the first of five runs with the whole model on the card, UD-Q2_K_XL of Qwen3.6 wrote it at 231: after a
slow start its guessing had stopped for the rest of the answer, which is what the change is about.

That morning, before the last four changes to reading a prompt ([What was tuned](#what-was-tuned)), the three
new models' first sizes read this prompt at 3,835 to 3,860 tokens/s with the whole model on the card and at
3,063 to 3,417 held to 8 GB, with the first token after 7.9 to 8.3 and 9.3 to 10.0 s, and GLM read it at 529
to 531, with the first token after 53.5 to 53.6 s. Before the two changes before those the three read it at
2,214 to 2,408 and at 1,941 to 2,251, with the first token after 13.1 to 13.8 and 14.1 to 15.7 s: more slowly
than a 10,000-token one, and held to 8 GB nearly as fast as whole, because what took the time was the
attention and not the experts' copies. They wrote as the builds after them did, up to the change to what a
guess costs.

A run with the picture encoder left on the card is not
in the table: the encoder holds 1.6 GB there, no expert fit beside it, and Qwen3.6 wrote 127 tokens/s and read
the prompt at 2,023, with the build before a prompt batch's attention took its rows a block at a time.

### On a laptop with 8 GB of VRAM

The kind of PC the three new models are for: an RTX 4070 Laptop with 8 GB, an i7-14700HX (8 performance cores
of two threads each and 12 efficiency cores, AVX2 and no AVX-512), 32 GB of RAM, Windows 10, driver 610.74,
CUDA 13.0, the engine compiled with Visual Studio's 2022 compiler. Setup installed the nine sizes of the three
models: a context of 32,768, the picture encoder on the CPU, and 13 threads for the experts that are not in
VRAM. The numbers are the benchmark's through the server, as on the first machine, with a browser and the
desktop on the same card and in the same RAM. They are one pass over the nine sizes with one build of the
engine, the one with every change in [What was tuned](#what-was-tuned) and the merge of the Strata project's
0.1.41.

The README's test, `--prompt-tokens 24000`, three runs a size. A size's first run is the first after another
model was in memory, and it starts without a profile of the experts' use. The table has the two runs after it.
The card's memory was at 7.4 to 7.5 GiB of its 8 in every one of them.

| Model | Prompt, tokens | First token, s | Reads, tokens/s | Writes a short chat | Writes after the prompt | Experts from VRAM | Machine's RAM, GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 26B-A4B UD-Q3_K_M | 31,209 | 16.4 | 1,907 | 70 to 71 | 51 to 53 | 51% to 62% | 23.4 to 24.9 |
| UD-Q2_K_XL | 31,209 | 17.0 | 1,844 to 1,845 | 86 to 87 | 60 to 62 | 70% to 78% | 21.3 to 22.8 |
| UD-Q4_K_M | 31,209 | 17.0 to 17.1 | 1,837 to 1,839 | 67 | 53 to 54 | 36% to 50% | 27.3 to 28.9 |
| Qwen3.6-35B-A3B UD-Q3_K_M | 31,364 | 22.6 | 1,390 to 1,392 | 73 to 75 | 60 to 62 | 49% to 55% | 26.3 to 27.1 |
| UD-Q2_K_XL | 31,364 | 23.8 | 1,320 | 91 | 70 to 72 | 62% to 76% | 22.2 to 23.0 |
| UD-Q4_K_M | 31,364 | 23.2 and 28.6 | 1,354 and 1,096 | 58 to 61 | 56 | 42% to 55% | 25.6 to 31.3 |
| Ornith 1.5 APEX-I-Compact | 31,364 | 23.4 | 1,343 to 1,345 | 70 | 59 | 57% to 58% | 26.4 to 27.2 |
| APEX-I-Mini | 31,364 | 24.0 to 24.1 | 1,302 to 1,308 | 63 to 68 | 58 | 60% to 72% | 23.6 to 24.4 |
| APEX-I-Quality | 31,364 | 23.8 to 24.3 | 1,294 to 1,322 | 42 to 48 | 55 to 56 | 43% to 49% | 27.3 to 31.4 |

The prompt is longer than in the tables above because its text is the start of two of this repository's
documents, which have grown. For Gemma 4 it is more than the context, and the benchmark sends 95% of it.

The first runs, with a cache filled without counts, wrote the short chat at 43 to 67 tokens/s (Gemma 4
UD-Q2_K_XL at 83) and after the prompt at 40 to 60. For Gemma 4 that is within a tenth of the table. Qwen3.6
UD-Q2_K_XL wrote the short chat at 66 where the table has 91, and UD-Q4_K_M wrote 44 and 40 where 58 to 61 and
56.

The two largest files, Qwen3.6 UD-Q4_K_M and Ornith APEX-I-Quality, take the machine's RAM to 31.3 and 31.4 GiB
with what else runs on it. Part of a file then comes from the disk again: one of Qwen3.6 UD-Q4_K_M's two runs
read the prompt at 1,096 tokens/s, and in the default benchmark below the two read the middle prompt at 458 and
655.

The default benchmark, the second of two runs. The first wrote within a seventh of it.

| Model | Prompt, tokens | First token, s | Reads, tokens/s | Writes, tokens/s | Experts from VRAM | Guesses right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 26B-A4B UD-Q3_K_M | 26 | 0.61 | 48 | 72 | 62% | 69% |
| | 2,847 | 1.98 | 1,482 | 70 | 49% | 79% |
| | 10,980 | 4.65 | 1,952 | 70 | 61% | 75% |
| UD-Q2_K_XL | 26 | 0.48 | 62 | 88 | 78% | 82% |
| | 2,847 | 1.82 | 1,602 | 81 | 69% | 81% |
| | 10,980 | 4.70 | 1,926 | 82 | 78% | 82% |
| UD-Q4_K_M | 26 | 0.83 | 35 | 67 | 50% | 68% |
| | 2,847 | 2.38 | 1,228 | 59 | 38% | 63% |
| | 10,980 | 5.08 | 1,787 | 70 | 48% | 83% |
| Qwen3.6-35B-A3B UD-Q3_K_M | 21 | 0.41 | 54 | 72 | 49% | 83% |
| | 2,743 | 2.51 | 1,097 | 69 | 52% | 81% |
| | 10,600 | 5.92 | 1,448 | 74 | 64% | 87% |
| UD-Q2_K_XL | 21 | 0.28 | 76 | 92 | 64% | 89% |
| | 2,743 | 2.27 | 1,211 | 90 | 72% | 84% |
| | 10,600 | 5.98 | 1,434 | 89 | 81% | 90% |
| UD-Q4_K_M | 21 | 0.57 | 38 | 63 | 42% | 84% |
| | 2,743 | 6.00 | 458 | 68 | 46% | 83% |
| | 10,600 | 6.46 | 1,326 | 72 | 58% | 84% |
| Ornith 1.5 APEX-I-Compact | 21 | 0.32 | 66 | 68 | 59% | 79% |
| | 2,743 | 2.59 | 1,062 | 78 | 59% | 88% |
| | 10,600 | 6.19 | 1,383 | 74 | 67% | 80% |
| APEX-I-Mini | 21 | 0.28 | 76 | 67 | 63% | 75% |
| | 2,743 | 2.40 | 1,143 | 81 | 69% | 83% |
| | 10,600 | 6.18 | 1,386 | 76 | 80% | 81% |
| APEX-I-Quality | 21 | 0.66 | 32 | 51 | 48% | 82% |
| | 2,743 | 4.20 | 655 | 59 | 47% | 72% |
| | 10,600 | 6.78 | 1,263 | 71 | 50% | 89% |

What guessing gains here, over the engine's protocol and from the profile the benchmark left: two answers of
400 tokens after short prompts, the second of two runs, in tokens/s.

| Model | With guesses | Without |
| --- | ---: | ---: |
| Gemma 4 26B-A4B UD-Q3_K_M | 81 and 79 | 51 and 50 |
| Qwen3.6-35B-A3B UD-Q3_K_M | 75 and 80 | 55 and 56 |
| Ornith 1.5 APEX-I-Compact | 72 and 70 | 62 and 65 |

Ornith's second answer ends after 332 tokens with guesses and after 122 without.

Without guesses and over the engine's protocol, against llama.cpp's 64 tokens after the five prompts of
[the check](#how-each-backend-was-checked), whose references are from the first machine:

| Model | Same tokens as llama.cpp (of 64) |
| --- | --- |
| Gemma 4 26B-A4B UD-Q3_K_M | 64, 64, 64, 64, 64 |
| UD-Q2_K_XL | 64, 53, 22, 50, 55 |
| UD-Q4_K_M | 64, 26, 64, 64, 46 |
| Qwen3.6-35B-A3B UD-Q3_K_M | 64, 64, 64, 64, 64 |
| UD-Q2_K_XL | 64, 64, 64, 64, 64 |
| UD-Q4_K_M | 64, 64, 64, 64, 43 |
| Ornith 1.5 APEX-I-Compact | 60 of 60, 7, 64, 2, 30 |
| APEX-I-Mini | 23 of 39, 64, 64, 13, 28 |
| APEX-I-Quality | 61 of 61, 64, 64, 22, 64 |

The table there is of the first machine's build, from before the two changes to an answer's attention, which
sum in another order. Through the server each size answered a picture, with the encoder on the CPU: in 3.9 to
4.7 s, and the two largest files of Qwen3.6 and Ornith in 6.5 s.

A pass of the build from before the merge and before the last four changes of
[What was tuned](#what-was-tuned), run the night before with nobody at the laptop, read the prompt 1% to 1.5%
faster in every size that was not held up by the disk (1,925 tokens/s where 1,907, 1,412 where 1,391, 1,360
where 1,344). It wrote the short chat at 47 to 93 tokens/s where this one wrote it at 42 to 91, and the answer
after the prompt at 47 to 71 where 51 to 72. The tokens against llama.cpp's were the same lists.

Setup, the engine and the benchmark each did something on this PC that the first machine could not show, and
were changed for it:

- No expert of Gemma 4's went to VRAM. The staging buffer took 2 GiB before the expert cache, and less than
  the reserve was left. It is now no larger than a layer's experts, and the reserve on a card of 8 GB is 512
  MiB where it was 1,024. As setup installs them, the first sizes start with 637 of Gemma 4's 3,840 experts
  in VRAM, 1,973 of Qwen3.6's 10,496 and 2,448 of Ornith's; the smallest sizes with 1,104, 3,421 and 3,286,
  and the largest with 403, 1,422 and 1,548.
- Some six seconds into a load on every core, this laptop stops running threads on its 16 performance threads
  and goes on with the 12 efficiency cores, for 40 seconds in one run. Gemma 4 with its experts on the CPU and
  20 threads then wrote 0.7 tokens/s where it had written 37. [What was tuned](#what-was-tuned) has what the
  pool of those threads did, and what it does now, and Windows' power throttling, which may have been what moved
  the threads.
- The engine took a thread for every logical CPU, 28, which is the slowest count on this processor: Gemma 4
  with every expert on the CPU wrote 24.5 tokens/s with 28 threads, 30.0 with 8, 32.6 with 12, 34.3 with 16
  and 35.1 with 20, and once the performance cores were gone 12 threads wrote 26 where 20 wrote 16 to 17.
  Setup now writes a count for a processor with more efficiency cores than performance cores: the
  performance cores but one and half of the efficiency cores, 13 here.
- Setup did not find the compiler in Visual Studio 2026 and would have installed the 2022 Build Tools beside
  it. It now takes a newer Visual Studio that has the 2022 toolset.
- The benchmark ended with the server's refusal of Gemma 4's long prompt. It now cuts a refused prompt to 95%,
  90% and 85% and says so.
- No run had left a profile of the experts' use, so every start filled its cache without counts. On Windows the
  server's engine ends with the server, before it saves, and a save over an old profile failed there.
- Setup said of the three models that they had run on no card under 8 GB, on the card they had run on: it
  reports 8,188 MiB. Setup now compares the size in whole GB.

What an answer waits for on this PC is other than on the first machine: the card's memory, which moves a
seventh of the 5090's bytes a second, and the experts the CPU computes. The changes that came of it are in
[What was tuned](#what-was-tuned): a window whole or in halves, the reserve by the card's size, a prompt in one
read, an answer's attention after a long prompt, the profile, the longer window for an answer that copies, and
the engine out of Windows' power throttling. Before them the first sizes wrote a short chat at 45 to 54
tokens/s here and after the long prompt at 21 to 32, and read it at 870 to 1,610.

Not run on this laptop: GLM 5.3 Flash, for which it has not the RAM.

## How each backend was checked

Against llama.cpp on the same file: greedy, 64 tokens after five chat prompts of about 25 to 8000 tokens, the
token ids compared and the probabilities of the reference's five likeliest tokens at every step.

| Model | Same tokens as llama.cpp (of 64) | Largest probability difference |
| --- | --- | --- |
| Qwen3.6-35B-A3B | 64, 64, 40, 64, 64 | 0.10 |
| Ornith 1.5 | 8 of 60, 7, 64, 2, 64 | 0.37 |
| Gemma 4 26B-A4B | 64, 64, 64, 64, 43 | 0.12 |
| Qwen3.6-35B-A3B UD-Q2_K_XL | 64, 46, 64, 64, 64 | 0.15 |
| Qwen3.6-35B-A3B UD-Q4_K_M | 64, 64, 33, 64, 19 | 0.08 |
| Ornith 1.5 APEX-I-Mini | 40 of 40, 64, 64, 13, 28 | 0.25 |
| Ornith 1.5 APEX-I-Quality | 61 of 61, 7, 21, 22, 34 | 0.16 |
| Gemma 4 26B-A4B UD-Q2_K_XL | 64, 53, 22, 50, 33 | 0.09 |
| Gemma 4 26B-A4B UD-Q4_K_M | 64, 26, 52, 25, 64 | 0.15 |

The first three rows are the sizes setup takes first, UD-Q3_K_M and APEX-I-Compact. With the build of
2026-10-10 they were 64, 64, 40, 29 and 64 within 0.12, 60 of 60, 7, 56, 2 and 64 within 0.29, and five times
64 within 0.09. The six below them were
compared once, with the current build, each against llama.cpp on its own file. Wherever one leaves llama.cpp's
tokens, the two likeliest tokens at that step are within 0.1 of each other in the engine or in llama.cpp. Their
states after a read and after the same calls one by one are equal byte for byte, held to 8 GB after 7,700
tokens and with the whole model on the card after 30,000. Their answers to pictures are in the table further
down.

Two correct implementations do not write the same tokens for long. Activations are quantized to 8 bits at every
quantized matrix, in a layout that depends on the kernel, so the two differ in the last bits and a near tie
between two tokens goes either way. Ornith's file has 3-bit experts and 4-bit attention and is the most sensitive:
llama.cpp's own GPU and CPU runs of it agree for 17 of 50, 7 of 64 and 15 of 64 tokens on the three short prompts
and differ by up to 0.19, which is as far apart as this engine is from either.

The numbers move with every change to a kernel that changes the order of a sum or how a number is rounded.
The last one gives the products of a prompt batch's attention halves, as llama.cpp's attention has them:
Gemma 4's fifth was 64 before it and leaves llama.cpp's after 43 tokens now, where the two likeliest tokens
are at 0.498 and 0.495 here and at 0.488 and 0.506 there. The largest differences after the two long prompts
are 0.07, 0.23 and 0.07 for the three models where 0.06, 0.13 and 0.07. The one before takes that attention
a block of rows at a time: Ornith's fifth was 30 before it, and those differences 0.06, 0.12 and 0.07. The one
before that made a prompt batch's float matrices cuBLAS products, which only what follows a prompt of 32
tokens or more can show: Ornith's fifth was 64 before it, and those differences 0.07, 0.19 and 0.07. Before
the change before that, the sums with more threads a block, they were 64, 64, 64, 64, 64 within 0.07 for
Qwen3.6, 60 of 60, 7,
39, 40, 64 within 0.48 for Ornith and 64, 64, 32, 64, 64 within 0.07 for Gemma 4; before the one before that
Ornith's were 60, 7, 56, 2, 64 and Gemma 4's fourth was 44. Where a change kept the arithmetic (the first
rewrite of the delta net's step, the threads of the routing, the sum of a call's experts), the numbers and the
tokens stayed as they were, and that was the check of the change.

The two changes to an answer's attention in [What was tuned](#what-was-tuned) sum a score in another order and
a value in more shares. They were built on the laptop only, and the table above is not of them:
[the laptop's section](#on-a-laptop-with-8-gb-of-vram) has what the nine sizes write with them. Before them the
three first sizes wrote 64, 46, 32, 64, 64 there for Gemma 4, 64, 64, 20, 64, 64 for Qwen3.6 and 8 of 60, 64, 56,
64, 64 for Ornith. What was checked of the two: on made-up caches a token comes out with the same bits alone and
beside others, and with every expert on the CPU 200-token answers with windows of guesses and without are the
same tokens, after short prompts and after 31,000 tokens.

How far one changed sum goes was looked at once, with GLM REAP50's first token. The first layer norm of the two
builds differs by a unit in the last place. Two of its 4,096 activations then round to the 8-bit value next to
the one they had, and the layer's three projections differ by 1.5e-4 of their size, nearly every one of their
24,576 values a little. After the last layer the two builds are about a percent apart.

Also checked, for each model: with windows of guessed tokens and without, 300-token answers are the same tokens;
with every expert on the CPU the short prompts still follow the reference to a near tie. With part of the
experts cached, two runs can only be compared when they hold the same experts: a run adds its counts to the
profile it is given, and holds more experts without the draft block than with it. With the profile copied for
each run and the cache left as it was loaded, the build whose windows go a token behind the other writes the
tokens of the build before it: GLM on three prompts, and the three new models held to 8 GB. GLM wrote the same
tokens as before the engine was split, with every expert on the CPU, until the sums' order changed. Since then
its REAP50 file is compared with llama.cpp as the others are: on three short prompts 4, 39 and 9 of 64 tokens
are the same, within 0.29 (4, 39 and 9 within 0.27 before), after a prompt of 1,947 tokens 28 of 64 within 0.12
(38 within 0.23 before a prompt batch's float matrices were cuBLAS products; it leaves at a near tie both
times), and with every expert on the CPU 200-token answers are the same tokens with and without guessed ones.

A prompt read several calls at a time, layer after layer, is checked against the same calls one by one. A
build for the check does every read twice from the same state, once each way, and compares the two states byte
for byte, with the states at the checkpoints inside the read: GLM with calls of 1,024, 2,048, 512 and 896
tokens, 3 to 12 a read, and the three new models held to 8 GB and with every expert in VRAM, 66 to 210 MB of
state each, all the same. That check found one difference, in the first version of this reading: a call of
fewer than 256 tokens does not copy a layer's experts ahead, and when it took the ones the call before it had
copied, its launch had other matrices in it than alone, the batched kernels divided the work another way, and
35 of Qwen3.6's 66 MB of state came out with other last bits. Such a call staged its own experts again, until
every call in a read of several was made to copy ahead. The calls taken one by one are told to do the same
since, and the two states are equal with a small call at the end of a read (20, 100 and 230 tokens), alone
after a checkpoint, and with every call small. The check also covers a layer that takes the staging buffer in
more than one round and keeps one for its next call: GLM's six, and Qwen3.6 held to 8 GB with staging buffers
of 256 and 128 MiB and Gemma 4 with 128, which with the smaller buffer is two or three rounds in every layer.
Since a prompt batch's attention takes its rows a block at a time, the three new models have also been checked
on reads of 30,000 tokens, with the whole model on the card and held to 8 GB, and again since its products
take halves.

With the cache left as it was loaded and both builds holding the same experts, the 200 tokens after a prompt
were those of the build that read call by call: GLM after 1,947 and 7,788 tokens, the three new models held to
8 GB after their two long prompts each. The shorter of those two ends in a call of 63 or 163 tokens, and since
small calls copy ahead its sums are not the earlier build's: Qwen3.6 writes the same 120 tokens after it
without guesses and parts at token 77 with them, and Gemma 4 parts at token 75, where the two likeliest tokens
were at 0.3946 and 0.3933 and are at 0.3562 and 0.4070. After the prompts without a small call the tokens are
the earlier build's still.

Over the engine's protocol, held to 8 GB or with GLM's half cache: a prompt of about 7,700 tokens cut at 7,000,
5,000 and 3,000 and continued another way is read from the checkpoints at 6,144, 4,096 and 2,048, which one
read wrote, and GLM, Ornith and Gemma 4 write what a fresh engine writes; Qwen3.6's answers there are one token,
the same one. A prompt, its answer and a question after it are read from the answer's end, and GLM and Ornith
then write what a fresh engine writes. Qwen3.6 and Gemma 4 do not, here and in the builds before these changes:
the fresh engine reads the answer in a prompt batch, whose sums are in another order than the ones the answer
was written with.

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
| Qwen3.6-35B-A3B | 280, 440, 160 | 0 of 11, 33 of 64, 64 of 64 | 0.14 |
| Ornith 1.5 | 280, 440, 160 | 17 of 17, 4 of 35, 59 of 59 | 0.17 |
| Gemma 4 26B-A4B | 117, 194, 77 | 14 of 14, 19 of 19, 55 of 55 | 0.12 |
| Qwen3.6-35B-A3B UD-Q2_K_XL | 280, 440, 160 | 12 of 12, 64 of 64, 64 of 64 | 0.11 |
| Qwen3.6-35B-A3B UD-Q4_K_M | 280, 440, 160 | 11 of 11, 11 of 64, 64 of 64 | 0.07 |
| Ornith 1.5 APEX-I-Mini | 280, 440, 160 | 6 of 18, 54 of 54, 16 of 64 | 0.29 |
| Ornith 1.5 APEX-I-Quality | 280, 440, 160 | 18 of 18, 40 of 40, 50 of 50 | 0.22 |
| Gemma 4 26B-A4B UD-Q2_K_XL | 117, 194, 77 | 14 of 14, 7 of 25, 55 of 55 | 0.11 |
| Gemma 4 26B-A4B UD-Q4_K_M | 117, 194, 77 | 14 of 14, 19 of 19, 55 of 55 | 0.07 |

Ornith's second answer leaves llama.cpp's where that has a near tie: 0.504 against 0.495 there and 0.444
against 0.555 here. Qwen3.6's first answer starts with another token than llama.cpp's, which has five tokens
between 0.10 and 0.14 there. The engine's first step is 0.14 from that, and not all of its five likeliest are
the same five. The differences are what the same models show on text.

The six rows below the first three are the other sizes, compared once with the current build, each against
llama.cpp on its own file with its model's encoder file. Where one leaves llama.cpp's answer the two likeliest
tokens are close on both sides: UD-Q4_K_M of Qwen3.6 in its second answer at 0.51 against 0.49 here and 0.46
against 0.54 there, APEX-I-Mini in its first at 0.58 against 0.42 and 0.47 against 0.53 and in its third at
0.55 against 0.43 and 0.48 against 0.50, UD-Q2_K_XL of Gemma 4 in its second at 0.53 against 0.39 and 0.41
against 0.47. APEX-I-Mini's 0.29 is a step inside its second answer, whose 54 tokens are llama.cpp's.

The counts and the differences have moved with the sums. Before the products of a prompt batch's attention
took halves, Ornith's three were 17, 4 and 9 within 0.22, its third leaving at a near tie too, Qwen3.6's first
step was 0.03 from the reference's and Gemma 4's largest difference 0.06. While that attention took 32 tokens
at a time, Ornith's three were 17, 35 and 59 within 0.19, Qwen3.6's first step was 0.14 from the reference's
and Gemma 4's largest difference 0.12: Qwen3.6's first step goes between the same two places. Before a prompt
batch's float matrices
were cuBLAS products Ornith's were 17, 4 and 9 within 0.15, at the near ties above, and the other two within
0.07. Before the sums had more threads a block Qwen3.6's were 11 of 11, 11 of 64 and 64 of 64, with its second
answer leaving at a near tie.

A fourth prompt has the picture after a list of 180 lines, about 3,600 tokens: past a prompt batch, a checkpoint
and the rows Gemma 4's window layers keep. Qwen3.6 writes llama.cpp's 64 tokens, which name the shapes and get
as far as quoting the right line of the list. Ornith leaves llama.cpp's answer after 11 of 38 tokens, where
three tokens are at 0.37, 0.34 and 0.28 here and at 0.24, 0.38 and 0.37 there, and Gemma 4 starts with another
token: 0.500 against 0.472 here, 0.457 against 0.517 there. Both answers name the shapes and give the code of
the right line, as llama.cpp's do, in other words. Before the attention's products took halves all three wrote
llama.cpp's tokens, 64, 38 and 31 of them. While a prompt batch's attention took 32 tokens at a time
Qwen3.6 left after four tokens, at a near tie, and Ornith after eleven. Before a prompt batch's float matrices
were cuBLAS products Qwen3.6 wrote llama.cpp's 64 and Gemma 4 started with another token at a near tie, and
before the sums had more threads a block Ornith left at its fourth. A fifth has 43 lines ahead of the picture,
so that its rows start in one prompt batch of Qwen3.6's and end in the next: 64 of 64 tokens.

The other sizes on the fourth prompt: UD-Q4_K_M of Qwen3.6 and both of Gemma 4's write llama.cpp's tokens, 64,
28 and 25 of them. UD-Q2_K_XL of Qwen3.6 leaves after 40 of 64 (0.39 against 0.33 here, 0.32 against 0.42
there) and APEX-I-Mini after 48 of 64 (0.52 against 0.48 here, 0.35 against 0.65 there). APEX-I-Quality starts
with another token (0.43 against 0.28 here, 0.35 against 0.37 there). These three answers name the shapes and
give the code of the right line too. On the fifth prompt both other sizes of Qwen3.6 write llama.cpp's 64
tokens.

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
| The experts' pages read once after load, so they are mapped when a token needs them | GLM 5.3 Flash, the first answer after a load | 43 | 47 |
| | Qwen3.6 held to 8 GB, writing after a short prompt | 97 to 106 | 121 to 125 |
| Q2_K experts eight at a time too | GLM 5.3 Flash, one token at a time | 43 to 44 | 44 |
| A vector summed by up to 1,024 threads a block | GLM 5.3 Flash, one token at a time | 43 to 44 | 50 |
| | Gemma 4, one token at a time, 64 tokens | 123 to 124 | 134 to 139 |
| The recurrent attention's step with eight threads a column of its state | GLM 5.3 Flash, one token at a time | 50 | 49 to 52 |
| | GLM 5.3 Flash, reading 9,446 tokens | 321 | 341 |
| The last four together | GLM 5.3 Flash, writing after its three prompts | 43, 43 to 46, 46 to 48 | 50 to 52, 48, 53 to 54 |
| A window's tokens one behind the other, the card's layers during the CPU's experts; the policy gets no time from a round that recorded, and counts the asking | GLM 5.3 Flash, writing after its three prompts | 52 to 53, 48, 51 to 55 | 56 to 59, 55 to 56, 53 to 55 |
| | Qwen3.6 held to 8 GB, with guesses, after a short prompt | 134 to 135 | 144 to 152 |
| | Gemma 4 held to 8 GB, with guesses, after about 2,200 tokens | 109 | 123 |
| | Qwen3.6 with every expert in VRAM, after 2,568 tokens | 191 | 217 to 222 |
| The experts to swap listed in one pass over them | GLM 5.3 Flash, a 600-token answer | 182 ms of 9,736 | 33 ms |
| A token picked, and a guess's logits read, without a candidate for each of the vocabulary | the same answer | 159 and 96 ms | 25 and 29 ms |
| The two together | GLM 5.3 Flash, writing after its three prompts | 55 to 57, 53 to 54, 52 to 55 | 59, 56 to 57, 56 to 57 |
| | Qwen3.6 with every expert in VRAM, after its three prompts | 267, 222, 167 | 309, 247, 180 |
| | Gemma 4 with every expert in VRAM, after its three prompts | 232, 202, 146 | 276, 229, 160 |
| | Qwen3.6 held to 8 GB, with guesses, after a short prompt | 144 to 152 | 156 to 159 |
| | Gemma 4 held to 8 GB, with guesses, after a short prompt | 128 to 134 | 150 to 152 |
| A prompt batch's float matrices as cuBLAS products | GLM 5.3 Flash, reading 7,788 tokens | 363 | 419 |
| | Qwen3.6 with every expert in VRAM, reading 10,039 tokens | 3,375 | 3,810 |
| | Gemma 4 with every expert in VRAM, reading 10,393 tokens | 3,137 | 3,445 |
| A prompt's calls through a layer one after the other, the layer's experts copied to the card once | GLM 5.3 Flash, reading 7,788 tokens | 417 | 516 |
| | Qwen3.6 held to 8 GB, reading 7,709 tokens | 2,407 | 2,520 |
| The two together | GLM 5.3 Flash through the server, 9,446 tokens: seconds to the first token | 22.8 | 15.9 |
| A read past its checkpoints, up to 8,192 tokens, the calls' residual waiting in host memory | GLM 5.3 Flash, reading 7,788 tokens | 514 to 531 | 602 to 616 |
| | GLM 5.3 Flash through the server, 7,398 tokens read: seconds to the first token | 15.7 to 15.9 | 14.5 to 14.6 |
| | Qwen3.6 held to 8 GB, reading 7,709 tokens | 2,740 to 2,800 | 3,740 to 3,790 |
| | Gemma 4 held to 8 GB, reading 8,001 tokens | 3,180 to 3,190 | 4,030 to 4,040 |
| A round of a layer's experts left in the staging buffer for the layer's next call | GLM 5.3 Flash, reading 7,788 tokens | 599 to 602 | 639 to 643 |
| | GLM 5.3 Flash through the server, 7,398 tokens read: seconds to the first token | 14.2 to 14.5 | 13.8 to 13.9 |
| A small call in a read of several copies ahead like the others | GLM 5.3 Flash, reading 7,398 tokens that end in a call of 230 | 582 to 585 | 629 to 630 |
| | GLM 5.3 Flash through the server, the same number read: seconds to the first token | 13.8 to 13.9 | 12.7 to 12.8 |
| | Qwen3.6 held to 8 GB, reading 2,112 tokens | 2,060 to 2,080 | 2,250 to 2,300 |
| | Gemma 4 held to 8 GB, reading 2,212 tokens | 2,050 to 2,130 | 2,340 |
| A prompt batch's attention a block of rows at a time, the call's tokens together | Qwen3.6, reading 30,126 tokens | 2,214 to 2,218 | 3,835 to 3,860 |
| | Gemma 4, reading 31,211 tokens | 2,389 to 2,408 | 3,846 to 3,852 |
| | Qwen3.6 held to 8 GB, reading 30,126 tokens | 1,975 to 1,987 | 3,156 to 3,190 |
| | Qwen3.6, reading 10,039 tokens | 3,798 to 3,811 | 4,841 to 4,862 |
| The attention's rows as floats in a block's worth of scratch | Gemma 4 held to 8 GB: experts in VRAM at a start | 182 | 271 |
| The products of a prompt batch's attention on the tensor cores, halves in and floats summed | Qwen3.6, reading 30,126 tokens | 3,835 to 3,860 | 5,114 to 5,173 |
| | Gemma 4, reading 31,211 tokens | 3,846 to 3,852 | 5,529 to 5,710 |
| | Qwen3.6 held to 8 GB, reading 30,126 tokens | 3,156 to 3,190 | 3,918 to 3,999 |
| | Qwen3.6, reading 10,039 tokens | 4,841 | 5,529 |
| | Gemma 4 held to 8 GB: experts in VRAM at a start | 271 | 289 |
| GLM's recurrent step for a prompt batch in three launches, a column of the state in registers | GLM 5.3 Flash, reading 7,788 tokens | 644 | 708 |
| GLM's indexer scores of a prompt batch as a cuBLAS product | GLM 5.3 Flash, reading 28,354 tokens | 580 | 725 |
| | GLM 5.3 Flash, reading 7,788 tokens | 712 | 761 |
| GLM's sparse selection of a prompt batch on the card | GLM 5.3 Flash, reading 28,354 tokens | 736 | 813 |
| | GLM 5.3 Flash, reading 7,788 tokens | 755 | 806 |
| The three together | GLM 5.3 Flash through the server, 28,354 tokens: seconds to the first token | 53.5 to 53.6 | 36.2 to 36.3 |
| | GLM 5.3 Flash through the server, 7,398 tokens read: seconds to the first token | 12.7 to 12.8 | 9.7 to 9.9 |
| A guess's token and its probability found on the card | Qwen3.6 with every expert in VRAM, after its three prompts | 312, 249, 177 | 323, 268, 179 |
| | Ornith with every expert in VRAM, after its three prompts | 230, 192, 158 | 264, 223, 158 |
| | Gemma 4 with every expert in VRAM, after its three prompts | 273, 225, 160 | 276, 233, 166 |
| | Ornith, a 256-token answer about code | 238 | 291 |
| | Ornith held to 8 GB, the same answer | 152 | 166 |
| | Qwen3.6 UD-Q2_K_XL, an answer whose first 16 rounds take 3 ms longer each | 243 to 310 | 332 to 342 |
| The staging buffer no larger than a layer's experts | Gemma 4 on the laptop's 8 GB card: experts in VRAM at a start | 0 | 625 |
| A worker of the CPU's pool leaves a finished batch alone and sleeps on the batch count | Gemma 4 on the laptop, one token at a time, its experts on the CPU: 20 threads on 12 cores | 0.33 | 16.6 |
| | the same, 12 threads | 16.3 | 23.3 |
| A window with experts on the CPU in two halves or whole, by what each has taken | Gemma 4 on the laptop, writing two 400-token answers | 48 to 50 | 62 to 69 |
| | Qwen3.6 there | 54 to 57 | 70 to 76 |
| | Ornith there | 61 to 65 | 66 to 75 |
| The reserve a sixteenth of the card, 512 to 1,024 MiB | Gemma 4 on the laptop: experts in VRAM at a start | 435 | 635 |
| | Qwen3.6 there | 1,574 | 1,958 |
| A prompt in one read | Qwen3.6 on the laptop, reading 30,836 tokens | 1,130 to 1,209 | 1,365 to 1,465 |
| | Gemma 4 there, reading 31,000 tokens | 1,420 to 1,564 | 1,772 to 1,829 |
| | Ornith there, reading 30,836 tokens | 1,107 to 1,196 | 1,361 to 1,362 |
| | Qwen3.6 there: seconds from a stop to the engine's answer | up to 7 | 0.13 |
| An answer's attention scores read a key side by side | Gemma 4 on the laptop, writing 300 tokens after a 31,000-token prompt | 32 | 56 |
| | Qwen3.6 there, after 30,836 tokens | 35 to 36 | 51 to 52 |
| | Ornith there | 37 | 50 to 52 |
| A window's tokens read the keys and values once between them | Gemma 4 there | 56 | 62 to 72 |
| | Qwen3.6 there | 50 to 52 | 56 |
| | Ornith there | 49 to 51 | 55 to 58 |
| A time of the window way that is behind counts for half | Gemma 4 there, the runs that kept to halves | 62 | 68 |
| The expert profile saved over the old one, and after answers | Gemma 4 there, the run after one that left a profile | 68 | 79 |
| A repeat fills a window of eight where a guess costs no VRAM | Gemma 4 there, giving a file back with a name changed | 88 | 100 to 107 |
| | Gemma 4 there, quoting a function and explaining it | 74 to 78 | 83 to 88 |
| A long window leaves the recurrent state after three of its tokens | Qwen3.6 there, giving the file back | 90 to 92 | 112 to 113 |
| | Qwen3.6 there, quoting the function | 77 | 83 to 85 |
| | Ornith there, giving the file back | 99 to 102 | 120 to 121 |
| | Ornith there, quoting the function | 82 to 83 | 86 to 90 |
| A window past four tokens only from a repeat of eight | Gemma 4 there, a story of 400 tokens | 72 to 79 | 76 to 86 |
| The engine out of Windows' power throttling | Qwen3.6 there, the answer after a 31,000-token prompt, through the server | 39 | 57 to 62 |
| | the same after 7,709 tokens, the engine started from a shell | 48 to 50 in four runs of six | 66 to 72 |

All but Gemma 4's draft model and GLM's recurrent step, indexer and selection are in the shared runtime or its
kernels, so every architecture with that kind of layer has them. Gemma 4's row for the routing has the two
changes before it in it as well. The rows for writing are with guessed tokens, as setup installs the models,
unless they say one token at a time, and Gemma 4's from its draft model on are with that. The 8 GB row and
GLM's reading have the changes around them in them too: they were measured before the four and after.

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

GLM 5.3 Flash was looked at last. One token at a time it took 23 ms. With a wait for the card after each stage
of a layer, 11 of them were the CPU's experts and 11 the card's work on a layer before its router has its
scores, one after the other; the experts the card computes meanwhile took 1.7.

The CPU's 11 ms are memory. 24 threads read the experts' bytes at 181 GiB/s when they only add them up and at
130 when they compute, and smaller tasks for the threads made it slower. A row product for AVX-512 with VNNI and
VBMI was written, with ggml's bits, and is not in the engine: on the test machine's Xeon it is no faster than
ggml's AVX2 one, 8.1 against 8.4 GiB/s a thread from the cache and the same from memory. What did cost was an
expert's first use. The file's pages come into the process's page table as they are first read, and a new
process computed four experts a layer in 21 ms a token where it took 11 with the pages mapped. A thread now
reads a byte of every page of the experts after load, 4 seconds for the 100 GiB of GLM's that the file cache
holds, so the first answers after a load are as fast as the later ones, and so is each run of the 8 GB table.

The card's 11 ms were not its matrices. Timed a launch at a time, a layer norm took 16 microseconds, the two
launches that mix the model's four residual streams 41 and the recurrent attention's step 33, where a launch
that only quantizes a vector takes 8. Each gave a thread 16 to 128 values to read one after the other, and a
kernel this small takes as long as those reads. With up to 1,024 threads a block, four values a thread, and
eight threads a column of the recurrent state, the three take 9, 24 and 16. Two more were measured and left
out: the 4 by 4 matrix that mixes the streams made by a warp's threads together (its launch took 25
microseconds where 16), and a layer's three projections in one launch (no faster).

These two change the order of sums, which is why the checks above moved. A GLM token is now 19 ms: 11 for the
CPU's experts, 6 for the card before the routers. The two cannot overlap within a token, because a layer's
router needs the layer before it finished; what the card does during the CPU's experts, and the CPU during the
card's layers, is wait.

A window of guessed tokens can fill that wait. Its tokens used to go through a layer together, which shares the
card's launches and leaves the waiting as it was. With experts on the CPU they now go one at a time and one
behind the other: the engine's thread only routes a token and computes its experts, and while it does, one of
the experts' threads hands the card the token before, from its experts' sum to the next layer's router. A first
version left those launches on the engine's thread, between two tokens' experts, and wrote no faster than the
tokens together: a token's launches for a layer are 50 to 110 microseconds, a fifth to two fifths of the time
its experts take. With both builds holding the same experts and windows of two, GLM wrote 43, 52 and 59
tokens/s on three prompts where the tokens together wrote 38, 44 and 51. With every expert in VRAM there is
nothing to wait for, and a window goes together as before.

Windows of two are what pays for GLM now. One more token costs most of a token's CPU time, so a second guess has
to be nearly certain, and the block was still asked for it in most rounds. Two changes to what the policy is told
ended that. A round that recorded launches gives it no time: its first figure for a window's size was many
times too high, and the sizes above looked cheap beside it. And the block's own run, 1 to 2 ms, counts against a
guess before the block is asked for it. The two are also what Qwen3.6 gained with every expert in VRAM.

Timed, a round of two tokens was 30 ms for GLM: 24 for the CPU's experts, 5 waiting for the card and 2 for the
rest, the block's run among it. The waiting was in the later layers. In that run most of a token's experts there
were cached and the CPU had its part done in about 200 microseconds, while the card had the other token's layer,
its shared expert and its cached experts to get through, and was 60 to 90 microseconds behind at every layer.

That was the wrong run to time. With the cache as the profile fills it, which held 39% of what the answer
routed to, a later layer leaves the CPU four to six experts and 250 to 390 microseconds of work, and the card is
done first: a token of a window waits 1 ms for it and computes experts for 15. With the cache following the
answer, as in the benchmark, it waits 1.5 to 3 ms and computes for 13. A second stream for a token's shared and
cached experts, beside the other token's layer, was built for the card being behind. It wrote the same tokens
and no faster, 50.0 and 55.3 tokens/s where 49.7 and 55.3 with the same experts cached, and through the server
56 to 57, 54 and 54 to 55 where 56 to 57, 53 to 54 and 54 to 56. It is not in the engine.

The CPU's experts are at what the memory gives. With 32 to 52 threads on the node's 52 cores an expert takes
50 microseconds where 56 with 24, and three to seven of the 42 layers take 300 to 500 each: the machine is
shared. What the cache follows was tried with other constants too (an expert coming in after one use, a longer
memory, more uploads at a time). None wrote faster than the shipped ones' 60 to 63 tokens/s on the same prompt;
the one with the most uploads had 51% of the experts from VRAM where 48% and wrote 59.

What was left was outside the layers, a tenth of an answer's time. Before each window the engine went through
every expert of the model once for each upload it could start, to find the one to bring in and the one to
replace: 0.64 ms a window. It goes through them once now. Each pick of a token made a pair of every token of
the vocabulary and its logit and compared them all, 0.3 ms, and a guess read the block's logits twice, a compare
or a branch a value, another 0.3. The most likely token is now the first value equal to the largest, found as
eight maxima side by side; with a temperature the best few are a heap that a token joins only when it beats the
worst of them; the guess's sum skips 64 logits at a time when none is near the best. No answer changed: 36
through the server's protocol with six sampling settings, for GLM and Qwen3.6, and every pair of experts the one
pass picks is the pair the passes picked (4,211 for GLM, 1,538 for Qwen3.6 and 552 for Gemma 4 held to 8 GB).
The models with every expert in VRAM gain most, since a token is 3 to 6 ms for them.

GLM read a prompt at 325 to 365 tokens/s, and that was timed last, a 1,024-token call with a wait for the card
after each part. A layer's uncached experts, 223 of a Q2_K layer's 288, are 1.9 GiB through the staging buffer: 30
ms, the rate of the card's link, while the card works on the layer up to its router. That took a recurrent
layer 19 ms and a sparse-attention layer 75, so the sparse ones were what the card was slow at, and not at
attending: 18.7 of the 75 ms were the query's projections, 16.3 the output's, 15.4 the indexer's scores and 7.3
the rows it stores, and in a recurrent layer 10.5 were its two gates. These are float matrices, most of them one
a head, and the kernel that multiplies them has a block for every output of every token: 32,768 a token for
the per-head key projection. From 32 tokens on they are cuBLAS products now, a BF16 matrix made floats for it
first. The four parts take 0.9, 1.1, 8.6 and 0.3 ms and the gates 0.8, a sparse layer 29 and a recurrent one 9.
The models with every expert in VRAM read a tenth faster for it; which of their matrices that is was not
timed.

After that every layer waited for its copy, and each call made the same copies again. So a prompt's calls now
go through a layer one after the other before any goes on to the next (`Engine::read()`), and the second finds
the layer's experts in the staging buffer. Only the rows of the residual have to be kept for each call between
two layers, 64 MiB a call for GLM. A read ended at a conversation's next checkpoint then, 2,048 tokens on, so
it was two calls of the default size; with calls of 512 tokens it was four, and GLM read at 371 tokens/s where
235.
Its six layers of Q4_K and Q3_K experts need the buffer twice a call, and went on copying both times for every
call. Reading with calls of 2,048 tokens (`--prompt-batch`) was 710 tokens/s before this change, for 185 experts
fewer in the cache; it is the same copy for twice the tokens.

A read stopped at a checkpoint because a checkpoint is the state between two calls, and a read's calls are not
one after the other. Reads of 8,192 tokens, tried with every call's residual in VRAM, were 630 tokens/s where
533. So a read now goes past the checkpoints inside it and takes them a layer at a time: when a layer has run
the call that ends at one, the layer's state is what a call-by-call reading would find there, and it is copied
before the next call changes it (`save_layer()`). The residual of more than two calls waits in pinned host
memory, 512 MiB for GLM, and goes through the two halves of the card's buffer in turn, copied out and in on a
stream of its own beside the card's work on the call between. That keeps the VRAM as it was, and GLM reads at
602 to 616 tokens/s, 3% under what the residual in VRAM gave.

A call at the end of a read can be small, and one of fewer than 256 tokens copied its own experts as it does
alone (see the checks): GLM's longest benchmark prompt ends in a call of 229 tokens, which copied the experts it
routes to once more for every layer. A prompt has such a call whenever fewer than 256 tokens are left after its
last checkpoint or after a call of 1,024. In a read of several calls every call now copies ahead, whatever its
size, so a small one finds the layer's experts where the call before left them. GLM reads 7,398 tokens that end
in a call of 230 at 629 to 630 tokens/s where 582 to 585, and 6,244 that end in a call of 100 after a checkpoint
at 625 where 550; through the server the benchmark's longest prompt went from 534 to 539 to 579 to 582. The
small call's launches are a larger call's then, and its sums are split another way than they were, so the
answer after such a prompt can leave the earlier build's at a near tie.

This page said of an earlier timing that a layer spent 12.5 ms on the shared expert, the rows' quantization and
the cached experts, and read that as work. Timed part by part, the cached experts' rows are 1.9 ms and the
staged ones' 5.6. About 8 of the 12.5 passed before the first of those launches, with nothing queued for the
card but the routing tables' copy: a small copy, which appears to wait on the link behind the experts'. What it
holds up is the cached experts' rows, which could have run meanwhile. What is left of a 1,024-token call, with
the copies made once in up to eight calls: 9 ms a recurrent layer and 29 a sparse one up to the router, of
which 8.6 are the indexer's scores, 6.6 the host's sort of them and 11.4 the attention; 9 for the experts; and
the six layers of larger experts, whose copies were made for every call then, 70 to 95 ms each.

Those six pass their experts through the staging buffer in two rounds, and a read of eight calls copied both
for each: 96 copies. The buffer still holds a call's last round when the layer's next call comes, so that call
now takes the round first, without a copy, if its experts are exactly one of its own rounds. The experts copied
ahead are the same for every call. The second round is what a call routes to of the rest, so the next call has
it only if it routes to the same experts; when it does not, it copies both, the experts that go ahead last, and
the call after finds them. Of the 42 calls after a layer's first, 18 found the experts that go ahead, 15 the
other round and 9 neither, which is 63 copies, and GLM reads 7,788 tokens at 639 to 643 tokens/s where 599 to
602. With calls of 512 tokens the other round is there for 6 of the 42 calls that look for it, and GLM reads at
447 to 449 where 409 to 411. Those 7,788 tokens are one document of 1,947 four times over. The benchmark's
longest prompt, 7,398 tokens read through the server, went from 511 to 522 tokens/s to 533 to 535.

A call that looks at its routing first copies nothing ahead, and the copy ahead is free while the card is busy
up to the routing. With a staging buffer of 128 MiB (`--stage-mib`) every layer of Qwen3.6 held to 8 GB has two
or three rounds and almost never the same later one, and a build that always kept the round read 7,709 tokens
at 1,920 to 2,000 tokens/s where 2,180 to 2,230. So the engine times both, and keeps a round only if the card got to the
routing in under half the time the layer's copy ahead took. Qwen3.6 with that buffer copies ahead in all but a
few calls and reads as it did.

A cache that an answer has swapped experts into reads a prompt a little slower. A prompt batch takes a layer's
cached experts in runs of adjacent slots, and swaps scatter a layer's: after a 300-token answer a layer's cached
experts took 63 launches where 9, and in the build that times the parts a 7,788-token prompt took 13.8 s where
13.3.

Qwen3.6, Ornith and Gemma 4 read a long prompt ever more slowly: Qwen3.6 took a prompt's first 4,096 tokens at
4,190 tokens/s and the ones after 28,000 at 1,240, and read 30,000 only a tenth more slowly held to 8 GB than
with the whole model on the card. It was the attention of a prompt batch. That took as many tokens at a time
as had their scores, for every row they attend, in a scratch of 64 MiB: 32 tokens at a context of 32,768,
whatever the prompt's length, and cuBLAS is slow on a product with so few columns. Timed part by part, 30,000
tokens were 6.3 s of the values' products, 2.0 s of the scores' and 0.5 s of the cache's rows made floats, once
for every 32 tokens, in a read of 13.5 s. Gemma 4's window layers, which attend 1,024 rows, took 32 tokens at a
time as well.

A prompt batch's attention now takes the call's tokens together and the rows a block at a time, as many rows as
have their scores in the same scratch: 1,024 for a call of 1,024 tokens. A token's softmax goes on from block
to block: the largest score so far, the sum of the exps against it, and the values so far shrunk when a block
has a larger one. The products have every token of the call for their columns whatever the context, and the
same 30,000 tokens are 1.0 s of the values' products, 1.6 s of the scores' and 0.06 s of the rows. A prompt of
30,000 tokens is read at 3,835 to 3,860 tokens/s where 2,214 to 2,218, one of 10,000 at 4,841 to 4,862 where
3,798 to 3,811, and the three models' prompt of about 2,600 at 4,127 to 4,272 where 3,500 to 3,972.

The scratch for the rows as floats held a context's rows and needs a block's now, 16 MiB where 128 for
Qwen3.6, and what it gave back went to the expert cache. That mattered on a small card: the scratch for a
call's tokens had grown with the change before, and Gemma 4 held to 8 GB had 182 experts in VRAM at a start
where 207, and wrote a short chat at 125 tokens/s where 137. It has 271 now and writes at 135 to 142.

After that the two products were still 2.6 of the 7.9 s Qwen3.6 took to read 30,000 tokens: cuBLAS did them in
single precision, at 30 and 41 TFLOPS on this card. They now take halves and sum floats, which the card does on
its tensor cores, at 155 and 135 TFLOPS for the shape of a 1,024-token call. Floats rounded to TF32 gave 79 and
77. The keys and values are halves in the cache, so the products read the cache's rows where they lie, and the
copy of a block's rows as floats is gone with its scratch. The queries and the softmax's weights are rounded to
halves, as llama.cpp's attention has them, and the scores stay floats. Two things had to be found on the way.
cuBLAS is half as fast when a block's rows are not a multiple of 8, 62 TFLOPS on 691 rows and 138 on 688, so a
block is cut to a multiple of 16. And it takes 75 ms over a process's first product of halves, which a prompt
of 2,100 tokens felt, so the engine does a small one when it starts. The two products are 0.7 s of the 30,000
tokens now, and the read 5.9 s.

GLM read a prompt of 28,354 tokens in 51.9 s from the command line, at 546 tokens/s. Timed by layer, with a
wait for the card after each part, a call of 1,024 tokens was 1.0 s of layers up to their routers and 0.9 s of
experts. The 34 recurrent layers took 10.8 ms each up to the router, 7.2 of them the state's step. The 11
sparse ones took 58 ms: 32 for the indexer's scores, which grow with the context, 9 for the host's sort of
them and 13 for the attention.

The recurrent step had one block of 1,024 threads a head walk the tokens: 64 blocks on a card with 170
processors, each waiting for its warps three times a token and reading and writing the state in the card's
memory. For a prompt batch it is three launches now: the norms of every token at once, the state with a
column's values in registers and 16 columns a block, and the output's norm. That is 1.5 ms where 7, with the
same bits. The sums add in the old order, and which product is fused into which sum is written out in both
kernels, because the compiler fused the new one's another way. The state after a read has the same hash from
both.

The indexer scores every pool of four rows for every token, with 32 heads of 128 values: 59 Gflop for a call
at 28,000 tokens, which a kernel with a block a pool and token did at under 2 TFLOPS. One cuBLAS product of
the queries and the pools' keys, which lie four rows apart in the indexer's cache, does it in 2 ms. A kernel
that kept the old one's order of sums, a thread a pool and token, was no faster than the old one. The scores
have other last bits, and where pools tie at the cut another one can be selected: 5 selections of 63,096 on a
prompt that is one document four times over. GLM's attention reads only the rows that were selected, so after
such a prompt the tokens can leave the earlier build's with probabilities far apart, not at a near tie. After
that prompt they do at token 77 of 200.

The selection went to the card as well. The host got 33 MB of scores a layer and call, sorted each token's
pools on its threads, which copy the layer's experts at the same time, and sent the lists back. A kernel now
finds the key that 512 pools reach a bit at a time, takes the pools above it and the lowest of those on it,
and gives each its place by counting. The lists are the host's, in the host's order, for all 63,096, and the
state's hash is unchanged.

GLM reads the 28,354 tokens in 34.8 s from the command line, at 813 tokens/s, and at 784 to 785 through the
server. What a call of 1,024 tokens still takes there: 0.4 s up to the routers, 13 ms of each sparse layer
in its attention, whose products are floats, and 0.86 s of experts. Of those, 0.32 s are the six layers of
larger experts. Their experts outside the cache are 2.6 to 3.8 GiB a layer and the staging buffer is 2 GiB, so
every call copies one of a layer's two rounds.

A guess was asked for more seldom than it paid. After the block's kernels, which take 0.46 ms, the engine
copied the guess's logits to the host and summed the softmax there on one thread. That took 0.1 ms where the
block is sure, because few tokens are within 20 of the best, and 1.2 to 2.3 ms where it is not. The guessing
weighs what a guess costs against what it brings. Before a round's first guess it asks whether a guess the
block were sure of would pay for the asking, by the time of the last guesses and of the last windows of two
tokens, and while the answer is no neither is measured again. With Ornith the second window of two of a
process was enough, 8.8 ms where 6.7 is usual: a 256-token answer about code then had 68 guesses offered and
40 of them right, and came at 238 tokens/s. A short chat of Qwen3.6 UD-Q2_K_XL that came at 231 tokens/s, where
four other runs had 321 to 335, was the same thing after a slow start.

A kernel now finds the first of the largest logits and its probability, one block of 1,024 threads, and eight
bytes come back. A guess of Ornith's takes 0.56 to 0.60 ms and one of Qwen3.6's 0.48. Ornith's answer about
code has 153 guesses offered and 127 right, and comes at 291 tokens/s. The question before a round's first
guess is still asked, and a window size that looks slow is still not measured again. With guesses at this
price neither stopped the guessing in these runs, also not when an answer's first 16 rounds were made 3 ms
longer each: UD-Q2_K_XL then wrote at 332 to 342 tokens/s in four runs, where the build before wrote at 243 to
251 in three of four and at 310 in the fourth. GLM writes as it did: a 400-token answer came at 60 to 66
tokens/s before and at 60 to 67 after.

Both are measured again now (`src/poly/guesses.hpp`). A time older than 32 rounds is not believed: until its
length is run again a token more counts as one the policy has not seen, and asking as free. One slow round
among fresh ones moves a time by a tenth. A process of Qwen3.6 held to 8 GB that was held still eight times
for 0.3 s from its first answer's 40th token had 68 guesses offered in that answer of 400 tokens and 22 in the
next; with the change it had 283 and 321. That was one run each, on Linux.

The last two changes are from [the laptop](#on-a-laptop-with-8-gb-of-vram). The staging buffer was 2 GiB for
every model, about what a layer of GLM's leaves out of the cache. A layer of Gemma 4 UD-Q3_K_M has 328 MiB of
experts in all and one of Qwen3.6 UD-Q3_K_M 341, so the rest of the buffer was never written, and it is
allocated before the expert cache, which takes what is left. On the laptop's card, with 394 MiB of it in use by
the desktop, Gemma 4 at a context of 32,768 left 801 MiB free, less than the 1,024 the engine leaves alone. The
buffer is now the smaller of `--stage-mib` and what the largest layer's experts take. A layer's experts still
pass in one round, and GLM keeps the buffer it had.

The pool that computes the CPU's experts posts a batch, takes tasks beside its workers and takes the batch
down when its tasks are taken. Between a batch's last task and that, a worker that found the batch still posted
went round it without a pause: it counted itself in, found no task and counted itself out. While the thread
that posted has a CPU, the gap is a few instructions long. It has none when there are more threads than CPUs
that run them, and then the workers spent their time slices going round, at every batch, of which a token has
60 or more. The first machine never had fewer CPUs than threads. The laptop has once its performance cores
stop taking threads. A worker now keeps the batch count at which it found nothing to do and only reads the
count until it moves. It sleeps on the count too, so the mutex and the condition variable are gone, and with
them the turns the woken workers took at the lock. With every core allowed and 20 threads, through the
laptop's own loss of its cores, the answer went from 37 tokens/s to 0.7 before the change and to 16 to 17
after it, and with 12 threads from 35 to 26. Both changes were built and measured on Windows only.

The three changes after those are from the laptop too, and are about what its card is slow at. By their
specifications an RTX 4070 Laptop reads its memory at a seventh of the 5090's speed (256 against 1,792 GB/s),
and a layer's dense weights are read for every call: 1.67 GiB over Gemma 4's 30 layers. Timed on the card's own
clock, from a call's experts to its next router, a layer took 0.46 to 0.50 ms for one token, 0.54 to 0.58 for two
and 0.56 to 0.61 for two to four. A window of
guessed tokens went a token at a time, one behind the other, so the card read every layer once for each token: a
window of two took 36 ms and one of three 53, and the card was what the answer waited for.

A window now goes in two halves, each a call of its tokens, or whole. In halves the card works on one half's
layer while the CPU computes the other's experts, and reads a layer twice. Whole, the card's part and the CPU's
take turns, and the layer is read once. Which takes less depends on the card, the CPU and how much of a token is
left to the CPU: on the laptop a window of two took 31 to 32 ms whole and 36 in halves, one of four 47 to 48 whole
and 43 to 46 in halves, and with 300 more experts in VRAM the whole one took 43. So the engine times both for
each window size, three windows each, takes the faster and gives the other every 32nd window. With the cache held
still, the answers are token for token the build before's.

The cache took what VRAM was free but for 1,024 MiB on any card. What that reserve is for is what the engine
allocates later, the launches it records as it runs: 200 MiB over prompts of 25 to 31,000 tokens on the 8 GB
card. It cannot be nothing. Under Windows a card that is asked for more than it has moves memory to the system's
and goes on: started with a reserve of 96 MiB the engine had no VRAM free after a 31,000-token prompt, read it at
1,108 tokens/s where it reads at 1,387 to 1,450 and wrote after it at 13 where it writes at 28 to 31. The reserve
is now a sixteenth of the card, between 512 and 1,024 MiB: 512 on 8 GB, and what it was from 16 GB on. The engine
looks before each request, and when less than a quarter of the reserve is free it sizes the cache again. The
server's `POST /v1/vram` does the same on request ([DETAILS.md](DETAILS.md)): the cache is emptied and filled
for another reserve with the experts used most so far, in 11 ms to give up 1.6 GiB and 262 ms to take it back.

A read took up to 8,192 tokens, and each read copies every layer's experts outside the cache to the card once:
four times for a 31,000-token prompt, on a card that has four fifths of the experts outside its cache. A read is
now as many tokens as leave their rows of the residual in 512 MiB of host memory, which is 8,192 for GLM 5.3
Flash, as before, and more than a context of 32,768 for the three new models. Between two layers a read asks its
caller, who answers with a progress line or, when the request was stopped, gives the read up; the conversation is
then back at its last checkpoint. A stop used to wait for the read under way. Of the checkpoints inside a read
only the ones the conversation keeps get a state: fifteen are due in a 31,000-token prompt and five are kept.

After a long prompt an answer came at half a short chat's speed, and the four changes after those are about
that. An answer's attention reads every row of a layer's keys and of its values: after 31,000 tokens that is
127 MB of each in five of Gemma 4's layers, and 32 MB of each in ten of Qwen3.6's. The laptop's card moves
256 GB a second, so Gemma 4's keys cannot be had in less than half a millisecond a layer. The kernel that
scores them took 1.74 ms. A thread of it was a row and a head and summed that row's key by itself, so the 16 or
32 threads that run in step each read another place of the cache. A warp now has a row and a group of heads, and
its threads read the key next to each other, four values each at a step: 0.43 ms a layer, and for one token the
attention's three kernels together 1.12 ms where 2.25, for Qwen3.6's shape 0.45 where 1.12.

A window paid that for each of its tokens. A launch of the kernels now takes its tokens together, over the rows
they attend between them. A key and a value are read once. A token that does not attend one of those rows has
the weight zero there, which adds nothing to its sum, and its sum is put together as it is when the token goes
alone, so it has the same bits in any company: with every expert on the CPU, answers with windows and without
are the same tokens. What these kernels wait for is the card's memory, and how much of its speed they get is a
matter of how many reads are on their way at a time, which is why a value's sum is in 32 shares now, a warp
each, where it was in 8. A call at 31,000 rows, for one to four tokens, in ms:

| | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4's shape, before | 1.12 | 1.96 | 3.18 | 4.45 |
| now | 1.12 | 1.16 | 1.25 | 1.40 |
| Qwen3.6's shape, before | 0.51 | 0.99 | 1.43 | 1.87 |
| now | 0.45 | 0.53 | 0.91 | 1.05 |

At 8,000 rows a call of four takes 0.35 ms where 0.65 for Gemma 4 and 0.26 where 0.51 for Qwen3.6, and in a
window layer of Gemma 4's, which attends 1,024 rows, 0.054 where 0.103.

A window in one piece then costs less than its halves after a long prompt, 39 to 41 ms against 50 for four
tokens of Gemma 4, and the engine did not come to that within an answer. Right after a prompt the cache has few
of the experts the answer will use, 16% of those of its first windows without a profile and 70% some 25 windows
later. A whole window waits for the experts the CPU computes, and one in halves has the card busy meanwhile: the
first whole windows took 61, 53 and 52 ms, the halves 55, 49 and 49. The way that is behind gets every 32nd
window, and its time counted an eighth against what it had, so the whole way needed 128 windows to be seen as
the faster, and an answer of 300 tokens has 80. Its time now counts for half.

That start is what the expert profile is for: the counts of the experts' use that a run leaves and the next
start fills its cache from. On Windows no run had left one. The engine saved it when it was told to quit, and
there the server's engine ends with the server, at once, when the server is ended in any other way than with
Ctrl+C in its console. And a save over an existing profile failed, because the C library's rename takes no file's
place on Windows. The engine now saves after its first answer and then after an answer when the last save is
ten seconds ago, and moves the file in a way that replaces one. With the profile of a run before it, Gemma 4
started the same answer with 66% to 70% of its experts in VRAM and wrote it at 79 tokens/s where 68.

What a profile gives depends on what came before an answer. That answer was its start's first request. In the
README's test a short chat comes first and the cache follows the chat's answer: in one run of Gemma 4 it held
39% of what the answer to the long prompt then used when that prompt came, and 60% at the answer's end. Between
a size's first run there, without a profile, and the runs after it, Gemma 4's short chat went from 66 to 70 to
72 tokens/s and its answer to the long prompt from 51 to 52 to 55. Qwen3.6's, whose cache holds a smaller share
of its experts, went from 53 to 74 and from 44 to 59, with its file coming from the disk in the first run.
Filling the cache by what the prompt used was looked at and left: the cache's slots hold 64% of that answer's
uses at the most, and 47% when they are chosen by the prompt's.

An answer that copies from its prompt, a file given back with a change or a function quoted, is where the repeat
drafter is nearly always right, and a window held four tokens: Gemma 4 wrote such a copy at 88 tokens/s, with 3.5
tokens a window. A longer window was not free for every model. A guess needs the recurrent layers' state as it
was before it, and that was a copy in VRAM for every guess, 62 MiB each for Qwen3.6. With a window of eight and
its seven copies Qwen3.6's cache starts with 1,784 experts where 1,976, and it writes the copy at 111 tokens/s
where 90, a short chat at 70 to 74 where 72 to 76 and an answer after a long prompt at 60 to 62 where 62 to 63.
Gemma 4 has no such layers, so its window went to eight first. The draft block still guesses three at most, each
of its guesses being a run of the block: with it free to fill eight, Gemma 4's short chat came 1% to 3% slower.
Gemma 4 then writes the copy at 100 to 107 tokens/s and the quote at 83 to 88 where 74 to 78, with its short chat
and its answer after a long prompt where they were. The two were asked with the model's thinking off, through
the server, after 21,900 tokens that are three source files. With thinking on, the first 600 tokens of the same
answers are the model's reasoning, which copies nothing, and came no faster with a window of eight.

Qwen3.6 and Ornith then got the window of eight with the three copies a window of four has. The delta net's
kernels leave the state after the tokens they are told, and a window of eight leaves it after its first, third
and fifth. When the last right token is none of those, the engine goes back to the nearest state before it and
puts the one or two right tokens after it through the layers again, which come out as they did. Qwen3.6 then
writes the copy at 112 to 113 tokens/s where 90 to 92 and the quote at 83 to 85 where 77, and Ornith at 120 to
121 where 99 to 102 and at 86 to 90 where 82 to 83. Qwen3.6's cache starts with 1,972 experts where 1,976, and
the card's memory in use after the two answers was 7.73 to 7.75 GiB where 7.65 to 7.66: the longer windows'
launches are recorded too, out of the reserve the cache leaves. With every expert on the CPU, where two runs
write the same tokens, the three models wrote the same text for the two requests as without guesses, and
Qwen3.6 also with a switch for the test that cut its windows short at two to seven tokens. GLM 5.3 Flash's
recurrent kernels leave the state after a call's first tokens only, so its window stays at four.

At first any repeat could fill a window of eight. A repeat of three or four tokens is in any text, and what
followed it the first time is seldom what follows now for seven tokens, so the later guesses of such a window
were computed for nothing: Gemma 4 wrote a story at 68 to 71 tokens/s where 76 to 79 with a window of four. A
window is now longer than the draft block's four only when the repeat's match is eight tokens or more. Gemma 4
then wrote the story at 76 to 86 where 72 to 79 in the same hour, and gives the file back at 103 tokens/s where
101.

The laptop's processor has cores of two kinds, and Windows may hold the threads of a process it takes for
background work to the slower kind. The experts that are not in VRAM are computed on those threads. Since the
merge of the Strata project's 0.1.41 the server takes the engines it starts out of that power throttling, and
strata-poly does the same at its start, for an engine that anything else starts. In one hour, with the server
from before the merge, Qwen3.6 wrote the answer after a 31,000-token prompt at 39 tokens/s in four runs of
four, with the server of the merge at 57 to 62, and with the old server and its engine taken out from outside
at 58. Started from a shell, the engine wrote 300 tokens after 7,709 at 48 to 50 tokens/s in four runs of six
and at 66 in the other two, and with its own leaving at 66 to 72 in six of six. The laptop was on mains power,
on the High performance plan, and its owner was at work on it. The pass of the night before, with nobody at
it, wrote that answer at 59: it had not met the throttling. The threads that left the performance cores in
the first measurements here ([On a laptop with 8 GB of VRAM](#on-a-laptop-with-8-gb-of-vram)) may have been
this, which was not tried again. Without the throttling, 16 and 20 threads wrote two short answers 2% to 7%
faster than setup's 13 in three rounds of four and no faster in the fourth: the count stays.

These eight were built and measured on Windows only, and GLM 5.3 Flash, whose attention has kernels of its own,
was not run with them.

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

- One NVIDIA card. Built and run on one Linux machine, and the three new models' nine sizes on one Windows
  laptop as well. GLM 5.3 Flash was not run on Windows.
- The changes that came from the laptop were built on Windows first, and on Linux on 2026-10-10 (gcc 11.4,
  CUDA 12.8), where the first sizes were checked against llama.cpp and with and without guesses again, and the
  benchmark of [A prompt of 30,000 tokens](#a-prompt-of-30000-tokens) was run. GLM 5.3 Flash, which only that
  machine has the RAM for, stopped in the middle of an answer with them until the pool of the CPU's threads was
  fixed that day. The engine's leaving of the power throttling is Windows' alone. The pool sleeps with
  `std::atomic`'s `wait`, which on Linux needs libstdc++ 11 or newer.
- The target of 8 GB of VRAM with 32 GB of RAM was measured on that one laptop. Each model's largest file
  takes its RAM to 28.6 to 31.4 GiB of 31.8 with what else runs there, and the two largest of all then read
  part of a prompt from the disk.
- On the laptop an answer on another subject than the one before starts with fewer of its experts in VRAM
  than it ends with: the cache makes that up over the answer's first windows. That was not worked on.
- Writing on the laptop still moves with what else it does. On a day with its owner at work on it, runs of
  two short prompts wrote at 40 to 84 tokens/s.
- Setup installed the three new models from files that were already on the disk, on both machines. Of the new
  files its own download fetched Gemma 4's draft model (0.46 GB, through a mirror) and found the pinned
  SHA-256; the three addresses of the model files were asked for their headers only, and each states the
  pinned size. For the laptop the files were fetched by setup's download and checksum functions, called from
  a script. The CUDA Toolkit and Visual Studio were on the laptop before setup ran, so its install of the
  two was not run for this engine.
- Setup's RAM figures for them are estimates.
- Pictures were read with the encoder on the GPU of the first machine. Setup puts it on the CPU of a card with
  less than 16 GB: that was run through the server with Qwen3.6 on the first machine (a new picture took 3 to
  5 s with 12 threads), and on the laptop with the nine sizes, a picture each (3.8 to 7.4 s).
- Setup's install with images ran on the test machine for the three models, with the encoder files already
  beside the models' files: it compiled the encoder and wrote the run configs, and the server read pictures
  with them. The files had been fetched before with setup's download and checked against the pinned SHA-256.
- A picture of Gemma 4's goes into the model in one prompt batch, 1,024 rows unless `--prompt-batch` says more;
  one with more rows is refused. Setup holds pictures to 1,024 rows (`--vision-tokens` changes it).
- Video and audio are not read.
- With everything in VRAM, llama.cpp reads prompts faster on the same card, 6,000 to 9,000 tokens/s against
  4,100 to 5,800. It writes 165 to 180 tokens/s; this engine writes 150 to 205 a token at a
  time and more with guesses ([Benchmark](#benchmark)). The router's scores still go to the host and back in
  every layer, which a model that fits does not need.
- Ornith's guessed tokens pay less than Qwen3.6's, though the two are one architecture. Its block offers fewer
  guesses and fewer of them are right: in a 256-token answer about code 127 of 153, at 291 tokens/s, where
  Qwen3.6 has 166 of 190 and 342, and in a short story 28 of 41 and 237 against 160 of 187 and 335. Their
  matrices are of other formats too, and what a window costs each of them was not compared.
- GLM's recurrent layers use the step the delta net had before it was rewritten for Qwen3.6 in every call of
  fewer than 32 tokens, which is all of them while it writes. Its writing is bound by the experts that are not
  in VRAM, so it was left. Its window of guessed tokens stays at four: its kernels were not given the choice of
  the tokens that leave a state, as GLM could not be run where that was made.
- No test runs the engine without a GPU and the model files: what is said of it here was measured by hand on the
  two machines.
- The benchmark's VRAM and RAM figures are the machine's, not the engine's alone.
