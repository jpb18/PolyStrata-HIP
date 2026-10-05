# Models and backends

PolyStrata runs more than one model family. This page says which, how the engine is put together so that it can,
how a model is recognized, how to add one, and how to change the model that answers.

One model is in memory at a time. The measurements below are from one machine: an RTX 5090 (32 GB), 24 threads of
a Xeon Platinum 8470Q, 120 GiB of RAM, Linux, CUDA 12.8. On that card the three new models fit in VRAM whole, so
the numbers say nothing yet about a card with 8 GB.

## What runs

| Model | Architecture | Engine | File it was checked with | Guesses ahead |
| --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next and fine-tunes | `qwen4exp` | `strata` | the packs setup builds | separate MTP model |
| GLM 5.3 Flash | `glm5-next` | `strata-glm` | GSQ-RCO 3.0bit, REAP50-Q3_K_M, UD-IQ2_XXS | draft block |
| Qwen3.6-35B-A3B | `qwen35moe` | `strata-glm` | Unsloth UD-Q3_K_M, 17.1 GB | draft block in the file |
| Ornith 1.5 35B-A3B | `qwen35moe` | `strata-glm` | APEX-MTP-I-Compact, 17.4 GB | draft block in the file |
| Gemma 4 26B-A4B | `gemma4` | `strata-glm` | Unsloth UD-Q3_K_M, 12.7 GB | repeats of the text only |

`strata` is the first engine and runs Qwen3.8 only; nothing in it changed. `strata-glm` is the engine that reads a
GGUF in place, and it now has three architectures. Its name is from its first model and will change.

Setup installs Qwen3.8 and GLM. The three new models have no setup step yet: see
[Adding a model](#adding-a-model-of-an-architecture-that-is-there).

## How a backend works

There are two levels.

**The process.** The server starts an engine as a child process and talks to it over lines of text: `GEN` with
token ids goes in, `T` lines with tokens and a `DONE` line with timings come out. Any program that speaks those
lines is a backend to the server. Changing the model stops the child and starts another, so its VRAM, its pinned
memory and its file mappings are given back by the operating system when the process ends, not by cleanup code
that could miss something.

**The engine.** Inside `strata-glm`, `Engine` (src/glm/engine.hpp) is what every architecture shares:

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
| `row_bytes()`, `save_rows()`, `load_rows()`, `extra_bytes()`, `save_extra()`, `load_extra()` | what a conversation keeps |

A layer's own weights live in the class's `Block`, one a layer, beside the shared `Layer`. The engine calls a hook
once a layer or once a call, never inside a kernel, so the calls cost nothing that can be measured: GLM writes at
the same speed before and after the split (51.8, 50.6 and 46.1 tokens/s against 50.7, 49.5 and 45.8).

Nothing in the shared code asks which model it runs. What differs between architectures is a hook or a value the
architecture sets (`gelu_` for the experts' activation, the distance between experts in a fused tensor).

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

## The registry

`serve/models.py` lists the installation's models. A model is a run config, `strata-<name>.json`, beside
`setup.py`: the engine, its arguments, the tokenizer directory. An entry has:

| Field | From |
| --- | --- |
| `id` | the config's `model_name` |
| `display_name` | the GGUF's `general.name` |
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
{"qwen3.6-35b-a3b": {"display_name": "Qwen3.6 35B-A3B", "notes": "the daily one"}}
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

Tokens per second on the machine above, context 32768, every expert in VRAM for the three new models.
"Before" is the first build of each backend, "after" has the changes under [What was tuned](#what-was-tuned).

| Model | Prompt, tokens | First token, s | Reads, tokens/s | Writes, tokens/s | Experts from VRAM | Guesses right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 21 | 0.20 | 106 | 110 to 108 | 100% | 88% |
| | 2,486 | 3.0 to 2.9 | 831 to 859 | 97 to 106 | 100% | 94% |
| | 10,083 | 12.7 to 9.1 | 636 to 889 | 68 to 84 | 100% | 86% |
| Ornith 1.5 | 21 | 0.20 | 105 | 87 to 90 | 100% | 93% |
| | 2,486 | 3.0 to 2.9 | 825 to 854 | 81 to 86 | 100% | 93% |
| | 10,083 | 12.8 to 9.2 | 632 to 877 | 54 to 68 | 100% | 78% |
| Gemma 4 26B-A4B | 22 | 0.25 | 88 | 95 | 100% | |
| | 2,544 | 10.9 to 1.2 | 235 to 2,192 | 69 to 82 | 100% | 21% |
| | 10,465 | 39.5 to 3.7 | 213 to 2,346 | 55 to 75 | 100% | 45% |
| GLM 5.3 Flash 3.0bit | 23 | 0.87 | 27 | 43 | 48% | 87% |
| | 2,337 | 7.4 | 317 | 43 | 50% | 80% |
| | 9,497 | 23.4 | 319 | 47 | 59% | 85% |

Where a cell has two numbers they are before and after. GLM's numbers did not change. A short prompt is read a
few tokens at a time, which is why its reading speed is low. Gemma 4's guesses come from repeats of the text
alone, so there are few of them and fewer are right.

llama.cpp (Unsloth's fork at 86ebfef) with every layer on the same card writes about 180 tokens/s with Qwen3.6
and Ornith and 165 with Gemma 4, and reads prompts at 6000 to 9000. It cannot do that on a card the model does
not fit, which is the case this engine is for; that case has not been measured for the new models yet.

## How each backend was checked

Against llama.cpp on the same file: greedy, 64 tokens after five chat prompts of about 25 to 8000 tokens, the
token ids compared and the probabilities of the reference's five likeliest tokens at every step.

| Model | Same tokens as llama.cpp (of 64) | Largest probability difference |
| --- | --- | --- |
| Qwen3.6-35B-A3B | 64, 64, 64, 64, 64 | 0.08 |
| Ornith 1.5 | 60 of 60, 7, 56, 2, 64 | 0.35 |
| Gemma 4 26B-A4B | 64, 64, 32, 44, 64 | 0.08 |

Two correct implementations do not write the same tokens for long. Activations are quantized to 8 bits at every
quantized matrix, in a layout that depends on the kernel, so the two differ in the last bits and a near tie
between two tokens goes either way. Ornith's file has 3-bit experts and 4-bit attention and is the most sensitive:
llama.cpp's own GPU and CPU runs of it agree for 17 of 50, 7 of 64 and 15 of 64 tokens on the three short prompts
and differ by up to 0.19, which is as far apart as this engine is from either.

Also checked, for each model: with windows of guessed tokens and without, 300-token answers are the same tokens;
with every expert on the CPU the short prompts still follow the reference to a near tie. GLM writes the same
tokens as before the engine was split, with every expert on the CPU.

The tokenizers give llama.cpp's ids on the same prompts (27 to 8001 tokens) for Qwen3.6, Ornith and Gemma 4.

## What was tuned

| Change | Model | Before | After |
| --- | --- | --- | --- |
| The batched kernels also take matrices whose rows end inside one of their steps | Gemma 4, reading 8001 tokens | 229 tokens/s | 1216 |
| A prompt batch's attention through cuBLAS | Gemma 4, reading 8001 tokens | 1216 | 3200 |
| | Qwen3.6, reading 7709 tokens | 750 | 931 |
| The attention's sum over values in eight shares | Qwen3.6, writing after 7709 tokens | 57 | 78 |
| | Gemma 4, writing after 8001 tokens | 53 | 68 |

All three are in the shared kernels, so every architecture with that kind of attention has them.

## Adding a model of an architecture that is there

1. Get its GGUF. Check `general.architecture` is one of the table above.
2. Write its tokenizer directory:

   ```bash
   python tools/strata_tokenizer.py --gguf model.gguf --out packs/<name>
   ```

   It writes `packs/<name>/tokenizer/` with the vocabulary, the merges and the model's chat template. Byte-level
   BPE (`gpt2`: Qwen, GLM) and Gemma 4's `gemma4` are read.
3. Write `strata-<name>.json` beside `setup.py`:

   ```json
   {"exe": "engine/strata-glm", "args": ["--model", "/path/model.gguf", "--max-context", "32768"],
    "cwd": ".", "tokenizer": "packs/<name>/tokenizer", "model_name": "<name>", "port": 8080}
   ```
4. `python polystrata.py --list-models` should list it as available, and `python polystrata.py --model <name>`
   should answer.
5. Check it against llama.cpp before trusting it: the same prompt's token ids into both, greedy, and compare the
   tokens that come out.

## Adding an architecture

1. Read llama.cpp's graph for it (`src/models/<arch>.cpp`) and the ggml ops it builds. That is the reference.
2. Derive a class from `Engine`. `qwen35.cpp` is the shortest example with both a recurrent layer and attention;
   `gemma4.cpp` shows the hooks for a model that departs further.
3. Put what the layers need that no kernel does yet in `src/glm/kernels.cu`, next to the others. Keep a kernel's
   result for a token independent of the tokens beside it when it runs in a window of guessed tokens.
4. Name the class in `Engine::open()` and the file in `CMakeLists.txt`.
5. Check it against llama.cpp on the same file before any tuning: tokens and probabilities on short and long
   prompts, with every expert on the CPU, and with and without guessed tokens.

## Limits

- Linux and one NVIDIA card. Built and run on one machine.
- The three new models have only run with all their experts in VRAM. The expert cache, the CPU experts and the
  staging buffer are the ones GLM uses with 45% of its experts in VRAM, but for these models a small card is
  untested, and so is the target of 8 GB of VRAM with 32 GB of RAM.
- Setup does not install the new models. Their files, tokenizer directories and run configs are made by hand.
- Gemma 4: its draft model (a separate small GGUF that shares the caches) is not run, so it guesses from repeats
  only; its tool calls are not parsed; images are not read.
- Qwen3.6 and Ornith: images are not read.
- Qwen3.6 and Ornith read prompts at about 900 tokens/s. A prompt batch launches the batched kernel once per
  cached expert, and with 256 experts a layer that is most of the time.
- A layer with attention over the cache is not recorded as a CUDA graph, because its launches take the position.
  Gemma 4 has such a layer everywhere, Qwen in every fourth.
- With everything in VRAM, llama.cpp is faster on the same card (see above). The router's scores go to the host
  and back in every layer, which a model that fits does not need.
- The benchmark's VRAM and RAM figures are the machine's, not the engine's alone.
