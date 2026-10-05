# GLM 5.3 Flash (`strata-glm`)

This fork adds a second engine, `strata-glm`, that runs GLM 5.3 Flash. It sits beside `strata`, which still runs
Qwen3.8-Flash-Next as before. The server, the web app and the API are the same for both: a run config names the
engine it starts.

It is early. It has been built and run on one machine, with Linux and one NVIDIA card. Setup does not offer it yet,
so the steps below are done by hand.

## What it was measured on

| | |
| --- | --- |
| Graphics card | RTX 5090, 32 GB |
| CPU | Xeon Platinum 8470Q, 24 threads used |
| RAM | 120 GiB |
| Model | `GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf`, 113.6 GB, from [neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF) |
| Context | 32K |

| | `strata-glm` | llama.cpp ([neurall's fork](https://github.com/neurall/llama.cpp)), stock settings |
| --- | ---: | ---: |
| Writes answers | 26.7 to 27.7 tokens/s | 26 tokens/s |
| Reads a prompt of 3,059 tokens | 373 tokens/s | about 265 tokens/s |
| Reads a prompt of 11,673 tokens | 334 tokens/s | |
| Reads a prompt of 29,069 tokens | 312 tokens/s | |
| Starts | about 6 s with the file in the page cache | |

Both ran on the machine above. Nothing here has been run on a smaller card.

With greedy sampling the two write the same tokens until two candidates are nearly tied. The model's activations
are quantized to 8 bits between layers, so a difference in rounding between two correct implementations can decide
such a tie.

## What you need

- Linux and an NVIDIA card with the CUDA toolkit. It has only run on 32 GB of VRAM.
- RAM for the model file. The file is mapped, and the experts that are not in VRAM are read from it while the
  model answers, so the 113.6 GB should fit in RAM beside everything else.
- The GGUF above. It is one file of 113.6 GB.

## Build

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON
cmake --build build --target strata-glm
```

## Run

Write the tokenizer out of the GGUF once:

```
python tools/strata_tokenizer.py --gguf /models/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf --out glm-tokenizer
```

Then write a run config, `glm.json`:

```json
{
  "exe": "build/strata-glm",
  "cwd": "/path/to/Strata",
  "args": ["--model", "/models/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf",
           "--max-context", "32768",
           "--expert-profile", "glm.profile",
           "--conversation-cache-mib", "8192"],
  "tokenizer": "glm-tokenizer/tokenizer",
  "model_name": "glm-5.3-flash",
  "log": "glm-engine.log"
}
```

and start the server with it:

```
python serve/server.py --engine strata --config glm.json --port 8095
```

The web app is at `http://127.0.0.1:8095`, and the API is the one the README describes. Chat, thinking, tool
calls and streaming were checked through `/v1/chat/completions`. `"enable_thinking": false` in
`chat_template_kwargs` turns thinking off.

## How it uses the card

The model has 45 layers. 42 of them choose 8 of 288 experts for every token, which makes 12,096 experts and most
of the file. The rest of the weights are always in VRAM.

- **The expert cache.** The experts used most often are kept in VRAM: 2,573 of them, 19.9 GiB, on the 32 GB card
  at a 32K context. `--expert-profile FILE` is where the engine keeps count of which ones are used. It reads the
  file at the start to choose, and writes it back when it stops. Without a profile it has no counts yet and takes
  the experts in layer order, so the second start is the one to measure.
- **Writing an answer.** An expert that is not in VRAM runs on the CPU while the card runs its share.
- **Reading a prompt.** The prompt goes in up to 1,024 tokens at a time. For each layer the experts outside the
  cache are copied through a buffer in VRAM and run there. A larger batch reads faster and leaves less VRAM for the
  expert cache, so answers get slower: 4,096 at a time read the 11,673-token prompt at 396 tokens/s and wrote 18.5
  tokens/s.

## Several conversations at once

The engine holds one conversation. A coding agent rarely sends just one: subagents and short side requests come in
between the turns of the main one. Each switch used to make the engine read the whole prompt again.

`--conversation-cache-mib N` gives the engine N MiB of RAM to park conversations in. When another conversation
needs the engine, the current one is copied to RAM. When it comes back it is copied in again and goes on where it
stopped. Two conversations that start with the same text, a shared system prompt for one, share what is saved for
that part.

Measured through the server with 8,192 MiB: a main conversation of 3,884 prompt tokens, a subagent of 3,323 and a
short request took turns. Every later turn of the first two had 97.8 to 99.2% of its prompt cached and took 1.3 to
3.9 s. The first turns, which read the prompt, took 13 s.

What it costs: a checkpoint is 153 MB of RAM, and a parked conversation takes 28 KB for each token. At the end of
that run two parked conversations held 1.1 GB.

The API reports what was not read again as `usage.prompt_tokens_details.cached_tokens` (Responses:
`input_tokens_details.cached_tokens`, Messages: `cache_read_input_tokens`), and the web app's Monitor has a
Conversation cache card.

It is off by default, as it is in `strata`. Without it the engine still keeps the last conversation: a next turn
goes on from where it stopped, and an edit goes back to the nearest checkpoint.

## Options

| Option | Default | What it does |
| --- | ---: | --- |
| `--model FILE.gguf` | | the model |
| `--serve` | | take requests from the server (the server passes this) |
| `--max-context N` | 8192 | tokens a conversation can hold |
| `--threads N` | all usable | CPU threads for the experts that are not in VRAM |
| `--vram-reserve-mib N` | 1024 | VRAM left free beside the expert cache |
| `--expert-profile FILE` | | expert use counts, read at the start and written back at the end |
| `--no-gpu-experts` | | keep every routed expert on the CPU |
| `--prompt-batch N` | 1024 | prompt tokens read at a time, 1 to 4096; about 1.4 MiB of VRAM each |
| `--stage-mib N` | 2048 | VRAM the experts outside the cache pass through while a prompt is read |
| `--stage-ahead N` | 256 | from N tokens, a batch copies those experts before the layer has chosen; 0 = never |
| `--prompt-cache N` | 6 | checkpoints kept of a conversation; 0 = only a next turn is not read again |
| `--prompt-cache-every N` | 2048 | a checkpoint every N prompt tokens, besides the one at a prompt's end; 0 = none |
| `--conversation-cache-mib N` | 0 | RAM for parked conversations; 0 = off |
| `--conversation-cache-slots N` | 4 | parked conversations at most |
| `--conversation-cache-min-tokens N` | 1024 | a shorter conversation is not parked |
| `--conversation-cache-min-free-mib N` | 2560 | park only while this much physical RAM stays free |
| `--eos-ids ID,ID,...` | from the GGUF | tokens that end a reply |

`strata-glm` with no arguments prints the list, with the options for running it without the server.

## What is missing

- Setup does not offer GLM, and there are no release builds of `strata-glm`.
- Windows, AMD and Intel cards, and more than one card.
- Pictures. The model's image input is not read.
- Several requests decoded together (`"parallel"`). Requests wait their turn.
- `--prompt-cache-root` and `--turn-token`. Checkpoints are taken by position.
