# Strata on an Intel Arc

Strata's engine is CUDA. On an Intel Arc the same model runs behind the same Strata server, with
llama.cpp's SYCL backend doing the arithmetic. Everything above the engine - the OpenAI and Anthropic
APIs, streaming, tool calls, MCP, the web app - is unchanged.

Written for and measured on an **Arc Pro B70 (32 GB)** running the Coder (IQ1_M) on Ubuntu 24.04.

## What you get

| | NVIDIA (Strata engine) | Intel Arc (llama.cpp engine) |
|---|---|---|
| model files | the same GGUFs | the same GGUFs |
| where the model lives | experts in RAM, hot ones on the card | **all of shard 1 on the card** (it fits a 32 GB card); shard 2's lookup table paged from disk |
| RAM needed | 32-64 GB | little: the model is on the card (23 GB was fine) |
| decode, Coder IQ1_M | 44-51 tok/s on an RTX 5070 | 23-25 tok/s on the B70 |
| prompt reading | ~1,870 tok/s | ~150 tok/s |
| speculative decoding (MTP) | yes | no (the GGUF carries no draft layer llama.cpp can use) |
| images | yes | not yet |
| context | up to 262K | 32K by default; more as VRAM allows |

The speed gap is the SYCL backend's known state on Battlemage: general matrix multiplies do not use
the XMX units yet (only the oneDNN flash-attention path does). It is upstream llama.cpp work, not
something in Strata. Decode is GPU-bound at 90%+ busy, so nothing is falling back to the CPU.

## Setup

    ./setup.sh

Setup notices the Arc (no `nvidia-smi`; an Intel GPU under the `xe` or `i915` driver in sysfs), says
so, and takes the Intel path: no compiler, no CUDA toolkit. It downloads the model, exports the
tokenizer, finds a `llama-server`, and writes a config with a `"llama"` block and a start script.

Where the `llama-server` comes from, in order:

1. `--llama-server PATH` - a llama.cpp you built yourself with `-DGGML_SYCL=ON` (needs oneAPI).
2. `llama-server` on PATH.
3. A container: `ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable`, a community SYCL
   build for Battlemage (llama.cpp 0.4.1, IntelLLVM 2026.1, AOT for bmg-g31). Needs docker and the
   `render` and `video` groups. Setup writes `start-llama.sh`, which runs it with the GPU passed
   through and the right environment. The run script starts it when nothing is answering yet.

Then `run-<model>.sh` (or `./setup.sh` again) starts the model. First start of a 30 GB model: about
two minutes.

## The config

```json
{
 "engine": "llama",
 "tokenizer": ".../packs/coder-iq1_m/tokenizer",
 "llama": {
  "url":   "http://127.0.0.1:8081",
  "model": ".../Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
  "start": ".../start-llama.sh",
  "image": "ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable"
 }
}
```

`url` attaches to a running llama-server; `start` is run first if nothing answers there. A config
with `"exe"` and `"args"` instead spawns a native binary directly. See `serve/engine_llama.py`.

## Things that matter on this GPU

- **`SYCL_CACHE_PERSISTENT` must be 0.** The persistent JIT cache segfaults on Xe2 during the first
  compile. The start script sets it; if you run llama-server by hand, do too.
- **The whole model goes on the card** (`--n-gpu-layers 999`), except `per_layer_token_embd.weight`,
  the single 28.8 GB tensor of shard 2. `--override-tensor per_layer_token_embd=CPU` keeps it in host
  memory, mmapped and paged from the SSD by row, which is how the model's authors serve it. Host RSS
  stays around 2 GB.
- **Thinking.** The model reasons before it answers. Strata's web app has the setting; for an API
  client, put `{"reasoning_effort": "none"}` in `strata-<model>.shared-settings.json` next to the
  config, or send `chat_template_kwargs: {"enable_thinking": false}` per request. Without one of
  those a short `max_tokens` is spent entirely inside the think block and the answer looks empty.
- **`/health` says 503 while loading**; the server polls `/props` instead.

## Not done

- Images. Strata's vision path encodes with `strata-vision` into embeddings the CUDA engine reads;
  llama-server takes the image itself (`--mmproj`). Wiring that is the next piece.
- Speculative decoding. Strata's MTP draft is its own packed format; the GGUF has none.
- The expert-residency profile and n-gram lookup: moot with the model on the card.

## Measured, 2026-09-29, Arc Pro B70, Coder IQ1_M, 32K context, q8_0 KV

| | |
|---|---|
| load | ~100 s |
| VRAM | 28.4 of 32 GB |
| decode | 23-25 tok/s; GPU 92% busy at 165 W, CPU idle |
| prefill | 149 tok/s on a 2,701-token code prompt |
| quality | correct code on every test; matches the NVIDIA path token for token in spirit, not measured |
