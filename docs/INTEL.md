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
| context | up to 262K | 32K by default; 131K is the measured ceiling on a 32 GB card (see below); 262K does not fit |

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

Two flags worth knowing on a box that already runs things: `--port N` (the default 8080 is also what
open-webui takes; the llama-server container gets N+1) and `--host 0.0.0.0` to reach it from other
machines (the default binds localhost only, as upstream does).

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

## How much context fits

The architecture keeps this small: only every fourth layer is full attention (12 of 48); the other
36 are gated-delta-net layers with a fixed-size recurrent state that does not grow with context.
Per token, the 12 attention layers keep 2 KV heads x 256 x (K + V) at q8_0, plus the
sparse-attention indexer's keys. **Measured: about 20 KiB per token** (VRAM grows 0.6 GB per 32k
of context with everything else unchanged).

| context | VRAM in use, model loaded | headroom on 32 GB | status |
|---|---|---|---|
| 32,768 | 28.4 GB | 3.5 GB | measured, the default |
| 65,536 | 29.0 GB | 2.9 GB | measured, loads |
| 98,304 | 29.6 GB | 2.3 GB | measured, loads |
| 131,072 | 30.3 GB | 1.5 GB | **measured, the practical ceiling**: a 104,798-token prompt (79% of the window) read in 247 s at 424 tok/s, then answered |
| 163,840 | ~31.0 GB | ~0.9 GB | not attempted: under the 1.2 GB safety margin |
| 262,144 | ~32.6 GB | none | **does not fit - asking for it took the host down** |

**Do not ask for more than fits.** On this driver a GPU allocation past VRAM does not fail: the xe
driver evicts buffers into host RAM, the kernel runs out of memory, and the machine livelocks until
its hardware watchdog resets it. A 262K request did exactly that on 2026-09-29, and llama.cpp's own
"failed to fit params" check fired too late to prevent it. Compute the KV size first and leave
1.5 GB free.

Prompt reading gets faster with size, because the work batches better: ~150 tok/s on a 2.7k-token
prompt, 424 tok/s on a 105k-token one. So a full 128k window costs about five minutes to read,
not fifteen.

## The engine itself on Intel: the SYCL port (`sycl/`)

Everything above runs the model through llama.cpp. This is the other half: Strata's own engine - the 50 CUDA
kernels and the host code that drives them (streams, events, graph capture, pinned memory) - built for the
Arc with oneAPI. It is a migration of the tree, not a new backend: the engine has no backend seam to slot
into.

**Layout.** `sycl/src` and `sycl/include` mirror the tree and hold only the files the port changes; the
SYCL build (`sycl/CMakeLists.txt`) takes every other source from the original location. No upstream file
is edited. `sycl/include/dpct/` is the vendored SYCLomatic helper library, so the port builds without the
migration tool.

**How it was made, so it can be redone.**

1. `sycl/tools/Dockerfile` - the dev image: the llama.cpp SYCL image plus SYCLomatic (`dpct` 2025.3),
   ninja, and the CUDA 12.8 headers that `sycl/tools/get-cuda-headers.sh` pulls out of NVIDIA's pip wheels
   (dpct parses CUDA; it needs the headers, not the toolkit).
2. `sycl/tools/migrate.sh` - writes a compilation database for the 86 CUDA-touching translation units and
   runs dpct over them. 85 migrate; dpct reports no line it could not migrate, and about 1,400 advisory
   notes.
3. `sycl/tools/fixups.py` - what dpct got wrong or could not do, as an idempotent script with a reason per
   item. The ones that mattered:
   - CUDA's null stream means the default stream; dpct turned it into a null `sycl::queue*`. Every
     stream cast now goes through `strata::q_of()` (`sycl/include/strata/sycl_queue.hpp`).
   - `__ldg((const float*) p)` came out as `*p`, reading one byte of a float scale (two sites, s_gemv).
   - `__fadd_rn(a, b ? c : d)` lost its parentheses (two sites).
   - the ggml lookup tables were threaded through kernel parameters with the wrong table per template;
     they are plain `static const` arrays read from device code, as ggml-sycl does.
   - helper headers renamed since dpct 2025.3 (`entangle`, `chunked_partition`), graph introspection and
     `cudaGraphUpload` (no SYCL equivalents), `%globaltimer` (the stage profiler reads zeros).
4. `sycl/tools/build.sh` - configure + build with icpx inside the image. Two compiler flags are load-bearing:
   `-fp-model=precise` (icpx defaults to a fast FP model) and `-cl-fp32-correctly-rounded-divide-sqrt`
   for the device compiler - the Arc's fp32 divide is not correctly rounded by default (OpenCL allows
   2.5 ulp), and Strata's quantizers are byte-exact against ggml through `amax / 127`. Measured: without
   it `quantize_act_parity` has 303k mismatches, with it none.

**Where it stands (2026-09-29).** The whole tree builds and links: the `strata` binary and 22 kernel
parity tests. On the B70, 19 of the 22 pass byte-for-byte or within their tolerances; the other three
need model fixtures (`iq_parity`, `ple_parity`) or the tensor-core kernel below (`qsa_prompt_attn`).

| parity test | result |
|---|---|
| bf16_gemv, cvec, dequant_s2, elementwise, gdn, gr, kv_q4, kv_q8, kv_stream, qsa, quantize_act, rope, router_top10, s2_gemv, s2_gemv_q8, s_gemv, s_gemv_q8k, sampler, shared_expert | pass |
| iq_parity, ple_parity | need fixtures (a `logs/iq_fixture` directory, a Q2_0 shard) |
| qsa_prompt_attn_parity | needs the tensor-core kernel (below) |

**The engine end to end (2026-09-29, later the same day).** It generates. Prompt "Write a Python function
that returns the n-th Fibonacci number", greedy, 64 tokens:

> `<think>` The user wants a Python function that returns the n-th Fibonacci number. Let me consider the
> options: 1. **Recursive approach** - Simple but O(2^n) time complexity, very slow for large n. 2.
> **Iterative approach** - O(n) time, O(1) ...

Measured: **~13 tok/s steady state** (64 tokens in 52.0 s of which ~47 s is the first window: the SYCL
runtime JIT-compiles every kernel on first use; 6 tokens take 47.5 s, 64 take 52.0 s). llama.cpp's SYCL
build does 23-25 tok/s on the same card; Strata's kernels have had no tuning for Xe2 yet and its
tensor-core paths are still the scalar fallbacks. Prompt reading is 2 tok/s and unoptimised.

How to run it (all of this is what `coderiq1/sycl/run-engine-test.sh` does):

```
STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 \
build-sycl/strata --pack <iq pack> --native <shard1> --ple-gguf <shard2> \
    --expert-profile data/expert-profile-coder.bin --expert-cache auto --stream-experts \
    --prefill auto --spec 4 --spec-min-p 0.5 --max-context 8192 --tokens <ids> --max-new 64 --greedy
```

- `--stream-experts` (this port): no resident host copy of the experts; every one of the 12,288 goes from
  the GGUF into the VRAM cache through a small staging ring (`GgufExpertSource`). Upstream needs 32 GB of
  RAM for this model; with the flag the engine runs on 23 GiB.
- `STRATA_VERIFY_DEVICE_PLAN=1`: the GPU plans each layer itself (upstream's E-6; off by default there).
- `STRATA_VERIFY_NO_HOST=1` (this port): the host waits for the whole window graph instead of per-layer
  rings. Only valid with every expert resident, which is the case on a 32 GB card.

What the port had to get right beyond compiling (each is an entry in `sycl/tools/fixups.py` or a flag):

- **Host-mapped flags.** Strata's decode is a GPU/CPU handshake through host-mapped memory. `volatile`
  device loads do not bypass the caches on Intel; system-scope atomics do, and they are the only thing
  measured to work (`sycl/probe/doorbell.cpp`, six variants). Host-to-device visibility *during* a kernel
  stays unreliable on this platform, which is why the all-resident path waits for the window instead.
- **Bounded spins.** A device spin that never sees its flag is not a hang of one process: the xe driver
  times the queue out and resets the GT node by node - a window graph has 2,400 - and the card stays
  wedged until a reboot (twice). Every spin is capped (`kSpinMax`).
- **32-lane sub-groups.** The kernels are written for warps; dpct pinned 134 of 289 launches, the rest
  would run at Xe2's default 16 (`-fsycl-default-sub-group-size=32`).
- **Synchronous copies.** `cudaMemcpy` blocks; dpct's default-queue `memcpy` did not wait. With the
  streaming ring that filled the expert cache from overwritten buffers: non-deterministic residuals, NaN by
  layer 5. Found with the per-layer residual ladder (`STRATA_VERIFY_DEBUG=1` prints R after every layer).
- `native_expert_parity` hand-ported: the GPU native expert kernel matches ggml's float reference on real
  IQ1_M rows (rel 1.1e-2, the same class as the CPU path).

Known: the process segfaults at exit (teardown order; the output is complete by then), and the first
window's JIT (an AOT build with `-DSTRATA_SYCL_AOT=bmg-g31` is the fix, in progress).

**Not ported yet.**

- Three kernels carry inline PTX (`mma.sync` tensor-core matrix ops, `ldmatrix`, `cp.async`):
  `qsa_prompt_attn`, `qsa_select`'s block scores, `native_qsa_score`. Each already has the "older card"
  fallback the CUDA build uses below sm_80, and the SYCL build takes that path: the launchers refuse the
  device and their callers use the older kernels. The XMX versions (`joint_matrix`, bf16/f16 on Xe2; no
  tf32) are the next piece of work and the one that decides prompt speed.
- The ggml MMQ prefill path (`moe_mmq.cu`) needs llama.cpp's ggml-cuda sources; not built.
- AOT device code (`-DSTRATA_SYCL_AOT=bmg-g31`) is wired but untested; the port JITs from SPIR-V.

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
| prefill | 149 tok/s on a 2,701-token prompt; 424 tok/s on a 104,798-token prompt at 131k context |
| quality | correct code on every test; matches the NVIDIA path token for token in spirit, not measured |
