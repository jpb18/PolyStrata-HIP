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
| kv_hybrid_parity (0.1.27, hybrid K8V4 KV) | appends, gathers and attention pass; its last step is that same tensor-core kernel |

**The engine end to end (2026-09-29, later the same day).** It generates. Prompt "Write a Python function
that returns the n-th Fibonacci number", greedy, 64 tokens:

> `<think>` The user wants a Python function that returns the n-th Fibonacci number. Let me consider the
> options: 1. **Recursive approach** - Simple but O(2^n) time complexity, very slow for large n. 2.
> **Iterative approach** - O(n) time, O(1) ...

Measured on the same card, same 2,185-token prompt, 64 greedy tokens (AOT build, `--spec 2`,
`--no-prefill-borrow`), against llama.cpp's SYCL build serving the same GGUF:

| | llama.cpp SYCL | Strata SYCL port |
|---|---|---|
| prompt reading | 138-319 tok/s | **560 tok/s** (693 at 2,000 tokens, 180 at 300) |
| decode, suffix drafter only (`--spec 2`) | 24.2-26.0 tok/s | 20.2-20.8 tok/s |
| decode with the MTP draft layer (`--spec 4 --mtp`) | - | **42.7 tok/s** at the 2,185-token context (81% accepted, 2.9 tokens/round); 45.3 on a short prompt |

Prompt reading is where Strata's design pays (oneMKL GEMM over dequantised experts, the whole model on
the card). Decode is at ~80%: a speculative round costs ~53 ms whatever its size and the suffix drafter
is accepted 9% of the time, so nearly every round yields one token; `--spec 4` (16.9 tok/s) was worse
than `--spec 2` (20.8) for that reason. The kernels themselves run at 130-160 GB/s of weights on a card
that streams 600 GB/s (`sycl/probe/bw.cpp`, incompressible data) - the same class llama.cpp reaches -
and five variants of the hot Q6_K matvec (lanes per row, rows per warp, unroll, 16-byte loads, software
pipelining; `mmvq_bench`) all landed in that band. The draft source was the lever: the base
Qwen3.8-Flash-Next checkpoint's MTP layer (`tools/mtp_fetch.py fetch`, `mtp_pack.py --experts q2_0`,
`mtp_rt.py`; 4.9 GB downloaded, 809 MiB of VRAM) drafts for the Coder fine-tune at 87% acceptance:
3.4 tokens per 53 ms round, 45 tok/s, the same greedy tokens. The `mtp.cpp` path ran on the first try
under the same flags.

Without AOT the runtime JIT-compiles every kernel on first use, ~47 s the first time a process runs;
`SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=<dir>` keeps that across runs (15 MB).

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
- `--no-prefill-borrow`: the prompt path must not lend expert slots (a lent expert is served from the host
  behind a flag the GPU does not see reliably here; long prompts hung without it).

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

Profiling: `sycl/tools/Dockerfile.unitrace` builds the dev image with Intel's unitrace; `unitrace -d` around
the engine plus `sycl/rank_kernels.py` gives device time per kernel. `mmvq_bench` and `native_expert_parity
NATIVE_BENCH=1` time the two hot kernel families in isolation (warm the clocks first: a 5 ms run measures
the ramp, not the kernel).

**After the 0.1.25-0.1.27 merge (2026-09-30, same card, same flags, AOT).** Fibonacci prompt (19 tokens),
64 greedy tokens: decode 46.5 tok/s (85% of drafts accepted). The 2,184-token prompt: 566 tok/s prompt,
38.3 tok/s decode (77% accepted on that text); the draft layer's prompt pass is now upstream's batched one
(38.6 ms for 2,184 tokens, from 105 ms for 19 before it). Parity: 19 of 22, byte-exact `quantize_act` back.

**Speed work, 2026-09-30 (after the 0.1.27 merge).** What moved the numbers, measured on the B70 with the
same prompts as above (outputs identical before and after each change):

| change | effect |
|---|---|
| every window graph and the drafter's graphs captured at load (`STRATA_WARM_GRAPHS=0`: as before) | the first request no longer pays 250-290 ms of captures |
| the commit graph left running while the drafter's round runs on its own queue (`Verifier::commit(wait=false)` + `commit_finish`) | decode 38.3 -> 43.0 tok/s at the 2,184-token context, 46.5 -> 49.8 short |
| upstream 0.1.29 merged (GDN recurrence pipelining, the block-scores read, sampler) | 2,184-token prompt 765 -> 792 tok/s, decode at that context 43.1 -> 45.2 |
| the draft layer's batched prompt pass takes prompts under 64 rows (it fell back to per-6-token graphs) | its prompt cost on 19 tokens 106 -> 16 ms |
| one device module per kernel (`-fsycl-device-code-split=per_kernel`) | the first launch in the prompt path (the embedding gather) 245 -> 1 ms |
| the expert dequant writes each thread's run of FP16 values as one vector store (it wrote them one 2-byte store at a time) | dequant per expert 0.085 -> 0.030 ms; prompt 496 -> 575 tok/s at 2,184 tokens, 720 -> 841 at 8,000 |
| SWAR sign compare/subtract in the expert dots, a local-memory resident-plan kernel, a split-K fused down kernel (bit-identical to the single-token one) | parity-clean, no measurable decode change; kept |
| the PLE row reader's default queue depth 16 -> 64 blocking O_DIRECT threads (`STRATA_IO_THREADS`; 128 and 256 change nothing: the drive tops out near 85k IOPS on the 27k random 4 KB reads a 2,184-token prompt needs) | the rows of that prompt 466 -> 330 ms |
| a short first prompt chunk (`STRATA_PREFILL_FIRST`, default 256 tokens) so the GPU starts while the remaining rows are still being read | 2,184 tokens: 711 -> 765 tok/s, 8,000: 857 -> 986; time to first token about 150 ms less. The chunk boundary moves rounding in the expert GEMMs, so a long greedy continuation can diverge late (token 45 of 64 on the test prompt) |

Draft policy sweep (2,184-token prompt, 128 tokens): windows of 6 (`--spec 4`) at 41.6-43.5 tok/s; `--spec 2`
33.6, `--spec 6` 35-37; `--spec-min-p` 0.3-0.7 within noise. Profiles (unitrace, `strata-sycl-dev:metrics` with
Intel's metrics libraries and `dev.xe.observation_paranoid=0`): the expert dot kernels are ALU-bound (77% XVE
active), the dense projections memory-latency bound (92-128 GB/s at 84-93% occupancy), a decode round is 80-85%
kernel time and ~5 us of launch gap per node over 2,400-2,600 nodes.

**XMX.** oneMKL's FP16 GEMMs already run on the XMX units (30-60 TFLOP/s in `xmx_gemm_bench`); the prompt path is
bound by the dequant that feeds them, not by the products. Two joint_matrix kernels were written and are correct
but lose to the existing paths on this card, so both stay opt-in: `xmx_gemm_iq` (a fused dequant + GEMM straight
from the quantized rows, 4-5x slower than dequant + oneMKL) and `qsa_prompt_attn_xmx` (the port of the mma.sync
prompt attention; `qsa_prompt_attn_parity` passes at 1e-6 of scale, 3x slower than the FP32 fallback per chunk;
`STRATA_PROMPT_ATTN_XMX=1`). Where the prompt time goes now (2,184 / 8,000 tokens): dequant 28% / 27%, GEMMs
22% / 20%, the per-layer host grouping 9.5% / 9%, attention 7% / 14%, embeddings + PLE rows 13% / 5%.

A trap worth knowing: with a native pack `--prefill-until N` does not feed the rest of the prompt through the
token loop (it is skipped for native packs), so the tokens after N are dropped and the model free-runs. Compare
output tokens between paths, never only timings.

**An 80,000-token prompt (2026-09-30).** Where the time goes changes completely at this scale:

- **The VRAM plan.** `--expert-cache auto` fills the card down to `--vram-reserve-mib` *before* the KV state
  (about 1 GB in INT8 at 81,920 cells) and the prompt chunk buffers (about 2 GB at `--prefill 8192`) exist. With
  the 1,536 MiB default that put 30.95 of the B70's 32.6 GB in use, the driver started migrating buffers and the
  run never finished. `--max-context 81920 --kv int8 --vram-reserve-mib 3072 --prefill 4096` peaks at 28.8 GB.
  (80k ids also exceed Linux's 128 KB single-argument limit: `--tokens-file`.)
- **Streamed experts.** The reserve costs cache slots (10,348 of 12,288 at 3 GB), and the prompt path streams the
  missing experts from the GGUF for every chunk. Two port bugs sat on this path and are fixed: the stream plan
  held `GgufExpertSource::blob()` pointers across ~1,900 reads of a 512-slot ring (so blobs were overwritten
  before they were copied: the 75 s / 1,065 tok/s measured first was computed partly with the wrong experts), and
  the stream-all walk hangs on this card in its first large chunk (the copy engine stops on a barrier). The stager
  threads now read the blobs themselves, and the port uses the per-layer routed-only walk (`STRATA_PREFILL_RING=8`
  upstream, the port's default; `STRATA_PREFILL_STREAM_ALL=1` restores the other for debugging).
- **Correct result: 80,000 tokens in 101 s = 790 tok/s.** GPU time: expert down GEMM 26.0 s, attention 15.4 s,
  dequant 12.7 s, QSA block selection 9.3 s (0.2 s at 8k: it scans every block of the context per query), gate/up
  GEMM 8.9 s, gather 6.1 s, the per-layer grouping sync 5.2 s, GDN recurrence 4.8 s. The PLE rows are free at this
  scale (97.6% row-cache hits).
- **Decode after such a prompt, and the fix.** Without lending cache slots to the prompt path (`--no-prefill-borrow`,
  which the port used from the start) the reserve evicts ~1,900 experts for good and decode after the prompt runs at
  2 tok/s. Borrowing lends ~940 slots to the prompt path and refills them in about a second afterwards: 80,000
  tokens in 75 s = **1,062 tok/s, decode 35-40 tok/s** after it, at `--vram-reserve-mib 2048 --prefill 4096`.
  It costs ~1 s on short prompts (2,184 tokens: 610 vs 792 tok/s), so the port borrows by default only above a
  32K context (`--prefill-borrow` / `--no-prefill-borrow` decide explicitly).

**Long contexts by KV type (2026-09-30, borrowing on, `--vram-reserve-mib 2048 --prefill 4096`).** Same text
repeated to length, 64 greedy tokens after it:

| context | KV | prompt | decode after | peak VRAM | experts in VRAM (of 12,288) |
|---|---|---|---|---|---|
| 128K | int8 | 960 tok/s | 8.5 tok/s | 30.9 GB | 12,002 |
| 128K | q4_0 | 951 tok/s | 32.7 tok/s | 30.4 GB | 12,288 |
| 128K | k8v4 | 983 tok/s | 23.4 tok/s | 30.7 GB | 12,241 |
| 256K | q4_0 | 778 tok/s | 5.0 tok/s | 30.9 GB | 11,814 |
| 256K | k8v4 | 752 tok/s | 3.8 tok/s | 30.7 GB | 11,294 |
| 256K | int8 | 718 tok/s | 3.7 tok/s | 30.7 GB | 10,923 |

Every configuration completes and answers coherently; decode after the prompt is set by how many experts the KV
leaves room for. At 128K use `--kv q4_0`. At 256K all three evict 470-1,400 experts; KV streaming
(`--kv-resident`) is the next thing to try there. k8v4 works through the FP32 attention fallback; its dedicated
prompt kernel is the XMX one below.

**XMX prompt attention v2.** 64-cell chunks, vector-packed K^T and V, hi+lo Q in one accumulator per scale group,
one accumulator update per chunk: 1.4-1.5x faster than v1 and correct in all modes (`qsa_prompt_attn_parity`,
and `kv_hybrid_parity` passes completely with it), but still ~2x slower than the FP32 fallback (13.4 vs 5.5 ms per
chunk, INT8, 32K context). This attention is gather-bound: each query position selects its own ~2,000 cells, so
the K/V fetch dominates and only 12 of the 16 matrix rows are real heads. Opt-in: `STRATA_PROMPT_ATTN_XMX=1`
(64-cell chunks, 120 KB of local memory) or `=32`.

**XMX v2 in the full matrix, and why a grouped kernel is not next (2026-09-30).** With `STRATA_PROMPT_ATTN_XMX=1`
(stage timing on, ~8-10% overhead) the long prompts run 23-33% slower than with the FP32 attention: 128K int8 666 vs
960 tok/s, k8v4 596 vs 983; 256K k8v4 506 vs 752, int8 536 vs 718 (q4_0 does not use the XMX kernel). The obvious
fix, gathering the union of neighbouring positions' cells once, depends on how much their selections overlap.
Measured on the last chunk of an 80K prompt (first QSA layer, `STRATA_DUMP_SEL=<file>`): the union of 8 consecutive
positions is 3.3x one position's 2,051 cells (16: 5.1x), and only 12% of a selection is shared by all 8. Grouping
would cut the K/V gather to ~40% but multiply the arithmetic by 3-5x: at best 5-8 s of an 80K prompt. Not built.

**Serving the port (2026-09-30).** `serve/server.py --engine strata` runs the SYCL engine unchanged through
`sycl/serve/strata-sycl.sh`, an `exe` that starts the binary inside the oneAPI runtime image with the serve pipes
attached (paths in the config's `args` are the container's, the data root mounted at `/work`). A config:

```json
{"engine": "strata", "exe": "<repo>/sycl/serve/strata-sycl.sh",
 "args": ["--pack", "/work/pack", "--native", "<shard 1>", "--ple-gguf", "<shard 2>",
          "--expert-profile", "data/expert-profile-coder.bin", "--expert-cache", "auto", "--stream-experts",
          "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/work/mtp/rt",
          "--max-context", "32768", "--kv", "int8", "--vram-reserve-mib", "1024"],
 "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05}, ...}
```

Measured through the OpenAI API with those sampling defaults: 36 tok/s decode on a first turn, 33 on a follow-up
(which reuses the conversation's cached prompt), against 24-26 for llama.cpp on the same card. The reserve matters:
with `--stream-experts` there is no host copy of the experts, so any expert left out of VRAM is read from the SSD
and computed on the CPU whenever it is routed. At 1,536 MiB the cache came up 128 experts short and decode fell to
5-10 tok/s; 1,024 MiB fits all 12,288 with 2 GB of VRAM still free. `setup.py` still writes the llama.cpp config
for Intel cards; generating this one is the next step.

**Keeping up with upstream.** A merge of upstream `main` into `b70` leaves the copies in `sycl/` behind
wherever upstream touched a file they mirror. They are refreshed by re-migration, not by hand (done for
0.1.25-0.1.27, 2026-09-30):

1. Merge upstream into `b70` and resolve `setup.py` (the Intel path lives beside upstream's AMD one).
2. Migrate the *old* upstream tree (a `git archive` of the pre-merge commit) with `migrate.sh` into
   `sycl-base`, and the merged tree into `sycl-new`; about 3 minutes each in the dev image.
3. `sycl/tools/normalize.sh <dir> <commit>` on both: dpct's file names, the unchanged files, the
   message serials, then `fixups.py`. Two runs of dpct on the same source now differ only in the kernel
   name hashes it generates.
4. For every file present in both: `git merge-file sycl/F sycl-base/F sycl-new/F`. Files upstream did not
   change are left alone. Copies dpct never produced (`verify.cpp`, `mtp.cpp`, `remote_experts.cpp`
   include no CUDA header directly) take upstream's diff by hand; new parity tests are copied in and
   listed in `sycl/CMakeLists.txt`.
5. A fixup whose pattern upstream changed shows up as a compile error (the PTX gate became an `#elif`
   under upstream's `__HIPCC__` guard); extend the fixup, re-run it, rebuild.

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
