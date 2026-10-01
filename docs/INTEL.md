# Strata on an Intel Arc

Strata's engine is CUDA. On an Intel Arc there are two ways to run the same model behind the same Strata
server (OpenAI and Anthropic APIs, streaming, tool calls, MCP, the web app all unchanged):

- **Strata's own engine, ported to SYCL** (`sycl/`, the section "The engine itself on Intel" below). The
  default once it is built: setup.py picks it on an Intel card when `build-sycl-aot/strata` and the
  `strata-sycl-dev` image exist.
- **llama.cpp's SYCL backend** (`--intel-engine llama`, or when the port is not built). Simpler to set
  up, about a third of the speed.

Written for and measured on an **Arc Pro B70 (32 GB)** running the Coder (IQ1_M) on Ubuntu 24.04, in a
PCIe 3.0 x8 slot (the card trains at Gen3 x8 there; it can do Gen5 x16).

## What you get (2026-10-01)

| | NVIDIA (Strata engine) | Intel Arc, Strata SYCL port | Intel Arc, llama.cpp |
|---|---|---|---|
| model files | the same GGUFs | the same GGUFs + a native pack | the same GGUFs |
| where the model lives | experts in RAM, hot ones on the card | every expert in VRAM (`--stream-experts`: no host copy), shard 2's lookup table read from the SSD by row | all of shard 1 on the card, shard 2 paged from disk |
| RAM needed | 32-64 GB | little (23 GB is fine) | little (23 GB is fine) |
| decode, Coder IQ1_M | 44-51 tok/s on an RTX 5070 | **76.1 tok/s** (19-token prompt, 256 greedy tokens), 63.6 after a 2,184-token prompt (before the IQ4_NL change); ~56 through the API on a chat request | 23-25 tok/s |
| prompt reading | ~1,870 tok/s | **799 tok/s** at 2,184 tokens, 1,062 at 80K | ~150 tok/s (424 at 105K) |
| speculative decoding (MTP) | yes | yes (the base checkpoint's draft layer; 70-85% of drafts accepted on code) | no |
| images | yes | not yet | not yet |
| context | up to 262K | 256K measured (`--kv-resident`: the KV in pinned host memory, the attended window in VRAM) | 131K measured ceiling |

Decode speed with speculative decoding depends on the text: code drafts well, prose less so (see "Speed depends
on the text"). The SYCL numbers are greedy runs of the engine test rig on the final 2026-09-30 build (commit
ccdae3e); "Decode round 2" below lists what each change bought.

## Setup

    ./setup.sh

Setup notices the Arc (no `nvidia-smi`; an Intel GPU under the `xe` or `i915` driver in sysfs), says
so, and takes the Intel path: no CUDA toolkit. With the SYCL port built it writes a `"strata"` config for it
(see "Serving the port" below); otherwise, or with `--intel-engine llama`, it finds a `llama-server` and writes a
config with a `"llama"` block and a start script. The rest of this section and the next two are that llama.cpp
path.

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

(Measured before the 2026-09-30 decode work in "Decode round 2"; decode after a long prompt is higher now, not
re-measured.) Every configuration completes and answers coherently; decode after the prompt is set by how many
experts the KV leaves room for. **KV streaming fixes it** (`--kv-resident 32768`: the whole KV in pinned host memory, only the
attended window in VRAM): every expert stays in VRAM and decode after the prompt is 38.7 tok/s at 128K int8
(from 8.5), 35.3 at 128K q4_0, 31.3 at 256K int8 (from 3.7) and 31.5 at 256K q4_0 (from 5.0); the prompt pays
5-13% (725-860 tok/s). setup.py turns it on from 64K up for the SYCL engine and keeps INT8. k8v4 works through the FP32 attention fallback; its dedicated
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
(which reuses the conversation's cached prompt), against 24-26 for llama.cpp on the same card. After the 2026-09-30
decode work: 55.7 tok/s on a 300-token answer (prompt included), 55-59 tok/s over 5,000-18,000-token answers. The reserve matters:
with `--stream-experts` there is no host copy of the experts, so any expert left out of VRAM is read from the SSD
and computed on the CPU whenever it is routed. At 1,536 MiB the cache came up 128 experts short and decode fell to
5-10 tok/s; 1,024 MiB fits all 12,288 with 2 GB of VRAM still free.

`setup.py` writes this config by itself on an Intel card once the SYCL engine is built (`build-sycl-aot/strata`
and the `strata-sycl-dev` image): a native pack, the MTP draft layer (from an existing `mtp-q2_0.gguf` without the
5 GB download), the container's paths, the reserve (1,024 MiB up to 32K, 2,048 with 4,096-token chunks above),
INT8 KV with `--kv-resident 32768` from 64K up (when the RAM holds the KV), and the `sampling` block of an earlier
config kept. `--intel-engine llama` keeps llama.cpp.

**Speed depends on the text.** Decode with speculative decoding tracks how often the draft layer guesses right.
The served engine at 32K context: 45-51 tok/s on the rig's test prompt (continuing a Fibonacci function, 77-85%
of drafts accepted), 30-39 tok/s on a chat answer with prose (55-72%); sampling and the repetition penalty cost
nothing measurable. The engine used to abort at exit in serve mode (a queue wait in a destructor after the
runtime's teardown began); it now exits directly once its requests are done.

**Planned (2026-09-30), in order.** Ranked by payoff on this card; the first three were taken first.

1. Expert dot products on XMX in integer mode: a decode window (up to 6 tokens) fits one INT8 DPAS (1-8 rows), the
   i-quant grids decode to small integers, the activations are already INT8. Today: scalar dp4a, ALU-bound (77%).
   **Parked**: three `joint_matrix` versions (opt-in `STRATA_EXPERT_XMX=1`, do not enable) were 1.4x and 2-3x slower
   than dp4a, and the third (B filled in place with `joint_matrix_apply`) hung the GPU. At 1-6 rows the grid decode
   and the packed-B layout cost more than the DPAS saves.
2. Experts missing from VRAM read from pinned host memory over PCIe instead of the SSD (the 256K decode collapse).
   **Done**: at start-up every expert without a VRAM slot is read into a pinned host mirror (`STRATA_MIRROR_MIB`, by
   default free RAM less 4 GiB), and the device-built verify plan points the expert kernels straight at it. Before,
   that plan dropped non-resident experts, so output with misses was wrong. Forced test (`--expert-cache 8000`,
   1,879 experts / 3.6 GiB out of VRAM): decode 2.6 -> 40.9 tok/s (full cache 43.3), prompt 451 -> 640 tok/s
   (776), output tokens identical to the full-cache run. About 3.6 GiB of VRAM can go to context for ~6% of decode.
3. KV streaming (`--kv-resident`) from 64K up: only the attended window of the KV in VRAM. **Done**: 256K decodes at
   31 tok/s (from 4-5).
4. QSA block selection on XMX: every query against every pooled block, a dense product that grows with the context.
5. The hot decode kernels re-tuned for Xe2's native 16-wide sub-groups (twice the registers per thread).
   **Tried, no gain**: SIMD16 builds of the multi-column mmvq and the wide Q6_K kernel (`STRATA_MMVQ_SG`: 16 all,
   1 IQ4_XS only, 2 short outputs only; default 32). Alone (`mmvq_sg_bench`) SIMD16 is up to 1.45x faster on IQ4_XS
   and 1.3x on the 640-row shared-expert projections, 10-25% slower on the large K-quant ones; in the engine no
   setting beats SIMD32 beyond run-to-run noise (SIMD32 itself lands at 39 or 45 tok/s). Output tokens identical.
   The same bench shows the real headroom: the dense decode kernels stream 100-280 GB/s of a 608 GB/s card
   (Q6_K, the most common type, ~146 GB/s at 2-4 columns).
   **Found and fixed: misaligned loads.** A Q6_K block is 210 bytes, so every block's `ql`/`qh` runs start only
   2-byte aligned, and the B70 splits a misaligned 16-byte load into pieces. The same weights repacked at a
   224-byte stride ran 2.3-4.7x faster (`q6k_align_bench`). Without changing the layout, the wide Q6_K kernel now
   does two aligned 16-byte loads and a shift (`load16_a2`; both stay inside the block) and takes every Q6_K
   shape and window width (it used to be gated to n_out >= 4096, 1-4 columns): 2560->10240 at 1/2/4/6 columns
   160/145/141/101 -> 493/432/331/273 GB/s, the 248K-row head 150 -> 407 GB/s at 1 column. Output tokens
   identical. **Coder decode 44.9 -> 54 tok/s** (2k prompt, greedy). `STRATA_MMVQ_A2=0` restores the old path.
   Still misaligned: IQ4_XS (136 B, 8-aligned, ~100 GB/s), IQ4_NL (18 B), Q8_0 (34 B); Q4_K/Q5_K are aligned
   but use 4-byte loads in the multi kernel (230-280 GB/s).

**Decode round 2 (2026-09-30), from a unitrace of 256 decoded tokens.** Each step measured on its own, 19-token prompt,
256 tokens, two runs each, output tokens identical throughout:

| | decode |
|---|---|
| after the Q6_K alignment fix | 53.9 tok/s |
| + `resident_plan` grouping in parallel (thread 0 alone took 73 us per layer, ~3.5 ms a round) | 57.1 |
| + the GR down kernel reads the activations directly (the 60 KB SLM tile per group capped occupancy) | 58.7 |
| + wide 16-byte-load kernels for Q4_K, Q5_K, IQ4_XS (codebook in registers) | 62.6 |
| + wide kernels for Q8_0 (MTP dense, shexp down of layer 0) and IQ4_NL (shexp down); Q6_K's activation loads aligned | 65.6 |
| + the GR down projection sliced by columns (each group stages one 128-column slice of xn, 4 lanes per row) + a fixed-order reduce | 70.6 |
| + IQ4_NL expert dots (the down projection in 39 of 48 layers): codebook in registers, weights as aligned 4-byte loads, bitwise the same ints (2026-10-01) | **76.1** |

The last step changes the output: identical for 146 tokens, then a near-tie after a comma goes the other way (the new
kernels sum in a different order). Q8_0 kernels 4-6x (189 -> 31 us at 2560 x 10240, 2 columns), IQ4_NL 640 -> 2560
28 -> 6.6 us. The aligned-load helper only loads its second chunk when the address is unaligned, so it never reads a
16-byte chunk without a needed byte and cannot cross a page at the end of an allocation. `STRATA_MMVQ_WIDE_32=0`
restores the old Q8_0/IQ4_NL kernels. The sliced GR down kernel (GR read 108.6 -> 76.5 us at 6 tokens, sums equal to
~1e-7) flips that near-tie back: its output is the original one. `STRATA_GR_DOWN_SLICED=0` restores the direct kernel.

Kernel level: IQ4_XS 107 -> 323 GB/s at 2 columns (61 -> 212 at 6), Q4_K/Q5_K 1.3-1.6x; the GR read 135 -> 110 us at
6 tokens (bitwise equal). Switches to the old paths: `STRATA_PLAN_PARALLEL=0`, `STRATA_GR_DOWN_DIRECT=0`,
`STRATA_MMVQ_WIDE_K=0`. (`iq_parity` reports 10 "missing fixture" failures with and without the change: the oracle
fixtures are not in the port's tree.)

**Two-speed runs, explained (2026-09-30).** Identical greedy runs decode at either ~45 or ~39 tok/s. A per-gather
trace of the PLE reader (`STRATA_PLE_TRACE=1`) shows the slow runs pay one 226 ms PLE read stall in the first
decode round, after the window graphs are captured; every other read and round matches the fast runs. Prompt time
plus decode time is the same in both modes (4.95-5.17 s): the stall lands either in the prompt's PLE wait or in the
first decode round, so it is a once-per-process cost, not lost throughput. Ruled out: NVMe APST, the I/O scheduler
(`none`), CPU starvation (93% idle during decode), I/O thread count. Compare runs on time to first token + decode.
6. INT8 prompt GEMMs: experts dequantized to INT8, oneMKL/oneDNN INT8 on XMX (half the dequant bytes, 2x rate).
7. Fewer graph nodes per decode round (~2,500 at ~5 us): norm+rope, scores+top-k, gate+quantize fused.
8. Wider speculation (two draft branches per verify window): the kernels are latency-bound, so it is nearly free.
9. A model bigger than VRAM: the original Qwen3.8-Flash-Next IQ2_XS (35.5 GB of experts; the same path suits Q2_0
   and Swift 1.5) on the B70 with 23 GB of RAM. ~24 GB of experts in VRAM, the rest (~10-12 GB) in the pinned host
   mirror the device plan reads over PCIe (item 2). The port has the IQ2_XS/IQ3_XXS/Q2_0 expert kernels. Open: whether
   a 10+ GB pinned mirror fits beside everything else in 23 GB, and decode with that share of experts on a Gen3 x8
   link (3.6 GB mirrored measured 40.9 tok/s; expect less). Needs the 68 GB download.

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

**Not ported yet (2026-10-01).**

- Three kernels carry inline PTX (`mma.sync` tensor-core matrix ops, `ldmatrix`, `cp.async`):
  `qsa_prompt_attn`, `qsa_select`'s block scores, `native_qsa_score`. The SYCL build takes the "older card"
  fallback the CUDA build uses below sm_80. `qsa_prompt_attn` also has an XMX version (`joint_matrix`, opt-in
  `STRATA_PROMPT_ATTN_XMX=1`): correct, but slower than the fallback (see "XMX prompt attention v2").
- The ggml MMQ prefill path (`moe_mmq.cu`) needs llama.cpp's ggml-cuda sources; not built.
- AOT device code is what runs: `AOT=bmg-g31 BUILD_DIR=.../build-sycl-aot` (the JIT build costs ~47 s of
  compiling on the first window).

## Not done

- Images, on both Intel engines. Strata's vision path encodes with `strata-vision` into embeddings the CUDA
  engine reads; neither the SYCL port nor the llama.cpp config wires it yet.
- Speculative decoding on the llama.cpp path (the GGUF carries no draft layer llama.cpp can use). The SYCL
  port has it (MTP draft layer, `--mtp`).

## Measured, 2026-09-30, Arc Pro B70, Coder IQ1_M, 32K context, INT8 KV: the SYCL port

| | |
|---|---|
| decode | 76.1 tok/s on a 19-token prompt (2026-10-01); 63.6 after a 2,184-token prompt before the IQ4_NL change (256 greedy tokens, MTP + suffix drafts) |
| prompt | 799 tok/s on 2,184 tokens |
| through the API | 55.7 tok/s on a 300-token chat answer, prompt included |
| VRAM | all 12,288 experts resident, ~1.9 GB free with everything loaded (`--vram-reserve-mib 1024`) |
| RAM | no host copy of the experts (`--stream-experts`) |

## Measured, 2026-09-29, Arc Pro B70, Coder IQ1_M, 32K context, q8_0 KV: llama.cpp

| | |
|---|---|
| load | ~100 s |
| VRAM | 28.4 of 32 GB |
| decode | 23-25 tok/s; GPU 92% busy at 165 W, CPU idle |
| prefill | 149 tok/s on a 2,701-token prompt; 424 tok/s on a 104,798-token prompt at 131k context |
| quality | correct code on every test; matches the NVIDIA path token for token in spirit, not measured |
