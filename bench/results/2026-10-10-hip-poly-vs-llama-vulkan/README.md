# strata-poly (HIP) against llama.cpp (HIP and Vulkan) on an RX 9060 XT (2026-10-10/11)

The second phase's comparison ([AMD_PORT_PHASE2.md](../../../AMD_PORT_PHASE2.md)): the same GGUF, the same token
ids, greedy, 128 tokens an answer, a context of 49,152, measured with `tools/hip/bench_compare.py`.

- RX 9060 XT 16 GB (gfx1200, Adrenalin 32.0.31041.1004), Ryzen 5 5600X (6 threads used by every engine), 31.9 GB of
  RAM, Windows 11 Pro. Other programs open: before each run 20.9-24.0 GB of RAM were free and other processes held
  2.4-3.0 GB of VRAM (recorded per run in `full.json`).
- Model: `Qwen3.5-35B-A3B-Q3_K_M.gguf` (Unsloth; experts IQ3_XXS gate/up and IQ4_XS down, dense Q8_0).
- strata-poly: `build-hip-poly-win/strata-poly.exe` from `hip-poly` at `3e466ad1` (ROCm 10.2.0a20260930), with the
  routing on the card (on by default on Windows). `--expert-profile <pack>/expert-profile.txt --threads 6`.
- llama.cpp Vulkan: LM Studio's runtime `llama.cpp-win-x86_64-vulkan-avx2-2.55.0` (b11511, `fc9ce6b`), found on the
  PC, not built. llama.cpp HIP: Unsloth's build b11160 (`a3c12db9d`). Both `llama-server -np 1 --no-webui`, through
  `/completion` with `cache_prompt`.

## Prompts (`prompts.json`)

Two sets of four, from this repository's documents and code, each put in the model's chat template and tokenized by
llama-server (`bench_compare.py prepare`): A-2k 2,061, A-8k 8,216, A-31k 31,296, A-45k 45,236 tokens; B-2k 2,068,
B-8k 8,235, B-31k 31,296, B-45k 45,355. The two sets start with different text, so B is not read from what A left.

## A run (`bench_compare.py run`)

One process per run, started and loaded first (`load_s`: from start to ready). Then, in that process: the A set
("cold": the first requests the process serves), the B set ("warm"), then B-2k and B-8k again ("cache": what the
engine kept of them). TTFT is from the request to the first token, prefill is the tokens the engine read over its
prompt time, decode the 128 tokens over their time. VRAM is the process's dedicated and shared GPU memory
(Windows' counters), RAM its private bytes and working set, the highest seen while it ran.

## Files

| File | What |
| --- | --- |
| `full.json` | 3 runs of each of 4 configurations, interleaved (poly default, Vulkan, poly `--window 1`, llama.cpp HIP), and 3 of poly with `--conversation-cache-mib 6144` |
| `sweep-vulkan.json`, `sweep-hip.json`, `sweep-poly.json` | the first sweep of llama.cpp's `--n-cpu-moe` / `-ub` (A-8k, A-31k), before other programs' VRAM was recorded |
| `sweep2-vulkan.json`, `sweep2-hip.json` | the second sweep, with it recorded |
| `server-benchmark-48k-phase2.json` | `polystrata.py benchmark --prompt-tokens 2000,8000,24000,34000` through the server (`--turn-token 248045`), as `../2026-10-10-hip-poly-rx9060xt-baseline/server-benchmark-48k.json` was in the first phase |

`python tools/hip/bench_compare.py summary --json full.json` prints the tables below.

## Medians of 3 runs (range in AMD_PORT_PHASE2.md)

| Engine | Load s | Prefill tok/s 2K / 8K / 31K / 45K (warm) | Decode tok/s 2K / 8K / 31K / 45K (warm) | VRAM ded. / shared MiB | RAM private MiB |
| --- | ---: | --- | --- | --- | ---: |
| strata-poly HIP, default | 16.2 | 1,271 / 1,561 / 1,375 / 1,235 | 80.1 / 76.1 / 61.3 / 54.9 | 13,451 / 2,042 | 12,524 |
| strata-poly HIP, `--window 1` | 16.0 | 1,345 / 1,592 / 1,385 / 1,252 | 80.4 / 76.3 / 61.5 / 54.9 | 13,521 / 2,034 | 12,767 |
| llama.cpp HIP, `--n-cpu-moe 12 -ub 2048` | 8.4 | 1,097 / 1,195 / 1,243 / 1,191 | 16.7 / 16.7 / 16.6 / 16.2 | 13,803 / 354 | 19,905 |
| llama.cpp Vulkan, `--n-cpu-moe 12 -ub 2048` | 8.8 | 940 / 958 / 911 / 870 | 34.9 / 36.5 / 35.8 / 34.6 | 13,026 / 438 | 20,861 |

The shared memory of strata-poly is its pinned host memory (the experts' staging buffers and uploads), which
Windows counts as shared GPU memory; llama.cpp's configurations were chosen where they did not spill (see the sweeps).
