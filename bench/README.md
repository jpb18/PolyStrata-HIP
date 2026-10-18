# Benchmarks and evidence

Benchmark entry points stay in their established locations so invocation and JSON outputs remain compatible. `bench/results/` contains historical campaigns; new generated result directories are ignored by default. Local raw evidence may include private prompts and machine paths. Keep originals unchanged, and publish only a reviewed copy with its hardware/build/model metadata. Existing tracked evidence remains versioned.

## Choose the measurement

| Tool | Measures |
| --- | --- |
| `tools/hip/bench_compare.py` | Same token-ID prompts on `strata-poly` and llama-server; load, prefill, decode, TTFT, cold/warm/cache cases and process memory |
| `tools/hip/bench_prefill.py` | API prompt timing and native log counters on an already running server |
| `tools/hip/poly_serve_timing.py` | Engine line-protocol/server timing |
| `tools/hip/kv_q8_compare.py` | FP16/Q8 distributions, forced continuations, NLL/KL and expert-placement controls; not a general throughput claim |
| `tools/hip/poly_llama_parity.py`, `kv_parity.py` | Numerical/reference comparisons with their explicit fixtures |
| `poly_kernel_bench` | Isolated kernels; results do not establish end-to-end model speed |
| `tools/hip/functional_quality/` | Paired task quality, API/tool cycles and scoring; separate from speed |
| `bench/bench_vs_llama.py`, `tools/needle_bench.py`, `tools/bench_eviction.py` | Inherited engine comparisons, retrieval and cache studies; inspect each tool's requirements |

Use each script's `--help` for its full current interface. Do not change benchmark algorithms, thresholds or output field meanings for release cleanup.

## Example comparison

Run one arm at a time on the same machine. Supply your own llama-server build and compatible GGUF; preparation starts the reference tokenizer server. Review the script before allocating long contexts.

```bat
mkdir bench\results\local-example
.venv\Scripts\python.exe tools\hip\bench_compare.py prepare --llama-server "C:\tools\llama-server.exe" --model "models\model.gguf" --out bench/results/local-example/prompts.json
.venv\Scripts\python.exe tools\hip\bench_compare.py run --prompts bench/results/local-example/prompts.json --label poly --engine poly --exe build-hip-poly-win/strata-poly.exe --model "models\model.gguf" --ctx 49152 --args "--threads 6 --window 1" --json bench/results/local-example/comparison.json
.venv\Scripts\python.exe tools\hip\bench_compare.py run --prompts bench/results/local-example/prompts.json --label reference --engine llama --exe "C:\tools\llama-server.exe" --model "models\model.gguf" --ctx 49152 --args "-ngl 99 --n-cpu-moe 8 -t 6 -fa on" --json bench/results/local-example/comparison.json
.venv\Scripts\python.exe tools\hip\bench_compare.py summary --json bench/results/local-example/comparison.json
```

These are example configurations, not equivalent placement/performance guarantees. `--window 1` disables speculative windows for the poly arm. State how speculation is configured in the reference. The script distinguishes cold, warm and reused prompt cases; do not combine them into one speed number. Use a fresh output file for an independent campaign because the comparison tool appends records.

## Record enough to reproduce

- GPU model/VRAM/architecture, CPU, physical RAM, OS, AMD driver, ROCm and BLAS versions; clock/power settings and other GPU/RAM consumers.
- Git revision **and local changes**, compiler/CMake options, binary SHA-256, pinned ggml revision and relevant `STRATA_*`/device environment variables.
- Exact GGUF revision, SHA-256, split files, actual tensor quantizations and tokenizer/template. Weight quantization and KV format are different variables.
- Allocated maximum context **and actual prompt tokens**, reused tokens, generated tokens, batch size, KV type/residency and sampling/seed.
- MTP block/draft file and hash, whether it loaded, `--window`, `--no-draft`, `--lookup`, draft counts and acceptance. A configured MTP flag alone is insufficient.
- Expert profile copy/hash, resident experts and set hash, hit rate denominator, swaps, reserve/floor/ceiling, free VRAM and shared GPU memory. The engine writes usage profiles back: use separate copies for paired arms.
- Prefill tokens/s, decode tokens/s, TTFT and load time with their timing boundaries. Preserve native versus wall-clock counters; see [INFERENCE_METRICS.md](../docs/INFERENCE_METRICS.md).
- Repetitions, order, warmup, median/spread, failures/skips and missing data. Keep a same-day comparison build and report what was not tested.

For controlled Q8 comparisons, `kv_q8_compare.py` copies the supplied expert profile for each run. Keep context and placement controls equal and separate numerical drift from cache-placement noise. Forced-token results measure a different experiment from free generation; neither proves general task quality. See the [Q8 guide](../docs/AMD_HIP_KV_Q8.md) and [community benchmark guidance](../docs/COMMUNITY_BENCHMARKS.md).

New measurements were not taken as part of this organization task. Historical performance remains tied to the report's original machine, build and workload.
