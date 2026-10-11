# strata-poly HIP with Qwen3.6-35B-A3B UD-Q4_K_M, and Qwen3.5 against it (RX 9060 XT)

The report is [AMD_QWEN36_IMPLEMENTATION.md](../../../AMD_QWEN36_IMPLEMENTATION.md). One PC: RX 9060 XT 16 GB (gfx1200),
Ryzen 5 5600X, 32 GB RAM, Windows 11, ROCm 10.2.0a20260930 (TheRock), with other programs open (their VRAM and the
free RAM are in each JSON). Models: Unsloth's Qwen3.6-35B-A3B-UD-Q4_K_M (SHA-256 `0b21525e…`) and
Qwen3.5-35B-A3B-Q3_K_M.

| Files | What |
| --- | --- |
| `refs-llama-hip-cpu.json`, `refs-llama-hip-gpu.json`, `refs-llama-vulkan-gpu.json` | llama.cpp's greedy answers on the Qwen3.6 file, 5 prompts (A: HIP `--cpu-moe`, B: HIP `--n-cpu-moe 22`, C: Vulkan `--n-cpu-moe 22`) |
| `parity-*.json`, `logs/parity*.log` | `tools/hip/poly_llama_parity.py`: strata-poly in each setting against A or B, and the references against each other |
| `diag-steps*.json` | every step's 40 likeliest tokens of two documents in several settings (prompt batch 1,024 / 256 / 16, no MMQ, CPU experts) |
| `q35-parity-w1-vs-*.json` | Qwen3.5 with this phase's binary against the phase 2 references |
| `card-route-check.*` | `STRATA_POLY_CARD_ROUTE_CHECK=1`: the routing on the card against the host's |
| `bench.json`, `bench-summary.md`, `bench-*.log` | `tools/hip/bench_compare.py`: Qwen3.5 and Qwen3.6, default and `--window 1`, 3 alternated runs |
| `regression/` | Qwen3.5: the binary of `3f0e79aa` against this phase's, 3 alternated runs |
| `longctx/` | `tools/hip/kv_needles.py`: notes at 5/50/95 % at 48K, 128K and 262K, KV in VRAM and streamed (`summary.md`) |
| `api-check-qwen36*.json` | `tools/hip/poly_api_check.py` against the Qwen3.6 server |
| `ctest.txt`, `logs/kb1.log` | ctest (103 tests), `poly_kernel_bench` |
| `gguf-headers/` | `tools/hip/gguf_head.py`: metadata of the Qwen3.5 file and of 10 Qwen3.6 sizes, read over HTTP without the data |
