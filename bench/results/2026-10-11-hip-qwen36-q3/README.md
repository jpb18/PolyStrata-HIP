# strata-poly HIP with Qwen3.6-35B-A3B UD-Q3_K_M (RX 9060 XT)

The report is [AMD_QWEN36_Q3_MIGRATION.md](../../../AMD_QWEN36_Q3_MIGRATION.md). One PC: RX 9060 XT 16 GB (gfx1200),
Ryzen 5 5600X, 32 GB RAM, Windows 11, ROCm 10.2.0a20260930 (TheRock), with other programs open (their VRAM and the
free RAM are in each JSON). Model: Unsloth's Qwen3.6-35B-A3B-UD-Q3_K_M (revision `5bc3e238`, SHA-256
`8966dd0c…ae7b8e1`). The same binary as [2026-10-11-hip-qwen36](../2026-10-11-hip-qwen36/) (commit `00ded250`), whose
numbers for UD-Q4_K_M and Qwen3.5 Q3_K_M these are compared with.

| Files | What |
| --- | --- |
| `gguf-headers/` | `tools/hip/gguf_head.py` on the downloaded file: the quantization of every tensor |
| `logs/expert-parity.log` | `poly_expert_parity` on this file: each expert format against fp64, one token and a batch |
| `logs/first-run.*` | the first load and a short answer |
| `refs-llama-hip-cpu.json`, `refs-llama-hip-gpu.json` | llama.cpp HIP b11160's greedy answers on this file, 5 prompts (A: `--cpu-moe`, B: `--n-cpu-moe 22`) |
| `parity-*.json`, `logs/parity.log` | `tools/hip/poly_llama_parity.py`: strata-poly in each setting against A or B, and B against A |
| `diag-step29.json` | doc-2.5k, step 29 and the steps before it, in five settings (prompt batch 1,024 / 256 / 16, no MMQ, CPU experts) |
| `card-route-check.*` | `STRATA_POLY_CARD_ROUTE_CHECK=1`: the routing on the card against the host's |
| `bench.json`, `bench-summary.md`, `logs/bench-*.log` | `tools/hip/bench_compare.py`: default (MTP) and `--window 1`, 3 alternated runs, the phase 2 prompts |
| `longctx/` | `tools/hip/kv_needles.py`: notes at 5/50/95 % at 48K, 128K and 262K, KV in VRAM and streamed (`summary.md`) |
| `api-check.json` | `tools/hip/poly_api_check.py` against the server (48K profile) |
| `logs/opencode-*.txt` | OpenCode v2.0.18 (OpenChamber's) against the server, with the global configuration |
| `glm-4.7-flash-headers/` | GLM-4.7-Flash: the official `config.json` and the metadata of three Unsloth GGUFs, read over HTTP without the data (nothing downloaded) |
