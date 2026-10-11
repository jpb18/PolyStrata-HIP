# strata-poly HIP: KV streaming and long contexts on an RX 9060 XT

What [AMD_KV_STREAMING.md](../../../AMD_KV_STREAMING.md) reports, as measured: RX 9060 XT 16 GB (gfx1200), Ryzen 5
5600X, 31.9 GB of RAM, Windows 11 Pro, ROCm 10.2.0a20260930 (TheRock), `Qwen3.5-35B-A3B-Q3_K_M.gguf` (Unsloth),
6 CPU threads, on 2026-10-11, with other programs holding 4.2-4.4 GB of VRAM and 10-13 GB of RAM.

| Path | What |
| --- | --- |
| `matrix/<config>-<n>.json`, `.err` | `tools/hip/kv_needles.py --summary 128`, one process each, alternated: 48K (49,152) with the KV in VRAM, 128K (131,072) in VRAM, 128K with `--kv-resident 49152`, the same with `STRATA_POLY_KV_AHEAD=0` (zero-copy), 262K (262,144) in VRAM and with `--kv-resident 49152`. Each length: the document read, three notes asked for one by one and together, a 128-token summary. `.err` is the engine's log (memory budget at load, copies at exit) |
| `matrix-before-sync-fix/` | 128K with `--kv-resident 49152` before the commit that stopped heads below the split from waiting on the copy ahead (5-12 % slower below 49K) |
| `needles-first/` | the first two needle runs to 261,000 tokens: in VRAM, and streaming with zero-copy reads (before the copies ahead) |
| `summary.md` | `tools/hip/kv_summary.py` over the folders above |
| `parity/kvpar_*.json` | `tools/hip/kv_parity.py`: logits of every step against a reference run (CPU experts: bit for bit; GPU experts: against run-to-run spread) |
| `parity/llama_vs_*.json` | `tools/hip/poly_llama_parity.py` with streaming against the llama.cpp references of `2026-10-10-hip-poly-parity` |
| `regression/reg_*.json` | `tools/hip/bench_compare.py`: the build of `d947977c` (`base`) against this one (`new`), alternated, 3 runs each |
| `poly_kernel_bench-attn.txt` | one attention layer's decode and 1,024-token batch, 8K-262K, in VRAM / split / in RAM, bit for bit |
| `hip_poly_probe-kv.txt` | VRAM, zero-copy and copy-engine bandwidth; pinned memory Windows gives |
| `ctest.txt` | the full ctest of the `poly tests` build |
| `api-check-kvstream.json`, `api-long-kvstream.json` | the server's API with a 131,072 context and `--kv-resident 49152`: `tools/hip/poly_api_check.py`, and one 108,264-token chat request with a note in its middle |

Decode, tokens/s of the 128-token summary (median of 2-4 runs; ranges in `summary.md`):

| Prompt | 48K, KV in VRAM | 128K, in VRAM | 128K, streaming | 262K, in VRAM | 262K, streaming |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 68.2 | 60.4 | 65.0 | 52.9 | 58.5 |
| 32,768 | 54.9 | 54.1 | 53.3 | 49.0 | 50.7 |
| 65,536 | - | 43.4 | 25.2 | 39.3 | 25.2 |
| 130,900 / 131,072 | - | 29.4 | 8.1 | 26.0 | 6.9 |
| 261,000 | - | - | - | 16.7 | 2.8 |

Every note was found in every run at every length.
