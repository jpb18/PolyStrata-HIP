# strata-poly (HIP) against llama.cpp, and llama.cpp against itself (2026-10-11)

The second phase's precision check ([AMD_PORT_PHASE2.md](../../../AMD_PORT_PHASE2.md)), with
`tools/hip/poly_llama_parity.py`: the same GGUF (`Qwen3.5-35B-A3B-Q3_K_M.gguf`), the same token ids, greedy, 64
tokens, and for every step the reference's five likeliest tokens looked up in the other side's probabilities. The
largest difference over the steps compared is `max_diff`, beside it the mean and the 95th percentile; `same` is how
many tokens agree before the first that does not. RX 9060 XT, Ryzen 5 5600X, Windows 11, `hip-poly` at `3e466ad1`.

References (llama.cpp, `llama-server`, context 16,384 unless said):

| Name | What |
| --- | --- |
| A | `../2026-10-10-hip-poly-rx9060xt-baseline/llama-hip-refs.json`: Unsloth's HIP build b11160, `--cpu-moe -ngl 99` (every expert on the CPU) |
| B | `refs-llama-hip-gpu.json`: the same build, `-ngl 99 --n-cpu-moe 10` (most experts on the GPU) |
| C | `refs-llama-vulkan-gpu.json`: LM Studio's Vulkan build b11511, `-ngl 99 --n-cpu-moe 10` |
| A30 | `refs-llama-hip-cpu-30k.json`: as A, `docs/DETAILS.md` cut to 35,575 tokens, context 40,960 |

| File | Compared | Result |
| --- | --- | --- |
| `llama-cpu-vs-llama-hipgpu.json` | A against B | 64/64 on 4 prompts; largest 0.061, mean 0.0021-0.0034 |
| `llama-cpu-vs-llama-vulkan.json` | A against C | 64/64; largest 0.110, mean 0.0022-0.0083 |
| `llama-hipgpu-vs-llama-vulkan.json` | B against C | 64/64; largest 0.122, mean 0.0016-0.0082 |
| `poly-window1-vs-A.json` | strata-poly `--window 1` (routing on the card) against A | 64/64; largest 0.079, mean 0.0019-0.0072 |
| `poly-window1-vs-B.json` | the same against B | 64/64; largest 0.042, mean 0.0014-0.0049 |
| `poly-hostroute-vs-A.json` | `STRATA_POLY_CARD_ROUTE=0` against A | 64/64; largest 0.092, mean 0.0020-0.0070 |
| `poly-cpuexperts-vs-A.json` | `--no-gpu-experts` against A | 64/64; largest 0.107, mean 0.0022-0.0079 |
| `poly-default-ids-vs-A.json` | the default (windows of guessed tokens), tokens only | 64/64 on 3 prompts; the short one parts at token 19, at a near tie on A's side |
| `poly-vs-hipgpu.json`, `poly-vs-vulkan-ids.json` | `--window 1` against B and C, tokens only | 64/64 on all 4 |
| `poly-window1-vs-A-30k.json` | `--window 1` against A30 | 64/64; largest 0.079 (at a near tie in A30: 0.419 and 0.412), mean 0.0052 |
| `poly-cpuexperts-vs-A-30k.json` | `--no-gpu-experts` against A30 | 64/64; largest 0.067, mean 0.0069 |

llama.cpp's own builds and settings differ from each other by up to 0.122 on these prompts; strata-poly differs from
them by no more. The tolerance `poly_llama_parity.py` applies (every token the same, or the first difference at a
near tie, and the largest difference at most 0.15) is within that spread.
