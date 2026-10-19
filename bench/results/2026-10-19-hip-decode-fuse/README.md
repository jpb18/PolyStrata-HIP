# strata-poly HIP: a token's call with fewer launches (`STRATA_POLY_FUSE=1`), 19 October 2026

RX 9060 XT 16 GB (gfx1200), Ryzen 5 5600X (6 threads), 32 GB of RAM, Windows 11, ROCm 10.2.0a20260930 (TheRock).
Qwen3.6-35B-A3B UD-Q3_K_M, context 49,152. Built from `240e69bf` plus this change, for gfx1200 only, in
`build-hip-poly-fuse` (`strata-poly.exe` SHA-256 `d139b762...`). **The PC was in use the whole time** (the owner
working beside the runs): the absolute speeds move with that, and only the alternated pairs are compared.

## What it changes

A call of up to 8 tokens (a token, a window of guesses) launched ~1,250 kernels, 30-35 a layer, and on this card a
kernel that waits for the one before costs ~3.6 us whatever it does (`hip_poly_probe`, AMD_HIP_DECODE_OPT.md 3.5).
With `STRATA_POLY_FUSE=1` (off by default) these launches become one each, on Qwen3.5/3.6:

| Where | Before | With the switch |
| --- | --- | --- |
| a layer's first norm | `rms_norm`, `native_quantize_q8_1` | `rms_norm_q8` |
| the residual and the FFN's norm | `axpy`, `rms_norm`, `native_quantize_q8_1` | `rms_norm_q8` with the add |
| gated delta net, projections | wqkv, wz (Q8_0); alpha, beta (F32); `gdn_gates_kernel` | `native_mmvq_group` (wqkv, wz); `gdn_ab` (alpha, beta and their gates) |
| gated delta net, output | `gdn_out_kernel`, wo's `native_quantize_q8_1` | `gdn_out_q8_kernel` |
| attention, projections | wq, wk, wv (Q8_0) | `native_mmvq_group` |
| attention, gate | `mul_sigmoid`, wo's `native_quantize_q8_1` | `mul_sigmoid_q8` |
| shared expert | gate, up; `swiglu_clamp`, down's `native_quantize_q8_1` | `native_mmvq_group`; `swiglu_clamp_q8` |
| routed experts on the card | `swiglu_clamp_rows`, `native_quantize_q8_1` | `swiglu_clamp_q8` |
| their sum, the CPU's, the layer's tail | `add_scaled_dev`, `wait_add`, `axpy` | `card_tail` (up to 4 tokens) |

About 12 launches fewer in a gated-delta-net layer and 11 in an attention layer. Each fused kernel computes each
value with the same expression, the same thread and the same order of sums as the kernels it replaces; the Q8_1
blocks are `native_quantize_q8_1`'s (a warp a block of 32). `native_mmvq_group` runs, for each matrix, the block code
of that matrix's own launch (`small_mmvq_block`, `mmvq_multi_block`, moved out of their kernels unchanged). A prompt
batch (more than 8 tokens) and the other models are not touched.

## Same bits (`scripts/fuse_parity.py`, `parity-b2k.json`)

B-2k (2,068 tokens), `--expert-swaps 0 --expert-cache-max 4000` (so that every run caches the same experts and the
CPU's experts and `card_tail`'s wait are used: 63 % from VRAM), every step's logits compared as bits:

| Comparison | Steps with the same bits |
| --- | ---: |
| decode, `--window 1`: switch off against on | 48 of 48 |
| decode: the operational `build-hip-poly-win/strata-poly.exe` (`770ac295...`) against this build, switch off | 48 of 48 |
| windows (`--force-window 4,3,2,4/2,4/1`): off against on | 48 of 48 |
| windows against the decode, switch off | 48 of 48 |

With the switch, `STRATA_POLY_TRACE=2` (B-2k, `--window 1`; the trace's events inflate every part) shows the card's
time a call at 17.1 -> 15.3 ms: the layers' heads 7.2 -> 5.6 ms, the attention heads 2.2 -> 1.8 ms, the tail launch
(0.45 ms) gone into the experts' part.

## Speed (`scripts/fuse_ab2.py`, `ab2.json`)

One process a run, `--tokens` with 256 tokens greedy, `--expert-swaps 0` from a fresh copy of the same profile,
`STRATA_POLY_WARM=1`, the engine's own decode tok/s. Three pairs a setting, the order alternated pair by pair.

| Prompt | Window | Off -> on, each pair (tok/s) | Change, each pair | Median change |
| --- | --- | --- | --- | ---: |
| B-2k | `--window 1` | 68.92 -> 71.77, 69.08 -> 73.07, 67.26 -> 70.76 | +4.1, +5.8, +5.2 % | **+5.2 %** |
| B-45k | `--window 1` | 51.74 -> 56.59, 52.74 -> 56.02, 53.86 -> 55.08 | +9.4, +6.2, +2.3 % | **+6.2 %** |
| B-2k | default (guesses) | 95.54 -> 99.62, 87.84 -> 95.07, 78.93 -> 91.78 | +4.3, +8.2, +16.3 % | **+8.2 %** |
| B-45k | default (guesses) | 73.53 -> 61.80, 76.92 -> 78.97, 80.95 -> 82.62 | -16.0, +2.7, +2.1 % | **+2.1 %** |

11 of 12 pairs are faster. With guesses, the windows the policy picks follow its measured times, so the guesses
taken differ between the arms even where the tokens are the same; in the one slower pair (B-45k, pair 0) the tokens
differed and the switch's run took 129 of 160 guesses against 156 of 182. Free VRAM moved with the other programs,
so the experts in VRAM (80-87 %) and in some pairs the tokens differ between the two runs of a pair: these pairs
are not the same computation to the bit, the parity runs above are.

## The first attempt (`scripts/fuse_ab.py`, `ab.json`, `ab-first-attempt.log`)

`tools/hip/decode_trace.py` processes (default expert swaps): the run-to-run spread (+-7 %, as in earlier phases)
hid the difference, and the 11th process, **with the switch off**, stopped on the engine's fail-closed check:
`GPU_FATAL ... the card waited more than a second for the CPU's experts; new process required` (B-45k,
`--window 1`). That is the path every build has had (HIP_GPU_FAILURES.md); the switch was not on in that process.

## Not done

- The build for CUDA (`native_mmvq.cu` and `src/poly/kernels.cu` are shared; the fused kernels compile there but
  `card_tail` is HIP-only and nothing was built or run on NVIDIA), and other AMD cards.
- An idle PC. The switch stays off by default until a measurement on an idle machine.
- B-45k parity at the bit: the attention is untouched by the switch, and the parity was run at 2K only.
