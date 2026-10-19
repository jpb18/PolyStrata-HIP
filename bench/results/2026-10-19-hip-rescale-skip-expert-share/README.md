# strata-poly HIP: the attention rescale skip and the experts' shared reads, 19 October 2026

RX 9060 XT 16 GB (gfx1200), Ryzen 5 5600X (6 threads), 32 GB of RAM, Windows 11, ROCm 10.2.0a20260930 (TheRock).
Qwen3.6-35B-A3B UD-Q3_K_M. Built from the `main` after `STRATA_POLY_FUSE` became the default, plus these two changes,
for gfx1200 only. The "before" build is `build-hip-poly-win\strata-poly.exe` as it was (SHA-256 `26b7d90a...`), the
"after" build `d25a39a5...`. **The PC was in use.** The absolute speeds move with that, and only the alternated runs
are compared.

## 1. The attention rescale skip (on, no switch)

A prompt's attention goes block by block of keys, and after each block `attn_scale_kernel` multiplies the output of every
(token, head) by the factor the block's softmax gives, the weights of the blocks before against the new largest score.
When a block did not raise the largest score the factor is `expf(0)`, exactly 1, and `x * 1.0f` is `x` for every float,
so the kernel now returns for such a block without reading or writing its `dim` values. The launch stays (the host does
not know which blocks are 1). Nothing else changed: the bits are the same by construction, and checked below.

| Check | Result |
| --- | --- |
| B-45k, 8 greedy tokens, `--window 1`, Q8 KV: logits of the after build against the before build, two runs | 8 of 8 steps' bits, both |
| B-2k, 48 steps, decode and the windows below: before against after | 48 of 48 steps' bits |

Prefill, the engine's own tok/s, one process a run, order base, new, new, base (`scripts/ab_exe.py`):

| Prompt | Before (tok/s) | After (tok/s) | Change (means) |
| --- | --- | --- | ---: |
| B-45k | 1453.2, 1485.0 | 1517.2, 1520.6 | **+3.4 %** |
| B-2k | 635.9, 634.3, 634.5 | 631.1, 638.0, 630.7 | 0 (three runs a side) |

Two pairs at 45K are few; the sign was the same in both pairs, and the two arms did not overlap. At 2K the attention is
a small part of the prefill, as expected. The skip matters only for prompts, not for decode.

## 2. The experts' shared reads (`STRATA_POLY_EXPERT_SHARE=1`, **off by default**)

In a window of guessed tokens each (token, pick) is a slot of its own, so an expert two tokens pick is read twice.
With the switch, the slots that share a matrix are one read of it: every fourth of them takes the matrix and the next
three as columns (`row_dot_multi` for IQ3_XXS, the gate and up; `load` once and `apply` per column for IQ4_XS, the down,
in `native_mmvq_slots_half_dev_share_kernel`). Each column is summed in the order its own launch used, so the bits are
the same; the other slots return at once. Only calls of more than 8 slots take it (a token on its own never repeats an
expert), and only these formats (IQ3_XXS gate/up in 39 of the model's 41 gate/up tensors, IQ4_XS down in 37 of 41;
the 41 are the 40 layers and the draft block); the rest keep the old kernels.

**How often a window repeats an expert** (B-2k, default windows, 200 tokens; counted at the host over every pick, cached or
not): in a 3-token window 26.6 % of the picks are an expert another token of the window also picks, 36.5 % in a 4-token
window and 25.7 % in a 5-token one.

Same bits (`scripts/share_parity.py`, B-2k, 48 steps, `--expert-cache-max 4000`, windows `--force-window 4,3,2,4/2,4/1`):

| Comparison | Steps with the same bits |
| --- | ---: |
| decode: before against after | 48 of 48 |
| windows: before against after, `STRATA_POLY_EXPERT_SHARE=0` | 48 of 48 |
| windows: before against after, `STRATA_POLY_EXPERT_SHARE=1` | 48 of 48 |
| windows against the decode, before | 48 of 48 |

**Speed: no gain here.** `scripts/share_ab.py`, forced windows of 4 over the same 128 tokens (so every arm does the same
work), three sets, the order alternated, the engine's own tok/s:

| Arm | Sets (tok/s) | Median |
| --- | --- | ---: |
| before | 99.59, 98.26, 97.70 | 98.26 |
| after, `STRATA_POLY_EXPERT_SHARE=0` | 99.03, 98.25, 105.88 | 99.03 |
| after, `STRATA_POLY_EXPERT_SHARE=1` | 97.88, 99.12, 102.62 | 99.12 |

And the experts' part of a 4-token call in the trace (`STRATA_POLY_TRACE=2`, default windows, 200 tokens, two runs a
side; the trace's events inflate it and it includes the wait for the CPU's experts): 27.2 and 27.5 ms with the switch,
25.7 and 26.0 ms without.

So the shared read did not make a window faster on this card. Two reasons fit the numbers, neither separated here: a
second read of an expert a few microseconds after the first is probably served by the L2 and the 32 MB Infinity Cache,
and in these runs 34 % of the picks were CPU experts, whose sum the card waits for, so a faster card side is hidden
behind the CPU's. That is why it is off: it is available with `STRATA_POLY_EXPERT_SHARE=1` for a card or a cache size
where it might pay, not measured to.

## Not done

- An idle PC; more pairs at 45K and at 100K, where the rescale skip should matter most (attention is 40 % of the prefill
  at 64K and 52 % at 100K in `2026-10-13-hip-prefill-kernels`).
- The shared reads where all the experts are in VRAM (a smaller model, a bigger card), and for the other formats
  (Q4_K, Q6_K down; Q3_K gate/up in two layers).
- The build for CUDA (`iq_kernels.cu` and `native_mmvq.cu` are shared; both changes are inside `#if defined(__HIPCC__)`
  for the shared reads, and the rescale skip is in `src/poly/kernels.cu`, nothing was built or run on NVIDIA).
