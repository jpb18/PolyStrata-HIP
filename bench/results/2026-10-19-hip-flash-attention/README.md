# strata-poly HIP: flash attention for a prompt batch, 19 October 2026

RX 9060 XT 16 GB (gfx1200), Ryzen 5 5600X (6 threads), 32 GB of RAM, Windows 11, ROCm 10.2.0a20260930 (TheRock).
Qwen3.6-35B-A3B UD-Q3_K_M. Built from `main` at `48c0d927` plus this change, for gfx1200 only. **The PC was in use** for the engine
runs, so only the alternated pairs of one process each are compared, and the first run of a series can be cold.

## What it changes

A prompt batch's attention (`attend_batch()`, `attend_batch_kv()`) took its scores through the scratch: a hipBLAS product
for a block of keys, a softmax kernel, a rescale of the values so far, a second product. At 100K positions that is half
the prompt's time (`2026-10-13-hip-prefill-kernels`: 40 % at 64K, 52 % at 100K). Now, on gfx12 at Qwen3.5/3.6's shape
(head dimension 256, no window, ring or host part), `attend_flash()` does it in one launch (`src/poly/kernels.cu`):

- a CTA takes 64 rows (a token's 8 heads of one key-value head, 8 tokens) and walks the keys 32 at a time;
- the scores are computed on the matrix cores as S^T = K Q^T, so that a lane's fragment of S^T is the B operand of the
  product with V^T and the weights never leave the registers; the largest score and the sum of the weights are one
  lane's (and its partner's), a running softmax;
- the two waves of a 16-row group split the 256 dimensions (the first version, one wave a group, spilled 34 to 182
  registers; this one has none), the partial scores are added through the LDS in the same order in both;
- the next step's keys and values are fetched into registers while this one is computed;
- a q8_0 cache is read as codes times scales inside the kernel: no stage, no dequantization kernel, and the halves are
  the ones `kv_dequant_q8()` gives (checked bit for bit below), computed with packed half arithmetic.

The sums are in another order than the products', so **the bits are not the same**: the same inputs rounded to halves
(the queries, the weights, the keys and values) and the same sums in floats. `STRATA_POLY_FLASH=0` gives the products
back (read at each call).

## Kernel (`poly_kernel_bench attn-flash`)

A batch of 1,024 tokens ending the cache, 16 heads of 256 and 2 key-value heads; random keys and values in [-1, 1), queries
in [-8, 8) (a peaked softmax). Max |error| against a double-precision attention of the same half-rounded inputs on four
tokens and four heads (the values are up to 0.3 to 0.6):

| Context | Cache | Products (ms) | Flash (ms) | Speed | Error: products | Error: flash |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8,192 | fp16 | 9.13 | 4.18 | 2.19x | 3.4e-5 | 3.2e-5 |
| 8,192 | q8_0 | 9.28 | 4.25 | 2.19x | 7.2e-5 | 7.6e-5 |
| 32,768 | fp16 | 35.78 | 17.16 | 2.09x | 2.9e-5 | 3.3e-5 |
| 32,768 | q8_0 | 36.27 | 17.99 | 2.02x | 4.5e-5 | 4.9e-5 |
| 65,536 | fp16 | 72.19 | 34.49 | 2.09x | 2.5e-5 | 4.2e-5 |
| 65,536 | q8_0 | 73.18 | 36.20 | 2.02x | 3.4e-5 | 5.0e-5 |
| 131,072 | fp16 | 143.12 | 68.26 | 2.10x | 3.1e-5 | 6.6e-5 |
| 131,072 | q8_0 | 145.89 | 71.07 | 2.05x | 1.9e-5 | 6.0e-5 |

The error grows with the context about twice as fast as the products' (one fp16 weight is 5e-4 of its value; both stay
far under that). Batches of other sizes and positions (32, 33, 64, 97, 100, 129, 200 and 777 tokens from position 0, 5,
31, 1,000, 8,000 and 50,000), fp16 and q8_0: the same error as the products to the digit or lower in every one. Flash
over the q8_0 rows gives the same bits as flash over the halves `kv_dequant_q8()` makes of them, at every context.

At 131,072 positions that is 32 TFLOP/s for the flash and 15 for the products (2.19e12 floating-point operations a batch).

## In the engine (`scripts/flash_ab.py`)

The Q8 launch's arguments (q8_0 KV, 262,144 positions, 3,000 experts in VRAM, `--window 1`), one process a run, the
engine's own prefill tok/s (the whole prompt) and the logits of the first greedy tokens compared with the first
(products) run: the largest difference, the first token, and the KL divergence of the two distributions.

| Prompt (tokens) | Products (tok/s) | Flash (tok/s) | Change | KL mean / max (nats) | Same first token, same tokens |
| --- | --- | --- | ---: | --- | --- |
| B-2k (2,068) | 561, 336 (cold) | 809, 818 | +45 % to +143 % | 1.9e-4 / 2.4e-3 | yes, yes |
| B-45k (45,355) | 1,115, 1,426 | 1,989, 1,998 | **+40 %** (best against best) | 1.9e-4 / 1.3e-3 | yes, yes |
| A-45k + A-31k (76,532) | 1,244 | 1,711 | **+38 %** | 1.1e-4 / 5.8e-4 | yes, yes |
| A-45k + B-45k (90,591) | 1,067 | 1,620 | **+52 %** | 6.8e-4 / 5.2e-3 | yes, yes |
| A-45k + B-45k + A-31k (121,887) | 935 | 1,437 | **+54 %** | 1.1e-2 / 8.8e-2 | yes, yes |
| the four sets (130,122) | 907 | 1,397 | **+54 %** | 14.4 | no, no |

The last row is a prompt of four documents one after the other that ends in another document's last words: its next
token is not stable. The same prompt with the products over an fp16 KV cache instead of the q8_0 one, nothing else
changed, differs from the products over q8_0 by 17.5 nats and also gives another token, so that difference is the
prompt's, not the flash's (`flash-kv.log`: `--order 0,fp16:0,fp16:1 --ctx 140000`).

## Not done

- An idle PC; more pairs at the long prompts; the fp16 and the KV streaming (`--kv-resident`) profiles in the engine
  (the kernel is checked for fp16 caches, and a streaming layer's call keeps the products).
- Other shapes (head dimension 128, a window or ring: Gemma 4 keeps the products), and other cards than gfx12.
- Tuning: 32 TFLOP/s is well under what the matrix cores can do (a CTA of 8 waves with 54 KB of LDS); a bigger tile or a
  deeper pipeline might raise it. Not tried.
- CUDA: the code is inside `#if defined(STRATA_USE_HIP)`, nothing was built or run on NVIDIA.
