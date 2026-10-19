# Changelog

## Unreleased — HIP repository preparation

- Cleaned identifying paths throughout the branch history before publication, mapped the repository owner's commit identity to their public GitHub handle/noreply email and preserved other contributors' attribution. Original history and the commit map are retained in a private local bundle/archive; see `docs/PRIVACY_CLEANUP.md`.

- Added a HIP-focused README, build/configuration guide, testing categories, benchmark instructions and contribution guidance.
- Moved 14 AMD technical reports into `docs/reports/`, preserving old links through root pointers. Generalized personal paths in the publication reports; retained original evidence locally.
- Preserved the previous main README in `docs/UPSTREAM_README.md` to retain upstream context and measurements.
- Made `START_HIP_POLY.bat` the documented Windows HIP entry point. Removed `START-POLY-HIP.bat`, which only forwarded to it: call `START_HIP_POLY.bat` instead. Startup now reports missing prerequisites instead of installing dependencies or building automatically.
- Protected launcher profile selection from conflicting replacements and made FP16 `check` reuse profiles without creating them. Explicit custom configs do not require the default build directory.
- Made the configuration writer refuse existing outputs unless `--overwrite` is explicit, and discover tokenizer tools in CMake's pinned fetched checkout.
- Included the frozen Q8 expert profile under `data/hip/` with its original SHA-256. SHA calculation also supports Python 3.10.
- Extended ignore rules for local binaries, weights, scratch runs, configuration backups, one-off machine-specific vision experiments and newly generated benchmark campaigns. Existing tracked evidence is retained.
- Anonymized the Windows user-directory component in 65 tracked benchmark publication copies, retaining original bytes locally and recording both hashes. Numeric JSON values and token-ID arrays were verified unchanged.
- Removed a personal-machine tokenizer fallback from a test; `STRATA_TOKENIZER` remains supported.
- Isolated checksum tests from installed engines/release network requests and mock quality campaigns from a live GPU campaign's lock. Assertions and validation thresholds are retained.

- Added fused decode kernels for strata-poly (`STRATA_POLY_FUSE`): a norm with its Q8_1 quantization, projections that read one input in one launch, and the experts' sums with the layer's tail, about 470 fewer launches per call of up to 8 tokens. The bits match the unfused path (48 of 48 steps' logits at B-2k and Q8 262K with a fixed expert cache). On by default on HIP; `STRATA_POLY_FUSE=0` turns it off. CUDA builds keep it off unless `STRATA_POLY_FUSE=1`; they are not built or measured for it. Decode gain measured on a PC in use: +2% to +8%.

- Added to strata-poly on HIP: a prompt's attention skips the rescale of a block of keys whose largest score did not change (factor exactly 1; same bits), B-45k prefill +3.4% on a PC in use; and `STRATA_POLY_EXPERT_SHARE=1` (off by default: bit-identical, but not faster on the RX 9060 XT), which reads an expert once for the tokens of a window that pick it.
- Flash attention for a prompt batch on HIP (gfx12, head dimension 256, fp16 and q8_0 KV): one kernel in place of the products, the softmax and the rescale, 2.0 to 2.2 times faster than them and the prompt +38 % to +54 % at 76K to 130K positions (+40 % at 45K), on a PC in use. Another order of sums, so not the same bits: the next-token distributions differ by 1e-4 to 1e-2 nats up to 122K positions. `STRATA_POLY_FLASH=0` gives the products back.
- Flash attention for decode on HIP (gfx12, head dimension 256, fp16 and q8_0 KV): the same kernel with the keys split 512 a CTA, added by a second launch. 1.1x to 1.3x faster for a token and 7x to 8x for a window of 4 tokens than the rows kernels at 8K to 262K positions; in the engine +4% to +8% a token at 45K to 91K and +91% for windows of 4 at 91K, on an idle PC. A window's tokens are the same bits as each alone. Rounds keys, values, queries and weights to halves: KL 1e-3 nats against the rows kernels, about twice the q8_0 cache's distance from fp16. `STRATA_POLY_FLASH=0` gives the rows kernels back.
- One engine: `START_HIP_POLY.bat`'s Q8 menu runs `build-hip-poly-win\strata-poly.exe`, like every other menu. The frozen Q8 executable, its SHA pin and the "promoted binary" rule are gone, so a rebuild is in the next start; the launcher prints and logs the engine's SHA-256.

No inference kernels, quantization rules, model loading, memory layouts, expert scheduling, MTP policy or numerical defaults were changed by this cleanup. See [the audit](docs/RELEASE_AUDIT.md) for verification results and remaining release limits.
