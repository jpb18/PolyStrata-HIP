# Changelog

## Unreleased — HIP repository preparation

- Cleaned identifying paths throughout the branch history before publication, mapped the repository owner's commit identity to their public GitHub handle/noreply email and preserved other contributors' attribution. Original history and the commit map are retained in a private local bundle/archive; see `docs/PRIVACY_CLEANUP.md`.

- Added a HIP-focused README, build/configuration guide, testing categories, benchmark instructions and contribution guidance.
- Moved 14 AMD technical reports into `docs/reports/`, preserving old links through root pointers. Generalized personal paths in the publication reports; retained original evidence locally.
- Preserved the previous main README in `docs/UPSTREAM_README.md` to retain upstream context and measurements.
- Made `START_HIP_POLY.bat` the documented Windows HIP entry point. Removed `START-POLY-HIP.bat`, which only forwarded to it: call `START_HIP_POLY.bat` instead. Startup now reports missing prerequisites instead of installing dependencies or building automatically.
- Protected launcher profile selection from conflicting replacements and made FP16 `check` reuse profiles without creating them. Explicit custom configs do not require the default build directory.
- Made the configuration writer refuse existing outputs unless `--overwrite` is explicit, and discover tokenizer tools in CMake's pinned fetched checkout.
- Included the frozen Q8 expert profile under `data/hip/` with its original SHA-256; the frozen executable verification remains in place. SHA calculation also supports Python 3.10.
- Extended ignore rules for local binaries, weights, scratch runs, configuration backups, one-off machine-specific vision experiments and newly generated benchmark campaigns. Existing tracked evidence is retained.
- Anonymized the Windows user-directory component in 65 tracked benchmark publication copies, retaining original bytes locally and recording both hashes. Numeric JSON values and token-ID arrays were verified unchanged.
- Removed a personal-machine tokenizer fallback from a test; `STRATA_TOKENIZER` remains supported.
- Isolated checksum tests from installed engines/release network requests and mock quality campaigns from a live GPU campaign's lock. Assertions and validation thresholds are retained.

- Added fused decode kernels for strata-poly (`STRATA_POLY_FUSE`): a norm with its Q8_1 quantization, projections that read one input in one launch, and the experts' sums with the layer's tail, about 470 fewer launches per call of up to 8 tokens. The bits match the unfused path (48 of 48 steps' logits at B-2k and Q8 262K with a fixed expert cache). On by default on HIP; `STRATA_POLY_FUSE=0` turns it off. CUDA builds keep it off unless `STRATA_POLY_FUSE=1`; they are not built or measured for it. Decode gain measured on a PC in use: +2% to +8%.

No inference kernels, quantization rules, model loading, memory layouts, expert scheduling, MTP policy or numerical defaults were changed by this cleanup. See [the audit](docs/RELEASE_AUDIT.md) for verification results and remaining release limits.
