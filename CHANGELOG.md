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

No inference kernels, quantization rules, model loading, memory layouts, expert scheduling, MTP policy or numerical defaults were changed by this cleanup. See [the audit](docs/RELEASE_AUDIT.md) for verification results and remaining release limits.
