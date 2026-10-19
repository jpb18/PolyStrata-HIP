# HIP release preparation audit — 18 October 2026

## Scope and initial state

This audit covered the repository inventory, tracked files, non-ignored additions, entry points, build/setup files, runtime and cache implementations, tests, benchmarks, documentation, licenses and Git settings. Generated environments and build trees were inventoried without treating them as source or deleting them. The checkout was already on `hip-poly`, 24 commits ahead of `origin/hip-poly`, with extensive uncommitted HIP/server changes and validation work. Those changes are the baseline for this cleanup, not changes introduced by it.

A filesystem inventory during the cleanup counted **561,425 files and 47.89 GiB of logical file sizes**, including ignored files and the new CPU validation build; traversal reported zero read errors. This is a snapshot, not disk blocks: hardlinks can be counted more than once. The large areas are SDKs, build/dependency trees and raw measurements, rather than evidence of unused code. The earlier [repository audit](REPOSITORY_AUDIT.md) remains a historical snapshot with its own date and counts.

No commit, push, merge, publication, staging, Git cleanup or destructive reset was performed. Git remotes and branch settings were retained. Read-only searches of open issues and PRs in the configured `jpb18/PolyStrata` repository returned no entries; no issue or comment was created.

## Structure and actual entry points

| Area | Role and execution path |
| --- | --- |
| `START_HIP_POLY.bat` | Main Windows HIP launcher; menu/config → `tools/hip/poly_launch.py` → `serve.server` → configured engine |
| `START-HERE.bat`, `SETUP.bat`, `setup.py`, `setup.sh` | Inherited general installation route; kept separate from the experimental poly HIP instructions |
| `UPDATE.bat`, `update.sh` | Inherited update route |
| `polystrata.py`, `chat.py` | Python model/client interfaces retained at their existing paths |
| `serve/server.py`, `serve/web/` | HTTP APIs, model process management and browser app |
| `src/program/generate.cpp`, shared `src/`/`include/` | Original `strata` engine and shared components |
| `src/poly/main.cpp`, `engine.*`, model implementations | `strata-poly` GGUF engine, protocol, memory, expert placement and speculative decoding |
| `src/poly/qwen35.*`, `kernels.cu` | Qwen attention, Q8 KV and FP16 streaming implementation |
| `cmake/hip_backend.cmake`, HIP compatibility headers | AMD compilation/runtime mapping; `STRATA_POLY_HIP` is opt-in |
| `tools/hip/` | Explicit build, config, environment, parity, quality and measurement tools |
| `tests/`, tests under `src/`, `serve/`, `tools/` | Existing test interfaces; categorized in [TESTING.md](TESTING.md) without moving their source paths |
| `bench/`, `bench/results/` | Measurement tools, frozen fixtures and historical campaigns |
| `sycl/`, `ref/`, `data/`, `third_party/` | Inherited backend, reference/runtime data and licensed upstream components |

The [technical guide](HIP_GUIDE.md) documents allocation/reservation, usage-based expert replacement and resizing, Q8 cache representation, MTP loading/verification, and their actual controls. It distinguishes this fork's measured RX 9060 XT path from other upstream hardware claims.

## Cleanup decisions and preserved work

**No inference source files were moved or changed.** The existing source architecture, imports, engine protocols and CMake paths were retained. Hash comparisons against the initial working tree verified **all 567 existing files under `src/`, `include/` and `sycl/` unchanged**, including the user's uncommitted implementations. `CMakeLists.txt`, `setup.py`, the server implementation and the web app were also retained. License files at the root and both ggml locations matched their initial hashes.

Fourteen root AMD reports moved to `docs/reports/`; their [index](reports/README.md) lists every report. Root files now point to the corresponding report to preserve links. Relative Markdown references were adjusted, and personal paths were generalized in publication copies. Historical measurements were retained, and reports are marked as historical instructions. The previous main README was preserved in `docs/UPSTREAM_README.md`; translated READMEs remain inherited documents rather than translations of the new HIP guide.

The new public documentation is `README.md`, `docs/README.md`, `docs/HIP_GUIDE.md`, `docs/TESTING.md`, this audit, `bench/README.md`, `CONTRIBUTING.md` and `CHANGELOG.md`. `data/hip/` now contains the frozen Q8 expert profile and its provenance note; its bytes and SHA match the original campaign profile. No model weights or executables were added.

There was insufficient evidence to call operational binaries, rollback builds, campaigns, patches, test fixtures or the one-off vision work obsolete. They were preserved. Local scratch/backup/compiler products are ignored. The root vision experiments (`debug_vision.py`, download/search helpers, `enable_vision.py`, `test_vision.py`, `test_image_download.py`, `build-vision-test.bat`) were identified as machine-specific local experiments and excluded from publication by default; reusable encoder source remains `tools/vision/`. Nothing was deleted.

New raw campaign directories under `bench/results/` are ignored by default, while already tracked evidence stays versioned. **Ignore rules do not untrack files.** A privacy scan found the local Windows user-directory component in 65 tracked benchmark files. Original bytes were backed up locally and identifying text was anonymized in the publication copies. JSON numeric values and token-ID arrays were compared and remained identical; original/publication hashes are recorded in [PUBLICATION_REDACTIONS.json](../bench/results/PUBLICATION_REDACTIONS.json). See the [publication note](../bench/results/PUBLICATION_NOTES.md) for effects on historical content hashes. Private originals and new raw campaigns should not be force-added as release content.

`.rgignore` additionally excludes relocated historical reports from routine source searches. They remain available through `rg --no-ignore` and are not deleted or made untrackable by that search rule.

## Startup and configuration changes

- The launcher resolves and temporarily enters its own directory, restores the caller's directory, and offers `--help`. It reports missing Python/packages/engine with explicit setup commands. It never installs packages, downloads dependencies or builds automatically.
- The existing model/context/KV/chat/check/nocap selections and compatibility alias remain available. `POLY_CONFIG` directly selects the user's existing FP16 configuration; it no longer requires the default engine path when that config uses another executable. Its engine arguments, including cache and MTP options, are preserved.
- Launcher preset generation refuses a conflicting existing profile. FP16 `check` does not create a missing profile. The standalone configuration utility refuses existing output unless `--overwrite` is explicit. Its tokenizer lookup includes the selected build's pinned CMake checkout and `STRATA_GGML_DIR`, addressing a fresh-clone setup gap.
- Q8 keeps its exact executable SHA, model restrictions, expert-profile SHA, admission checks, INFO/READY/health checks, port protection and isolated run configs. The expert profile's source path moved to `data/hip/` without content changes. Streaming SHA computation supports the documented Python 3.10 minimum.
- A test's author-specific tokenizer fallback was removed; environment/local-fixture discovery remains. Two test-isolation problems were fixed: checksum refusal tests now mock release HEAD requests and use a temporary engine directory; a fake quality campaign now takes a temporary lock rather than a live campaign's lock. No assertions or quality thresholds were relaxed.

## Verification performed

Final results, after correcting sandbox restrictions and the test-isolation issues above:

| Check | Result |
| --- | --- |
| New profile/config/launcher regression tests | **8 passed**, including external working directory, spaces in paths, custom MTP/cache arguments, read-only check, missing environment and overwrite refusal |
| Existing HIP launcher tests | **33 passed**, including the Q8 policy, resource refusals, occupied ports, no fallback and configuration preservation |
| Q8 T4 policy unit tests | **8 passed** |
| Python server suite | **664 run, 9 skipped, all remaining passed**; mocks/loopback, no real model inference |
| Paired functional-quality harness CPU/mock suite | **64 passed** |
| Setup test collection | **30 scripts, 467 tests passed**; final checksum/ModelScope rechecks included |
| Fresh CPU CMake configure/build and CTest | **12/12 targets passed**, MSVC 19.51, Visual Studio 18 2026; separate `build-release-cpu-check` |
| Source/license preservation | **567 backend/source hashes unchanged**; CMake and license checks passed |
| Q8 artifact checks | Included profile matches its pinned SHA |
| Publication text scan | No recognizable GitHub/AWS/OpenAI token patterns or this machine's username in non-ignored publication candidates; no candidate read errors |
| Diff whitespace and current guide links | Passed; 94 local guide links resolve in the working tree; historical local-only archive links are documented below |

The CPU CMake selection used `STRATA_ENABLE_CUDA=OFF`, `STRATA_ENABLE_HIP=OFF`, `STRATA_NATIVE_EXPERTS=OFF`, `STRATA_BUILD_TESTS=OFF`, `STRATA_BUILD_CONVERSATION_TESTS=ON`. Turning native experts off avoids an unnecessary ggml download for this standalone test selection. No operational build directory was reconfigured.

Windows sandbox temp-file and localhost restrictions initially prevented some tests. Final process/HTTP suites ran with normal localhost access; scratch files stayed in the repository. The ModelScope test served a small fixture locally, not model weights from the internet. Non-fatal resource warnings appeared in the server suite. Test logs, original README/launcher copies, hash baselines, filesystem inventory, report move records and original redacted evidence are retained in ignored `tmp/release-preparation/`.

## Remaining release limits

1. A **clean-clone HIP compilation and real AMD/model regression run were not performed** for this cleanup. No new GPU numerical, VRAM-pressure, long-context, MTP-acceptance, vision or performance measurement is claimed. Run the hardware/model checks in [TESTING.md](TESTING.md) for the exact source revision intended for release, including the pre-existing uncommitted implementation work. CUDA/SYCL were not rebuilt because their source files were not changed by the cleanup.
2. The Q8 menu runs the one engine, `build-hip-poly-win\strata-poly.exe`, which is ignored and built locally, not supplied by a clone. Its profile ships as source data.
3. Historical report links to local raw campaigns may not resolve in a clone. Original evidence remains local. Publication copies can have different hashes after path redaction; do not replace numerical evidence or claim original-file identity. New prompt preparation reads the changed documentation corpus, so paired comparisons must share a frozen token-ID file.
4. The nine server skips depend on unavailable optional fixtures/capabilities. Python 3.10 compatibility was reviewed in the changed hashing path but tests ran under Python 3.14.6; other Python versions, AMD architectures and Linux were not exercised.
5. The checkout still contains the user's original unstaged/untracked work. Release review should include those implementation files; this audit certifies preservation during cleanup, not independent approval of all pre-existing code.
6. At the time of the initial cleanup, anonymization covered the working tree and history was preserved under the instruction to make no commits. The user's subsequent request authorized a [history privacy cleanup](PRIVACY_CLEANUP.md), which removes the identifying information from the branch history intended for publication and documents its checks and private recovery archive.

No CI services, runtime dependencies, model downloads, performance changes or publishing actions were introduced by this cleanup.
