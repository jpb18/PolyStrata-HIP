# PolyStrata HIP

Run GGUF mixture-of-experts models with an AMD GPU and system RAM on Windows. This fork extends PolyStrata's `strata-poly` engine with an experimental HIP backend. Experts that fit are cached in VRAM; the CPU computes the others.

The main development and measurements for this fork used **Windows 11, Radeon RX 9060 XT 16 GB (gfx1200), Ryzen 5 5600X and 32 GB RAM**. This is a measured configuration, not a guarantee for every AMD card or model. See the [hardware and feature limits](docs/HIP_GUIDE.md).

Built on [PolyStrata](https://github.com/VecSzn/PolyStrata) and [Strata by Niko1221](https://github.com/Niko1221/Strata). The inherited installer, server, web app, CUDA engine and SYCL tree are retained. The [original README](docs/UPSTREAM_README.md) describes those routes; this README covers the HIP fork.

## What this fork adds

- An opt-in `STRATA_POLY_HIP` build of `strata-poly`, using HIP/hipBLAS and the existing CUDA-shaped source architecture.
- Qwen attention KV in FP16, optional Q8_0 KV, and optional FP16 KV streaming into pinned RAM.
- Expert cache sizing and adaptation to available VRAM, with explicit reservation, ceiling and optional floor controls.
- The existing MTP draft-block and speculative verification pipeline on HIP, with controls for disabling drafts or all speculative windows.
- Windows launch/configuration tools, failure handling, native timing metrics and reproducible comparison tools.

These are implemented features, not claims of identical output or faster inference on every configuration. Q8 changes numerical behavior. Kernel, quality and performance evidence is linked from the [technical guide](docs/HIP_GUIDE.md) and [report index](docs/reports/README.md).

## Prerequisites

For the source-build route below:

- Windows 10/11 x64; this fork's measurements used Windows 11. A discrete AMD GPU supported by the selected ROCm package and a current AMD driver are required. Only the RX 9060 XT configuration above was validated for this fork's full path.
- An x64 CPU with AVX2 for the portable build, enough RAM for the selected GGUF and its caches, and disk space for the weights, compiler and build artifacts. The tested Qwen3.6 UD-Q3_K_M GGUF is **17,104,402,720 bytes**; ROCm installation alone takes several GB. Long contexts require more memory.
- Git, Python **3.10 or newer**, and Visual Studio 2022 (or 2019) Build Tools with the C++ workload and Windows SDK.
- Internet access during the explicit dependency/build steps. The runtime does not need a model download if the GGUF is already on disk.

The root HIP launcher **does not install packages, download models, or build the engine**. `START-HERE.bat` is the inherited general installer; use the steps below for `strata-poly` on HIP.

## Quick start: clone to first inference

Commands below are for **Command Prompt (`cmd.exe`)**, from the repository folder unless stated otherwise.

```bat
git clone --branch hip-poly https://github.com/jpb18/PolyStrata.git
cd PolyStrata
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Build explicitly. For the tested RX 9060 XT, select `gfx1200`; choose your card's architecture rather than copying this value for another card. The build script documents the available targets and overrides.

```bat
set STRATA_HIP_ARCHS=gfx1200
tools\hip\build_windows.bat poly
```

This command installs the pinned ROCm **10.2.0a20260930** TheRock SDK in `.rocm-win`, fetches pinned llama.cpp sources, configures CMake and produces `build-hip-poly-win\strata-poly.exe`. It may also install CMake/Ninja into the SDK environment. No administrator rights are requested by the script. Details and offline/local-checkout overrides: [build and dependencies](docs/HIP_GUIDE.md#build-and-dependencies).

Obtain a compatible GGUF yourself and keep its model license. A measured starting point is `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` from `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, revision `5bc3e238`; the [model report](docs/reports/AMD_QWEN36_Q3_MIGRATION.md#3-modelo-novo-e-verificaÃ§Ã£o) records its SHA-256 and tensor formats. The filename alone does not prove compatibility. Put it in `models\` or use its full path; weights are ignored by Git.

Check device detection, then start with a modest context:

```bat
tools\hip\poly_windows.bat probe
.venv\Scripts\python.exe tools\hip\poly_config.py --model "models\Qwen3.6-35B-A3B-UD-Q3_K_M.gguf" --context 32768 --name qwen36-hip
set POLY_CONFIG=strata-qwen36-hip.json
START_HIP_POLY.bat fp16
```

`poly_config.py` extracts the tokenizer from the GGUF using llama.cpp's `gguf-py` and writes a local run configuration. Use a new `--name` for a new configuration. The utility refuses an existing output unless you explicitly pass `--overwrite`. The launcher reuses existing profiles and refuses conflicting profile replacements.

If the desired GPU has a HIP ordinal other than zero, set `"hip_ordinal"` to that number in the generated JSON before launching; see [GPU selection](docs/HIP_GUIDE.md#configuration).

Open [the web app](http://127.0.0.1:8090). After the server reports ready, enter a short prompt. In a second window, verify the API if desired:

```bat
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/v1/models
```

Stop the server with Ctrl+C. The launcher binds to `127.0.0.1`; exposing a manually started server on another interface requires `--api-key`.

## Launch and configure

**`START_HIP_POLY.bat` is the main Windows HIP launcher.** It resolves its own repository folder and also works when called from another directory. Existing installation/update batch files serve their separate inherited tasks.

```bat
START_HIP_POLY.bat --help
START_HIP_POLY.bat
START_HIP_POLY.bat qwen3.6 128k
START_HIP_POLY.bat chat 48k
START_HIP_POLY.bat check 128k
```

Without `POLY_CONFIG`, the menus select configured models and context profiles. FP16 profiles are 48K (49,152), 128K (131,072), 262K (262,144), and 262K with FP16 KV streaming. These labels are presets, not a guarantee that your model supports that context or that it fits memory. `check` starts no engine and will not create a missing FP16 profile.

For a custom configuration, set `POLY_CONFIG` to its JSON path (relative to the repository or absolute). Its `args` array carries context, expert cache, MTP and other engine options; the FP16 launcher preserves it. `POLY_NO_OPEN=1` disables the browser and `POLY_NO_PAUSE=1` disables error pauses. Clear `POLY_CONFIG` with `set POLY_CONFIG=` to return to the menus.

The engine's current options are available through:

```bat
tools\hip\poly_windows.bat run --help
```

See [configuration examples and feature limits](docs/HIP_GUIDE.md#configuration), including GPU selection, cache bounds, MTP and streaming. Unknown batch arguments are model-name filters, not arbitrary engine flags: put engine flags in the configuration's `args` or use `poly_windows.bat run` for direct token-ID inference.

### Q8 KV is opt-in

The **source engine** implements `--kv q8_0` for Qwen3.5/3.6 attention, without KV streaming. FP16 remains the default. Weight quantization and KV quantization are independent.

The launcher's `q8_0` menu is for Qwen3.6 UD-Q3_K_M. It runs the same engine as every other menu, `build-hip-poly-win\strata-poly.exe`: there is one engine, no separate Q8 binary and no pinned executable, so a rebuild is in the next start. The matching expert profile is included in `data/hip/` and pinned by its SHA. Missing artifacts stop startup. Q8 also enforces resource checks and disables expert swaps and conversation parking for this route. See [Q8 launch policy](docs/HIP_LAUNCH_KV.md) and [technical limits](docs/HIP_GUIDE.md#q8-kv-and-streaming).

## Tests and benchmarks

[Testing](docs/TESTING.md) separates CPU unit tests, process/API integration tests, local installation checks and GPU/model validation. [Benchmarking](bench/README.md) gives commands, measurement variables and output conventions. Kernel benchmarks and quality tests are separate from throughput measurements.

Historical measurements live in [docs/reports/](docs/reports/README.md). Raw campaigns remain in `bench/results/`; new generated campaigns are ignored by default because they can contain local paths, process command lines and prompts. Existing tracked evidence is retained. Review and redact a separate copy before publishing new raw data; do not change the original measurement archive.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing Python environment/packages | Run the explicit venv and pip steps above. Check that `.venv\Scripts\python.exe` is Python 3.10+. |
| Build cannot find MSVC or SDK | Install the C++ workload and Windows SDK; the build uses ROCm Clang with MSVC headers/linker. |
| HIP runtime/DLL failure | Run `tools\hip\poly_windows.bat probe`; keep the SDK runtime DLLs beside the build products and the matching ROCm `bin` on PATH. Avoid mixing driver and SDK runtimes. |
| Wrong GPU selected | Inspect `strata-device --list-devices` via the probe wrapper and set `hip_ordinal` in the server configuration. |
| Out of memory or very slow long context | Reduce `--max-context` and close other GPU applications. Read the cache/reservation guide before setting an expert floor. |
| Q8 missing artifact or SHA mismatch | Use the normal FP16 route, or supply the exact validation artifacts; do not remove the checks. |
| Port 8090 already occupied | Stop the known server yourself or select another `port` in the config. The launcher does not switch or kill another server. |
| Profile conflict | Select another configuration name or edit the file explicitly; the launcher will not replace it. |

More: [HIP GPU failures](docs/HIP_GPU_FAILURES.md), [general troubleshooting](docs/TROUBLESHOOTING.md), and the [release audit](docs/RELEASE_AUDIT.md).

## Repository layout

```text
START_HIP_POLY.bat        Windows HIP launcher
src/poly/                GGUF engine, architecture implementations, kernels
src/ + include/          shared engine, platform layer and kernels
serve/                   Python API server and web app
cmake/                   backend and build support
tools/hip/               HIP build, config, validation and measurement tools
tools/vision/            image encoder source
tools/test_*.py           inherited setup/tool unit tests
tests/                   core, platform, CUDA, HIP and quality tests
bench/                   comparisons, microbenchmarks and historical results
data/                    small runtime data and the Q8 expert profile
docs/                    user/developer guides; detailed reports in reports/
sycl/                    inherited Intel backend
third_party/ggml/        retained license and upstream metadata
```

Build folders, environments, weights, generated configs, scratch runs and binaries are local artifacts. No inference sources were relocated for this cleanup. See [CHANGELOG.md](CHANGELOG.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits and license

[Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors supplies the original Qwen engine, installer, server and web app. [PolyStrata](https://github.com/VecSzn/PolyStrata) supplies the GGUF engine extended by this HIP fork. The [MIT license](LICENSE) and original copyright notice are preserved. ggml/llama.cpp attribution and license are retained in [third_party/ggml/](third_party/ggml/LICENSE) and [sycl/third_party/ggml/](sycl/third_party/ggml/LICENSE). ROCm packages and model weights have their own licenses; this repository does not grant rights to redistribute them.
