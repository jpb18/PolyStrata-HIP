# HIP engine and configuration

This guide describes the implementation in this working tree. Start with the [Windows quick start](../README.md). Historical reports describe their own builds and dates, and are not current installation instructions.

## Hardware and limits

The complete fork path was measured on Windows 11 Pro, RX 9060 XT 16 GB (`gfx1200`), Ryzen 5 5600X and 31.9 GB RAM, using ROCm 10.2.0a20260930. See the [Qwen3.6 report](reports/AMD_QWEN36_Q3_MIGRATION.md). Other architectures listed by the build script are compilation targets, not a validation matrix for this fork. The inherited `strata` backend has its separate [AMD support record](AMD_HIP.md).

`Engine::open()` in `src/poly/engine.cpp` dispatches GGUF architectures `qwen35moe`, `glm5-next`/`glm5next` and `gemma4`. Qwen3.6, Ornith and the tested Empero distill use the Qwen path. This does not make every GGUF or every quantization of those models supported. See each [model report](reports/README.md), [BACKENDS.md](BACKENDS.md) and [GLM.md](GLM.md) for evidence and unsupported cases. Large GLM models need much more RAM than the measured smaller Qwen configuration.

This Windows path runs one GPU per engine. Inherited multi-GPU and SYCL capabilities should not be inferred to apply to this experimental `strata-poly` HIP route. Linux HIP build support exists in CMake, but the new Windows instructions do not validate Linux or additional cards. Vision requires a separately built encoder and a matching projector; it is not configured by the text quick start.

## Build and dependencies

`tools/hip/build_windows.bat poly` builds the GGUF HIP engine; `poly tests` also enables the broader test targets. The script explicitly installs ROCm wheels and can install CMake/Ninja. The launcher never calls this script automatically.

| Component | Source of version/settings |
| --- | --- |
| Python packages | Pinned `requirements.txt`, Python 3.10+ |
| Windows ROCm | `STRATA_ROCM_VERSION`, default `10.2.0a20260930` |
| ROCm package index | `STRATA_ROCM_INDEX` in `tools/hip/build_windows.bat` |
| AMD targets | `STRATA_HIP_ARCHS`, semicolon-separated; select your actual architecture |
| Private SDK environment | `ROCM_VENV`, default `.rocm-win` |
| Build directory | `BUILD_DIR`, default `build-hip-poly-win` for `poly` |
| ggml/llama.cpp | Commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` in `CMakeLists.txt` |
| Local upstream checkout | `STRATA_GGML_DIR` (llama.cpp root, containing `ggml/`) |

For reproducibility, do not point `STRATA_GGML_DIR` at a different revision without recording it. CMake normally fetches the pinned checkout under the build tree. Tokenizer extraction also needs its `gguf-py` directory. The config utility searches the normal third-party locations, `STRATA_GGML_DIR`, and the selected build's fetched checkout. Use `poly_config.py --gguf-py` to specify another location. A fetched checkout is normally under `build-hip-poly-win/_deps/strata_llamacpp-src/gguf-py`; confirm the folder before using it.

```bat
.venv\Scripts\python.exe tools\hip\poly_config.py --model "models\model.gguf" --name my-model-hip --context 32768 --gguf-py "build-hip-poly-win\_deps\strata_llamacpp-src\gguf-py"
```

The build requires the Visual Studio C++ headers, linker and Windows SDK even though ROCm Clang compiles the HIP and host sources. HIP runtime DLLs are copied beside the executables; other SDK libraries are found through `lib_dirs` in the generated server configuration or through `tools/hip/poly_windows.bat`.

On Linux, the equivalent CMake switches are `STRATA_ENABLE_HIP=ON`, `STRATA_ENABLE_CUDA=OFF`, `STRATA_POLY_HIP=ON`, `STRATA_NATIVE_EXPERTS=ON`, and `STRATA_PREFILL_MMQ=ON`, with a matching installed ROCm toolchain. See [AMD_HIP.md](AMD_HIP.md) before adapting that build; no new Linux validation was performed for the cleanup.

## Configuration

The generated `strata-<name>.json` is local and ignored by Git. `exe`, `cwd`, `tokenizer`, `log` and `lib_dirs` describe the local installation. Keep engine switches as separate strings in `args`, for example:

```json
{
  "args": [
    "--model", "C:/models/model.gguf",
    "--max-context", "32768",
    "--threads", "6",
    "--vram-reserve-mib", "1024",
    "--expert-profile", "C:/models/model-pack/expert-profile.txt",
    "--window", "8"
  ]
}
```

This is an `args` fragment, not a complete server configuration. Edit a configuration generated for your installation. Six threads and a 1,024 MiB reserve are example values, not recommendations for every PC. `POLY_CONFIG` tells the launcher to use the existing file directly, preserving its arguments. Without it, preset generation carries the selected model, tokenizer, expert profile, threads, port, window, sampling and applicable vision settings; use `POLY_CONFIG` for other custom switches.

For server GPU selection, `hip_ordinal` chooses the single HIP device (and takes precedence over `gpu`). The server sets `HIP_VISIBLE_DEVICES` using its device mapping. Use the probe wrapper to inspect HIP numbering; an integrated adapter can occupy ordinal zero. A direct engine run can use the existing `HIP_VISIBLE_DEVICES` environment variable. No new device-selection policy is introduced here.

### Q8 KV and streaming

The Qwen implementation is in `src/poly/qwen35.cpp/.hpp`, with kernels in `src/poly/kernels.cu` and declarations in `include/strata/poly/kernels.hpp`. `--kv fp16` is the engine default. `--kv q8_0` stores blocks of 32 values with an FP16 scale; `--kv-k` and `--kv-v` permit separate key/value formats for investigation. For the tested Qwen shape, the help text reports 11,968 bytes per position versus 22,528 in FP16. These figures do not describe arbitrary models.

Q8 is supported on the Qwen attention path, requires all KV positions in VRAM and rejects incompatible streaming settings. Unsupported formats/architectures fail during loading rather than silently falling back. Q8 affects attention numerics; do not claim FP16 token identity. See [the implementation and validation record](AMD_HIP_KV_Q8.md) and its [manual decisions](AMD_HIP_KV_Q8_MANUAL_APPROVAL.md).

FP16 streaming uses `--kv-resident N|auto|all`, `--kv-ram-mib` and `--kv-ram-floor-mib`. Positions outside the resident prefix use pinned host RAM and PCIe traffic. The `262k-stream` preset keeps 49,152 positions resident; it is not a Q8 preset.

The launcher's Q8 route is narrower than source support. It verifies a particular executable SHA and model, uses the included frozen expert profile, checks free RAM/VRAM, checks INFO/READY and HTTP health, and refuses an occupied port. It creates a separate run directory under `tmp/hip-launch-q8`, disables swapping and conversation parking, and never promotes a newly compiled binary automatically. The included `data/hip/qwen36-q3-q8-expert-profile.txt` is byte-identical to the campaign profile (SHA-256 `1bc70f7edf706578a24aca5a6e8beee4f308e192142ae88339968f5da1924a9d`). The engine may update expert profiles, so the launcher copies the frozen file per run.

### Expert cache and VRAM safety

`src/poly/engine.cpp/.hpp` owns allocations, expert placement, usage counts, swaps and cache resizing. It chooses resident experts from a usage profile; misses execute through the CPU expert path. `--expert-swaps N` limits replacements during decoding (default 16; zero freezes placement). This is usage-based caching, not a newly introduced LRU algorithm.

The cache fills against available memory after allocations and reservation. `--vram-reserve-mib` leaves headroom, `--expert-cache-max` caps the number of experts, and `--expert-cache-min` optionally limits how far a resize may shrink it. The floor defaults to zero and can leave the requested reservation short. Inspect `vram_reserve_short_mib` in INFO and `/v1/vram`; do not raise a floor to conceal memory pressure. Between requests, memory checks can shrink the cache; the VRAM protocol can request resizing. [Measured floor/residency behavior](reports/AMD_HIP_VRAM_FLOOR.md) records the limitations.

Fatal HIP failures are handled by `include/strata/poly/gpu_failure.hpp` and the server's recovery policy. See [HIP_GPU_FAILURES.md](HIP_GPU_FAILURES.md). The cleanup does not change allocations, layouts, synchronization, kernels, cache scheduling or numerical policy.

### MTP and speculative decoding

Draft-block loading is architecture-specific; the shared engine and `src/poly/main.cpp` manage guessing, target verification and accepted-token accounting. A compatible embedded MTP block can be used, or `--draft-model FILE` supplies the same model's missing block (Gemma's assistant is its own architecture-specific route).

`--window N` controls speculative window size; `--window 1` disables speculative windows. `--no-draft` disables the draft block; repeated-token lookup is separately controlled by `--lookup` (zero disables it). There is no generic `--mtp on` option. Record these controls and confirm an actual draft load/acceptance in logs when reporting MTP measurements. `--window-fixed` is a diagnostic instrument, not the normal adaptive policy. See [INFERENCE_METRICS.md](INFERENCE_METRICS.md).

The launcher does not rewrite MTP options in an explicitly selected FP16 config. The frozen Q8 adapter preserves the source config's MTP switches while applying its documented cache/parking policy. A missing or incompatible MTP block must not be presented as an MTP-enabled benchmark.

## Code map

| Area | Files |
| --- | --- |
| Engine CLI and line protocol | `src/poly/main.cpp` |
| GGUF loading, memory, expert cache | `src/poly/engine.cpp`, `engine.hpp`, `cpu_experts.cpp` |
| Model implementations | `qwen35.*`, `glm5next.*`, `gemma4.*` in `src/poly/` |
| HIP kernel compilation | `cmake/hip_backend.cmake`, `include/strata/platform/hip_compat/` |
| Quantized operations | `src/poly/kernels.cu`, `mmq.cu`, shared `src/kernels/` |
| Windows launch/config | `START_HIP_POLY.bat`, `tools/hip/poly_launch.py`, `poly_profile.py`, `poly_config.py` |
| API and browser | `serve/server.py`, `serve/web/` |
| Timing/telemetry | `serve/inference_metrics.py`, `serve/prometheus.py` |

Keep these locations stable: CMake, packaging, imports and external tooling already refer to them.
