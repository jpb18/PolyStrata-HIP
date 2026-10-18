# Tests and validation

Run from the repository root with the environment installed from `requirements.txt`. Tests retain their existing locations to preserve direct invocation, imports and CMake targets. Their locations do not imply they all need hardware.

| Category | Locations and requirements |
| --- | --- |
| Python unit tests | `tools/test_*.py` (setup, tokenization and utilities), selected `tools/hip/test_*.py`, `tests/test_*.py` (quality/scoring policies); usually no GPU |
| Server/process integration | `serve/test_*.py`, launcher tests; mock engines and loopback HTTP, usually no GPU |
| CPU C++ tests | Standalone conversation/speculative-policy targets in `src/core/` and `src/poly/`; broader tests under `tests/core/`, `tests/platform/` and `tests/prefill/` |
| HIP/CUDA hardware checks | `tests/hip/`, `tests/cuda/`, kernel parity/bench targets under `src/kernels/` and `src/poly/`; matching GPU/runtime |
| Local installation checks | `tools/hip/test_vision_profiles.py` requires the author's configured models/encoder/projector; `serve/test_detok.py` needs a tokenizer and otherwise skips |
| Model integration/quality | `poly_api_check.py`, `poly_llama_parity.py`, `kv_parity.py`, `kv_q8_compare.py`, functional-quality campaigns; explicit models, reference artifacts and substantial memory |
| Benchmarks | `bench/`, `tools/hip/bench_*.py`, `poly_kernel_bench`; see [benchmarking](../bench/README.md) |

## CPU and mock checks

```bat
.venv\Scripts\python.exe tools\hip\test_poly_profile.py
.venv\Scripts\python.exe tools\hip\test_poly_launch.py
.venv\Scripts\python.exe tools\hip\test_kv_q8_t4_policy.py
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe -m unittest discover -s serve -p "test_*.py"
.venv\Scripts\python.exe tools\test_setup_amd.py
.venv\Scripts\python.exe tools\test_setup_choices.py
```

The setup scripts are independent tests; each `tools/test_setup_<name>.py` runs without a GPU or model download. Use separate Python processes when running the whole collection, so module patches and test fixtures do not leak between scripts. Do not indiscriminately discover all `tools/hip/test_*.py`: `test_gpu_failure.py` requires `--exe` pointing to the compiled driver-free mock, and the vision profile checks require local fixtures.

`STRATA_TOKENIZER` can point to an installed tokenizer for detokenizer tests. Tests use local temporary files and loopback ports. A sandbox that forbids loopback or child processes can prevent those checks; record that limitation rather than disabling assertions.

A separate CPU-only CMake build can exercise standalone conversation and speculative-policy tests without modifying operational build folders:

```bat
.venv\Scripts\cmake.exe -S . -B build-release-cpu-check -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=ON
.venv\Scripts\cmake.exe --build build-release-cpu-check --config Release
.venv\Scripts\ctest.exe --test-dir build-release-cpu-check -C Release --output-on-failure
```

This still needs a C++20 compiler. Other CPU tests are gated by the broader CMake options and may require ggml or model fixtures; inspect the targets rather than assuming the standalone selection covers them.

## Hardware and model validation

Build the touched backend before requesting a code review. For the Windows HIP engine:

```bat
tools\hip\build_windows.bat poly tests
tools\hip\poly_windows.bat probe
tools\hip\poly_windows.bat tests
tools\hip\poly_windows.bat parity "models\model.gguf"
.venv\Scripts\python.exe tools\hip\poly_api_check.py --base http://127.0.0.1:8090
```

The wrapper's `tests` command runs CTest names matching `poly|conversation`, not the entire HIP suite. It can include GPU tests. To run other targets explicitly, use the same SDK environment, for example `tools\hip\poly_windows.bat cmd .venv\Scripts\ctest.exe --test-dir build-hip-poly-win --output-on-failure` after inspecting `ctest -N`. Some tests need external fixtures; a missing fixture is not a kernel pass.

For Q8: compare the same GGUF, context, prompt token IDs, expert placement and MTP settings against FP16 and a same-day reference build. Test keys/values, prompt continuation, cache restore, long context, failures and speculation. Follow the existing [Q8 validation record](AMD_HIP_KV_Q8.md), [paired-quality protocol](AMD_HIP_KV_Q8_PAIRED_TESTS.md) and [functional harness](../tools/hip/functional_quality/README.md). CPU/mock checks do not establish GPU numerical correctness, VRAM safety or throughput. Do not change validators or thresholds to obtain a pass.

The [release audit](RELEASE_AUDIT.md) records checks actually run for this cleanup and checks left to hardware validation.
