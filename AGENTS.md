# AGENTS.md

PolyStrata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

A second engine, `strata-poly` (`src/poly/`), reads GLM 5.3 Flash and smaller models from GGUF files:
[docs/GLM.md](docs/GLM.md), [docs/BACKENDS.md](docs/BACKENDS.md); experimental AMD support: [docs/AMD_HIP.md](docs/AMD_HIP.md).
The Qwen engine, the installer, the server and the web app come from
[Strata](https://github.com/Niko1221/Strata).

## Installing PolyStrata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, PolyStrata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- Start searches in `src/`, `include/`, `serve/`, `tools/`, `tests/`, `sycl/` and the relevant top-level build/setup files.
  `.rgignore` omits historical results and root AMD reports; use `rg --no-ignore <pattern> bench/results/<run>`
  (or a specific report) when evidence is needed. See [docs/REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md) for the inventory.
- Skip `build*/`, `engine*/`, `.venv/`, `.rocm-win/`, `.lcm/` and generated logs in routine source searches.
- There is one engine: `build-hip-poly-win\strata-poly.exe`, which every `START_HIP_POLY.bat` menu runs. No promoted, frozen or
  "to be promoted" binaries, no pinned engine hashes: after a change, build it there (`tools\hip\build_windows.bat poly` builds into that folder), so the next
  start of `START_HIP_POLY.bat` has it. Do not keep rollback copies of the engine. Preserve validation reports, patches and existing local work.
- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Text taken from the Strata project names that project. After merging from it,
  `python tools/polystrata_names.py` writes PolyStrata again; its first lines say what it leaves alone.
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A feature is on by default, and reachable from `START_HIP_POLY.bat`. It gets a flag only when it was measured to cost
  speed (say by how much). Say how you checked the bits.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
