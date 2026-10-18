# Contributing

Search existing issues and pull requests before opening a report. Use the issue templates for bugs, features and questions, and keep each pull request focused on one change. Do not include local model weights, environments, build products, credentials, personal prompts or unreviewed benchmark archives.

Start with [the code map](docs/HIP_GUIDE.md#code-map) and [tests](docs/TESTING.md). Preserve engine interfaces, configuration keys and existing script paths. New features are opt-in. Kernel, memory-layout, scheduling and default changes need separate measurements and backend validation; documentation cleanup is not a reason to modify inference.

Build every backend a changed source file touches (CUDA, HIP and SYCL as applicable). Report commands, toolchain/hardware, failures, skipped tests and untested configurations. If you cannot run a backend, say so before asking for review. Never replace validation thresholds with easier ones.

For performance reports, compare against a same-day reference and include the variables in [bench/README.md](bench/README.md). Use copies of expert profiles: inference updates their counts. Preserve original evidence locally and review a separate publication copy for paths and private prompts.

Keep the MIT notice and upstream attribution. Documentation should use plain words and attach every measured number to its machine and workload. The inherited naming utility `tools/polystrata_names.py` is for upstream merges; inspect its exclusions before applying it.
