# Documentation

Start with the [HIP fork README](../README.md).

| Guide | Purpose |
| --- | --- |
| [HIP guide](HIP_GUIDE.md) | Build dependencies, hardware limits, Q8 KV, expert cache, MTP and code map |
| [Tests](TESTING.md) | CPU, mock/API, local-fixture and GPU/model checks |
| [Benchmarks](../bench/README.md) | Comparison commands, metrics and reproducibility |
| [Release audit](RELEASE_AUDIT.md) | Cleanup scope, files preserved, checks and release limits |
| [Privacy cleanup](PRIVACY_CLEANUP.md) | History anonymization, public Git identity, verification and recovery |
| [Technical reports](reports/README.md) | Historical AMD measurements and investigations |
| [Q8 implementation/validation](AMD_HIP_KV_Q8.md) | Detailed evidence, with subsequent decision records linked inside |
| [Frozen Q8 launcher](HIP_LAUNCH_KV.md) | Local validation route and required artifacts |
| [Metrics](INFERENCE_METRICS.md) | Native timing fields, semantics and missing-data rules |
| [GPU failure handling](HIP_GPU_FAILURES.md) | Process failure and recovery policy |

The inherited [AMD backend](AMD_HIP.md), [installer](INSTALL.md), [model backends](BACKENDS.md), [GLM](GLM.md), [architecture/details](DETAILS.md), [Intel backend](INTEL.md) and [original README](UPSTREAM_README.md) describe broader or different routes. Their hardware matrices and measurements do not automatically validate this fork's `strata-poly` HIP route.

Historical report links can refer to raw local campaign archives omitted from a clone. The source, curated expert profile, build pins and benchmark procedures needed for a new run are documented independently. No model weights or operational executables are distributed by this cleanup.
