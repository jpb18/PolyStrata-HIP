# AMD technical reports

These reports record experiments on their stated dates, builds and hardware. Follow the [current README](../../README.md) for installation; historical commands can describe superseded startup behavior. Personal paths in these publication copies were generalized without changing measurement values. Original raw evidence and pre-cleanup copies remain local.

| Report | Subject |
| --- | --- |
| [Port feasibility](AMD_PORT_FEASIBILITY.md) | Initial hardware/toolchain investigation |
| [Port implementation](AMD_PORT_IMPLEMENTATION.md) | HIP build and first engine validation |
| [Phase 2](AMD_PORT_PHASE2.md) | Runtime integration and measurements |
| [Qwen implementation](AMD_QWEN36_IMPLEMENTATION.md) | Qwen model path |
| [Qwen Q3 migration](AMD_QWEN36_Q3_MIGRATION.md) | Exact model, quantizations, API and context measurements |
| [KV streaming](AMD_KV_STREAMING.md) | FP16 host/VRAM residency |
| [Decode optimization](AMD_HIP_DECODE_OPT.md) | Historical kernel/workload experiments |
| [Decode validation](AMD_HIP_DECODE_VALIDATION.md) | Validation evidence and limitations |
| [VRAM residency](AMD_HIP_VRAM_RESIDENCY.md) | Allocation and cache behavior |
| [Expert cache floor](AMD_HIP_VRAM_FLOOR.md) | Cache floor, reserve and resize measurements |
| [Empero Qwen distill](AMD_EMPERO_QWEN38.md) | Tested distill configuration |
| [Gemma](AMD_GEMMA4_26B_A4B.md) | Gemma investigation and measurements |
| [GLM](AMD_GLM47_FLASH.md) | GLM-4.7 Flash investigation; do not infer support from the report name |
| [North Mini Code](AMD_NORTH_MINI_CODE.md) | Model feasibility/investigation |

The root files with these names are compatibility pointers. References to raw campaigns under `bench/results/` may require the local evidence archive, which is not automatically included in a clone. New raw campaign directories are ignored by default; existing tracked historical results remain versioned. Current Q8 implementation and decisions are in [AMD_HIP_KV_Q8.md](../AMD_HIP_KV_Q8.md) and its linked records.
