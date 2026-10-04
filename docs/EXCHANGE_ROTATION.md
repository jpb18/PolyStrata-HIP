# Resident expert exchange buffer rotation

When the resident RAM complement exchanges an expert with the GPU cache, the
evicted expert lands in a temporary host buffer. The existing path then copies
those bytes into the promoted expert's old RAM slot. Rotation removes that last
host copy: the temporary buffer becomes resident, and the old RAM slot becomes
the next temporary buffer.

## Enable

Rotation is **off by default**. Set `STRATA_EXCHANGE_ROTATE=1` before starting the
engine with your existing arguments:

```bash
export STRATA_EXCHANGE_ROTATE=1
```

```powershell
$env:STRATA_EXCHANGE_ROTATE = '1'
```

Only the exact value `1` opts in. Set it to `0` or remove it to restore the copy
path. This setting does not select resident RAM mode or change any preset.
It applies only when all layers have equal-size expert blocks, the resident
complement is fully pinned and GPU-mapped, and the exchange buffers can also be
pinned and mapped. Other layouts keep the copy path and log why rotation could
not be enabled. Configurations without a resident complement do not use it.

The startup diagnostic confirms activation:

```text
FileExpertSource: exchange buffer rotation enabled: ...; no host commit memcpy
```

Serving diagnostics report the cumulative rotated block count and avoided
`memcpy` payload bytes. This counts the copied payload once, not read plus write
traffic, and does not include the GPU transfers, which still occur.

## Ownership and synchronization

The original allocations own their memory until `FileExpertSource::close()`.
Rotation updates slot ownership and keeps each host pointer paired with its GPU
alias. Atomic slot IDs let background residency queries observe those immutable
pairs. Compute readers and GPU transfers must still finish before the existing
serialized commit; rotation adds no overlapping compute or new synchronization
policy. Reserve the maximum exchange capacity before rotation starts: growing a
live exchange arena is rejected because it can now hold resident experts.

## Tests

The ownership test can run without CUDA:

```bash
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -pthread -Iinclude tests/core/exchange_storage_test.cpp -o /tmp/exchange_storage_test
/tmp/exchange_storage_test
```

With an existing CUDA build configured:

```bash
cmake --build build --target strata exchange_storage_test file_expert_source_test
ctest --test-dir build -R '^(exchange_storage_test|file_expert_source_test)$' --output-on-failure
./build/file_expert_source_test --rotation-gpu
compute-sanitizer --tool memcheck --error-exitcode 71 ./build/file_expert_source_test --rotation-gpu
```

The CPU test checks byte preservation, pointer/alias ownership, guards, invalid
operations and concurrent residency queries across 12,304 exchanges. The CUDA
fixture performs 64 exchanges in each of three modes: copy with pinned RAM,
rotation with pinned RAM, and requested rotation with pageable RAM (fallback).
It checks exact bytes, GPU alias reads, file fallback, capacity, and close/reopen.
The GPU fixture is explicit; the ordinary CTest invocation does not run it.

On 2026-10-04, the clean patch on upstream
`6f32ec070f23ced9f50e704d854d775da52591ab` built on Linux with GCC 15.2 and CUDA
13.3 for an RTX 4090. Both CTest tests, the ASan/UBSan ownership test, and the
CUDA fixture passed. Compute Sanitizer reported **0 errors**. AMD and Windows
GPU execution were not tested for this patch.

## One native token-identical A/B

This is one existing measurement from the original rotation implementation at
`1a50d913bf910a1f63fbc1a0788a7083e3ca5f8c`, dated 2026-10-03. It is **not a new
full-model benchmark of the clean patch on current main**. The available model
packs for the fresh checks have mixed expert block sizes and cannot activate
this deliberately narrow path.

Both arms used the same native binary and coding prompt: Q8_0 weights, FP16 KV,
65,536 input tokens, 1,024 output tokens, target-only decoding (MTP and suffix
drafting disabled), 16,400 GPU expert slots, and a 39.77 GiB pinned RAM
complement. Hardware: RTX PRO 6000 Blackwell 96GB, Ryzen 9 7950X, 128GB RAM.

| Rotation | Decode tokens/s | Resident exchanges | Avoided host copy payload | Output token IDs |
|---|---:|---:|---:|---|
| `0` | 70.07 | 2,377 | 0 bytes | 1,024, identical |
| `1` | 75.39 | 2,377 | 12,413,644,800 bytes | 1,024, identical |

The [A/B receipt](../bench/results/exchange-rotation-ab.json) contains the binary,
prompt and token hashes, diagnostics, and a pinned public source receipt. Both
token hashes are
`b1f0b9db418f250daa39ec15507b7e60afa8168c32bd23e7e7fdab178f95c25a`.
The original comparison reported no differing token; its published summary
retains counts and hashes, not raw token arrays. This single pair establishes
the recorded parity result, not a general speedup or speculative-decoding
parity guarantee.
