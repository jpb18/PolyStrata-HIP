# HIP GPU failures: stop the engine and recover in a new process

This policy applies to the experimental HIP build of `strata-poly`. The server's
fatal-error protocol and health handling are shared with the other engines.
It changes failure handling, without changing model arithmetic, sampling defaults,
MTP policy, driver settings or the four Qwen profile configurations.

## Detection and scope

Checked HIP runtime calls and hipBLAS failures stop the process. This covers launch
errors, errors returned by copies, events, streams and synchronization, and errors
from the thread that uploads experts. Launch errors are checked before the existing
stream synchronization boundaries, including prefill, decode, MTP windows and draft
logits. Synchronization checks execution errors before host logits are consumed.
The routing and draft paths accept `NotReady` as pending work but no longer discard
other query errors. MMQ's launch checks use the same termination policy.

The policy deliberately treats **every unexpected error in a checked inference
call** as fatal for that process. Illegal addresses, device assertions, launch
failure/timeouts, destroyed contexts and unknown runtime errors are examples; an
error code is not proof that the hardware itself is damaged. There is no reliable
rollback of partly written KV, recurrent state, graph state or expert-cache slots.
An exception after request validation, during generation or a VRAM cache mutation,
also makes that inference process unusable. Malformed requests and requests outside
the context or vocabulary are rejected before inference and remain recoverable.

Two existing optional startup cases remain specific: allocation OOM can skip the
small BLAS warmup; the Windows memory query can retry its initial `InvalidValue`.
The Windows hipBLAS stale `InvalidValue` workaround applies only after a successful
product, with a launch check before the product. Other HIP errors propagate. A
missing DXGI budget still uses HIP's memory figure; a HIP device-query failure must
not become successful fallback data.

## State, protocol and requests

`gpu_failure.hpp` latches the first failure, writes one stderr diagnostic and one
stdout `ERR GPU_FATAL ...` line, flushes them, and uses `_Exit(70)`. The diagnostic
includes backend, operation, numeric code, error name and reason. The existing
startup log records device/model/memory configuration; the server records PID and
the interrupted request. There is no per-kernel success logging.

Termination invalidates the entire process: KV in VRAM or mapped RAM, recurrent
states, parked conversation snapshots, MTP guesses, streams, graphs and caches.
No GPU reset, free, stream destruction, cache save or `atexit` cleanup is attempted
after the fatal path. Such cleanup can hang on a damaged context. A failing upload
thread uses the same path; token publication checks its failure latch.

The Python output pump recognizes the fatal marker before serial/batch routing,
marks the engine dead, discards queued tokens and completion lines, invalidates
held slot prefixes, and sends EOF to every consumer. Consumers also check the
failure latch and process generation before consuming pending output, retaining
the old process's queues across a restart. The active generation fails with
`EngineDied`, rather than a validation error or a successful completion. HTTP
nonstream requests return 503; streams end through their existing API error event
(OpenAI chat, Responses and Anthropic). A streamed `[DONE]` closes the transport;
it does not make an earlier error event a successful answer.

`/health` and `/api/health` return HTTP 503, `status: error`, `loaded: false` and
`recovery: new_engine_process` after a detected fatal failure. Unplanned process
exit is also unhealthy. Healthy responses and intentional idle/lazy unloading keep
their existing response shape. Unloading a failed engine does not clear its fault.

## Recovery

The affected request is never retried automatically. The existing server lifecycle
allows a subsequent request to start a **new native process**, after terminating
and waiting for the old one. Requests still waiting for admission may proceed in
that new process; none may use the failed context. The new process reloads weights,
starts empty inference caches and must reach `READY` before the fault latch clears.
A failed startup keeps the service unhealthy. The server already refuses a restart
when it cannot end the old process (`EngineStuck`), and bounds normal startup retries.
Restarting the whole server is also a valid recovery step. Repeated failures require
investigation of the engine and driver logs; this change does not repair their cause.

## Limits

This is protection against **observable failures**, not a GPU integrity monitor.
The preceding [decode validation](reports/AMD_HIP_DECODE_VALIDATION.md) recorded a Windows
WATCHDOG incident and incoherent generation without a HIP error in its log. That
does not establish which state was lost or whether HIP reported an error at all.
There is no claim that a TDR always returns a HIP failure. Successful runtime calls
and synchronization do not prove correct arithmetic or intact memory. Silent
corruption, stale yet plausible values, and resets that the runtime hides can pass
these checks. No Windows event-log monitor or independent numerical oracle was added.

Already transmitted stream bytes cannot be retracted. A client must discard a
response that ends in an error; errors detected after an answer was sent cannot
retroactively invalidate that answer on the network. A runtime call that never
returns cannot execute its error check. The existing server silence/stall handling
and routing timeout remain the fallback, with their configured allowances; disabling
those timeouts removes their bound. This policy is not a hard real-time guarantee.

AMD documents the distinction between [launch errors and runtime checks](https://rocmdocs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/error_handling.html)
and the [HIP error codes](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/error_codes.html).
Microsoft's [TDR description](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery)
provides Windows reset context, without specifying HIP's error-reporting guarantees.

## Tests

The driver-free fixture in `tests/poly_gpu_failure_mock/` compiles the production
failure header against a small HIP mock. `tools/hip/test_gpu_failure.py --exe PATH`
injects runtime/launch/synchronization/query/worker failures in subprocesses and
verifies exit 70, one diagnostic, no continuation, and no destructor recovery.
Benign `NotReady`, successful calls and unknown inference state are covered.
Optional OOM clearing is checked against an unrelated fatal last error too.
`python -m unittest serve.test_fatal_recovery` exercises real pipes/processes and
HTTP endpoints, buffered output rejection, fresh-process recovery and a fault latch
that survives failed startup. No test deliberately causes a TDR or illegal access
on a real GPU. Results and reproduction scripts for this PC are in
[`bench/results/2026-10-12-hip-failclosed-validation/`](../bench/results/2026-10-12-hip-failclosed-validation/).
