# Q8/FP16 functional comparison — protocol v2

Assessment policy: KV-Q8-paired-v1-2026-10-17. Absolute per-arm validators,
tasks and execution budgets below remain unchanged. Reports now use paired
outcomes: both pass, shared failure, Q8-only failure, Q8-only success. Shared
failures are not attributed to Q8. Q8-only failures require review even when
improvements elsewhere cancel the aggregate difference. Missing/infrastructure
or incompatible inputs never become comparable task scores. All shared failures
means BASELINE_LIMITED, not PASS. This user-requested revision is for new freezes;
old frozen protocols/records/reports stay intact. Replay is explicitly retrospective.

This is the complementary 36-task pilot: 14 code, 12 tool use, 6 technical
reasoning, 4 strict instruction tasks. It does not replace the original T8.
The legacy original T8 in docs/AMD_HIP_KV_Q8.md §8.1 required poly_api_check.py 15/15,
100% valid tool-call JSON, and 20 executed code tasks with Q8 >= FP16 - 1.
ORIGINAL_CODE.jsonl freezes those 20 tasks separately (the pilot's 14 plus 6).
The original code suite and API checks have separate result namespaces. Passing
the pilot never marks the original T8 or Q8 promotion approved.
Shared code tasks C01..C14 reuse the exact same hashed main result between suites;
they are never generated a second time for the original code suite.

## Inputs and order

The manifest hashes task files, protocol, CPU tests, harness, actual server Python
sources, tokenizer files, model bytes, runtime DLLs, candidate and frozen expert
profile. The candidate is 309aab05003e794a6f4975e1499afc77f8d26030aa85a453f2535b90e108f3c5;
profile is 1bc70f7edf706578a24aca5a6e8beee4f308e192142ae88339968f5da1924a9d.
The expected 5000-expert set is dea737b94e80ddcd. Every server gets a new copy of
the frozen profile, never the operational profile. Same GGUF and executable in
both arms. FP16 omits --kv, Q8 adds --kv q8_0. Both use context 49152, 6 threads,
--turn-token 248045, --expert-cache-max 5000, --expert-swaps 0, adaptive default
MTP (no window override), STRATA_POLY_ATTN_FIT=1, STRATA_POLY_CACHE_HASH=1,
STRATA_POLY_WARM=1. Conversation parking is disabled to avoid retaining tasks.

Temperature 0, reasoning none/thinking disabled, no custom stop strings. Token
and turn budgets are fixed per task. Maximum 8 tool calls per turn. Default
server retry/repetition recovery is explicitly disabled in isolated test configs.
No model switching, vision, operational settings or launcher changes.

Within each category, alternate tasks between first-FP16 and first-Q8 groups.
Four sequential server blocks: group 0 FP16, group 0 Q8, group 1 Q8, group 1 FP16.
Thus exactly 18 AB and 18 BA pilot pairs, balanced within category. No ABBA per
task. Task order is frozen; actual block order, restarts and resume are logged.
Each block first runs the same disposable warm-up (reply READY, 16 tokens),
which is excluded from scores. Main tasks run once per format. Completed FAIL
is as final as PASS and is never selectively rerun.

## Automatic validation

Code tasks expose only solution.py, immutable public_tests.json, and the
read_file/write_file/run_tests tools. Source returned in prose is never applied.
The harness actually writes the requested source and executes public tests in a
fresh subprocess; after completion it executes hidden and public tests again.
Passing requires a write, a public test invocation, no tool errors, final stop,
all public/hidden tests, and no input mutation. Hidden cases and oracle source
are never put in the model's project or messages. The worker accepts restricted
pure Python with a curated builtin/method set, no imports/reflection/I/O, 256 MiB
process memory, 3 CPU seconds and a 5-second parent deadline. Windows Job Objects
or Linux resource limits are mandatory. This is not a general-purpose shell or
an operating-system security sandbox for arbitrary programs. Tasks fit the
restricted language and reference solutions are executed before freeze.

Tool tasks each start a new loopback REST server with deep-copied initial state.
Every accepted call goes through HTTP; results return as actual tool messages,
and the model continues. Passing requires strict valid JSON arguments (no NaN,
duplicates or trailing text), schema/name correctness, a live read first, the
specified exact mutation sequence and arguments, exact final state and typed
JSON answer, final stop, and no tool errors. Additional read-only calls are
allowed. Every call including invalid ones counts in the JSON denominator.
The mock is destroyed between tasks and arms. No arbitrary URLs or reset tool.

Reasoning uses typed JSON numbers/booleans/algorithmic answers. Instruction tasks
use typed JSON or exact line/CSV constraints. Object-key order is immaterial;
array/line order matters. Exact line tasks allow one terminal LF or CRLF only.
There is no natural-language similarity grading or second model as judge.

CPU tests check all reference solutions, negative solutions, hidden boundaries,
mutation and escape rejection, real REST state/reset and sequence, malformed
JSON, streamed tool cycles, limits, idempotence and balanced order. Disposable
smoke fixtures are distinct from scored tasks; no real task is sent to any model
before the freeze checkpoint.

## Records, safety and failure policy

One exclusive OS GPU lock, dedicated loopback port 8197, no concurrent arms.
Admission requires >=16384 MiB available RAM, <=3072 MiB background dedicated
VRAM, known VRAM telemetry and no other inference process. These conservative
limits reserve room for the observed ~10894 MiB candidate GPU allocation and
warm GGUF pages on a 32 GiB machine; they are not performance measurements.
Unknown telemetry refuses launch. Checks are repeated before each block. During
execution, available RAM below 2048 MiB, engine restart/identity change, GPU_FATAL
or nonfinite/OOM evidence stops the campaign as infrastructure error. No service
operated by the user is stopped. Only PIDs created by this runner and verified
by creation time, executable and command line may be terminated. Locks are
released by the OS even on a runner crash; leftover server processes cause the
next admission to refuse, never a blind kill by name.

JSONL records contain full requests/events/tool results, effective order, UTC
times, wall duration, token usage, first streamed output and first answer time,
server-provided prefill/decode timings, finish reason, validators/errors and
memory snapshots. Native metrics absent from the pinned build remain null;
they are not estimated from wall time or borrowed from newer builds. Paging
counters/shared VRAM and swap are recorded; no claim of continuous peak sampling.
Task snapshots are taken before/after; OOM/GPU_FATAL startup/task log checks stop
the run. CPU workers have independent OS limits.

Results have PASS/FAIL (functional, terminal) or INFRA_ERROR (not a score).
Missing/interrupted results are NOT_RUN. Each task has a durable started event;
an interrupted attempt with no final record requires explicit --retry-infra.
Integrity mismatches always stop. Resume skips both PASS and FAIL with verified
result/event hashes. Attempt directories are unique and never overwritten.
Atomic checkpoint replacement follows fsync of result JSONL. Infrastructure
retries are explicit, recorded, and excluded from main score denominators.

The report gives per-category paired counts, shared failures, discordances, code delta, tool JSON
validity and performance distributions descriptively. The pilot has no frozen
noninferiority promotion margin and cannot establish superiority. The original
20-code margin is reported as legacy_original_t8 only for its complete suite and
original API/tool requirements. Current original_t8 is the paired assessment,
which can require review despite equal totals. API and tool JSON integration
checks retain a separate status; no missing check counts as PASS.

No task/criterion edits after either arm's output. Validator defects require an
append-only erratum with reproducer independent of arm outcomes, versioned
replacement, symmetric revalidation of all stored outputs, and preservation of
original scores; new generation requires a new campaign directory. Never patch
the frozen source to make one arm pass. Historical Phase 4 PENDING and Phase 4b
FAIL remain unchanged and retain their provenance; the fixed-window MTP study
is a T7 instrument and provides no T8 functional verdict.
