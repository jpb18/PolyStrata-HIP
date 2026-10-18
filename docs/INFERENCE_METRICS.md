# Inference metrics

The server keeps the last 500 native requests in memory. `GET /metrics` returns the newest 12;
`GET /metrics?requests=all` returns the retained history. `Accept: text/plain` or `?format=prometheus`
returns Prometheus text. Totals last for the lifetime of the server and are not limited to the retained history.

The additional request record is `requests[i].inference`. OpenAI chat responses and final streaming chunks that
already expose `timings` carry the same record in `timings.inference`. `/v1/status.last_timings` carries it too.
The completion log prints selected fields from that record once per request. Anthropic and Responses usage stays
in its existing format; detailed inference facts for those requests are available through `/metrics`.

## Enable native measurements

The server accepts older engines. Missing native facts are JSON `null`, `unavailable` in the completion log and
absent Prometheus samples. Rich native measurements are opt-in for `strata-poly` (HIP or CUDA):

```powershell
$env:STRATA_POLY_INFERENCE_METRICS = '1'
# Start a server using the candidate executable, in this process's environment.
```

A private run config can instead carry `"env": {"STRATA_POLY_INFERENCE_METRICS": "1"}`. This setting is passed to
the child engine through the existing environment mechanism. It does not need a global app setting.
The server also accepts `"inference_metrics": true` in that private config. It must be a JSON boolean and overrides
the inherited/env value for this engine; false explicitly disables the native extension. It does not replace an
executable or change a model profile by itself. Older executables cannot gain these measurements from a server flag.
An enabled new engine advertises `INFO inference_metrics_version=1 inference_metrics_enabled=1` before READY,
with its existing maximum read and batch sizes. The server announces this once at startup; it reports absence of
that announcement without assuming that an older version's DONE counters are absent.
Some earlier instrumented poly binaries emit the DONE extension without advertising it in INFO. The request's
`engine_metrics_status`, based on its own DONE, is the authoritative observation; a missing startup announcement
alone does not mean the extension is unsupported.
The default engine emits the original positional `DONE` line. Enabled, it appends `metrics_version=1` and named
fields. No GPU event, additional synchronization, per-token log, new checkpoint or inference policy is introduced.
The measurements add host clock reads around existing calls and at the first emitted token.

## Units, boundaries and origins

All token counts are integers. All `*_ms` fields are milliseconds, `*_tok_s` are tokens per second.
Engine times use C++ `steady_clock`; server times use Python `perf_counter`. Neither is a measurement of GPU
kernel time. The timestamps `time` and `totals.since` retain their existing Unix seconds format.

One API generation can run several native segments: a reasoning budget, forced tool opening or reasoning recovery
can cause another prefill and decode. Input/cache usage belongs to the first segment. Work and its elapsed times
sum over the acknowledged segments; each segment remains available in `engine_segments`. Injected wrap-up tokens
count towards API output but are not native generated tokens. MCP rounds remain separate history entries, as before.

| Field in `inference` | Meaning and origin |
| --- | --- |
| `version` | Record schema, currently 1. |
| `prompt_total_tokens` | Full original prompt rendered/tokenized by the server, even if prefill is cancelled. |
| `cached_tokens` | First segment's acknowledged reused prefix (`DONE.reused`). Unknown without its DONE. |
| `output_tokens` | Server/API output count, including existing forced/wrap-up tokens and observed stop tokens. |
| `engine_attempted_segments` | Calls made by `Service.run` to the engine generator. |
| `engine_reported_segments` | Segments with their own acknowledged DONE. |
| `engine_complete` | Every attempted segment has a DONE; cancellation can still be acknowledged. This is not a success flag. |
| `engine_prefill_complete` | All acknowledged segments completed their prompts; null for engines without the extension. Check `engine_complete` as well. |
| `engine_work_scope` | `reported_segments`: missing/error segments have no invented work or times. |
| `engine_metrics_status` | `reported`: every acknowledged segment carries the native extension; `extension_not_reported`: DONE exists but the extension is absent; `partial_extension`: mixed acknowledged versions; `no_done`: no native acknowledgement. This describes instrumentation, not request success. |
| `engine_prefill_tokens` | Sum of full prompt token positions actually evaluated after resume, including the last prompt token's logits evaluation. Native `Result.read`; null if a read stopped part way through layers. |
| `engine_prefill_committed_tokens` | Sum of completed read/eval blocks acknowledged by Session. They may subsequently have been rolled back after cancellation. This is not the final retained context. |
| `engine_reread_tokens` | Evaluated positions within the selected history's common prefix, after its resume point. From the existing `Conversations.last_resumed().same` and completed read count. |
| `engine_new_prompt_tokens` | Evaluated suffix positions after the selected common prefix: `prefill_tokens - reread_tokens`. “New” means outside that prefix; it does **not** prove the text has never occurred before. An old edit may force previously seen suffix text through prefill again. |
| `engine_common_prefix_tokens` | First segment's common prefix with the selected saved history, measured before resume mutates the cache. |
| `engine_best_common_prefix_tokens` | Largest first-segment common prefix among all histories examined, which can differ from the selected resumable history. |
| `engine_resume_tokens` | First segment's actual resume position, returned by the existing cache. |
| `engine_generated_tokens` | Sum of native `DONE.generated`, including native EOS and tokens drained after the client stops consuming. |
| `engine_decode_tokens` | Sum of `max(0, native_generated - 1)` per segment: generated tokens after that segment's first token. Excludes rejected speculative candidates and injected tokens. |
| `engine_context_start_tokens` | Actual `engine.pos()` after the first prompt is evaluated, immediately before generation; null if that prefill did not complete. |
| `engine_context_before_request_tokens` | Actual `engine.pos()` at first native generation entry, before cache resume/reset. This can describe the previous request's state; it is distinct from context after prefill. |
| `engine_context_end_tokens` | Actual `engine.pos()` at the last acknowledged segment's end, including rollback after cancellation. May be less than the logical context. |
| `engine_logical_context_end_tokens` | Last segment's full prompt plus its emitted native tokens, if its prompt completed. The terminal emitted token need not have been fed back into the engine. |
| `engine_cache_resume_ms` | Sum of time inside the existing resume operation: prefix search, reset, state restore, parking/branching as applicable. |
| `engine_prefill_ms` | Sum of native prompt phase wall time with the resume operation subtracted. Includes checkpoint management, prompt read/eval, prompt progress output and host overhead. Excludes queue/load and decode. |
| `engine_prefill_compute_ms` | Sum of time in existing prompt `Engine.read` and `Engine.eval` calls. Includes their internal staging, waits and checkpoints; not pure GPU time. |
| `engine_prefill_read_calls` / `engine_prefill_eval_calls` | Sum of attempted Session prompt read/eval calls, from differences of the existing PrefillTrace counters. Count each outer call once, including an interrupted call. A read can contain many internal batches/layers; these counts are not kernel launches or additional native generations. |
| `engine_prompt_total_ms` | Sum of original `DONE.prompt_ms`, including resume; retained for comparison with the legacy fields. |
| `engine_decode_ms` | Sum of elapsed time from just before each segment's first token is emitted until its generation finishes. Includes host sampling, verification, output pipe writes and cache maintenance. Excludes prompt and the initial sampling before first emission. |
| `engine_decode_total_ms` | Sum of original `DONE.decode_ms`, including initial sampling; retained for comparison. |
| `engine_ttft_ms` | First segment's generation entry to just before its first `on_token` call; includes its cache resume and prefill. Null if no token. |
| `server_queue_ms` | Wall time waiting for `Service.run`'s request FIFO. In parallel mode, slot waiting occurs later and is not measured here. |
| `server_control_wait_ms` | Time acquiring the model-loading coordination FIFO in parallel mode. |
| `server_load_ms` | Time in `ensure_loaded`, separated from control lock waiting. Includes any load/admission waiting that this existing operation performs. |
| `server_duration_ms` | `Service.run` entry at FIFO registration to the final accounting timestamp, including queue/load, host processing, consumer backpressure and STOP drain. Excludes HTTP body reading, initial template/tokenizer work, completion logging and final response serialization. |
| `server_inference_ms` | Post-load Service wall time to that accounting timestamp. Includes host work and any slot/control waiting in the engine generator; not an engine prefill/decode time. |
| `server_ttft_ms` | `Service.run` entry to receipt of the first native token, including queue/load. Can be EOS, reasoning or a partial text token; not necessarily visible text or first HTTP byte. |
| `server_inference_ttft_ms` | Post-load to receipt of the first native token. |
| `server_observed_tokens` | Native token IDs consumed by Service, excluding heartbeats and injected tokens. Can be smaller than native generated tokens after STOP drain. |
| `server_token_span_ms` | Receipt of first to receipt of last native token. Null for fewer than two tokens. Includes continuation gaps and host/consumer delays. |

For a completed single-segment prompt:

```text
cached_tokens + engine_prefill_tokens = prompt_total_tokens
engine_reread_tokens = max(0, min(resume + read, common_prefix) - resume)
engine_new_prompt_tokens = engine_prefill_tokens - engine_reread_tokens
engine_prefill_tok_s = 1000 * engine_prefill_tokens / engine_prefill_ms
engine_prefill_compute_tok_s = 1000 * engine_prefill_tokens / engine_prefill_compute_ms
engine_decode_tok_s = 1000 * engine_decode_tokens / engine_decode_ms
server_token_span_tok_s = 1000 * (server_observed_tokens - 1) / server_token_span_ms
```

Every rate is null if its count or time is unavailable or non-positive. One emitted token has zero decode intervals,
so it has no decode throughput. In several segments, rates use **summed matching work / summed matching time**,
not a mean of segment rates. Do not divide first-segment input counts by all segments' times.

## Existing fields and Prometheus

Existing names and API usage remain. History `prompt_tokens` is the old count of seen input positions (possibly
partial after cancellation, and falling back to full input when no DONE/read count exists); `prompt_total` is the complete input; `prompt_read` and `reused` belong to the first
segment. `prompt_ms`, `decode_ms` and `engine_generated` sum over native segments. The old `decode_tok_s` and
`timings.predicted_per_second` remain the generated count over the legacy decode phase; use the explicit
`inference.engine_decode_tok_s` for the after-first-token rate.

`timings.prompt_n` remains the original uncached input count. The added `prompt_work_n` sums reported native reads
over continuations; `prompt_per_second` and `prompt_per_token_ms` now use this count with the matching sum of legacy
prompt phase times. These are whole-prompt-phase rates (including cache resume), not pure compute rates. They are
unavailable when any segment's read count is unavailable. Extended engines that abort a layered read have no rate.
For older complete DONEs with known input and reuse, the legacy uncached-position count is retained as a fallback;
cancelled/unknown old reads have no rate. This fallback does not supply the new precise native work fields.

`live.prompt_read` retains the engine PP progress position, including the reused prefix. It may estimate layer progress.
`live.prompt_position_tokens` and `prompt_position_is_estimate` make this explicit. `live.prefill_tok_s_mean` remains
the legacy engine prompt phase estimate; `prefill_rate_scope` labels it. `live.tok_s`/`tok_s_mean` remain the existing
Monitor output-event rates and are labeled by `tok_s_scope`; they may include injected output. The added
`live.server_observed_tokens`, `server_token_span_ms` and `server_token_span_tok_s` use observed native token intervals.
The latter is a first-to-latest-token average, not GPU decode throughput. Parallel live fields describe the selected
latest Service request; old slot views retain their original scope.

Prometheus preserves existing metric names. `strata:last_<inference field>` exports available numeric token counts,
segment counts and rates. A `*_ms` field becomes `*_seconds` and is divided by 1000. No request-ID labels are added.
`totals.inference.<field>` holds `{sum, requests}` for available additive measurements; each Prometheus sum has a
matching `*_requests_total` coverage counter. These sums cover acknowledged work, including cancelled/error requests
when measurements arrived. They do not extrapolate to unavailable requests.

TTFT and E2E histograms now use the server clock at the Service boundary. ITL keeps its existing name but uses the
mean observed gap `(last_token - first_token)/(observed_tokens - 1)`, weighted by observed gaps. It is not a distribution
of individually timed gaps and cannot describe per-token tail latency. It no longer divides engine time by API output
that can contain injected tokens. The legacy `vllm:request_success_total` retains its historical “finished requests”
meaning, including errors/cancellations; use request `finish` for outcome analysis.

## Limits and validation

An interrupted layered prefill has no reliable full-token equivalent: completed blocks are reported separately,
actual total prefill/reread/suffix counts and their rates are null. No claim is made about FLOPs, partially completed
layers, rejected draft evaluations or recurrent-state replay inside speculative decode. Those need different counters.
Prefill tokens are model token positions evaluated by the engine, not a claim of exclusively GPU computation.
Dense/attention work uses the GPU, while expert work can also use the CPU depending on the existing engine path.
Host phase time includes that work and required transfers/waits. No GPU-event elapsed time or CPU/GPU FLOP split is
fabricated. Reread counts cover evaluated positions in the common prefix; a suffix after divergence may contain
previously seen text, but it is not identifiable as the same contextual evaluation without an explicit alignment.
The phase boundaries are host calls. Asynchronous operations can cross those boundaries; cache resume milliseconds
measure the host resume operation, not an exclusive GPU interval. Cancelled prefill phase time includes its rollback
at the existing boundary, while `engine_prefill_compute_ms` covers the read/eval calls themselves. No cancelled
partial-token throughput is reported.
An error without DONE provides only server observations, never the previous request's native numbers. A later failed
segment leaves preceding acknowledged work available with `engine_complete=false`. Final context then describes
the last acknowledged segment, not the unknown failed engine state.

Older `strata` and poly executables lack the extension. Their native prompt phase can include cache restoration and
their legacy read counts may exclude restoration replay. The new precise fields stay unavailable. GPU failure handling,
checkpoint selection, MTP and generation algorithms are unchanged. CUDA compilation/hardware validation is required
before claiming validation there; the shared poly source is not part of the SYCL engine.

Run `python -m unittest serve.test_inference_metrics serve.test_request_accounting serve.test_prometheus` for
protocol, API and unit checks. Hardware evidence, before/after JSON examples, limitations and preservation hashes
are in [the validation report](../bench/results/2026-10-13-inference-metrics/README.md).
Controlled native validation and an on/off overhead comparison are in
[the native metrics report](../bench/results/2026-10-13-native-metrics/README.md). Its isolated example profile uses
the existing GDN reference branch; it does not promote the operational executable or activate the other session's
experimental kernel path.
