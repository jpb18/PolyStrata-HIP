"""Versioned inference facts shared by history, API timings and the completion log.

Engine durations are host wall clocks, not GPU event times. Missing facts stay None.
The native extension is additive to DONE's positional fields; see docs/INFERENCE_METRICS.md.
"""
import math

TOKEN_FIELDS = (
    "prefill_tokens", "prefill_committed_tokens", "reread_tokens", "new_prompt_tokens",
    "common_prefix_tokens", "best_common_prefix_tokens", "resume_tokens", "decode_tokens",
    "context_start_tokens", "context_end_tokens", "logical_context_end_tokens", "prefill_complete",
    "context_before_request_tokens", "prefill_read_calls", "prefill_eval_calls",
)
TIME_FIELDS = ("cache_resume_ms", "prefill_ms", "prefill_compute_ms", "ttft_ms", "decode_ms")


def parse_done_fields(fields):
    values = dict(field.split("=", 1) for field in fields if "=" in field)
    if values.get("metrics_version") != "1":
        return None
    result = {"version": 1, "source": "strata-poly host steady_clock"}
    for key in TOKEN_FIELDS + TIME_FIELDS:
        try:
            value = int(values[key]) if key in TOKEN_FIELDS else float(values[key])
            result[key] = value if math.isfinite(value) and value >= 0 else None
        except (KeyError, ValueError, OverflowError):
            result[key] = None
    return result


def rate(tokens, ms):
    if tokens is None or ms is None or tokens <= 0 or ms <= 0 or not math.isfinite(ms):
        return None
    return round(1000.0 * tokens / ms, 3)


def request_inference(prompt_total, output_tokens, segments, attempted, server):
    """Aggregate only acknowledged native work. Counts/rates have the same segment scope as times."""
    native = []
    for segment in segments:
        facts = dict(segment.get("inference") or {})
        facts.update(prompt_total_tokens=segment.get("prompt_tokens"), cached_tokens=segment.get("reused"),
                     generated_tokens=segment.get("generated"), prompt_total_ms=segment.get("prompt_ms"),
                     decode_total_ms=segment.get("decode_ms"), finish=segment.get("finish"))
        native.append(facts)

    def total(key):
        values = [segment.get(key) for segment in native]
        return sum(values) if values and all(value is not None for value in values) else None

    first, last = (native[0], native[-1]) if native else ({}, {})
    result = {
        "version": 1, "prompt_total_tokens": prompt_total, "cached_tokens": first.get("cached_tokens"),
        "output_tokens": output_tokens, "engine_attempted_segments": attempted,
        "engine_reported_segments": len(native), "engine_complete": len(native) == attempted and attempted > 0,
        "engine_work_scope": "reported_segments", "engine_segments": native,
        "engine_metrics_status": "no_done" if not native else
        "reported" if all(segment.get("version") == 1 for segment in native) else
        "partial_extension" if any(segment.get("version") == 1 for segment in native) else "extension_not_reported",
        "engine_prefill_complete": all(segment.get("prefill_complete") == 1 for segment in native)
        if native and all(segment.get("prefill_complete") is not None for segment in native) else None,
        **{"engine_" + key: total(key) for key in
           ("prefill_tokens", "prefill_committed_tokens", "reread_tokens", "new_prompt_tokens",
            "generated_tokens", "decode_tokens", "cache_resume_ms", "prefill_ms", "prefill_compute_ms",
            "decode_ms", "prompt_total_ms", "decode_total_ms", "prefill_read_calls", "prefill_eval_calls")},
        **{"engine_" + key: first.get(key) for key in
           ("common_prefix_tokens", "best_common_prefix_tokens", "resume_tokens", "context_start_tokens", "ttft_ms",
            "context_before_request_tokens")},
        **{"engine_" + key: last.get(key) for key in ("context_end_tokens", "logical_context_end_tokens")},
        **server,
    }
    result["engine_prefill_tok_s"] = rate(result["engine_prefill_tokens"], result["engine_prefill_ms"])
    result["engine_prefill_compute_tok_s"] = rate(result["engine_prefill_tokens"], result["engine_prefill_compute_ms"])
    result["engine_decode_tok_s"] = rate(result["engine_decode_tokens"], result["engine_decode_ms"])
    result["server_token_span_tok_s"] = rate(
        max(0, server.get("server_observed_tokens", 0) - 1), server.get("server_token_span_ms"))
    return result


def log_fields(facts):
    def show(key):
        value = facts.get(key)
        return "unavailable" if value is None else str(round(value, 3) if isinstance(value, float) else value)
    return " ".join(f"{key}={show(key)}" for key in (
        "engine_metrics_status", "prompt_total_tokens", "cached_tokens", "engine_prefill_tokens", "engine_reread_tokens",
        "engine_resume_tokens", "engine_common_prefix_tokens", "engine_generated_tokens", "engine_decode_tokens", "output_tokens",
        "engine_context_before_request_tokens",
        "engine_context_start_tokens", "engine_context_end_tokens", "engine_logical_context_end_tokens",
        "engine_prefill_ms", "engine_decode_ms",
        "engine_cache_resume_ms", "engine_prefill_read_calls", "engine_prefill_eval_calls", "engine_prefill_complete",
        "engine_prefill_tok_s", "engine_decode_tok_s", "engine_ttft_ms", "server_ttft_ms", "server_queue_ms",
        "server_load_ms", "server_duration_ms", "engine_reported_segments", "engine_attempted_segments"))


def observe_totals(totals, facts):
    """Each sum has a coverage count: unavailable measurements never contribute a made-up zero."""
    for key in ("engine_prefill_tokens", "engine_reread_tokens", "engine_new_prompt_tokens",
                "engine_generated_tokens", "engine_decode_tokens", "engine_cache_resume_ms", "engine_prefill_ms",
                "engine_prefill_compute_ms", "engine_decode_ms", "server_queue_ms", "server_load_ms", "server_control_wait_ms",
                "server_duration_ms", "server_inference_ms", "server_ttft_ms"):
        value = facts.get(key)
        if value is not None:
            counter = totals.setdefault(key, {"sum": 0, "requests": 0})
            counter["sum"] += value
            counter["requests"] += 1
