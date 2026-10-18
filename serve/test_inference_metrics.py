"""Inference observability contracts, without model downloads or GPU work."""
import contextlib
import io
import json
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from serve.inference_metrics import parse_done_fields, rate, request_inference, observe_totals
from serve.prometheus import Latencies, render
from serve.server import (ByteTokenizer, ChatTemplate, EngineDied, Service, StrataEngine,
                          request_stats, request_timings, child_env)
from serve import test_request_accounting as accounting


def segment(n=100, resume=0, common=0, generated=5, prefill=40, decode=80):
    read = n - resume
    return dict(prompt_tokens=n, reused=resume, prompt_read=read, generated=generated,
                prompt_ms=prefill + 10, decode_ms=decode + 2, finish="length", inference=dict(
                    version=1, prefill_tokens=read, prefill_committed_tokens=read,
                    reread_tokens=max(0, common - resume), new_prompt_tokens=n - max(common, resume),
                    common_prefix_tokens=common, best_common_prefix_tokens=common, resume_tokens=resume,
                    cache_resume_ms=10, prefill_ms=prefill, prefill_compute_ms=prefill - 1,
                    ttft_ms=prefill + 12, decode_tokens=max(0, generated - 1), decode_ms=decode,
                    context_start_tokens=n, context_end_tokens=n + max(0, generated - 1),
                    logical_context_end_tokens=n + generated, prefill_complete=1))


class NativeContract(unittest.TestCase):
    def test_additive_done_and_old_done(self):
        engine = StrataEngine.__new__(StrataEngine)
        line = "DONE 5 100 50 82 length 0 0 20 0 0 0 0 0 80 0"
        engine._parse_done(line)
        self.assertNotIn("inference", engine.last)
        facts = segment(resume=20, common=30)["inference"]
        facts.update(context_before_request_tokens=107, prefill_read_calls=2, prefill_eval_calls=1)
        extension = " metrics_version=1 " + " ".join(f"{key}={value}" for key, value in facts.items())
        engine._parse_done(line + extension)
        self.assertEqual(engine.last["inference"]["resume_tokens"], 20)
        self.assertEqual(engine.last["inference"]["prefill_tokens"], 80)
        self.assertEqual(engine.last["decode_ms"], 82)
        self.assertEqual(engine.last["inference"]["decode_ms"], 80)
        self.assertEqual(engine.last["inference"]["context_before_request_tokens"], 107)
        self.assertEqual(engine.last["inference"]["prefill_read_calls"], 2)
        engine._parse_done(line)
        self.assertNotIn("inference", engine.last, "an old DONE must not inherit new facts")

    def test_unavailable_invalid_and_zero_rates(self):
        facts = parse_done_fields(["metrics_version=1", "prefill_tokens=-1", "decode_ms=nan", "ttft_ms=inf"])
        self.assertIsNone(facts["prefill_tokens"])
        self.assertIsNone(facts["decode_ms"])
        self.assertIsNone(facts["ttft_ms"])
        for tokens, ms in ((0, 0), (1, 0), (0, 1), (None, 1), (1, None), (1, -1)):
            self.assertIsNone(rate(tokens, ms))
        self.assertEqual(rate(20, 200), 100)
        json.dumps(facts, allow_nan=False)

    def test_context_work_at_short_and_long_lengths(self):
        for n in (1, 40, 8192, 100000):
            for resume, common in ((0, 0), (n - 1, n), (n // 2, n - 1), (0, n // 10)):
                with self.subTest(n=n, resume=resume, common=common):
                    stats = segment(n, resume, common)
                    facts = request_inference(n, 5, [stats], 1, {})
                    self.assertEqual(facts["cached_tokens"] + facts["engine_prefill_tokens"], n)
                    self.assertEqual(facts["engine_reread_tokens"] + facts["engine_new_prompt_tokens"], n - resume)
                    self.assertEqual(facts["engine_prefill_tok_s"], rate(n - resume, 40))
                    self.assertEqual(facts["engine_decode_tok_s"], rate(4, 80))
                    self.assertEqual(facts["engine_context_start_tokens"], n)

    def test_continuations_sum_matching_work_and_time(self):
        segments = [segment(100, 20, 30), segment(110, 104, 104, prefill=7)]
        facts = request_inference(100, 15, segments, 2, {})
        self.assertEqual(facts["cached_tokens"], 20)
        self.assertEqual(facts["engine_prefill_tokens"], 86)
        self.assertEqual(facts["engine_prefill_tok_s"], rate(86, 47))
        self.assertEqual(facts["engine_decode_tok_s"], rate(8, 160))
        self.assertEqual(facts["engine_generated_tokens"], 10)
        self.assertEqual(facts["output_tokens"], 15)
        stats = request_stats(segments)
        timings = request_timings(100, 15, stats)
        self.assertEqual(timings["prompt_n"], 80)
        self.assertEqual(timings["prompt_work_n"], 86)
        self.assertEqual(timings["prompt_per_second"], round(86000 / 67, 1))

    def test_partial_prefill_and_missing_done_stay_unknown(self):
        stats = segment()
        stats["inference"].update(prefill_tokens=None, reread_tokens=None, new_prompt_tokens=None,
                                  prefill_committed_tokens=20, prefill_complete=0, context_start_tokens=None,
                                  ttft_ms=None, decode_ms=None, context_end_tokens=0)
        stats.update(prompt_read=20, finish="cancel", generated=0)
        facts = request_inference(100, 0, [stats], 1, {})
        self.assertIsNone(facts["engine_prefill_tok_s"])
        self.assertEqual(facts["engine_prefill_committed_tokens"], 20)
        self.assertIsNone(request_timings(20, 0, request_stats([stats]))["prompt_per_second"])
        failed = request_inference(100, 5, [segment()], 2, {})
        self.assertFalse(failed["engine_complete"])
        self.assertEqual(failed["engine_reported_segments"], 1)
        self.assertEqual(failed["engine_work_scope"], "reported_segments")
        missing = request_inference(100, 0, [], 1, {})
        self.assertIsNone(missing["cached_tokens"])
        self.assertIsNone(missing["engine_generated_tokens"])

    def test_legacy_complete_prompt_and_zero_native_decode(self):
        old = dict(prompt_tokens=100, reused=20, generated=0, prompt_ms=40, decode_ms=10, finish="stop")
        timings = request_timings(100, 12, request_stats([old]))
        self.assertEqual(timings["prompt_per_second"], 2000)
        self.assertIsNone(timings["predicted_per_second"], "12 injected output tokens did not decode")
        old["finish"] = "cancel"
        self.assertIsNone(request_timings(100, 12, request_stats([old]))["prompt_per_second"])

    def test_status_explains_missing_extension_without_losing_done_counts(self):
        old = dict(prompt_tokens=100, reused=20, generated=8, prompt_ms=40, decode_ms=10, finish="length")
        facts = request_inference(100, 8, [old], 1, {})
        self.assertEqual(facts["engine_generated_tokens"], 8, "DONE count is real even without the extension")
        self.assertEqual(facts["engine_metrics_status"], "extension_not_reported")
        self.assertIsNone(facts["engine_prefill_ms"])
        self.assertEqual(request_inference(100, 0, [], 1, {})["engine_metrics_status"], "no_done")
        mixed = request_inference(100, 16, [old, segment()], 2, {})
        self.assertEqual(mixed["engine_metrics_status"], "partial_extension")
        self.assertIsNone(mixed["engine_prefill_tokens"])

    def test_existing_prefill_call_counts_sum_once_across_generations(self):
        segments = [segment(100, 20, 30), segment(110, 104, 104)]
        segments[0]["inference"].update(prefill_read_calls=2, prefill_eval_calls=1, context_before_request_tokens=80)
        segments[1]["inference"].update(prefill_read_calls=1, prefill_eval_calls=1, context_before_request_tokens=104)
        facts = request_inference(100, 10, segments, 2, {})
        self.assertEqual(facts["engine_prefill_read_calls"], 3)
        self.assertEqual(facts["engine_prefill_eval_calls"], 2)
        self.assertEqual(facts["engine_context_before_request_tokens"], 80)
        self.assertEqual(facts["engine_prefill_tokens"], 86)
        self.assertEqual(facts["engine_metrics_status"], "reported")
        text = render({"requests": [{"inference": facts}]}, Latencies().snapshot())
        self.assertIn('strata:last_engine_prefill_read_calls{model_name="strata"} 3', text)
        self.assertIn('strata:last_engine_metrics_reported{model_name="strata"} 1', text)

    def test_profile_switch_is_opt_in_and_preserves_other_engine_environment(self):
        with patch.dict("os.environ", {"STRATA_POLY_INFERENCE_METRICS": "0"}):
            self.assertEqual(child_env({})["STRATA_POLY_INFERENCE_METRICS"], "0")
            cfg = {"inference_metrics": True, "env": {"STRATA_POLY_GDN_STATE": "0"}}
            result = child_env(cfg)
            self.assertEqual(result["STRATA_POLY_INFERENCE_METRICS"], "1")
            self.assertEqual(result["STRATA_POLY_GDN_STATE"], "0")
            cfg["inference_metrics"] = False
            self.assertEqual(child_env(cfg)["STRATA_POLY_INFERENCE_METRICS"], "0")
            for bad in (1, "true", None):
                cfg["inference_metrics"] = bad
                with self.assertRaises(ValueError):
                    child_env(cfg)

    def test_prometheus_units_coverage_and_missing_samples(self):
        facts = request_inference(100, 5, [segment()], 1, {"server_ttft_ms": 150})
        totals = {}
        observe_totals(totals, facts)
        observe_totals(totals, request_inference(100, 0, [], 1, {}))
        self.assertEqual(totals["engine_prefill_tokens"], {"sum": 100, "requests": 1})
        text = render({"requests": [{"inference": facts}], "totals": {"inference": totals},
                       "live": {"server_token_span_ms": 200, "server_token_span_tok_s": 10}}, Latencies().snapshot())
        self.assertIn('strata:last_engine_prefill_seconds{model_name="strata"} 0.04', text)
        self.assertIn('strata:totals_engine_prefill_tokens_requests_total{model_name="strata"} 1', text)
        self.assertIn('strata:live_server_token_span_seconds{model_name="strata"} 0.2', text)
        missing = render({"requests": [{"inference": {"engine_ttft_ms": None}}]}, Latencies().snapshot())
        self.assertNotIn("strata:last_engine_ttft_seconds", missing)


class ClockedMetricsEngine(accounting.ClockedThinkingEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        before = len(self.done)
        try:
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
        finally:
            if len(self.done) > before and self.last is not None:
                old = self.last
                self.last = {**old, "inference": segment(len(ids), old["reused"], old["reused"],
                                                        old["generated"], old["prompt_ms"] - 1,
                                                        old["decode_ms"] - 2)["inference"]}


class HTTPMetrics(accounting.RequestAccounting):
    """Exercise all existing continuation/stream/error API tests with the additive native contract."""
    def setUp(self):
        super().setUp()
        self.engine = ClockedMetricsEngine(self.svc.tok)
        self.svc.engine = self.engine

    def assert_request_work(self, timings, completion_tokens, finish="stop"):
        super().assert_request_work(timings, completion_tokens, finish)
        facts = timings["inference"]
        row = self.svc.history[-1]
        self.assertEqual(row["inference"], facts)
        self.assertEqual(facts["engine_reported_segments"], 2)
        self.assertEqual(facts["engine_prefill_tokens"], sum(done["prompt_read"] for done in self.engine.done))
        self.assertEqual(facts["engine_decode_tokens"], sum(done["generated"] - 1 for done in self.engine.done))
        self.assertEqual(facts["engine_prefill_tok_s"], rate(facts["engine_prefill_tokens"], 45))
        self.assertEqual(facts["engine_decode_tok_s"], rate(facts["engine_decode_tokens"], 276))
        self.assertIsNotNone(facts["server_ttft_ms"])


class ServerClocks(unittest.TestCase):
    def service(self):
        tok = ByteTokenizer()
        engine = ClockedMetricsEngine(tok)
        return Service(engine, tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))

    def test_queue_and_load_are_separate_from_engine_prefill(self):
        svc = self.service()
        svc.fifo.acquire()
        timer = threading.Timer(0.05, svc.fifo.release)
        timer.start()
        svc.ensure_loaded = lambda: time.sleep(0.03)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            result = list(svc.run([1] * 40, False, [], 5, {}, threading.Event()))
        timer.join()
        facts = result[-1][1]["timings"]["inference"]
        self.assertGreater(facts["server_queue_ms"], 30)
        self.assertGreater(facts["server_load_ms"], 20)
        self.assertGreaterEqual(facts["server_ttft_ms"], facts["server_queue_ms"] + facts["server_load_ms"])
        self.assertEqual(facts["engine_prefill_ms"], 39)
        self.assertIn("engine_prefill_ms=39", log.getvalue())
        self.assertEqual(svc.latencies.snapshot()["itl"][2], facts["server_observed_tokens"] - 1)

    def test_error_never_inherits_previous_facts(self):
        svc = self.service()
        list(svc.run([1] * 40, False, [], 5, {}, threading.Event()))
        def fail(*args, **kwargs):
            raise EngineDied("no DONE")
            yield 1
        svc.engine.generate = fail
        with self.assertRaises(EngineDied):
            list(svc.run([1] * 40, False, [], 5, {}, threading.Event()))
        row = svc.history[-1]
        self.assertEqual(row["finish"], "error")
        self.assertIsNone(row["inference"]["engine_generated_tokens"])
        self.assertIsNone(row["inference"]["server_ttft_ms"])
        self.assertIsNone(svc.last_timings)

    def test_live_native_intervals_exclude_injected_output(self):
        svc = self.service()
        now = time.perf_counter()
        svc.status.update(busy=True, first_token=time.time(), generated=1000, started=time.time(),
                          _native_observation=(1, now, now))
        svc.engine.progress = (20, 40)
        live = svc.metrics()["live"]
        self.assertIsNone(live["server_token_span_tok_s"])
        self.assertEqual(live["prompt_position_tokens"], 20)
        self.assertTrue(live["prompt_position_is_estimate"])
        svc.status["_native_observation"] = (3, now, now + 0.25)
        self.assertEqual(svc.metrics()["live"]["server_token_span_tok_s"], 8)
        svc.engine.batch = 2
        self.assertIsNone(svc.metrics()["live"]["prompt_position_tokens"], "global PP cannot describe a batch slot")


if __name__ == "__main__":
    unittest.main()
