"""serve/test_engine_llama.py - the llama.cpp engine boundary, against a scripted llama-server (no GPU).

    python -m unittest serve.test_engine_llama -v
    STRATA_LLAMA_URL=http://<your-box>:8081 python -m unittest serve.test_engine_llama -v     (+ the live test)

The stub speaks exactly what the real llama-server was observed to send on 2026-09-29 (llama.cpp b29c606):
`/props` with default_generation_settings.n_ctx, and `/completion` streaming `data: {...}` lines with
`tokens`, `stop`, and on the last line `stop_type` and `timings`.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.engine_llama import EngineDied, LlamaEngine, engine_from_config  # noqa: E402

N_CTX = 4096


class Stub(BaseHTTPRequestHandler):
    """A llama-server that streams a scripted list of ids, one per chunk, slowly enough to test cancel."""
    script: list[int] = []
    delay_s = 0.0
    seen: list[dict] = []
    error: str | None = None
    stop_type = "eos"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/props":
            body = {"default_generation_settings": {"n_ctx": N_CTX}, "build_info": "stub-1",
                    "model_path": "/models/stub.gguf", "modalities": {"vision": False}}
        elif self.path == "/health":
            body = {"status": "ok"}
        else:
            self.send_response(404); self.end_headers(); return
        b = json.dumps(body).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        req = json.loads(self.rfile.read(n))
        Stub.seen.append(req)
        self.send_response(200); self.send_header("content-type", "text/event-stream"); self.end_headers()
        if Stub.error:
            self.wfile.write(f"data: {json.dumps({'error': Stub.error})}\n\n".encode()); self.wfile.flush(); return
        ids = Stub.script[: int(req["n_predict"])]
        for i, t in enumerate(ids):
            last = i == len(ids) - 1
            ev = {"tokens": [t], "content": "x", "stop": last, "tokens_predicted": i + 1,
                  "tokens_evaluated": len(req["prompt"])}
            if last:
                ev["stop_type"] = Stub.stop_type if len(ids) == len(Stub.script) else "limit"
                ev["timings"] = {"prompt_n": len(req["prompt"]), "prompt_ms": 12.5, "predicted_n": len(ids),
                                 "predicted_ms": 40.0, "cache_n": 3}
            try:
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                Stub.seen[-1]["broken_at"] = i
                return
            if Stub.delay_s and not last:
                time.sleep(Stub.delay_s)


class Offline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown(); cls.httpd.server_close()

    def setUp(self):
        Stub.script, Stub.delay_s, Stub.seen, Stub.error, Stub.stop_type = [11, 22, 33, 44, 55], 0.0, [], None, "eos"

    def test_attach_reads_context_and_info(self):
        e = LlamaEngine.attach(self.url)
        self.assertEqual(e.max_context, N_CTX)
        self.assertEqual(e.info["version"], "stub-1")
        self.assertEqual(e.info["model"], "stub.gguf")
        self.assertTrue(e.alive())

    def test_generate_streams_ids_and_reports_timings(self):
        e = LlamaEngine.attach(self.url)
        out = [t for t in e.generate([1, 2, 3], 10, {"temperature": 0.7, "top_k": 20}, threading.Event())
               if t is not None]
        self.assertEqual(out, [11, 22, 33, 44, 55])
        self.assertEqual(e.last["generated"], 5)
        self.assertEqual(e.last["prompt_tokens"], 3)
        self.assertEqual(e.last["finish"], "stop")
        self.assertEqual(e.last["reused"], 3)
        self.assertAlmostEqual(e.last["prompt_ms"], 12.5)
        req = Stub.seen[-1]
        self.assertEqual(req["prompt"], [1, 2, 3])            # ids, not text
        self.assertTrue(req["return_tokens"] and req["stream"] and req["cache_prompt"])
        self.assertAlmostEqual(req["temperature"], 0.7)
        self.assertEqual(req["top_k"], 20)

    def test_max_new_caps_and_maps_to_length(self):
        e = LlamaEngine.attach(self.url)
        out = [t for t in e.generate([1], 2, {}, threading.Event()) if t is not None]
        self.assertEqual(out, [11, 22])
        self.assertEqual(e.last["finish"], "length")

    def test_greedy_when_no_temperature(self):
        e = LlamaEngine.attach(self.url)
        list(e.generate([1], 1, {}, threading.Event()))
        self.assertEqual(Stub.seen[-1]["temperature"], 0.0)

    def test_cancel_stops_early_and_closes_the_stream(self):
        Stub.delay_s = 0.05
        e = LlamaEngine.attach(self.url)
        cancel = threading.Event()
        got = []
        for t in e.generate([1], 5, {}, cancel):
            if t is not None:
                got.append(t)
                if len(got) == 2:
                    cancel.set()
        self.assertEqual(got, [11, 22])
        self.assertEqual(e.last, {"finish": "cancelled"})
        time.sleep(0.3)                                       # the stub notices the closed socket on its next write
        self.assertIn("broken_at", Stub.seen[-1])

    def test_consumer_stopping_early_closes_the_stream(self):
        Stub.delay_s = 0.05
        e = LlamaEngine.attach(self.url)
        gen = e.generate([1], 5, {}, threading.Event())
        self.assertEqual(next(t for t in gen if t is not None), 11)
        gen.close()
        time.sleep(0.3)
        self.assertIn("broken_at", Stub.seen[-1])

    def test_server_error_event_is_a_value_error(self):
        Stub.error = "context too long"
        e = LlamaEngine.attach(self.url)
        with self.assertRaises(ValueError):
            list(e.generate([1], 1, {}, threading.Event()))

    def test_images_refused_clearly(self):
        e = LlamaEngine.attach(self.url)
        with self.assertRaises(ValueError):
            list(e.generate([1], 1, {}, threading.Event(), embeddings="/tmp/x.sve"))

    def test_sampling_body_mapping(self):
        b = LlamaEngine.sampling_body({"temperature": 0, "top_p": 0.9, "min_p": 0.05, "repetition_penalty": 1.1,
                                       "presence_penalty": 1.5, "penalty_last_n": 64, "seed": 7, "top_k": 0})
        self.assertEqual(b, {"temperature": 0.0, "top_p": 0.9, "min_p": 0.05, "repeat_penalty": 1.1,
                             "presence_penalty": 1.5, "repeat_last_n": 64, "seed": 7, "top_k": 0})

    def test_engine_from_config_attach(self):
        e = engine_from_config({"llama": {"url": self.url, "model": "/m/x.gguf"}})
        self.assertEqual(e.model_path, "/m/x.gguf")
        with self.assertRaises(ValueError):
            engine_from_config({"llama": {}})

    def test_attach_to_nothing_dies_quickly(self):
        import serve.engine_llama as M
        old = M.READY_TIMEOUT_S
        M.READY_TIMEOUT_S = 2
        try:
            with self.assertRaises(EngineDied):
                LlamaEngine.attach("http://127.0.0.1:9")
        finally:
            M.READY_TIMEOUT_S = old


@unittest.skipUnless(os.environ.get("STRATA_LLAMA_URL"), "set STRATA_LLAMA_URL to test against a live llama-server")
class Live(unittest.TestCase):
    def test_round_trip(self):
        e = LlamaEngine.attach(os.environ["STRATA_LLAMA_URL"])
        self.assertGreater(e.max_context, 0)
        ids = json.loads(urllib.request.urlopen(urllib.request.Request(
            e.url + "/tokenize", data=json.dumps({"content": "def fib(n):", "add_special": True}).encode(),
            headers={"content-type": "application/json"}), timeout=30).read())["tokens"]
        out = [t for t in e.generate(ids, 8, {"temperature": 0}, threading.Event()) if t is not None]
        self.assertEqual(len(out), 8)
        self.assertEqual(e.last["generated"], 8)
        self.assertGreater(e.last["decode_ms"], 0)


if __name__ == "__main__":
    unittest.main()
