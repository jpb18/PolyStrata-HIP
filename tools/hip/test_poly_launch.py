"""CPU-only launcher regression tests. No inference subprocess is started."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import poly_launch as launch


class LauncherTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / "tmp/hip-launch-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        assert Path(self.temp.name).resolve().parent == scratch.resolve()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.exe = self.root / "candidate.exe"
        self.exe.write_bytes(b"CPU fixture, never executed")
        self.profile = self.root / "frozen.txt"
        self.profile.write_bytes(b"frozen experts")
        self.model = self.root / "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"
        self.model.touch()
        self.cfg = {"exe": str(self.exe), "args": ["--model", str(self.model), "--max-context", "65536",
                    "--threads", "6", "--expert-profile", "operational.txt", "--turn-token", "248045",
                    "--image-min-tokens", "1024"],
                    "backend": "hip", "port": 8090, "model_name": launch.BASE,
                    "tokenizer": "existing-tokenizer", "lib_dirs": ["rocm"], "switch_on_request": True,
                    "vision": {"exe": "untouched-vision"}, "sampling": {"temperature": 0.6}}
        self.source = self.root / f"strata-{launch.BASE}.json"
        self.source.write_text(json.dumps(self.cfg))
        self.patches = patch.multiple(launch, ROOT=self.root, Q8_EXE=self.exe, Q8_SHA=launch.sha(self.exe),
                                      PROFILE=self.profile, PROFILE_SHA=launch.sha(self.profile))
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def prepare(self):
        out = launch.prepare_q8()
        return out, json.loads(out.read_text())

    def main(self, *args):
        out = io.StringIO()
        with patch.object(sys, "argv", ["poly_launch", *args]), contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            result = launch.main()
        return result, out.getvalue()

    def test_fp16_default_preserves_config_and_all_engine_arguments(self):
        before = self.source.read_bytes()
        with patch.object(launch, "run_server", return_value=0) as run, patch.object(launch, "port_free"):
            code, text = self.main("--config", str(self.source))
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args[1], self.cfg)
        self.assertEqual(run.call_args.args[2], "fp16")
        self.assertIn("FP16 - Estavel / Padrao", text)
        self.assertIn("65536", text)
        self.assertEqual(self.source.read_bytes(), before)

    def test_q8_independent_profile_and_fixed_context(self):
        before = self.source.read_bytes(), self.profile.read_bytes()
        out, cfg = self.prepare()
        args = cfg["args"]
        for key, value in {"--kv": "q8_0", "--max-context": "32768", "--expert-cache-max": "5000",
                           "--expert-swaps": "0", "--conversation-cache-mib": "0"}.items():
            self.assertEqual(launch.arg(args, key), value)
        self.assertEqual(cfg["exe"], str(self.exe))
        self.assertEqual(cfg["port"], self.cfg["port"])
        self.assertEqual(cfg["sampling"], self.cfg["sampling"])
        self.assertNotIn("vision", cfg)
        self.assertNotIn("--image-min-tokens", cfg["args"])
        self.assertFalse(cfg["switch_on_request"])
        self.assertEqual(Path(launch.arg(args, "--expert-profile")).read_bytes(), before[1])
        self.assertNotEqual(out.parent, self.root)
        self.assertNotEqual(self.prepare()[0].parent, out.parent)
        self.assertEqual((self.source.read_bytes(), self.profile.read_bytes()), before)
        # Automatic MTP and the ordinary model settings are preserved.
        for flag in ("--window", "--threads", "--turn-token"):
            self.assertEqual(launch.arg(args, flag), launch.arg(self.cfg["args"], flag))

    def test_q8_removes_streaming_and_parking_preserves_mtp(self):
        self.cfg["args"] += ["--kv-k", "fp16", "--kv-v", "fp16", "--kv-resident", "49152",
                             "--conversation-cache-mib", "4096", "--window", "8"]
        self.cfg.update(slot_save_path="old-slots", before_load="external-command")
        self.source.write_text(json.dumps(self.cfg))
        _, cfg = self.prepare()
        self.assertNotIn("--kv-resident", cfg["args"])
        self.assertNotIn("--kv-k", cfg["args"])
        self.assertEqual(launch.arg(cfg["args"], "--window"), "8")
        self.assertEqual(launch.arg(cfg["args"], "--conversation-cache-mib"), "0")
        self.assertNotIn("slot_save_path", cfg)
        self.assertNotIn("before_load", cfg)

    def test_missing_config_executable_profile_and_wrong_hash(self):
        self.assertEqual(self.main("--config", str(self.root / "absent.json"))[0], 1)
        for target in (self.exe, self.profile):
            data = target.read_bytes()
            target.unlink()
            with self.assertRaisesRegex(ValueError, "Ficheiro em falta"):
                self.prepare()
            target.write_bytes(data + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 inesperado"):
                self.prepare()
            target.write_bytes(data)

    def test_unsupported_model_and_incomplete_vision_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "apenas"):
            launch.prepare_q8(base="other-model")
        self.cfg["model_name"] += "-vision"
        self.source.write_text(json.dumps(self.cfg))
        with self.assertRaisesRegex(ValueError, "Ficheiro em falta"):
            self.prepare()

    def test_q8_vision_preserves_encoder_and_poly_does_not_get_vision_flag(self):
        cfg = dict(self.cfg, model_name=launch.VISION_BASE)
        encoder = self.root / "strata-vision.exe"
        projector = self.root / "mmproj.gguf"
        encoder.touch()
        projector.touch()
        cfg["vision"] = dict(exe=str(encoder), mmproj=str(projector), model=str(self.model))
        self.source.write_text(json.dumps(cfg))
        out = launch.prepare_q8(config=self.source)
        actual = json.loads(out.read_text())
        self.assertEqual(actual["vision"], cfg["vision"])
        launch.q8_policy(actual)
        uncapped = json.loads(launch.prepare_q8(config=self.source, caps="uncapped").read_text())
        self.assertEqual(uncapped["vision"], cfg["vision"])
        self.assertNotIn("--expert-cache-max", uncapped["args"])
        self.assertEqual(launch.q8_policy(uncapped), (32768, 0, None))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from serve.server import engine_args
        actual["exe"] = "build-hip-poly-q8-operational-win/strata-poly-q8.exe"
        self.assertNotIn("--vision", engine_args(actual))

    def test_invalid_cli_input_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.main("--kv", "q4")
        self.assertEqual(error.exception.code, 2)

    def test_model_selection_filters_q8_and_keeps_all_fp16_models(self):
        listing = SimpleNamespace(stdout=f"other-hip|Other|other.gguf\n{launch.BASE}|Qwen|qwen.gguf\n")
        with patch("subprocess.run", return_value=listing), patch.object(launch, "choose_menu", return_value=launch.BASE) as menu:
            self.assertEqual(launch.choose_model("q8_0"), launch.BASE)
            self.assertEqual(menu.call_args.args[1], [(launch.BASE, "Qwen")])
            launch.choose_model("fp16")
            self.assertEqual(len(menu.call_args.args[1]), 2)

    def test_menu_empty_eof_invalid_and_q8(self):
        for data, expected in (("\n", "fp16"), ("", "fp16"), ("invalid\n1\n", "fp16"), ("2\n", "q8_0")):
            with patch.object(sys, "stdin", io.StringIO(data)), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launch.choose_kv(), expected)

    @unittest.skipUnless(os.name == "nt", "Windows console")
    def test_menu_timeout_uses_fp16(self):
        with patch.object(sys, "stdin", Mock(isatty=lambda: True)), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launch.choose_kv(timeout=0), "fp16")

    def test_check_reports_selection_but_never_starts(self):
        out, _ = self.prepare()
        with patch.object(launch, "run_server") as run, patch.object(launch, "port_free"):
            code, text = self.main("--config", str(out), "--kv", "q8_0", "--mode", "check")
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertIn("Q8_0 - Aprovado", text)
        self.assertIn("INFO/READY por confirmar", text)
        self.assertIn("32768", text)

    def test_occupied_non_http_port_refuses_to_start(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with patch.object(launch, "health_at", side_effect=OSError), \
                    self.assertRaisesRegex(ValueError, "Nada foi parado ou trocado"):
                launch.port_free(listener.getsockname()[1])
            self.assertGreater(listener.fileno(), -1)

    def test_active_polystrata_only_receives_health_get_and_survives(self):
        requests = []
        class Health(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(("GET", self.path))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"service":"strata","model":"active","loaded":true}')
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Health)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.cfg["port"] = server.server_port
            self.source.write_text(json.dumps(self.cfg))
            with patch.object(launch, "run_server") as run:
                code, text = self.main("--config", str(self.source))
            self.assertEqual(code, 1)
            self.assertIn("PolyStrata, modelo active", text)
            run.assert_not_called()
            self.assertEqual(requests, [("GET", "/health")])
            self.assertTrue(thread.is_alive())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_effective_info_ready_and_expert_set_required(self):
        out, cfg = self.prepare()
        Path(cfg["log"]).write_text(f"strata-poly: expert set {launch.EXPERT_SET} (5000 experts in VRAM)\n")
        engine = SimpleNamespace(info={"kv": "q8_0", "context": 32768, "expert_slots": 5000,
                                      "conversation_cache_mib": 0}, max_context=32768, spawn=(str(self.exe),))
        with contextlib.redirect_stdout(io.StringIO()):
            launch.verify_ready(engine, cfg)
        for field, value in (("kv", "fp16"), ("context", 65536), ("expert_slots", 4999), ("conversation_cache_mib", 1)):
            old = engine.info[field]
            engine.info[field] = value
            with self.assertRaises(ValueError):
                launch.verify_ready(engine, cfg)
            engine.info[field] = old
        engine.max_context = 65536
        with self.assertRaises(ValueError):
            launch.verify_ready(engine, cfg)
        engine.max_context = 32768
        Path(cfg["log"]).write_text("wrong expert set")
        with self.assertRaisesRegex(ValueError, "conjunto de experts"):
            launch.verify_ready(engine, cfg)

    def test_startup_mismatch_closes_only_new_engine_before_http(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from serve import server
        out, cfg = self.prepare()
        engine = SimpleNamespace(info={"kv": "fp16"}, max_context=32768, close=Mock())
        def fake_main():
            server.start_model(cfg, None)
            self.fail("A mismatched Q8 engine must not reach the HTTP server")
        with patch.object(server, "start_model", return_value=(engine, None, {}, None)) as start, \
                patch.object(server, "serve") as serve, patch.object(server, "main", side_effect=fake_main), \
                patch.object(launch, "check_resources"):
            with self.assertRaisesRegex(ValueError, "INFO/READY inesperado"):
                launch.run_server(out, cfg, "q8_0")
            engine.close.assert_called_once()
            start.assert_called_once()
            serve.assert_not_called()
            self.assertIs(server.start_model, start)

    def test_failed_experimental_start_is_logged_without_fallback(self):
        out, cfg = self.prepare()
        with patch.object(launch, "port_free"), \
                patch.object(launch, "run_server", side_effect=RuntimeError("GPU_FATAL fixture")) as run:
            code, text = self.main("--config", str(out), "--kv", "q8_0")
        self.assertEqual(code, 1)
        run.assert_called_once()
        self.assertIn("Sem fallback automatico", text)
        log = Path(cfg["log"])
        self.assertIn(launch.sha(self.exe), log.read_text())
        self.assertIn("GPU_FATAL fixture", log.with_name("launcher-q8_0.log").read_text())

    def test_every_context_has_matching_profile_and_ready_limits(self):
        self.assertEqual(launch.Q8_CONTEXTS, {"32k": 32768, "48k": 49152, "128k": 131072, "262k": 262144})
        before = self.source.read_bytes(), self.profile.read_bytes()
        for name, context in launch.Q8_CONTEXTS.items():
            with self.subTest(context=name):
                out = launch.prepare_q8(profile=name)
                cfg = json.loads(out.read_text())
                cap, expert_set = launch.Q8_EXPERTS[context]
                self.assertEqual(launch.q8_policy(cfg), (context, cap, expert_set))
                self.assertEqual(launch.arg(cfg["args"], "--max-context"), str(context))
                self.assertEqual(launch.arg(cfg["args"], "--expert-cache-max"), str(cap))
                Path(cfg["log"]).write_text(f"strata-poly: expert set {expert_set} ({cap} experts in VRAM)\n")
                engine = SimpleNamespace(info={"kv": "q8_0", "context": context, "expert_slots": cap,
                                               "conversation_cache_mib": 0}, max_context=context, spawn=(str(self.exe),))
                with contextlib.redirect_stdout(io.StringIO()):
                    launch.verify_ready(engine, cfg)
                engine.max_context = context // 2
                with self.assertRaisesRegex(ValueError, "INFO/READY inesperado"):
                    launch.verify_ready(engine, cfg)
        self.assertEqual((self.source.read_bytes(), self.profile.read_bytes()), before)

    def test_context_menu_all_options_default_and_invalid(self):
        for data, expected in (("\n", "32k"), ("1\n", "32k"), ("2\n", "48k"), ("3\n", "128k"),
                               ("4\n", "262k"), ("", "32k"), ("5\n3\n", "128k")):
            with patch.object(sys, "stdin", io.StringIO(data)), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launch.choose_context(), expected)

    def test_caps_menu_options_default_invalid_and_cli(self):
        self.assertEqual(launch.Q8_CAPS, ("capped", "uncapped"))
        for data, expected in (("\n", "capped"), ("1\n", "capped"), ("2\n", "uncapped"), ("", "capped"),
                               ("3\n2\n", "uncapped")):
            with patch.object(sys, "stdin", io.StringIO(data)), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launch.choose_caps(), expected)
        with patch.object(sys, "stdin", io.StringIO("2\n")), contextlib.redirect_stderr(io.StringIO()):
            code, text = self.main("--choose-caps")
        # The menu goes to stderr and the answer to stdout, both captured; the answer is the last thing written.
        self.assertEqual(code, 0)
        self.assertTrue(text.rstrip().endswith("uncapped"), text)

    def test_uncapped_q8_drops_only_the_measured_ceiling(self):
        before = self.source.read_bytes(), self.profile.read_bytes()
        _, capped = self.prepare()
        config = launch.prepare_q8(caps="uncapped")
        actual = json.loads(config.read_text())
        ceiling = capped["args"].index("--expert-cache-max")
        # Each run copies the frozen profile into its own folder, so the path is not what is compared here.
        def same(args):
            args = list(args)
            args[args.index("--expert-profile") + 1] = "<profile copy>"
            return args
        self.assertEqual(same(actual["args"]), same(capped["args"][:ceiling] + capped["args"][ceiling + 2:]))
        for key, value in {"--kv": "q8_0", "--max-context": "32768", "--expert-swaps": "0",
                           "--conversation-cache-mib": "0", "--conversation-cache-slots": "0"}.items():
            self.assertEqual(launch.arg(actual["args"], key), value)
        self.assertEqual(launch.q8_policy(actual), (32768, 0, None))
        self.assertIn("sem tecto", actual["display_name"])
        self.assertEqual((self.source.read_bytes(), self.profile.read_bytes()), before)
        # The measured ceiling put back by hand is the capped profile again, not a third one.
        actual["args"] += ["--expert-cache-max", "5000"]
        self.assertEqual(launch.q8_policy(actual), (32768, 5000, launch.EXPERT_SET))

    def test_uncapped_every_context_and_unknown_caps_refused(self):
        for name, context in launch.Q8_CONTEXTS.items():
            with self.subTest(context=name):
                cfg = json.loads(launch.prepare_q8(profile=name, caps="uncapped").read_text())
                self.assertNotIn("--expert-cache-max", cfg["args"])
                self.assertEqual(launch.arg(cfg["args"], "--max-context"), str(context))
                self.assertEqual(launch.q8_policy(cfg), (context, 0, None))
        with self.assertRaisesRegex(ValueError, "Tecto Q8 desconhecido"):
            launch.prepare_q8(caps="nenhum")

    def test_uncapped_ready_confirms_the_count_and_refuses_an_empty_set(self):
        cfg = json.loads(launch.prepare_q8(caps="uncapped").read_text())
        engine = SimpleNamespace(info={"kv": "q8_0", "context": 32768, "expert_slots": 5097,
                                       "conversation_cache_mib": 0}, max_context=32768, spawn=(str(self.exe),))
        Path(cfg["log"]).write_text("strata-poly: expert set 3e276751da60c027 (5097 experts in VRAM)\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            launch.verify_ready(engine, cfg)
        self.assertIn("sem tecto: 5097", output.getvalue())
        for log, slots in (("strata-poly: expert set 3e276751da60c027 (5096 experts in VRAM)\n", 5097),
                           ("", 5097), ("strata-poly: expert set 3e276751da60c027 (5097 experts in VRAM)\n", 0)):
            engine.info["expert_slots"] = slots
            Path(cfg["log"]).write_text(log)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "sem tecto"):
                launch.verify_ready(engine, cfg)

    def test_a_ceiling_that_is_neither_measured_nor_absent_is_refused(self):
        _, cfg = self.prepare()
        cfg["args"][cfg["args"].index("--expert-cache-max") + 1] = "4000"
        with self.assertRaisesRegex(ValueError, "Perfil Q8 invalido"):
            launch.q8_policy(cfg)

    def test_streaming_and_unsupported_context_refused(self):
        for profile in ("262k-stream", "64k", "999999"):
            with self.assertRaisesRegex(ValueError, "Contexto Q8 nao suportado"):
                launch.prepare_q8(profile=profile)
        _, cfg = self.prepare()
        cfg["args"] += ["--max-context", "262144"]
        with self.assertRaisesRegex(ValueError, "Perfil Q8 invalido"):
            launch.q8_policy(cfg)

    def test_preflight_refuses_unsafe_ram_vram_and_other_inference(self):
        import psutil
        _, cfg = self.prepare()
        good = dict(ram_available_mib=20000, vram_used_mib=1000,
                    process_scan_complete=True, other_inference=[])
        cases = [dict(good, ram_available_mib=16000), dict(good, vram_used_mib=4000),
                 dict(good, vram_used_mib=None), dict(good, process_scan_complete=False),
                 dict(good, other_inference=[{"pid": 987, "name": "strata-poly.exe"}])]
        for state in cases:
            with patch.object(launch, "snapshot", return_value=state), patch.object(psutil, "process_iter", return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "guardas existentes"):
                launch.check_resources(cfg)
        other = SimpleNamespace(pid=988, info={"pid": 988, "name": "python.exe", "cmdline": ["python", "serve/server.py"]})
        with patch.object(launch, "snapshot", return_value=good.copy()), patch.object(psutil, "process_iter", return_value=[other]), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "another inference"):
            launch.check_resources(cfg)

    def test_postload_large_context_requires_ram_and_vram_headroom(self):
        import psutil
        out = launch.prepare_q8(profile="262k")
        cfg = json.loads(out.read_text())
        for ram, vram in ((3000, 5000), (6000, 100), (6000, None)):
            with patch.object(psutil, "virtual_memory", return_value=SimpleNamespace(available=ram * 2**20)), \
                    self.assertRaisesRegex(ValueError, "recusado apos carga"):
                launch.check_loaded_resources(SimpleNamespace(info={"vram_free_mib": vram}), cfg)

    def test_postload_reports_actual_ram_deficit_without_relaxing_guard(self):
        import psutil
        out = launch.prepare_q8(profile="262k")
        cfg = json.loads(out.read_text())
        with patch.object(psutil, "virtual_memory", return_value=SimpleNamespace(available=2527 * 2**20)), \
                self.assertRaisesRegex(ValueError, "faltam 1569 MiB de RAM disponivel") as error:
            launch.check_loaded_resources(SimpleNamespace(info={"vram_free_mib": 6201}), cfg)
        self.assertIn("minimo 4096", str(error.exception))
        self.assertIn("HTTP ainda nao foi aberto", str(error.exception))
        self.assertNotIn("faltam -", str(error.exception))

    def test_postload_reports_expert_cache_floor_leaving_reserve_short(self):
        import psutil
        out = launch.prepare_q8(profile="262k")
        cfg = json.loads(out.read_text())
        with patch.object(psutil, "virtual_memory", return_value=SimpleNamespace(available=20000 * 2**20)), \
                self.assertRaisesRegex(ValueError, "cache minima deixou a reserva curta em 300 MiB") as error:
            launch.check_loaded_resources(SimpleNamespace(info={"vram_free_mib": 6201, "vram_reserve_short_mib": 300}),
                                          cfg)
        self.assertIn("--expert-cache-min", str(error.exception))
        launch.check_loaded_resources(SimpleNamespace(info={"vram_free_mib": 6201, "vram_reserve_short_mib": 0}), cfg)
        launch.check_loaded_resources(SimpleNamespace(info={"vram_free_mib": 6201}), cfg)

    @unittest.skipUnless(os.name == "nt", "Windows venv redirector")
    def test_only_verified_venv_parent_is_owned(self):
        import psutil
        command = [sys.executable, "tools/hip/poly_launch.py", "--kv", "q8_0"]
        parent = Mock(pid=987)
        parent.name.return_value = "python.exe"
        parent.cmdline.return_value = command
        parent.exe.return_value = sys.executable
        current = Mock()
        current.parent.return_value = parent
        current.cmdline.return_value = command
        current.exe.return_value = str(self.root / "base-python.exe")
        with patch.object(psutil, "Process", return_value=current):
            self.assertEqual(launch.launcher_pids(), {os.getpid(), 987})
            parent.cmdline.return_value = [sys.executable, "serve/server.py"]
            self.assertEqual(launch.launcher_pids(), {os.getpid()})
            parent.cmdline.return_value = command
            parent.exe.return_value = str(self.root / "other-python.exe")
            self.assertEqual(launch.launcher_pids(), {os.getpid()})
            parent.exe.side_effect = psutil.AccessDenied(987)
            self.assertEqual(launch.launcher_pids(), {os.getpid()})

    def test_own_redirector_excluded_but_identical_other_launcher_blocks(self):
        import psutil
        _, cfg = self.prepare()
        good = dict(ram_available_mib=20000, vram_used_mib=1000,
                    process_scan_complete=True, other_inference=[])
        command = [sys.executable, "tools/hip/poly_launch.py", "--kv", "q8_0"]
        wrapper = SimpleNamespace(pid=987, info={"pid": 987, "name": "python.exe", "cmdline": command})
        other = SimpleNamespace(pid=988, info={"pid": 988, "name": "python.exe", "exe": "other-python.exe",
                                              "cmdline": [*command, "--api-key", "must-not-be-printed"]})
        with patch.object(launch, "launcher_pids", return_value={os.getpid(), 987}), \
                patch.object(launch, "snapshot", side_effect=lambda **kw: dict(good, other_inference=[])) as snap, \
                patch.object(psutil, "process_iter", return_value=[wrapper]), contextlib.redirect_stdout(io.StringIO()):
            launch.check_resources(cfg)
            snap.assert_called_once_with(owned={os.getpid(), 987})
        output = io.StringIO()
        with patch.object(launch, "launcher_pids", return_value={os.getpid(), 987}), \
                patch.object(launch, "snapshot", side_effect=lambda **kw: dict(good, other_inference=[])), \
                patch.object(psutil, "process_iter", return_value=[wrapper, other]), \
                contextlib.redirect_stdout(output), self.assertRaisesRegex(ValueError, "another inference"):
            launch.check_resources(cfg)
        self.assertIn("PID=988", output.getvalue())
        self.assertIn("other-python.exe", output.getvalue())
        self.assertNotIn("must-not-be-printed", output.getvalue())

    def test_preflight_never_starts_engine(self):
        out, _ = self.prepare()
        with patch.object(launch, "port_free"), patch.object(launch, "check_resources") as guard, \
                patch.object(launch, "run_server") as run:
            code, output = self.main("--config", str(out), "--kv", "q8_0", "--mode", "preflight")
        self.assertEqual(code, 0)
        guard.assert_called_once()
        run.assert_not_called()
        self.assertIn("PREFLIGHT aprovado", output)

    def test_health_checks_every_selected_context(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from serve import server
        for name, context in launch.Q8_CONTEXTS.items():
            out = launch.prepare_q8(profile=name)
            cfg = json.loads(out.read_text())
            svc = SimpleNamespace(engine=SimpleNamespace(close=Mock()))
            httpd = SimpleNamespace(shutdown=Mock())
            health = dict(service="strata", status="ok", loaded=True, model=cfg["model_name"], max_context=context)
            def fake_main():
                server.serve(svc, port=8090)
                return 0
            for reported in (context, context // 2):
                with patch.object(server, "serve", return_value=httpd), patch.object(server, "main", side_effect=fake_main), \
                        patch.object(launch, "check_resources"), patch.object(launch, "health_at", return_value=dict(health, max_context=reported)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if reported == context:
                        self.assertEqual(launch.run_server(out, cfg, "q8_0"), 0)
                    else:
                        with self.assertRaisesRegex(ValueError, "health inesperado"):
                            launch.run_server(out, cfg, "q8_0")
            httpd.shutdown.assert_called_once()
            svc.engine.close.assert_called_once()

    @unittest.skipUnless(os.name == "nt", "Windows batch")
    def test_real_batch_enter_invalid_input_and_explicit_q8(self):
        # Missing model/executable makes both branches fail before importing the inference server.
        self.cfg["exe"] = str(self.root / "missing.exe")
        self.model.unlink()
        self.source.write_text(json.dumps(self.cfg))
        repo = Path(__file__).resolve().parents[2]
        env = dict(os.environ, POLY_CONFIG=str(self.source), POLY_NO_OPEN="1")
        for data, expected in (("\n", "missing.exe"), ("invalid\n1\n", "Opcao invalida"),
                               ("2\n", "Modelo Q8 esperado")):
            result = subprocess.run(["cmd", "/d", "/c", "START-POLY-HIP.bat"], input=data, text=True,
                                    capture_output=True, cwd=repo, env=env, timeout=20)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn(expected, result.stdout + result.stderr)
        result = subprocess.run(["cmd", "/d", "/c", "START_HIP_POLY.bat q8_0 check"], input="\n", text=True,
                                capture_output=True, cwd=repo, env=env, timeout=20)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Modelo Q8 esperado", result.stdout + result.stderr)
        # "nocap" is an argument of its own (no ceiling), not a model name: the run reaches the Q8 config check.
        result = subprocess.run(["cmd", "/d", "/c", "START-POLY-HIP.bat", "q8_0", "nocap"], input="", text=True,
                                capture_output=True, cwd=repo, env=env, timeout=20)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Modelo Q8 esperado", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
