"""Profile preservation and Windows launcher checks; no engine or GPU is started."""
import contextlib
import io
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.hip import poly_profile, poly_config


class ProfileSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"fixture")
        self.cfg = {"args": ["--model", str(self.model), "--max-context", "49152",
                             "--draft-model", "draft.gguf", "--expert-cache-min", "2000"],
                    "tokenizer": "custom/tokenizer", "hip_ordinal": 1, "sampling": {"temperature": 0.6}}
        self.base = self.root / "strata-model-hip.json"
        self.base.write_text(json.dumps(self.cfg))

    def run_profile(self, *args):
        with mock.patch.object(poly_profile, "ROOT", self.root), \
                mock.patch.object(sys, "argv", ["poly_profile.py", *args]), \
                mock.patch.object(poly_profile.subprocess, "run") as child, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = poly_profile.main()
            child.assert_not_called()
            return code

    def test_matching_profile_preserves_every_setting(self):
        before = self.base.read_bytes()
        self.assertEqual(self.run_profile("48k", "--no-overwrite"), 0)
        self.assertEqual(self.base.read_bytes(), before)

    def test_conflicting_context_is_not_replaced(self):
        target = self.root / "strata-model-hip-128k.json"
        target.write_text(json.dumps(self.cfg))
        before = target.read_bytes()
        self.assertEqual(self.run_profile("128k", "--no-overwrite"), 2)
        self.assertEqual(target.read_bytes(), before)

    def test_read_only_does_not_create_missing_profile_or_tokenizer(self):
        before = set(self.root.iterdir())
        self.assertEqual(self.run_profile("128k", "--read-only"), 2)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_config_writer_refuses_existing_file_before_tokenizer_extraction(self):
        build = self.root / "build"
        build.mkdir()
        (build / ("strata-poly.exe" if os.name == "nt" else "strata-poly")).write_bytes(b"fixture")
        before = self.base.read_bytes()
        argv = ["poly_config.py", "--model", str(self.model), "--name", "model-hip", "--build", str(build)]
        with mock.patch.object(poly_config, "ROOT", self.root), mock.patch.object(sys, "argv", argv), \
                mock.patch.object(poly_config.subprocess, "run") as child:
            with self.assertRaisesRegex(SystemExit, "config already exists"):
                poly_config.main()
            child.assert_not_called()
        self.assertEqual(self.base.read_bytes(), before)

    def test_fresh_build_tokenizer_checkout_is_discovered(self):
        build = self.root / "build"
        gguf_py = build / "_deps/strata_llamacpp-src/gguf-py"
        (gguf_py / "gguf").mkdir(parents=True)
        (build / ("strata-poly.exe" if os.name == "nt" else "strata-poly")).write_bytes(b"fixture")
        data = self.root / "pack"

        def extract(command, **kwargs):
            self.assertEqual(Path(kwargs["env"]["STRATA_GGUF_PY"]), gguf_py)
            (data / "tokenizer").mkdir(parents=True)
            (data / "tokenizer/vocab.json").write_text('{}')
            return subprocess.CompletedProcess(command, 0)

        argv = ["poly_config.py", "--model", str(self.model), "--name", "fresh-hip", "--build", str(build),
                "--data", str(data), "--rocm-venv", str(self.root / "no-sdk")]
        with mock.patch.object(poly_config, "ROOT", self.root), mock.patch.object(sys, "argv", argv), \
                mock.patch.dict(os.environ, {"STRATA_GGML_DIR": ""}), \
                mock.patch.object(poly_config.subprocess, "run", side_effect=extract), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(poly_config.main(), 0)
        cfg = json.loads((self.root / "strata-fresh-hip.json").read_text())
        self.assertEqual(cfg["tokenizer"], str(data / "tokenizer"))

    @unittest.skipUnless(os.name == "nt", "Windows batch")
    def test_help_from_another_working_directory(self):
        result = subprocess.run(["cmd", "/d", "/c", str(ROOT / "START_HIP_POLY.bat"), "--help"],
                                cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("POLY_CONFIG", result.stdout)
        self.assertEqual(len(list(self.root.iterdir())), 2)

    @unittest.skipUnless(os.name == "nt" and (ROOT / ".venv/Scripts/python.exe").exists(),
                         "Windows launcher environment")
    def test_existing_config_check_preserves_paths_and_runtime_options(self):
        folder = self.root / "path with spaces"
        folder.mkdir()
        exe = folder / "not a real engine.exe"
        exe.write_bytes(b"must never execute in check mode")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cfg = dict(self.cfg, exe=str(exe), port=port)
        cfg['args'][cfg['args'].index('--draft-model') + 1] = 'draft with spaces.gguf'
        config = folder / "custom config.json"
        config.write_text(json.dumps(cfg))
        before = config.read_bytes()
        env = dict(os.environ, POLY_CONFIG=str(config), POLY_NO_OPEN="1", POLY_NO_PAUSE="1")
        result = subprocess.run(["cmd", "/d", "/c", str(ROOT / "START_HIP_POLY.bat"), "fp16", "check"],
                                cwd=self.root, env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CHECK: nao iniciado", result.stdout)
        self.assertIn('draft with spaces.gguf', result.stdout)
        self.assertIn('--expert-cache-min', result.stdout)
        self.assertEqual(config.read_bytes(), before)

    @unittest.skipUnless(os.name == "nt", "Windows batch")
    def test_missing_environment_only_reports_explicit_install_steps(self):
        launcher = self.root / "START_HIP_POLY.bat"
        launcher.write_bytes((ROOT / launcher.name).read_bytes())
        before = set(self.root.iterdir())
        result = subprocess.run(["cmd", "/d", "/c", str(launcher), "check"], cwd=self.root,
                                env=dict(os.environ, POLY_NO_PAUSE="1"), capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('Python environment missing', result.stdout)
        self.assertEqual(set(self.root.iterdir()), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
