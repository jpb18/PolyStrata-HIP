"""Tests for what setup keeps in a model's run config when it runs again (#629), the browser choice (#609) and the
image-token flag (#625).  Every outside effect is mocked (tools/test_setup_golden.py's install()).

    python -m unittest tools.test_setup_config
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


USER = {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "mcp_servers": {"files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}},
        "mcp": {"timeout_s": 60, "max_result_chars": 16000, "max_rounds": 8},
        "cors_origins": ["http://localhost:3000"], "reasoning_effort": "low"}


class CarryOver(unittest.TestCase):
    """#629: setup run again (a different --context) keeps the keys it does not write and keeps the old file as .bak."""

    def setUp(self):
        self.ram, self.found = PROFILES["64GB-1x32GB"]

    def run_setup(self, *extra, configs=()):
        return install(self.ram, self.found, ["--family", "qwen", "--model", "Q2_0", "--no-start", *extra],
                       configs=configs)

    def test_user_keys_survive_a_rerun(self):
        code, out, first, _ = self.run_setup("--context", "65536")
        self.assertEqual(code, 0, out)
        old = {**first, **USER, "args": first["args"] + ["--hand-option", "7"]}
        code, out, cfg, _ = self.run_setup("--context", "131072", configs=[("strata-q2_0.json", old)])
        self.assertEqual(code, 0, out)
        for k, v in USER.items():
            self.assertEqual(cfg[k], v, k)
        a = cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "131072")       # the new choice is used
        self.assertNotIn("--hand-option", a)                               # setup chooses the engine options...
        self.assertIn("kept from your earlier strata-q2_0.json: sampling, mcp_servers, mcp, cors_origins, "
                      "reasoning_effort", out)
        self.assertIn("kept as strata-q2_0.json.bak", out)
        self.assertIn("--hand-option", out)                                # ...and says which ones it did not keep

    def test_a_first_install_is_unchanged(self):
        code, out, cfg, _ = self.run_setup()
        self.assertEqual(code, 0, out)
        self.assertNotIn("kept from", out)
        self.assertNotIn(".bak", out)
        self.assertFalse(set(cfg) - setup.SETUP_KEYS)                     # setup writes only its own keys

    def test_setup_keys_are_rewritten(self):
        code, out, first, _ = self.run_setup()
        old = {**first, "model_name": "renamed", "port": 9999, "draft_vocab": "en", "host": "0.0.0.0"}
        code, out, cfg, _ = self.run_setup(configs=[("strata-q2_0.json", old)])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["model_name"], first["model_name"])
        self.assertEqual(cfg["port"], first["port"])
        self.assertNotIn("host", cfg)                                     # as before: --host is given to setup
        self.assertEqual(cfg["draft_vocab"], "en")                        # kept by its own rule (saved_draft_vocab)
        self.assertNotIn("kept from", out)

    def test_write_setup_config(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-x.json"
            mm = Path(d) / "mmproj-Q8_0.gguf"
            mm.write_bytes(b"")
            old = {"exe": "e", "args": ["--kv", "int8", "--threads", "8"], "sampling": {"temperature": 0.7},
                   "env": {"STRATA_RESIDENT_PIN": "0", "MY_VAR": "1"},
                   "vision": {"exe": "v", "mmproj": str(mm), "gpu": False, "max_tokens": 300, "min_tokens": 64}}
            p.write_text(json.dumps(old), encoding="utf-8")
            new = {"exe": "e2", "args": ["--kv", "int8"],
                   "vision": {"exe": "v2", "mmproj": str(Path(d) / "mmproj-BF16.gguf"), "gpu": False, "max_tokens": 300}}
            _, out = quiet(setup.write_setup_config, p, new)
            got = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(got["sampling"], {"temperature": 0.7})
            self.assertEqual(got["env"], {"MY_VAR": "1"})                 # setup's own env entry is setup's to write
            self.assertEqual(got["vision"]["mmproj"], str(mm))            # a Q8_0 mmproj of the user's own (#625)
            self.assertEqual(got["vision"]["min_tokens"], 64)
            self.assertEqual(got["exe"], "e2")
            self.assertEqual(json.loads((Path(d) / "strata-x.json.bak").read_text(encoding="utf-8")), old)
            self.assertIn("--threads", out)
            # the same again: nothing changes, so no new copy is made
            (Path(d) / "strata-x.json.bak").unlink()
            again = {"exe": "e2", "args": ["--kv", "int8"],
                     "vision": {"exe": "v2", "mmproj": str(Path(d) / "mmproj-BF16.gguf"), "gpu": False,
                                "max_tokens": 300}}
            quiet(setup.write_setup_config, p, again)
            self.assertFalse((Path(d) / "strata-x.json.bak").exists())
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), got)

    def test_an_unreadable_earlier_config_is_replaced_with_a_copy(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-x.json"
            p.write_text("{cut off", encoding="utf-8")
            quiet(setup.write_setup_config, p, {"exe": "e", "args": []})
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), {"exe": "e", "args": []})
            self.assertEqual((Path(d) / "strata-x.json.bak").read_text(encoding="utf-8"), "{cut off")

    def test_a_missing_mmproj_is_not_carried(self):
        old = {"vision": {"mmproj": "/nowhere/mmproj-Q8_0.gguf", "gpu": True}}
        new = {"vision": {"mmproj": "/data/mmproj-BF16.gguf", "gpu": True}}
        self.assertEqual(setup.carry_over(old, new), [])
        self.assertEqual(new["vision"]["mmproj"], "/data/mmproj-BF16.gguf")

    def test_a_copy_set_up_like_the_last_one_keeps_them_too(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "other" / "strata-x.json"
            src.parent.mkdir()
            src.write_text(json.dumps({"exe": "old", "mcp_servers": {"a": {"command": "x"}}}), encoding="utf-8")
            p = Path(d) / "strata-x.json"
            quiet(setup.write_setup_config, p, {"exe": "new"}, src)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")),
                             {"exe": "new", "mcp_servers": {"a": {"command": "x"}}})
            self.assertFalse((Path(d) / "strata-x.json.bak").exists())   # nothing here was replaced


if __name__ == "__main__":
    unittest.main()
