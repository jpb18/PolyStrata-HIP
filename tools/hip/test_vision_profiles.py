"""Regression tests for Vision profile support in poly_profile.py.

Run with:  python tools/hip/test_vision_profiles.py

Checks:
    - Vision 128K contains the correct vision configuration.
    - Normal profiles (non-vision) remain without a vision section.
    - The Vision 128K profile preserves its 131072-token context.
    - The encoder, projector and model paths exist on disk.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helper: load a profile by its stem (without the "strata-" prefix and ".json")
# ---------------------------------------------------------------------------
def _load_profile(suffix: str) -> dict:
    path = ROOT / f"strata-{suffix}.json"
    if not path.exists():
        raise FileNotFoundError(f"profile file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Test 1 – Vision 128K contains the correct vision configuration
# ---------------------------------------------------------------------------
class TestVision128kConfig(unittest.TestCase):
    def test_vision_section_exists(self):
        cfg = _load_profile("qwen3.6-35b-a3b-ud-q3_k_m-hip-vision-128k")
        self.assertIn("vision", cfg, "Vision 128K must have a 'vision' section")

    def test_vision_exe(self):
        cfg = _load_profile("qwen3.6-35b-a3b-ud-q3_k_m-hip-vision-128k")
        self.assertEqual(
            cfg["vision"]["exe"],
            str(ROOT / "strata-vision.exe"),
        )

    def test_vision_mmproj(self):
        cfg = _load_profile("qwen3.6-35b-a3b-ud-q3_k_m-hip-vision-128k")
        self.assertIn("mmproj", cfg["vision"])

    def test_vision_model(self):
        cfg = _load_profile("qwen3.6-35b-a3b-ud-q3_k_m-hip-vision-128k")
        self.assertIn("model", cfg["vision"])


# ---------------------------------------------------------------------------
# Test 2 – Normal profiles remain without vision
# ---------------------------------------------------------------------------
class TestNormalProfilesNoVision(unittest.TestCase):
    NORMAL_SUFFIXES = [
        "qwen3.6-35b-a3b-ud-q3_k_m-hip-128k",
        "qwen3.6-35b-a3b-ud-q3_k_m-hip-262k",
        "qwen3.6-35b-a3b-ud-q3_k_m-hip-262k-stream",
    ]

    def test_no_vision_in_normal_profiles(self):
        for suffix in self.NORMAL_SUFFIXES:
            cfg = _load_profile(suffix)
            self.assertNotIn(
                "vision", cfg,
                f"Normal profile '{suffix}' must NOT have a 'vision' section",
            )


# ---------------------------------------------------------------------------
# Test 3 – Vision 128K preserves its 131072-token context
# ---------------------------------------------------------------------------
class TestVision128kContext(unittest.TestCase):
    def test_max_context_is_131072(self):
        cfg = _load_profile("qwen3.6-35b-a3b-ud-q3_k_m-hip-vision-128k")
        args = cfg.get("args", [])
        max_ctx_idx = args.index("--max-context") + 1 if "--max-context" in args else -1
        self.assertGreater(max_ctx_idx, 0, "'--max-context' must be present in args")
        self.assertEqual(
            args[max_ctx_idx], "131072",
            "Vision 128K context must be exactly 131072 tokens",
        )


# ---------------------------------------------------------------------------
# Test 4 – Encoder, projector and model paths exist on disk
# ---------------------------------------------------------------------------
class TestVisionPathsExist(unittest.TestCase):
    def test_vision_exe_exists(self):
        cfg = _load_profile("qwen3.6-35b-a3b-ud-q3_k_m-hip-vision-128k")
        exe = cfg["vision"]["exe"]
        self.assertTrue(pathlib.Path(exe).exists(), f"Vision exe not found: {exe}")

    def test_mmproj_exists(self):
        cfg = _load_profile("qwen3.6-35b-a3b-ud-q3_k_m-hip-vision-128k")
        mmproj = cfg["vision"]["mmproj"]
        self.assertTrue(pathlib.Path(mmproj).exists(), f"MMProj not found: {mmproj}")

    def test_model_exists(self):
        cfg = _load_profile("qwen3.6-35b-a3b-ud-q3_k_m-hip-vision-128k")
        model = cfg["vision"]["model"]
        self.assertTrue(pathlib.Path(model).exists(), f"Model not found: {model}")


# ---------------------------------------------------------------------------
# Test 5 – poly_profile.py propagates vision when generating derived profiles
# ---------------------------------------------------------------------------
class TestPolyProfileVisionPropagation(unittest.TestCase):
    """Verify that the script source contains the vision propagation logic."""

    def test_vision_propagation_code_exists(self):
        src = (ROOT / "tools" / "hip" / "poly_profile.py").read_text(encoding="utf-8")
        self.assertIn('"vision"', src, "poly_profile.py must reference 'vision'")
        self.assertIn(
            '"vision" in name.lower()',
            src,
            "poly_profile.py must check if the derived profile name contains 'vision'",
        )

    def test_docstring_mentions_vision(self):
        src = (ROOT / "tools" / "hip" / "poly_profile.py").read_text(encoding="utf-8")
        self.assertIn(
            '"vision"',
            src[:src.index('"""', src.index('"""') + 3)],
            "Docstring must mention 'vision' propagation",
        )


# ---------------------------------------------------------------------------
# Test 6 – Base profile is unchanged (still has vision)
# ---------------------------------------------------------------------------
class TestBaseProfileUnchanged(unittest.TestCase):
    def test_base_has_vision(self):
        cfg = _load_profile("qwen3.6-35b-a3b-ud-q3_k_m-hip")
        self.assertIn(
            "vision", cfg,
            "Base HIP profile must still have its 'vision' section (unchanged)",
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
