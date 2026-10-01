"""Tests for setup.py's choices that depend on the PC (no GPU, no network, nothing installed): the image encoder of a
ready-made engine that has no code for the card (#331).

    python -m unittest tools.test_setup_choices
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class PrebuiltVision(unittest.TestCase):
    """#331: the 0.1.30/0.1.31 zip's encoder has no sm_75 code: an RTX 20 card gets the CPU encoder, not a compile."""
    META = {"version": "0.1.31", "archs": [75, 86, 89, 120], "ptx": True, "vision_archs": [86, 89, 120]}

    def test_a_card_the_encoder_has_no_code_for_gets_the_cpu_encoder(self):
        got, out = quiet(setup.prebuilt_vision, self.META, {"arch": 75}, "gpu")
        self.assertEqual(got, "cpu")
        self.assertIn("runs on the CPU instead", out)

    def test_covered_cards_keep_the_gpu_encoder(self):
        for arch in (86, 89, 120):
            self.assertEqual(quiet(setup.prebuilt_vision, self.META, {"arch": arch}, "gpu")[0], "gpu")
        # a newer card than the newest encoder code: only with PTX in the zip
        self.assertEqual(quiet(setup.prebuilt_vision, {**self.META, "vision_archs": [86, 89]}, {"arch": 120}, "gpu")[0],
                         "gpu")
        self.assertEqual(quiet(setup.prebuilt_vision, {**self.META, "vision_archs": [86, 89], "ptx": False},
                               {"arch": 120}, "gpu")[0], "cpu")
        # 0.1.32's zip: the encoder built with 75-real too
        self.assertEqual(quiet(setup.prebuilt_vision, {**self.META, "vision_archs": [75, 86, 89, 120]},
                               {"arch": 75}, "gpu")[0], "gpu")

    def test_other_choices_are_left_alone(self):
        for v in ("cpu", "none"):
            self.assertEqual(quiet(setup.prebuilt_vision, self.META, {"arch": 75}, v), (v, ""))
        # an older BUILD.json without vision_archs: the engine's archs
        self.assertEqual(quiet(setup.prebuilt_vision, {"archs": [75, 86]}, {"arch": 75}, "gpu")[0], "gpu")


if __name__ == "__main__":
    unittest.main()
