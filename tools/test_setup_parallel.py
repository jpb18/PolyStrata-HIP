"""#465: setup's recommendation for "parallel" (several requests decoded together in the engine's batch slots).
No GPU or download needed: python tools/test_setup_parallel.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup as S  # noqa: E402


class Parallel(unittest.TestCase):
    def test_slot_size(self):
        # measured: a slot's session at 32K, int8 KV: 0.56 GiB = 0.60 GB
        self.assertAlmostEqual(S.parallel_slot_gb(32768, "int8", False), 0.59, delta=0.03)
        # KV streaming keeps only the attention's 32K positions in VRAM, whatever the context
        self.assertEqual(S.parallel_slot_gb(262144, "int8", True), S.parallel_slot_gb(32768, "int8", False))
        self.assertLess(S.parallel_slot_gb(32768, "q4_0", False), S.parallel_slot_gb(32768, "int8", False))

    def test_recommendation_by_card(self):
        self.assertEqual(S.parallel_recommend(8, 32768, "int8", False), 0)      # too small: one at a time
        self.assertEqual(S.parallel_recommend(12, 32768, "int8", False), 2)     # the measured 5070
        self.assertEqual(S.parallel_recommend(24, 32768, "int8", False), 4)
        self.assertEqual(S.parallel_recommend(48, 32768, "int8", False), 4)     # capped at 4
        self.assertEqual(S.parallel_recommend(12, 262144, "int8", False), 0)    # a 262K KV cache per slot: no
        self.assertEqual(S.parallel_recommend(12, 262144, "int8", True), 2)     # ... unless it streams

    def test_notes_recommend_never_force(self):
        self.assertEqual(S.parallel_note(None, 8, 32768, "int8", False), [])
        self.assertIn("--parallel 2", S.parallel_note(None, 12, 32768, "int8", False)[0])
        lines = S.parallel_note(4, 12, 32768, "int8", False)
        self.assertIn("4 at once", lines[0])
        self.assertTrue(any("recommended for this card: 2" in x and "kept as you chose" in x for x in lines))
        self.assertTrue(any("at most 8" in x for x in S.parallel_note(12, 48, 32768, "int8", False)))
        self.assertEqual(len(S.parallel_note(2, 12, 32768, "int8", False)), 1)   # as recommended: no warning


if __name__ == "__main__":
    unittest.main()
