"""GLM 5.3 Flash behind the server: its pre-tokenizer.

    python -m unittest serve.test_glm
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402


class PreTokenizer(unittest.TestCase):
    TOKENS = ["1", "2", "3", "4", "5", "12", "123", "45", "a", "b", "ab"]
    MERGES = ["1 2", "12 3", "4 5"]

    def pieces(self, pre, text):
        tok = ST.Tokenizer(self.TOKENS, self.MERGES, pre=pre)
        return [self.TOKENS[i] for i in tok.encode(text)]

    def test_digits_in_runs_of_three(self):
        self.assertEqual(self.pieces("glm4", "12345"), ["123", "45"])
        self.assertEqual(self.pieces("qwen35", "12345"), ["1", "2", "3", "4", "5"])

    def test_a_whole_piece_skips_the_merges(self):
        # "ab" is a token no merge builds
        self.assertEqual(self.pieces("glm4", "ab"), ["ab"])
        self.assertEqual(self.pieces("qwen35", "ab"), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
