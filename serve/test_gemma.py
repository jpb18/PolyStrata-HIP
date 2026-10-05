"""Gemma 4 on the Python side: its tokenizer (a BPE over UTF-8 pieces with a visible space) and its reasoning,
which the model opens and closes with tokens of its own."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import strata_tokenizer as ST  # noqa: E402
from serve.frontend import OutputParser  # noqa: E402

SP = ST.SPACE
BYTES = ["<0x%02X>" % b for b in range(256)]
TOKENS = ["<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>", "\n", "\n\n", SP, "a", "b", "c", "ab", SP + "ab",
          SP + "c", "abc"] + BYTES
TYPES = [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [6] * 256
# a merge may pass through a piece the vocabulary does not keep ("bc"), and a piece may hold newlines
MERGES = ["a b", SP + " ab", SP + " c", "b c", "\n \n"]


class Tokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = ST.Tokenizer(TOKENS, MERGES, TYPES, pre="gemma4")
        self.id = {t: i for i, t in enumerate(TOKENS)}

    def test_spaces_are_pieces_and_merges_go_by_rank(self):
        ids = self.tok.encode("ab ab c")
        self.assertEqual([TOKENS[i] for i in ids], ["ab", SP + "ab", SP + "c"])
        self.assertEqual(self.tok.decode(ids), "ab ab c")

    def test_a_run_of_newlines_that_is_a_token_is_that_token(self):
        self.assertEqual([TOKENS[i] for i in self.tok.encode("a\n\nb\nc")], ["a", "\n\n", "b", "\n", "c"])

    def test_what_the_vocabulary_lacks_goes_byte_by_byte(self):
        ids = self.tok.encode("aé")
        self.assertEqual([TOKENS[i] for i in ids], ["a", "<0xC3>", "<0xA9>"])
        self.assertEqual(self.tok.decode(ids), "aé")
        # a merge to a piece it lacks: its bytes, as llama.cpp writes it, not the pieces it was merged from
        self.assertEqual([TOKENS[i] for i in self.tok.encode("bc")], ["<0x62>", "<0x63>"])
        self.assertEqual(self.tok.decode(self.tok.encode("bc")), "bc")

    def test_control_tokens_only_when_asked(self):
        self.assertEqual(self.tok.encode("<|turn>a", parse_special=True), [self.id["<|turn>"], self.id["a"]])
        self.assertNotIn(self.id["<|turn>"], self.tok.encode("<|turn>a"))

    def test_a_directory_keeps_merges_with_newlines(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "vocab.json").write_text(json.dumps(self.tok.ids, ensure_ascii=False), encoding="utf-8")
            (d / "merges.json").write_text(json.dumps(MERGES, ensure_ascii=False), encoding="utf-8")
            (d / "token_type.json").write_text(json.dumps(TYPES))
            (d / "tokenizer.json").write_text(json.dumps({"pre": "gemma4", "special_ids": {}}))
            again = ST.Tokenizer.from_dir(d)
        self.assertEqual(again.encode("ab ab\n\nc"), self.tok.encode("ab ab\n\nc"))
        self.assertEqual(again.ranks[("\n", "\n")], 4)


class Reasoning(unittest.TestCase):
    def events(self, pieces, thinking=True):
        p = OutputParser(thinking=thinking, call_format="gemma")
        out = [e for piece in pieces for e in p.feed(piece)] + p.finish()
        return ("".join(e.text for e in out if e.kind == "reasoning"),
                "".join(e.text for e in out if e.kind == "content"))

    def test_the_thought_channel_is_the_reasoning(self):
        text = "<|channel>thought\nThe user asks for a capital.<channel|>\nParis."
        for size in (1, 3, len(text)):
            pieces = [text[i:i + size] for i in range(0, len(text), size)]
            self.assertEqual(self.events(pieces), ("The user asks for a capital.", "Paris."))

    def test_a_reply_without_a_thought_is_content(self):
        self.assertEqual(self.events(["Par", "is."]), ("", "Paris."))
        self.assertEqual(self.events(["<|chan", "ged> x"]), ("", "<|changed> x"))
        self.assertEqual(self.events(["<|channel>thought\nx<channel|>y"], thinking=False),
                         ("", "<|channel>thought\nx<channel|>y"))


if __name__ == "__main__":
    unittest.main()
