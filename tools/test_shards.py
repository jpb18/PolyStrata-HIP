"""A missing or short model shard is named with its numbers - over a minimal GGUF written here (no download,
no model, no GPU).

    python -m unittest tools.test_shards
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402


class Stop(Exception):
    """setup.fail(), caught instead of exiting: the message is what the test reads."""


def write_gguf(path: Path, names=("blk.0.attn_q.weight", "blk.0.attn_k.weight")) -> int:
    """A GGUF v3 of F32[8] tensors 32 bytes apart from a 32-byte-aligned data start; returns its whole length."""
    b = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(names), 0))
    for i, n in enumerate(names):
        b += struct.pack("<Q", len(n)) + n.encode() + struct.pack("<IQIQ", 1, 8, 0, 32 * i)
    b += bytes(-len(b) % 32 + 32 * len(names))
    path.write_bytes(b)
    return len(b)


class ShardCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.saved = S.fail
        S.fail = lambda msg, hint=None: (_ for _ in ()).throw(Stop(msg))

    def tearDown(self):
        S.fail = self.saved
        self.tmp.cleanup()

    def test_whole_shards_pass(self):
        shards = [self.dir / f"m-0000{i}-of-00002.gguf" for i in (1, 2)]
        for s in shards:
            write_gguf(s)
        S.check_shards(shards)

    def test_missing_shard_is_named(self):
        shards = [self.dir / "m-00001-of-00002.gguf", self.dir / "m-00002-of-00002.gguf"]
        write_gguf(shards[0])
        with self.assertRaises(Stop) as cm:
            S.check_shards(shards)
        self.assertIn("m-00002-of-00002.gguf", str(cm.exception))

    def test_short_shard_names_file_and_sizes(self):
        s = self.dir / "m-00001-of-00002.gguf"
        whole = write_gguf(s)
        s.write_bytes(s.read_bytes()[:-40])
        with self.assertRaises(Stop) as cm:
            S.check_shards([s])
        msg = str(cm.exception)
        self.assertIn("m-00001-of-00002.gguf is short", msg)
        self.assertIn(f"{whole - 40:,} of {whole:,} bytes", msg)
        self.assertIn("40 missing", msg)

    def test_truncated_header_is_refused(self):
        s = self.dir / "m-00001-of-00002.gguf"
        write_gguf(s)
        s.write_bytes(s.read_bytes()[:12])
        with self.assertRaises(Stop) as cm:
            S.check_shards([s])
        self.assertIn("m-00001-of-00002.gguf is not a whole GGUF shard", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
