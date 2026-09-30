"""Focused pack compatibility tests; run .venv/bin/python -m unittest discover -s tools -p test_iq_pack.py."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

from _paths import add_gguf_py
add_gguf_py()
from gguf import GGUFWriter, GGMLQuantizationType as Q, quants
import iq_pack


def write_gguf(path, tensors, split=None, arch=True):
    """`split` = (no, count, tensors): llama.cpp's gguf-split keys (0-based split.no)."""
    writer = GGUFWriter(path, "qwen4exp")
    if not arch:                                   # a later shard of a split: no model metadata
        writer.kv_data[0].pop("general.architecture", None)
    if split is not None:
        writer.add_uint16("split.no", split[0])
        writer.add_uint16("split.count", split[1])
        writer.add_int32("split.tensors.count", split[2])
    for name, values, kind in tensors:
        writer.add_tensor(name, quants.quantize(values, kind), raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class CompatibilityTests(unittest.TestCase):
    def test_bf16_halfway_rounds_to_even(self):
        values = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000], dtype=np.uint32)
        got = np.frombuffer(iq_pack.bf16_bytes(values.view(np.uint8), "F32"), dtype=np.uint16)
        np.testing.assert_array_equal(got, [0x3F80, 0x3F82, 0xBF80, 0xBF82])

    def test_nonfinite_conversion_refused(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            iq_pack.bf16_bytes(np.array([np.nan], dtype=np.float32).view(np.uint8), "F32")

    def test_projection_conversion_and_native_bytes(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            values = np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 128)
            router = np.full((2, 128), 0.10001, dtype=np.float32)
            names = ["blk.0.hc_attn_down.weight", "output_hc_up.weight", "blk.1.ple_key.weight",
                     "blk.0.ssm_alpha.weight", "blk.3.indexer.q_proj.weight", "blk.1.ple_value.weight"]
            write_gguf(source, [(n, values, Q.Q8_0) for n in names] + [
                ("blk.0.ffn_gate_inp.weight", router, Q.F32),
                ("blk.0.attn_qkv.weight", values, Q.Q8_0),
                ("per_layer_token_embd.weight", values, Q.Q8_0),
            ])
            before = source.read_bytes()
            model = iq_pack.Model(source)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, model, True), 0)
            _, rows = iq_pack.read_index(root / "index.txt")
            dense = (root / "dense.bin").read_bytes()
            # A scalar oracle independent of the BF16 encoder: choose the nearest BF16 value,
            # using the even representation at exact ties.
            decoded = quants.dequantize(quants.quantize(values, Q.Q8_0), Q.Q8_0).ravel()
            def rounded(x):
                bits = int(np.float32(x).view(np.uint32))
                return (bits >> 16) + ((bits & 65535) > 32768 or
                                      ((bits & 65535) == 32768 and (bits >> 16) & 1))
            expected = np.array([rounded(x) for x in decoded], dtype=np.uint16)
            for name in names:
                row = rows[name]
                self.assertEqual(row[2], "4")
                offset, size = int(row[3]), int(row[4])
                np.testing.assert_array_equal(np.frombuffer(dense[offset:offset + size], dtype=np.uint16), expected)
            native = rows["blk.0.attn_qkv.weight"]
            self.assertEqual(native[4], "0")
            self.assertEqual(native[9], "8")
            self.assertNotIn("per_layer_token_embd.weight", rows)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(json.loads((root / "compat-bf16.json").read_text())["tensors"]), 7)

    def test_default_still_refuses_inexact_f32_router(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            write_gguf(source, [("blk.0.ffn_gate_inp.weight", np.full((2, 32), 0.10001, np.float32), Q.F32)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 1)

    def test_existing_bf16_pack_remains_byte_identical(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            values = np.linspace(-2, 2, 256, dtype=np.float32).reshape(2, 128)
            write_gguf(source, [("blk.0.hc_attn_down.weight", values, Q.BF16)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 0)
            self.assertEqual((root / "dense.bin").read_bytes(), quants.quantize(values, Q.BF16).tobytes())

    def test_quantized_control_weights_need_explicit_conversion(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            write_gguf(source, [("blk.0.hc_attn_down.weight", np.ones((2, 32), np.float32), Q.Q8_0)])
            (root / "dense.bin").write_bytes(b"previous pack")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 1)
            self.assertEqual((root / "dense.bin").read_bytes(), b"previous pack")

    def test_q2_ple_and_large_tensors_stay_native(self):
        self.assertFalse(iq_pack.needs_bf16("blk.1.ple_key.weight", "Q2_0"))
        for name in ["blk.0.ffn_gate_exps.weight", "token_embd.weight", "output.weight",
                     "per_layer_token_embd.weight", "blk.0.attn_qkv.weight"]:
            self.assertFalse(iq_pack.needs_bf16(name, "IQ3_XXS"))

    def test_snapshot_symlinks_keep_split_discovery(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            snap = root / "snapshot"
            snap.mkdir()
            first = snap / "model-00001-of-00002.gguf"
            second = snap / "model-00002-of-00002.gguf"
            write_gguf(root / "blob1", [
                ("blk.0.hc_attn_down.weight", np.ones((2, 32), np.float32), Q.Q8_0),
                ("blk.0.ffn_gate_inp.weight", np.ones((512, 32), np.float32), Q.BF16),
            ], split=(0, 2, 5))
            write_gguf(root / "blob2", [(f"blk.0.ffn_{r}_exps.weight", np.ones((512, 2, 32), np.float32), Q.Q8_0)
                                       for r in ("gate", "up", "down")], split=(1, 2, 5), arch=False)
            try:
                first.symlink_to(root / "blob1")
            except OSError:                              # Windows without Developer Mode or admin rights
                self.skipTest("symlinks are not available here")
            with self.assertRaisesRegex(FileNotFoundError, "missing model shards"):
                iq_pack.Model(first)
            second.symlink_to(root / "blob2")
            out = root / "pack"
            (out / "tokenizer").mkdir(parents=True)
            for name in ["vocab.json", "chat_template.jinja"]:
                (out / "tokenizer" / name).touch()
            with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(first), "--out", str(out), "--compat-bf16"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(iq_pack.main(), 0)
            expert_index = (out / "native_experts.txt").read_text()
            self.assertIn(second.name, expert_index)
            self.assertIn("n_expert 512,", expert_index)
            (root / "blob2").write_bytes((root / "blob2").read_bytes()[:-64])
            with self.assertRaisesRegex(ValueError, "truncated tensor"):
                iq_pack.Model(first)



# ---- split artifacts: Unsloth's UD-Q4_K_XL in miniature.  Shard 1 holds the metadata and no tensor; shard 2 the
# routers, layer 0 and layer 1's down; shard 3 layer 1's gate and up (a shard boundary inside a layer, like its
# layer 11); shard 4 one float tensor.  4 experts of 32 x 32, Q8_0 and F32.
N_EXPERT = 4


def expert(seed, shape):
    return np.random.default_rng(seed).standard_normal((N_EXPERT, *shape)).astype(np.float32)


def split_model(root, name="model", split_layer=True, dup=False, split_no=None, gate1=Q.Q8_0, seed=0):
    """Writes the 4 shards; returns their paths."""
    gu, dn = (32, 32), (32, 32)
    router = [(f"blk.{l}.ffn_gate_inp.weight", np.full((N_EXPERT, 32), 0.5, np.float32), Q.F32) for l in (0, 1)]
    l0 = [(f"blk.0.ffn_{r}_exps.weight", expert(seed + 10 + i, gu if r != "down" else dn), Q.Q8_0)
          for i, r in enumerate(("gate", "up", "down"))]
    l1_gu = [("blk.1.ffn_gate_exps.weight", expert(seed + 20, gu), gate1),
             ("blk.1.ffn_up_exps.weight", expert(seed + 21, gu), Q.Q8_0)]
    l1_d = [("blk.1.ffn_down_exps.weight", expert(seed + 22, dn), Q.F32)]
    shards = [[], router + l0 + (l1_d if split_layer else []), l1_gu + ([] if split_layer else l1_d),
              [("output_norm.weight", np.ones(32, np.float32), Q.F32)]]
    if dup:
        shards[3].append(l1_gu[1])
    total = sum(len(s) for s in shards)
    paths = []
    for i, tensors in enumerate(shards):
        p = root / f"{name}-{i + 1:05d}-of-00004.gguf"
        no = split_no.get(i, i) if split_no else i
        write_gguf(p, tensors, split=(no, 4, total), arch=i == 0)
        paths.append(p)
    return paths


def run_pack(first, out, *extra):
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    for n in ["vocab.json", "chat_template.jinja"]:
        (out / "tokenizer" / n).touch()
    buf = io.StringIO()
    with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(first), "--out", str(out), *extra]):
        with contextlib.redirect_stdout(buf):
            rc = iq_pack.main()
    return rc, buf.getvalue()


def expected_experts(model):
    """experts.bin from first principles: per layer, per expert, gate | up | down slices."""
    parts = []
    for l in (0, 1):
        ts = [model.bytes(f"blk.{l}.ffn_{r}_exps.weight").reshape(N_EXPERT, -1) for r in ("gate", "up", "down")]
        parts.append(np.concatenate(ts, axis=1).tobytes())
    return b"".join(parts)


class SplitArtifactTests(unittest.TestCase):
    def test_four_shards_and_a_layer_split_per_role(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root)
            rc, log = run_pack(paths[0], root / "pack", "--experts-bin")
            self.assertEqual(rc, 0, log)
            text = (root / "pack" / "native_experts.txt").read_text()
            head, line0, line1 = text.splitlines()
            self.assertTrue(head.startswith("# strata native experts v4:"), head)
            self.assertIn("(n_expert 4,", head)
            s2, s3 = paths[1].name, paths[2].name
            self.assertEqual(line0.split()[8:], [s2])
            self.assertEqual(line1.split()[8:], [f"{s3},{s3},{s2}"])
            # every role's offset is absolute in ITS shard: shard 3's data_start for gate/up, shard 2's for down
            m = iq_pack.Model(paths[0])
            for line, l in ((line0, 0), (line1, 1)):
                f = line.split()
                for r, role in enumerate(("gate", "up", "down")):
                    g, t, _, _ = m.where[f"blk.{l}.ffn_{role}_exps.weight"]
                    self.assertEqual(int(f[5 + r]), g.data_start + t.offset)
            self.assertNotEqual(m.where["blk.1.ffn_gate_exps.weight"][0].data_start,
                                m.where["blk.1.ffn_down_exps.weight"][0].data_start)
            self.assertEqual((root / "pack" / "experts.bin").read_bytes(), expected_experts(m))
            side = json.loads((root / "pack" / "experts.bin.src.json").read_text())
            self.assertEqual([s["name"] for s in side["shards"]], [p.name for p in paths])
            self.assertFalse(list((root / "pack").glob("*.tmp")))
            # the index: routers as BF16, nothing of the experts
            _, rows = iq_pack.read_index(root / "pack" / "index.txt")
            self.assertNotIn("blk.1.ffn_gate_exps.weight", rows)
            self.assertEqual(rows["blk.1.ffn_gate_inp.weight"][2], "4")

    def test_unsplit_layers_keep_the_v3_file(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root, split_layer=False)
            rc, log = run_pack(paths[0], root / "pack")
            self.assertEqual(rc, 0, log)
            head, line0, line1 = (root / "pack" / "native_experts.txt").read_text().splitlines()
            self.assertTrue(head.startswith("# strata native experts v3: layer gu_type d_type offset blob_bytes "
                                            "gate_off up_off down_off [shard] (n_expert 4, total "), head)
            self.assertEqual(line1.split()[8:], [paths[2].name])

    def test_missing_shard(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp))
            paths[3].unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing model shards.*00004-of-00004"):
                iq_pack.Model(paths[0])

    def test_duplicate_tensor_across_shards(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp), dup=True)
            with self.assertRaisesRegex(ValueError, "duplicate tensor blk.1.ffn_up_exps.weight"):
                iq_pack.Model(paths[0])

    def test_shard_of_another_split(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp), split_no={2: 1})
            with self.assertRaisesRegex(ValueError, "does not declare itself shard 3 of 4"):
                iq_pack.Model(paths[0])

    def test_truncated_shard(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp))
            paths[2].write_bytes(paths[2].read_bytes()[:-64])
            with self.assertRaisesRegex(ValueError, "truncated tensor"):
                iq_pack.Model(paths[0])

    def test_wrong_type_or_shape_leaves_the_pack_alone(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root, gate1=Q.F32)            # layer 1: gate F32, up Q8_0
            pack = root / "pack"
            pack.mkdir()
            (pack / "native_experts.txt").write_text("previous pack")
            rc, log = run_pack(paths[0], pack)
            self.assertEqual(rc, 1)
            self.assertIn("layer 1: gate and up differ in type", log)
            self.assertEqual((pack / "native_experts.txt").read_text(), "previous pack")
            self.assertFalse((pack / "dense.bin").exists())

    def test_interrupted_pack(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root)
            pack = root / "pack"
            real = np.concatenate
            calls = []

            def dies_on_layer_1(parts, axis):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt("stopped part-way")
                return real(parts, axis=axis)
            with patch.object(iq_pack.np, "concatenate", dies_on_layer_1):
                with self.assertRaises(KeyboardInterrupt):
                    run_pack(paths[0], pack, "--experts-bin")
            self.assertFalse((pack / "experts.bin").exists())
            self.assertFalse((pack / "experts.bin.src.json").exists())
            # the index stops part-way too: the old pack's files stay as they were
            before = {n: (pack / n).read_bytes() for n in ("index.txt", "dense.bin", "native_experts.txt")}
            with patch.object(iq_pack, "write_index", side_effect=KeyboardInterrupt("stopped")):
                with self.assertRaises(KeyboardInterrupt):
                    run_pack(paths[0], pack)
            self.assertEqual(before, {n: (pack / n).read_bytes() for n in before})
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertEqual(rc, 0, log)
            self.assertEqual((pack / "experts.bin").read_bytes(), expected_experts(iq_pack.Model(paths[0])))

    def test_stale_same_size_experts_bin(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            other = split_model(root / "a", name="other", seed=100)
            paths = split_model(root / "b")
            pack = root / "pack"
            rc, log = run_pack(other[0], pack, "--experts-bin")    # another model, the same geometry
            self.assertEqual(rc, 0, log)
            stale = (pack / "experts.bin").read_bytes()
            want = expected_experts(iq_pack.Model(paths[0]))
            self.assertEqual(len(stale), len(want))
            self.assertNotEqual(stale, want)
            # without --experts-bin the engine would read it: refused, nothing rewritten
            rc, log = run_pack(paths[0], pack)
            self.assertEqual(rc, 1)
            self.assertIn("another source", log)
            self.assertEqual((pack / "experts.bin").read_bytes(), stale)
            # with it: rewritten, not reused by its size
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertEqual(rc, 0, log)
            self.assertEqual((pack / "experts.bin").read_bytes(), want)
            # and now reused
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertIn("not rewritten", log)
            # an experts.bin without its sidecar (an older packer) is rewritten too
            (pack / "experts.bin.src.json").unlink()
            (pack / "experts.bin").write_bytes(stale)
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertEqual((pack / "experts.bin").read_bytes(), want)


if __name__ == "__main__":
    unittest.main()
