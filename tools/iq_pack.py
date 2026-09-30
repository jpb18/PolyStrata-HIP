"""tools/iq_pack.py - plan v0.3 P6: a native pack for any of the model files (Q2_0, IQ2_XS, IQ3_XXS).

    python tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --out pack/iq3_xxs            (standalone)
    python tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --base pack/full --out ...    (share dense.bin)

The i-quant experts cannot be re-expressed in the Q2_0 pack form, so this pack keeps every quantized tensor in
its GGUF form:

  experts.bin          optional (--experts-bin): per layer, 512 blobs of [gate rows | up rows | down rows], the
                       raw GGUF slices.  Blob size is per layer (the files mix IQ1_M ... IQ3_S gate/up and Q2_0 /
                       IQ4_NL down).  experts.bin.src.json says which source it was cut from (the shards' names and
                       sizes, the hash of native_experts.txt); an experts.bin is reused only when that matches.
  native_experts.txt   one line per layer: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard];
                       written last, so a pack without it is not finished
  index.txt            the table the engine loads.  Quantized dense tensors, token_embd and output are served
                       natively from the GGUF by the engine (--native): their rows carry shape only.
  dense.bin            standalone: the BF16/F16/F32 tensors exactly as the GGUF stores them (index kinds 4/5/2).
                       With --base: the base (Q2_0) pack's dense.bin, hard-linked - the float tensors are
                       byte-identical in all three model files (checked) - plus extra.bin for tensors that are
                       float here but quantized in the base pack (blk.1.ple_key).
  tokenizer/           exported from the GGUF (tools/strata_tokenizer.py), with the model's chat template.

Split files: every shard of the model is read (<name>-0000N-of-0000M.gguf beside --gguf; a missing shard is an
error), so the layers may be split anyhow (Swift 1.5's GGUFs put layers 13-47 in shard 2 and the PLE table in
shard 1).  A layer whose experts are not in shard 1 names its shard in native_experts.txt (v3).  A shard boundary
may even fall inside a layer (Unsloth's UD-Q4_K_XL: layer 11's down in shard 2, its gate and up in shard 3): that
layer's shard column is per role, `gate,up,down` (an empty field = the --gguf shard), and only then is the file
v4, so an older engine refuses it instead of misreading it.  Every other pack stays v3, byte for byte.  Router
tensors stored as F32 whose values are exactly BF16 (Swift 1.5) are written as BF16, the form the engine's router
takes; anything else is refused.

For ordinary quants, --compat-bf16 dequantizes the small projections that the engine reads as BF16, using
round-to-nearest-even. This introduces BF16 rounding; it does not reconstruct the original full-precision
weights. Experts, native attention projections, token embeddings and the disk-backed PLE table stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

FLOAT = {"BF16", "F32", "F16"}
ROUTERS = ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight")
NOT_IN_PACK = {"per_layer_token_embd.weight"}      # the 28.8 GB PLE table: read from its GGUF by the engine

# These small projections are read as BF16 by the residual, router, GDN, QSA and PLE kernels.
# GSQ-RCO files already store them that way. Ordinary GGUF quants (including OrcaRouter's IQ3_XXS)
# quantize them too; --compat-bf16 explicitly dequantizes and rounds ONLY these tensors.
BF16_PROJECTIONS = (
    "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
    "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
    "ssm_alpha.weight", "ssm_beta.weight", "indexer.q_proj.weight", "indexer.k_proj.weight",
    "ple_value.weight", *ROUTERS,
)
BF16_OUTPUT = {"output_hc_down.weight", "output_hc_up.weight"}


def needs_bf16(name: str, type_name: str) -> bool:
    if name in BF16_OUTPUT:
        return True
    if not name.startswith("blk."):
        return False
    # The existing native PLE key supports Q2_0 only. Other key encodings use the BF16 path.
    return name.endswith(BF16_PROJECTIONS) or (name == "blk.1.ple_key.weight" and type_name != "Q2_0")


def bf16_bytes(raw: np.ndarray, type_name: str) -> bytes:
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q, quants
    values = quants.dequantize(raw, Q[type_name])
    if not np.isfinite(values).all():
        raise ValueError("cannot convert non-finite weights to BF16")
    # ggml's round-to-nearest-even conversion, including correct halfway rounding.
    return quants.quantize(values, Q.BF16).tobytes()


class Model:
    """All shards of one model: name -> (GGUFFile, TensorInfo, memmap, shard path)."""

    def __init__(self, first: pathlib.Path):
        import re
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
        paths = [first]
        if m:
            total = int(m.group(2))
            paths = [first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, total))
                     for i in range(1, total + 1)]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing model shards (wait for the download): " + ", ".join(missing))
        self.paths = paths
        self.files = [G.GGUFFile(p) for p in paths]
        self.sizes = [p.stat().st_size for p in paths]
        check_split(self.files)
        self.where = {}
        for p, g in zip(paths, self.files):
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            for t in g.tensors:
                size = t.expected_bytes()
                if size is None or g.data_start + t.offset + size > mm.size:
                    raise ValueError(f"{p.name}: unsupported or truncated tensor {t.name}")
                if t.name in self.where:
                    raise ValueError(f"{p.name}: duplicate tensor {t.name}")
                self.where[t.name] = (g, t, mm, p)

    def bytes(self, name) -> np.ndarray:
        g, t, mm, _ = self.where[name]
        return tensor_bytes(mm, g, t)


def check_split(files) -> None:
    """The split keys, as the engine checks them (strata::GgufModel): shard 1 carries the metadata, and every shard
    declares split.count / split.no (and split.tensors.count) consistently - a shard of another model, or a shard
    renamed into the family, is refused rather than mixed in."""
    n = len(files)
    meta0 = files[0].metadata
    if n == 1:
        if int(meta0.get("split.count", 1)) > 1:
            raise ValueError(f"{files[0].path.name} is shard 1 of {meta0['split.count']}, but its name has no "
                             "-00001-of-0000N.gguf to find the others by")
        return
    if "general.architecture" not in meta0:
        raise ValueError(f"{files[0].path.name} has no general.architecture; the first shard of a split model "
                         "carries the metadata")
    total = meta0.get("split.tensors.count")
    for i, g in enumerate(files):
        md = g.metadata
        if md.get("split.count") != n or md.get("split.no") != i or \
                (total is not None and md.get("split.tensors.count") != total):
            raise ValueError(f"{g.path.name} does not declare itself shard {i + 1} of {n} of this model "
                             "(split.count / split.no / split.tensors.count)")
    if total is not None and sum(len(g.tensors) for g in files) != total:
        raise ValueError(f"the {n} shards hold {sum(len(g.tensors) for g in files)} tensors, but "
                         f"split.tensors.count is {total}")


ROLES = ("gate", "up", "down")
N_EXPERT = 512
ALIGN = 64


def read_index(path: pathlib.Path):
    rows, header = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            header.append(line)
            continue
        f = line.split()
        rows[f[0]] = f
    return header, rows


def tensor_bytes(mm, g, t) -> np.ndarray:
    n = t.expected_bytes()
    return mm[g.data_start + t.offset: g.data_start + t.offset + n]


def is_expert(name: str) -> bool:
    return name.startswith("blk.") and name.endswith(("_exps.weight",))


def index_standalone(src, out, model: Model, compat_bf16: bool = False) -> int:
    """Every non-expert tensor of the model: floats into dense.bin as stored (exact-BF16 F32 routers as BF16),
    quantized ones native-only."""
    if not compat_bf16:
        for name, (_, t, _, _) in model.where.items():
            if needs_bf16(name, t.type_name) and t.type_name != "BF16" and not (
                    t.type_name == "F32" and name.endswith(ROUTERS)):
                print(f"{name} is {t.type_name}, but the engine requires BF16; use --compat-bf16")
                return 1
    rows, at = [], 0
    served = 0
    converted = []
    with open(out / "dense.bin.tmp", "wb") as fo:
        for name, (g, t, mm, _) in model.where.items():
            if is_expert(t.name) or t.name in NOT_IN_PACK:
                continue
            if len(t.shape) > 2:
                print("tensor %s has %d dimensions; the index holds two" % (t.name, len(t.shape)))
                return 1
            ne0 = int(t.shape[0])
            ne1 = int(t.shape[1]) if len(t.shape) > 1 else 0
            convert = compat_bf16 and needs_bf16(t.name, t.type_name) and t.type_name != "BF16"
            if t.type_name in FLOAT or convert:
                raw = tensor_bytes(mm, g, t).tobytes()
                if convert:
                    raw = bf16_bytes(np.frombuffer(raw, dtype=np.uint8), t.type_name)
                    kind = "4"
                    converted.append({"name": t.name, "source_type": t.type_name, "bytes": len(raw)})
                else:
                    kind = {"BF16": "4", "F16": "5", "F32": "2"}[t.type_name]
                if not convert and t.type_name == "F32" and t.name.endswith(ROUTERS):
                    u = np.frombuffer(raw, dtype=np.uint32)
                    if np.count_nonzero(u & 0xFFFF):
                        print("router %s is F32 with values that are not BF16; the engine's router is BF16" % t.name)
                        return 1
                    raw = (u >> 16).astype(np.uint16).tobytes()     # the exact BF16 values
                    kind = "4"
                rows.append([t.name, "0", kind, str(at), str(len(raw)), "0", str(len(raw)), str(ne0), str(ne1),
                             "0", "0", "1"] + ["0"] * 7)
                fo.write(raw)
                pad = (-len(raw)) % ALIGN
                fo.write(b"\0" * pad)
                at += len(raw) + pad
            else:
                served += 1
                rows.append([t.name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7)
    write_index(out, rows, src, served, 0, publish=False)
    # published together, and the completion marker (native_experts.txt, written last by main) goes first: a stop
    # from here on leaves a pack that setup and the engine see as unfinished, never a new dense.bin under an old
    # index or the reverse
    (out / "native_experts.txt").unlink(missing_ok=True)
    (out / "dense.bin.tmp").replace(out / "dense.bin")
    (out / "index.txt.tmp").replace(out / "index.txt")
    if compat_bf16:
        (out / "compat-bf16.json").write_text(json.dumps({
            "source": str(src), "rounding": "nearest-even", "tensors": converted,
        }, indent=2) + "\n", encoding="utf-8")
        print("compat-bf16: %d tensors, %.2f GiB; expert and PLE table bytes unchanged"
              % (len(converted), sum(t["bytes"] for t in converted) / 2**30))
    return 0


def write_index(out, rows, src, served, n_extra, publish=True):
    at = 0
    for r in rows:
        r[5] = str(at)
        at += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
    with open(out / "index.txt.tmp", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from %s\n" % src.name)
        fo.write("# align %d pool %d tensors %d\n" % (ALIGN, at, len(rows)))
        for r in rows:
            fo.write(" ".join(r) + "\n")
    if publish:
        (out / "index.txt.tmp").replace(out / "index.txt")
    print("index.txt: %d tensors, %d served natively, %d in extra.bin, arena %.2f GiB"
          % (len(rows), served, n_extra, at / 2**30))


def index_from_base(a, src, base, out, g, T, mm) -> int:
    base_src = pathlib.Path(json.loads((base / "manifest.json").read_text(encoding="utf-8"))["source"]["shard1"])
    if not base_src.exists():
        print("cannot find the base pack's shard 1 from its manifest.json")
        return 1
    bg = G.GGUFFile(base_src)
    BT = {t.name: t for t in bg.tensors}
    bmm = np.memmap(base_src, dtype=np.uint8, mode="r")
    header, rows = read_index(base / "index.txt")
    new_rows, extra = [], []
    served = 0
    for name, f in rows.items():
        t, bt = T.get(name), BT.get(name)
        if t is None or bt is None:
            print("tensor %s missing from one of the models" % name)
            return 1
        if t.type_name in FLOAT and bt.type_name in FLOAT:
            if t.type_name != bt.type_name or t.shape != bt.shape or \
                    not np.array_equal(tensor_bytes(mm, g, t), tensor_bytes(bmm, bg, bt)):
                print("float tensor %s differs from the base model; this pack cannot reuse its dense.bin" % name)
                return 1
            new_rows.append(list(f))
        elif t.type_name in FLOAT:
            if t.type_name != "BF16":
                print("unexpected float type %s for %s" % (t.type_name, name))
                return 1
            nbytes = t.expected_bytes()
            off = sum(len(b) + (-len(b)) % ALIGN for b in extra)
            extra.append(tensor_bytes(mm, g, t).tobytes())
            # file 3 = extra.bin, raw BF16 (index kind 4)
            new_rows.append([name, "3", "4", str(off), str(nbytes), "0", str(nbytes), f[7], f[8]] + ["0"] * 10)
        else:
            served += 1
            new_rows.append([name, f[1], "0", "0", "0", "0", "0", f[7], f[8], "8", "0", "32"] + ["0"] * 7)
    write_index(out, new_rows, src, served, len(extra))
    with open(out / "extra.bin", "wb") as fo:
        for b in extra:
            fo.write(b)
            fo.write(b"\0" * ((-len(b)) % ALIGN))
    dense = out / "dense.bin"
    if not dense.exists():
        try:
            os.link(base / "dense.bin", dense)
        except OSError:
            shutil.copyfile(base / "dense.bin", dense)
    if (base / "tokenizer").exists() and not (out / "tokenizer").exists():
        shutil.copytree(base / "tokenizer", out / "tokenizer")
    return 0


def expert_layout(model: Model, src: pathlib.Path):
    """The pack's expert table: (layout rows, native_experts.txt text, n_expert, total bytes), or an error string.
    Each role is resolved by name in whichever shard holds it, and its offset is absolute in THAT shard: two
    shards do not start their data section at the same byte, so one role's data_start must not be used for
    another's (per-role data_start as in #255, gopinath87607)."""
    T = {n: w[1] for n, w in model.where.items()}
    exps = [n for n in T if n.startswith("blk.") and n.endswith("_exps.weight")]
    if not exps:
        return "the model has no expert tensors (blk.N.ffn_{gate,up,down}_exps.weight)"
    n_layers = 1 + max(int(n.split(".")[1]) for n in exps)
    n_expert = int(T["blk.0.ffn_gate_inp.weight"].shape[1])   # router rows = experts kept (pruned models ship < 512)
    if any(int(T["blk.%d.ffn_gate_inp.weight" % l].shape[1]) != n_expert for l in range(n_layers)):
        return "the routers disagree on the expert count; a per-layer pruned model cannot be packed"
    layout, lines, offset, n_split = [], [], 0, 0
    for l in range(n_layers):
        names = ["blk.%d.ffn_%s_exps.weight" % (l, r) for r in ROLES]
        if any(n not in T for n in names):
            return "layer %d: missing %s" % (l, ", ".join(n for n in names if n not in T))
        ts = [T[n] for n in names]
        if any(t.expected_bytes() is None or len(t.shape) != 3 or int(t.shape[2]) != n_expert for t in ts):
            return "layer %d: an expert tensor is not [*, *, %d] of whole blocks" % (l, n_expert)
        per = [t.expected_bytes() // n_expert for t in ts]
        if per[0] != per[1] or ts[0].type_name != ts[1].type_name:
            return "layer %d: gate and up differ in type" % l
        blob = per[0] + per[1] + per[2]
        layout.append((l, ts[0].type_id, ts[2].type_id, offset, blob, ts))
        ws = [model.where[n] for n in names]
        files = ["" if w[3] == src else w[3].name for w in ws]
        column = files[0] if len(set(files)) == 1 else ",".join(files)
        n_split += len(set(files)) != 1
        line = "%d %d %d %d %d %d %d %d" % (l, ts[0].type_id, ts[2].type_id, offset, blob,
                                            *[w[0].data_start + w[1].offset for w in ws])
        lines.append(line + ("" if not column else " " + column))
        offset += blob * n_expert
    if n_split:
        head = ("# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                "[shard | gate,up,down] (n_expert %d, total %d; absolute offsets in %s, or in the named shard "
                "beside it - per role where the column is gate,up,down)\n" % (n_expert, offset, src.name))
        print("%d layer(s) have their gate/up/down in different shards: native_experts.txt v4, per-role shard "
              "column for %s" % (n_split, ", ".join(str(l) for l, *_ in layout if "," in lines[l])))
    else:
        head = ("# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] "
                "(n_expert %d, total %d; absolute offsets in %s, or in the named shard beside it)\n"
                % (n_expert, offset, src.name))
    return layout, head + "".join(line + "\n" for line in lines), n_expert, offset


def experts_source(model: Model, text: str, total: int) -> dict:
    """What experts.bin is cut from: the shards (names and sizes) and the hash of native_experts.txt (every
    per-role file and offset, the formats and the blob sizes).  A same-size experts.bin of another model or
    another packing is not this one."""
    return {"schema": 1,
            "shards": [{"name": p.name, "size": s} for p, s in zip(model.paths, model.sizes)],
            "native_experts_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bytes": total}


def read_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the model's shard 1")
    ap.add_argument("--base", help="optional: a Q2_0 canonical pack whose dense.bin holds the shared float tensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-experts", action="store_true", help="rewrite the index only")
    ap.add_argument("--compat-bf16", action="store_true",
                    help="dequantize small non-native projections to BF16 for ordinary Qwen4Exp GGUFs "
                         "(rounds weights; leaves experts and the PLE table unchanged)")
    ap.add_argument("--experts-bin", action="store_true",
                    help="also write experts.bin (the engine otherwise reads the experts from the GGUF itself)")
    a = ap.parse_args()
    if a.compat_bf16 and a.base:
        ap.error("--compat-bf16 cannot reuse --base dense weights")
    # HF snapshot files are symlinks to hash-named blobs. Keep the shard filename for discovery: .absolute(), not
    # .resolve(), which would follow the link to the blob and lose the -0000N-of-0000M name.
    src = pathlib.Path(a.gguf).absolute()
    base = pathlib.Path(a.base).resolve() if a.base else None
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    g = G.GGUFFile(src)
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    model = Model(src)
    if len(model.paths) > 1:
        print("model shards: " + ", ".join(p.name for p in model.paths))
    # ---- the expert table first: a model that cannot be packed is refused before any file of the pack changes
    got = expert_layout(model, src)
    if isinstance(got, str):
        print(got)
        return 1
    layout, text, n_expert, offset = got
    path = out / "experts.bin"
    sidecar = out / "experts.bin.src.json"
    want = experts_source(model, text, offset)
    reuse = path.exists() and path.stat().st_size == offset and read_json(sidecar) == want
    if path.exists() and not reuse and not a.experts_bin:
        if sidecar.exists():
            print("%s was cut from another source than this model (%s): the engine would read it instead of "
                  "the GGUF.  Delete it, or rerun with --experts-bin to rewrite it." % (path, sidecar.name))
            return 1
        print("warning: %s has no %s, so nothing says it belongs to this model; the engine reads it instead of "
              "the GGUF (rerun with --experts-bin to rewrite it)" % (path, sidecar.name))
    if a.base:
        if any(w[3] != src for w in model.where.values() if not w[1].name in NOT_IN_PACK):
            print("--base needs a model whose tensors are all in shard 1")
            return 1
        (out / "native_experts.txt").unlink(missing_ok=True)
        rc = index_from_base(a, src, base, out, g, {t.name: t for t in g.tensors}, mm)
    else:
        rc = index_standalone(src, out, model, a.compat_bf16)
    if rc:
        return rc
    if not (out / "tokenizer" / "vocab.json").exists() or not (out / "tokenizer" / "chat_template.jinja").exists():
        subprocess.run([sys.executable, str(HERE / "strata_tokenizer.py"), "--gguf", str(src), "--out", str(out)],
                       check=True)   # writes <out>/tokenizer/

    # ---- the experts.  native_experts.txt is written to a temporary name and renamed only when every layer is
    # in: a stop part-way (a layer split across shards, #171) left a partial native_experts.txt that the next setup
    # run took as a finished pack (#172).  It is the pack's completion marker, so it is published last.
    tmp = out / "native_experts.txt.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(text)
    tmp.replace(out / "native_experts.txt")
    if a.skip_experts or not a.experts_bin:
        if path.exists() and not a.experts_bin:
            print("note: %s/experts.bin exists; the engine reads it instead of the GGUF" % out)
        return 0
    if reuse:
        print("experts.bin was cut from this model's shards (%s); not rewritten" % sidecar.name)
        return 0
    # written under a temporary name and renamed when complete, then the sidecar: an interrupted write leaves no
    # experts.bin, and an experts.bin without its sidecar is never reused
    sidecar.unlink(missing_ok=True)
    part = out / "experts.bin.tmp"
    with open(part, "wb") as fo:
        for l, gt, dt, off, blob, ts in layout:
            parts = [model.bytes(t.name).reshape(n_expert, -1) for t in ts]
            chunk = np.concatenate(parts, axis=1)          # (n_expert, blob): gate | up | down per expert
            assert chunk.shape == (n_expert, blob)
            fo.write(chunk.tobytes())
            if l % 8 == 0:
                print("  layer %2d  %-8s/%-7s blob %8d  at %.2f GiB" % (l, ts[0].type_name, ts[2].type_name, blob,
                                                                        off / 2**30), flush=True)
    part.replace(path)
    side_tmp = out / "experts.bin.src.json.tmp"
    side_tmp.write_text(json.dumps(want, indent=2) + "\n", encoding="utf-8")
    side_tmp.replace(sidecar)
    print("experts.bin: %d layers, %.2f GiB" % (len(layout), offset / 2**30))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
