"""Read a GGUF's metadata and tensor table from a URL (HTTP range requests) or a local file, without the data.
usage: python tools/hip/gguf_head.py URL_OR_PATH [--json out.json]"""
import json, struct, sys, urllib.request

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}
# (block elements, bytes per block)
BLK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22),
       "Q5_1": (32, 24), "Q8_0": (32, 34), "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
       "Q5_K": (256, 176), "Q6_K": (256, 210), "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ2_S": (256, 82),
       "IQ3_XXS": (256, 98), "IQ3_S": (256, 110), "IQ1_S": (256, 50), "IQ1_M": (256, 56), "IQ4_NL": (32, 18),
       "IQ4_XS": (256, 136), "MXFP4": (32, 17), "I32": (1, 4)}


class Src:
    def __init__(self, where):
        self.where, self.buf, self.local = where, b"", not where.startswith("http")
        self.f = open(where, "rb") if self.local else None

    def need(self, end):
        if end <= len(self.buf): return
        if self.local:
            self.f.seek(len(self.buf)); self.buf += self.f.read(max(end - len(self.buf), 1 << 20)); return
        want = max(end, len(self.buf) + (8 << 20))
        req = urllib.request.Request(self.where, headers={"Range": f"bytes={len(self.buf)}-{want - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r: self.buf += r.read()


def parse(where):
    s, p = Src(where), [0]

    def rd(n):
        s.need(p[0] + n); b = s.buf[p[0]:p[0] + n]; p[0] += n; return b

    u32 = lambda: struct.unpack("<I", rd(4))[0]
    u64 = lambda: struct.unpack("<Q", rd(8))[0]
    st = lambda: rd(u64()).decode("utf-8", "replace")
    scal = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4),
            7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}

    def val(t):
        if t in scal: f, n = scal[t]; return struct.unpack(f, rd(n))[0]
        if t == 8: return st()
        if t == 9:
            et, n = u32(), u64()
            if n > 64 and et in scal:   # long arrays (vocab scores, token types): skip the data, keep the length
                rd(scal[et][1] * n); return f"<array of {n}>"
            if n > 64 and et == 8:
                for _ in range(n): st()
                return f"<array of {n} strings>"
            return [val(et) for _ in range(n)]
        raise ValueError(t)

    assert rd(4) == b"GGUF"
    ver, nt, nkv = u32(), u64(), u64()
    meta = {}
    for _ in range(nkv):
        k = st(); meta[k] = val(u32())
    tensors = []
    for _ in range(nt):
        name = st(); nd = u32(); dims = [u64() for _ in range(nd)]; t = TYPES.get(u32(), "?"); off = u64()
        n = 1
        for d in dims: n *= d
        be, bb = BLK.get(t, (1, 0))
        tensors.append({"name": name, "dims": dims, "type": t, "bytes": n // be * bb})
    return {"version": ver, "meta": meta, "tensors": tensors, "header_bytes": p[0]}


if __name__ == "__main__":
    r = parse(sys.argv[1])
    if "--json" in sys.argv: json.dump(r, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    for k, v in r["meta"].items():
        if not k.startswith("tokenizer.ggml.") or not str(v).startswith("<"): print(k, "=", str(v)[:200])
    print("tensors:", len(r["tensors"]), "header bytes:", r["header_bytes"])
