"""strata-poly with its KV cache wholly in VRAM against the same with KV streaming (AMD_KV_STREAMING.md): the same
prompt, greedy, --window 1, every step's logits compared.

    python tools/hip/kv_parity.py --engine build-hip-poly-win/strata-poly.exe --model M.gguf --prompt ids.txt \
        --ctx 65536 --max-new 32 --ref "--kv-resident all" --test "--kv-resident 16384" --args "--threads 6" \
        [--json out.json]

The prompt is a file of comma-separated token ids.  --test may be given more than once.  For each test run and
each step: whether the logits are the same bits as the reference's, the largest difference of a logit, of a
probability (over the whole vocabulary) and the KL divergence of the test's distribution from the reference's.
The tokens are compared up to the first that differs, and there whether the reference's two likeliest were within
0.1 of each other (a near tie, where the last bits decide).  Also the softmax's largest logit and the entropy of the
reference, to see that the numbers stay sane at long contexts.  Exit 1 when a test run's tokens differ outside a
near tie.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _rocm_env import engine_env  # noqa: E402


def run(a, extra: str, tag: str, workdir: pathlib.Path) -> dict:
    out = workdir / f"{tag}.logits"
    cmd = [a.engine, "--model", a.model, "--max-context", str(a.ctx), "--tokens", "@" + a.prompt,
           "--max-new", str(a.max_new), "--window", "1", "--logits-out", str(out),
           *shlex.split(a.args, posix=False), *shlex.split(extra, posix=False)]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=engine_env(a.engine))
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        raise SystemExit(f"{tag}: strata-poly ended with {p.returncode}")
    ids = [int(x) for x in p.stdout.strip().splitlines()[-1].split(",") if x]
    info = [l for l in p.stderr.splitlines() if l.startswith(("prompt ", "strata-poly: KV", "strata-poly: VRAM MiB",
                                                                "strata-poly: loaded"))]
    logits = np.fromfile(out, dtype=np.float32)
    return {"args": extra, "ids": ids, "logits": logits, "info": info, "wall_s": round(time.time() - t, 1)}


def softmax(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64) - x.max()
    e = np.exp(z)
    return e / e.sum()


def compare(ref: dict, test: dict, n_vocab: int) -> dict:
    steps = min(len(ref["ids"]), len(test["ids"]))
    rl = ref["logits"].reshape(-1, n_vocab)
    tl = test["logits"].reshape(-1, n_vocab)
    same_bits = 0
    dlogit, dprob, kl = [], [], []
    first_diff = None
    near_tie = None
    for i in range(steps):
        if i >= len(rl) or i >= len(tl):
            break
        if np.array_equal(rl[i].view(np.uint32), tl[i].view(np.uint32)):
            same_bits += 1
        pr, pt = softmax(rl[i]), softmax(tl[i])
        dlogit.append(float(np.abs(rl[i] - tl[i]).max()))
        dprob.append(float(np.abs(pr - pt).max()))
        kl.append(float(np.sum(pr * (np.log(pr + 1e-300) - np.log(pt + 1e-300)))))
        if first_diff is None and ref["ids"][i] != test["ids"][i]:
            first_diff = i
            top2 = np.sort(pr)[-2:]
            near_tie = bool(top2[1] - top2[0] < 0.1)
            break   # after a different token the sequences are not the same prompt any more
    return {"args": test["args"], "steps": steps, "same_bits": same_bits, "first_diff": first_diff,
            "near_tie": near_tie, "max_dlogit": max(dlogit) if dlogit else None,
            "max_dprob": max(dprob) if dprob else None, "mean_dprob": float(np.mean(dprob)) if dprob else None,
            "max_kl": max(kl) if kl else None, "info": test["info"], "wall_s": test["wall_s"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True, help="a file of comma-separated token ids")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--args", default="", help="arguments for every run")
    ap.add_argument("--ref", default="--kv-resident all")
    ap.add_argument("--test", action="append", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    a.engine = str(pathlib.Path(a.engine).resolve())
    n_prompt = len(pathlib.Path(a.prompt).read_text().strip().split(","))
    with tempfile.TemporaryDirectory() as td:
        wd = pathlib.Path(td)
        ref = run(a, a.ref, "ref", wd)
        n_vocab = len(ref["logits"]) // max(len(ref["ids"]), 1)
        rl = ref["logits"].reshape(-1, n_vocab)
        stab = []
        for row in rl:
            p = softmax(row)
            stab.append({"max_logit": float(row.max()), "entropy": float(-(p * np.log(p + 1e-300)).sum()),
                         "finite": bool(np.isfinite(row).all())})
        print(f"prompt {n_prompt} tokens, context {a.ctx}, {len(ref['ids'])} steps")
        print(f"ref  [{a.ref}] {ref['wall_s']} s")
        for l in ref["info"]:
            print("     " + l)
        results = []
        bad = False
        for k, t in enumerate(a.test):
            r = compare(ref, run(a, t, f"test{k}", wd), n_vocab)
            results.append(r)
            ok = r["first_diff"] is None or r["near_tie"]
            bad |= not ok
            print(f"test [{t}] {r['wall_s']} s: {r['same_bits']}/{r['steps']} steps the same bits; "
                  f"tokens {'all the same' if r['first_diff'] is None else 'differ at step %d (%s)' % (r['first_diff'], 'near tie' if r['near_tie'] else 'NOT a near tie')}; "
                  f"largest logit difference {r['max_dlogit']:.3g}, probability {r['max_dprob']:.3g} "
                  f"(mean {r['mean_dprob']:.3g}), KL {r['max_kl']:.3g}")
            for l in r["info"]:
                print("     " + l)
        print("reference softmax: largest logit %.2f to %.2f, entropy %.3f to %.3f, all finite: %s" % (
            min(s["max_logit"] for s in stab), max(s["max_logit"] for s in stab),
            min(s["entropy"] for s in stab), max(s["entropy"] for s in stab), all(s["finite"] for s in stab)))
        if a.json:
            pathlib.Path(a.json).write_text(json.dumps({
                "prompt_tokens": n_prompt, "ctx": a.ctx, "ref": {"args": a.ref, "ids": ref["ids"], "info": ref["info"],
                                                                  "wall_s": ref["wall_s"], "softmax": stab},
                "tests": [dict(r, ids=None) for r in results]}, indent=1), encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
