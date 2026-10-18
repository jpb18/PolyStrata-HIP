"""strata-poly runs of the same prompt set against a reference run, step by step (docs/AMD_HIP_KV_Q8.md, section 8):
the error a setting brings (--kv q8_0, another build, another expert placement), kept apart from the noise of the
expert cache.

    python tools/hip/kv_q8_compare.py --model M.gguf --prompt ids.txt --ctx 49152 --max-new 64 \
        --profile C:/models/qwen36-q3-pack/expert-profile.txt --args "--threads 6" \
        --arm "ref=build-hip-poly-win/strata-poly.exe|" \
        --arm "again=build-hip-poly-win/strata-poly.exe|" \
        --arm "reserve4000=build-hip-poly-win/strata-poly.exe|--vram-reserve-mib 4000" \
        [--force] [--json out.json]

The first arm is the reference.  Each arm is EXE|ARGS: its own build and the arguments it adds.  Every run is greedy,
--window 1, with --logits-out.  --profile is copied from the file given before every run, so that every process
starts from the same expert profile (the engine writes its profile back on exit) and the file given is never
written.

Without --force, each run writes its own answer: the steps are compared up to the first token that differs, and
there whether the reference's two likeliest were within 0.1 of each other (a near tie).  With --force (an engine with
--force-tokens), every run is fed the reference's answer, so every step is compared, and the reference's next token
gives each run its negative log-likelihood: the mean difference (nats a token) and its standard error.

Per arm: the steps with the same logits bit for bit, the largest and mean probability difference, the mean and
largest KL divergence of the arm's distribution from the reference's, the share of steps whose likeliest token is
the reference's, and the load lines (experts in VRAM, the VRAM budget, STRATA_POLY_CACHE_HASH when set).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _rocm_env import engine_env  # noqa: E402

INFO = ("prompt ", "strata-poly: VRAM MiB", "strata-poly: loaded", "strata-poly: expert set", "strata-poly: KV")


def run(a, exe: str, extra: str, tag: str, wd: pathlib.Path, force: pathlib.Path | None) -> dict:
    out = wd / f"{tag}.logits"
    cmd = [exe, "--model", a.model, "--max-context", str(a.ctx), "--tokens", "@" + a.prompt,
           "--max-new", str(a.max_new), "--window", "1", "--logits-out", str(out)]
    if a.profile:
        prof = wd / "expert-profile.txt"
        shutil.copyfile(a.profile, prof)
        cmd += ["--expert-profile", str(prof)]
    if force is not None:
        cmd += ["--force-tokens", "@" + str(force)]
    cmd += [*shlex.split(a.args, posix=False), *shlex.split(extra, posix=False)]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=engine_env(exe))
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        raise SystemExit(f"{tag}: strata-poly ended with {p.returncode}")
    ids = [int(x) for x in p.stdout.strip().splitlines()[-1].split(",") if x]
    info = [l for l in p.stderr.splitlines() if l.startswith(INFO)]
    return {"exe": exe, "args": extra, "ids": ids, "logits": np.fromfile(out, dtype=np.float32), "info": info,
            "wall_s": round(time.time() - t, 1)}


def log_softmax(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64) - x.max()
    return z - math.log(np.exp(z).sum())


def compare(ref: dict, test: dict, n_vocab: int, forced: bool) -> dict:
    rl = ref["logits"].reshape(-1, n_vocab)
    tl = test["logits"].reshape(-1, n_vocab)
    steps = min(len(rl), len(tl))
    same_bits = top1 = 0
    dprob, kl, dnll = [], [], []
    first_diff = near_tie = gap = None
    for i in range(steps):
        if np.array_equal(rl[i].view(np.uint32), tl[i].view(np.uint32)):
            same_bits += 1
        lr, lt = log_softmax(rl[i]), log_softmax(tl[i])
        pr, pt = np.exp(lr), np.exp(lt)
        dprob.append(float(np.abs(pr - pt).max()))
        kl.append(float(np.sum(pr * (lr - lt))))
        top1 += int(np.argmax(rl[i]) == np.argmax(tl[i]))
        if forced:
            if i < len(ref["ids"]):   # step i's logits are the ones token i is picked from
                nxt = ref["ids"][i]
                dnll.append(float(lr[nxt] - lt[nxt]))   # the test's NLL minus the reference's
        elif first_diff is None and i < len(ref["ids"]) and i < len(test["ids"]) and ref["ids"][i] != test["ids"][i]:
            first_diff = i
            top2 = np.sort(pr)[-2:]
            gap = float(top2[1] - top2[0])
            near_tie = gap < 0.1
            steps = i + 1
            break   # after a different token the two are not on the same text any more
    n = len(kl)
    r = {"arm": test.get("name"), "args": test["args"], "exe": test["exe"], "steps": n, "same_bits": same_bits,
         "first_diff": first_diff, "near_tie": near_tie, "top2_gap": gap,
         "max_dprob": max(dprob) if dprob else None, "mean_dprob": float(np.mean(dprob)) if dprob else None,
         "mean_kl": float(np.mean(kl)) if kl else None, "p99_kl": float(np.percentile(kl, 99)) if kl else None,
         "max_kl": max(kl) if kl else None, "top1_agree": top1 / n if n else None,
         "info": test["info"], "wall_s": test["wall_s"]}
    if dnll:
        d = np.array(dnll)
        r["dnll_mean"] = float(d.mean())
        r["dnll_se"] = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True, help="a file of comma-separated token ids")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--args", default="", help="arguments for every run")
    ap.add_argument("--profile", help="an expert profile, copied before every run (never written)")
    ap.add_argument("--arm", action="append", required=True, help="NAME=EXE|ARGS; the first is the reference")
    ap.add_argument("--force", action="store_true", help="feed every arm the reference's answer (--force-tokens)")
    ap.add_argument("--json")
    a = ap.parse_args()
    arms = []
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        exe, _, extra = rest.partition("|")
        arms.append((name, str(pathlib.Path(exe).resolve()), extra.strip()))
    n_prompt = len(pathlib.Path(a.prompt).read_text().strip().split(","))
    with tempfile.TemporaryDirectory() as td:
        wd = pathlib.Path(td)
        name, exe, extra = arms[0]
        ref = run(a, exe, extra, "ref", wd, None)
        ref["name"] = name
        n_vocab = len(ref["logits"]) // max(len(ref["ids"]), 1)
        force = None
        if a.force:
            force = wd / "force.txt"
            force.write_text(",".join(str(t) for t in ref["ids"]))
            ref = run(a, exe, extra, "ref-forced", wd, force)   # the reference fed its own answer: the same path
            ref["name"] = name
        print(f"prompt {n_prompt} tokens, context {a.ctx}, {len(ref['ids'])} steps{' (forced)' if a.force else ''}")
        print(f"ref {name} [{extra}] {ref['wall_s']} s")
        for l in ref["info"]:
            print("     " + l)
        results = []
        for k, (name, exe, extra) in enumerate(arms[1:]):
            t = run(a, exe, extra, f"arm{k}", wd, force)
            t["name"] = name
            r = compare(ref, t, n_vocab, a.force)
            results.append(r)
            tok = ("all the same" if r["first_diff"] is None else
                   f"differ at step {r['first_diff']} ({'near tie' if r['near_tie'] else 'NOT a near tie'}, "
                   f"top-2 gap {r['top2_gap']:.3f})")
            nll = (f"; dNLL {r['dnll_mean']:+.4f} +- {r['dnll_se']:.4f} nats/token" if r.get("dnll_mean") is not None
                   and r.get("dnll_se") is not None else "")
            print(f"{name} [{extra}] {r['wall_s']} s: {r['same_bits']}/{r['steps']} steps same bits; tokens {tok}; "
                  f"top-1 {100 * r['top1_agree']:.1f}%; prob diff max {r['max_dprob']:.3g} mean {r['mean_dprob']:.3g}; "
                  f"KL mean {r['mean_kl']:.3g} p99 {r['p99_kl']:.3g} max {r['max_kl']:.3g}{nll}")
            for l in r["info"]:
                print("     " + l)
        if a.json:
            pathlib.Path(a.json).write_text(json.dumps({
                "prompt": a.prompt, "prompt_tokens": n_prompt, "ctx": a.ctx, "max_new": a.max_new, "forced": a.force,
                "args": a.args, "ref": {"arm": ref["name"], "exe": ref["exe"], "args": ref["args"], "ids": ref["ids"],
                                        "info": ref["info"], "wall_s": ref["wall_s"]},
                "arms": results}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
