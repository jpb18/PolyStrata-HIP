"""strata-poly against llama.cpp on the same GGUF: the check of docs/BACKENDS.md ("How each backend was checked").

    python tools/hip/poly_llama_parity.py --engine build-hip-poly-win/strata-poly.exe --model MODEL.gguf \
        --llama-server C:/path/llama-server.exe [--engine-args "--window 1 --no-draft"] [--json out.json]

A running llama-server (--llama-url) can stand in for --llama-server.  Each prompt is put in the model's chat template
and turned into token ids by llama-server, so both engines read the same ids.  Both answer greedily, `--tokens`
(64 by default) each, and for every step the reference's five likeliest tokens are looked up in the engine's own
probabilities.  The report has, per prompt, how many tokens are the same before the first that is not, the largest
probability difference over those steps, and at the step where they part the two likeliest tokens of each side:
two correct implementations part at a near tie (the two within 0.1 of each other on one side), as the page says.

The prompts are a short chat, a question about code, and two of this repository's documents cut to about 2,500 and
8,000 tokens, so that a prompt batch (1,024 tokens) and the reading of several calls (read()) are both crossed.
Exit 0 when every prompt either matches all its tokens or parts at a near tie and stays within --max-diff; else 1.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]


def post(url: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_up(url: str, proc: subprocess.Popen | None, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            raise SystemExit(f"llama-server ended (exit {proc.returncode}) before it was ready")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise SystemExit("llama-server was not ready in time")


def doc_text(name: str, chars: int) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")[:chars]


def prompts() -> list[tuple[str, str]]:
    code = (ROOT / "src" / "poly" / "cpu_experts.hpp").read_text(encoding="utf-8")
    return [
        ("short", "What is the capital of Portugal? Answer in one sentence."),
        ("code", "Explain in a few sentences what this header declares:\n\n" + code),
        ("doc-2.5k", "Summarize the following document in five bullet points.\n\n" + doc_text("BACKENDS.md", 10000)),
        ("doc-8k", "What are the main measurements in this document? Be brief.\n\n" + doc_text("BACKENDS.md", 32000)),
    ]


def ids_of(url: str, text: str) -> list[int]:
    templ = post(url + "/apply-template", {"messages": [{"role": "user", "content": text}]})["prompt"]
    return post(url + "/tokenize", {"content": templ, "add_special": True, "parse_special": True})["tokens"]


def reference(url: str, ids: list[int], n: int) -> list[dict]:
    """llama.cpp's greedy answer: per step the token and its five likeliest {id: probability}."""
    r = post(url + "/completion", {"prompt": ids, "n_predict": n, "temperature": 0.0, "top_k": 1, "n_probs": 5,
                                   "cache_prompt": False, "post_sampling_probs": False})
    steps = []
    for p in r.get("completion_probabilities", []):
        if "top_logprobs" in p:
            import math
            top = {int(t["id"]): math.exp(float(t["logprob"])) for t in p["top_logprobs"]}
        else:   # older servers
            top = {int(t["id"]): float(t["prob"]) for t in p["probs"]}
        steps.append({"id": int(p["id"]), "top": top})
    return steps


def engine_run(exe: str, model: str, ids: list[int], n: int, extra: list[str], ctx: int) -> tuple[list[dict], str]:
    """strata-poly's greedy answer through --tokens, with its 40 likeliest tokens a step (--top-probs)."""
    # the ids go in a file (--tokens @FILE): a long prompt's would not fit a Windows command line
    with tempfile.NamedTemporaryFile("w", suffix=".ids", delete=False) as f:
        f.write(",".join(map(str, ids)))
    cmd = [exe, "--model", model, "--max-context", str(ctx), "--tokens", "@" + f.name,
           "--max-new", str(n), "--top-probs", "40", *extra]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    finally:
        pathlib.Path(f.name).unlink(missing_ok=True)
    if p.returncode != 0:
        raise SystemExit(f"strata-poly failed (exit {p.returncode}):\n{p.stderr[-4000:]}")
    toks = [int(t) for t in p.stdout.strip().splitlines()[-1].split(",") if t]
    tops = []
    for line in p.stderr.splitlines():
        if line.startswith("top "):
            tops.append({int(a): float(b) for a, b in (x.split(":") for x in line.split()[1:])})
    steps = [{"id": t, "top": tops[i] if i < len(tops) else {}} for i, t in enumerate(toks)]
    stats = [l for l in p.stderr.splitlines() if l.startswith("prompt ") or "experts" in l.lower()]
    return steps, "\n".join(stats[-3:])


def two(top: dict) -> list[tuple[int, float]]:
    return sorted(top.items(), key=lambda kv: -kv[1])[:2]


def compare(ref: list[dict], got: list[dict]) -> dict:
    same, worst = 0, 0.0
    for r, g in zip(ref, got):
        for tid, p in r["top"].items():
            worst = max(worst, abs(p - g["top"].get(tid, 0.0)))
        if r["id"] != g["id"]:
            break
        same += 1
    out = {"same": same, "of": min(len(ref), len(got)), "max_diff": round(worst, 4)}
    if same < out["of"]:
        r2, g2 = two(ref[same]["top"]), two(got[same]["top"])
        out["parted_at"] = same
        out["llama_top2"] = r2
        out["engine_top2"] = g2
        gap = lambda t: t[0][1] - t[1][1] if len(t) > 1 else 1.0
        out["near_tie"] = min(gap(r2), gap(g2)) <= 0.1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="strata-poly(.exe)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama-server", help="llama-server(.exe) to start for the reference")
    ap.add_argument("--llama-url", help="a running llama-server with the same model")
    ap.add_argument("--llama-args", default="--cpu-moe -ngl 99", help="extra llama-server arguments")
    ap.add_argument("--engine-args", default="", help="extra strata-poly arguments, e.g. \"--window 1 --no-draft\"")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--max-diff", type=float, default=0.15)
    ap.add_argument("--only", help="run only the prompts whose name contains this")
    ap.add_argument("--json", help="write the results here")
    ap.add_argument("--refs", help="llama.cpp's answers: read from this file when it exists, else written to it")
    ap.add_argument("--ids-only", action="store_true",
                    help="compare the tokens only: with windows of guessed tokens (strata-poly's default) the "
                         "probabilities --top-probs prints are not the ones each token was picked from")
    a = ap.parse_args()

    proc = None
    url = a.llama_url
    refs = None
    if a.refs and pathlib.Path(a.refs).is_file():
        refs = [(r["name"], r["ids"], [{"id": s["id"], "top": {int(k): v for k, v in s["top"].items()}}
                                       for s in r["ref"]]) for r in json.loads(pathlib.Path(a.refs).read_text())]
        refs = [r for r in refs if not a.only or a.only in r[0]]
        print(f"reference: {a.refs} ({len(refs)} prompts)", flush=True)
    elif url is None:
        if not a.llama_server:
            ap.error("--llama-server or --llama-url is needed")
        port = 8187
        url = f"http://127.0.0.1:{port}"
        cmd = [a.llama_server, "-m", a.model, "--port", str(port), "-c", str(a.ctx), "--no-webui",
               *shlex.split(a.llama_args, posix=False)]
        print("reference:", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
    if refs is None:
        try:
            wait_up(url, proc, 600)
            work = [(n, t) for n, t in prompts() if not a.only or a.only in n]
            refs = []
            for name, text in work:
                ids = ids_of(url, text)
                refs.append((name, ids, reference(url, ids, a.tokens)))
                print(f"{name}: {len(ids)} prompt tokens, reference done", flush=True)
        finally:
            if proc is not None:   # the card is the engine's from here on
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if a.refs:
            pathlib.Path(a.refs).write_text(json.dumps([{"name": n, "ids": i, "ref": r} for n, i, r in refs]),
                                            encoding="utf-8")
    good = True
    for name, ids, ref in refs:
        got, stats = engine_run(a.engine, a.model, ids, a.tokens, shlex.split(a.engine_args, posix=False), a.ctx)
        c = compare(ref, got)
        c.update(name=name, prompt_tokens=len(ids))
        results.append(c)
        if a.ids_only:   # the engine's side of a near tie is not known here: the reference's must be one
            c["max_diff"] = 0.0
            if "parted_at" in c:
                r2 = c["llama_top2"]
                c["near_tie"] = len(r2) > 1 and r2[0][1] - r2[1][1] <= 0.1
                c["engine_top2"] = "(not compared)"
        ok = (c["same"] == c["of"] or c.get("near_tie", False)) and c["max_diff"] <= a.max_diff
        good = good and ok
        line = f"{name:9s} {len(ids):6d} tokens: {c['same']}/{c['of']} same, max diff {c['max_diff']:.3f}"
        if "parted_at" in c:
            line += (f", parts at {c['parted_at']}: llama {c['llama_top2']} engine {c['engine_top2']}"
                     f"{' (near tie)' if c['near_tie'] else ' (NOT a near tie)'}")
        print(("ok   " if ok else "FAIL ") + line, flush=True)
        if stats:
            print("     " + stats.replace("\n", "\n     "), flush=True)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("parity:", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
