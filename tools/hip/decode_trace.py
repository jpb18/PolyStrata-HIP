"""strata-poly's decode, part by part: one process, a warm-up prompt, then prompts of the phase 2 sets, each answered
greedy with STRATA_POLY_TRACE=1, which prints after each answer where its time went (src/poly/main.cpp, Trace).

    python tools/hip/decode_trace.py --model M.gguf --args "--expert-profile P --threads 6" \
        --prompts bench/results/2026-10-10-hip-poly-vs-llama-vulkan/prompts.json --only B-2k,B-8k --out out.json

Each request's line: prompt tokens, prefill and decode tok/s, the guesses, and the trace's lines as they were
printed.  --env KEY=VALUE (repeatable) sets more of the engine's environment; --exe picks another build.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _rocm_env import engine_env  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exe", default=str(ROOT / "build-hip-poly-win" / "strata-poly.exe"))
    p.add_argument("--model", required=True)
    p.add_argument("--ctx", type=int, default=49152)
    p.add_argument("--args", default="")
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--prompts", default=str(ROOT / "bench/results/2026-10-10-hip-poly-vs-llama-vulkan/prompts.json"))
    p.add_argument("--warm", default="A-2k", help="the prompt answered first and not counted ('' for none)")
    p.add_argument("--only", default="B-2k,B-8k,B-31k,B-45k")
    p.add_argument("--repeat", type=int, default=1, help="each prompt this many times (the conversation cache "
                   "then skips its prefill: the decode starts from the same place)")
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--label", default="")
    p.add_argument("--out", help="append the results to this JSON list")
    a = p.parse_args()

    sets = json.loads(pathlib.Path(a.prompts).read_text(encoding="utf-8"))["sets"]
    env = engine_env(a.exe)
    env["STRATA_POLY_TRACE"] = "1"
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    err = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False, encoding="utf-8", errors="replace")
    cmd = [a.exe, "--model", a.model, "--max-context", str(a.ctx), "--serve", *shlex.split(a.args, posix=False)]
    t = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                            env=env)
    for line in proc.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit("strata-poly ended before READY; its log: " + err.name)
    load_s = time.time() - t

    def gen(ids):
        pos = os.path.getsize(err.name)
        proc.stdin.write(f"GEN {a.max_new} {','.join(map(str, ids))}\n")
        proc.stdin.flush()
        tokens = []
        for line in proc.stdout:
            if line.startswith("T "):
                tokens.append(int(line.split()[1]))
            if line.startswith("ERR"):
                raise SystemExit("strata-poly: " + line.strip())
            if line.startswith("DONE"):
                f = line.split()
                time.sleep(0.2)
                err.flush()
                with open(err.name, encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    log = fh.read()
                trace = [ln.rstrip() for ln in log.splitlines()
                         if ln.startswith("strata-poly trace") or ln.startswith("  ")]
                produced, prompt_ms, decode_ms, read = int(f[1]), float(f[3]), float(f[4]), int(f[14])
                return {"produced": produced, "prompt_tokens": int(f[2]), "read": read,
                        "prefill_tps": round(1000 * read / prompt_ms, 1) if read and prompt_ms else None,
                        "decode_tps": round(1000 * produced / decode_ms, 2) if decode_ms else None,
                        "guessed_right": int(f[6]), "guessed": int(f[7]),
                        "vram_hits": round(int(f[9]) / max(int(f[10]), 1), 4), "trace": trace,
                        "tokens": tokens}
        raise SystemExit("strata-poly ended during a request; its log: " + err.name)

    runs = []
    try:
        if a.warm:
            r = gen(sets[a.warm])
            print(f"warm {a.warm}: decode {r['decode_tps']} tok/s", flush=True)
        for name in [s for s in a.only.split(",") if s]:
            for k in range(a.repeat):
                r = gen(sets[name])
                r["prompt"] = name
                runs.append(r)
                print(f"{name}#{k}: prefill {r['prefill_tps']} decode {r['decode_tps']} tok/s, guesses "
                      f"{r['guessed_right']}/{r['guessed']}, VRAM hits {r['vram_hits']}", flush=True)
                for ln in r["trace"]:
                    print("   " + ln, flush=True)
    finally:
        proc.stdin.close()
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            proc.kill()
    if a.out:
        out = pathlib.Path(a.out)
        allr = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        allr.append({"label": a.label, "exe": a.exe, "args": a.args, "env": a.env, "load_s": round(load_s, 1),
                     "runs": runs, "log": err.name})
        out.write_text(json.dumps(allr, indent=1), encoding="utf-8")
    print("engine log:", err.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
