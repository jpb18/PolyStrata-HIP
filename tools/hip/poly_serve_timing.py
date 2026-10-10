"""Prompts one after another through one strata-poly process (its --serve protocol), each timed: what a first
prompt costs beyond the ones after it, with no server in between.

    python tools/hip/poly_serve_timing.py --engine build-hip-poly-win/strata-poly.exe --model MODEL.gguf \
        --refs refs.json [--max-new 64] [--engine-args "--threads 6"] [--repeat 2]

The prompts are the token ids of a reference file that tools/hip/poly_llama_parity.py --refs wrote, each
`--repeat` times in a row.  A prompt sent again is read again only past what the engine keeps of the one before
(its checkpoints), so the repeats show the reading of a new prompt only in the first pass; the later passes time
the answer.  Each line: the prompt, its tokens, the seconds to the first token, the whole request, and the
engine's own DONE line.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _rocm_env import engine_env  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--engine-args", default="")
    a = ap.parse_args()
    refs = json.loads(pathlib.Path(a.refs).read_text())
    cmd = [a.engine, "--model", a.model, "--max-context", str(a.ctx), "--serve",
           *shlex.split(a.engine_args, posix=False)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1,
                         env=engine_env(a.engine))
    t0 = time.time()
    for line in p.stdout:
        if line.startswith("READY"):
            break
    print(f"ready after {time.time() - t0:.1f} s", flush=True)
    for r in refs:
        for k in range(a.repeat):
            ids = ",".join(map(str, r["ids"]))
            t = time.time()
            first = None
            p.stdin.write(f"GEN {a.max_new} {ids}\n")
            p.stdin.flush()
            done = ""
            for line in p.stdout:
                if line.startswith("T ") and first is None:
                    first = time.time() - t
                if line.startswith("DONE") or line.startswith("ERR"):
                    done = line.strip()
                    break
            print(f"{r['name']:9s} pass {k + 1}: {len(r['ids']):6d} tokens, first token {first or 0:.2f} s, "
                  f"all {time.time() - t:.2f} s | {done}", flush=True)
    p.stdin.close()
    p.wait(60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
