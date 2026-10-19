"""Windows of guessed tokens, flash attention against the rows kernels, in the engine: the same tokens (a decode's, forced
with --force-tokens) in windows of one size (--force-window), so every run does the same work; STRATA_POLY_FLASH=0 and
1 alternated, one process a run, the Q8 launch's arguments.  Prints the engine's own decode tok/s.

    python flash_win_ab.py --prompt A-45k+B-45k --windows 4 --pairs 2
"""
import argparse, json, pathlib, re, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools" / "hip"))
from _rocm_env import engine_env  # noqa: E402

MODEL = r"C:\Users\Joao_\models\Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"
PROFILE = ROOT / "tmp/hip-launch-q8/q8-ooc679i9/expert-profile.txt"

ap = argparse.ArgumentParser()
ap.add_argument("--exe", default=str(ROOT / "build-hip-poly-win" / "strata-poly.exe"))
ap.add_argument("--prompt", default="A-45k+B-45k")
ap.add_argument("--windows", default="4")
ap.add_argument("--pairs", type=int, default=2)
ap.add_argument("--new", type=int, default=128)
ap.add_argument("--cache", default="5000")
ap.add_argument("--ctx", default="262144")
a = ap.parse_args()

sets = json.loads((ROOT / "bench/results/2026-10-10-hip-poly-vs-llama-vulkan/prompts.json").read_text())["sets"]
ids = sum((sets[k] for k in a.prompt.split("+")), [])
work = pathlib.Path(tempfile.mkdtemp(prefix="flash-win-"))
prompt = work / "prompt.txt"
prompt.write_text(",".join(map(str, ids)))


def run(flash, extra, tag):
    prof = work / f"{tag}-profile.txt"
    shutil.copyfile(PROFILE, prof)
    cmd = [a.exe, "--model", MODEL, "--threads", "6", "--turn-token", "248045", "--max-context", a.ctx, "--kv", "q8_0",
           "--expert-profile", str(prof), "--expert-swaps", "0", "--expert-cache-max", a.cache,
           "--tokens", "@" + str(prompt), *extra]
    env = engine_env(a.exe)
    env["STRATA_POLY_WARM"] = "1"
    env["STRATA_POLY_FLASH"] = flash
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise SystemExit(f"{tag}: exit {p.returncode}")
    g = lambda pat: (re.search(pat, p.stderr) or [None, None])[1]
    return p, {"prefill": g(r"prompt \d+ tokens ([\d.]+) tok/s"), "decode": g(r"(?:generated|forced) \d+ tokens ([\d.]+) tok/s")}


p, _ = run("1", ["--max-new", str(a.new), "--window", "1"], "ids")
force = work / "force.txt"
force.write_text(p.stdout.strip().splitlines()[-1])
win = ["--force-tokens", "@" + str(force), "--force-window", a.windows, "--window", "8"]
rows = {"0": [], "1": []}
for i in range(a.pairs):
    for flash in (("0", "1") if i % 2 == 0 else ("1", "0")):
        _, r = run(flash, win, f"w{flash}{i}")
        rows[flash].append(r)
        print("flash", flash, i, json.dumps(r), flush=True)
for k, v in rows.items():
    d = sorted(float(r["decode"]) for r in v)
    print("flash", k, "decode tok/s:", d, "median", d[len(d) // 2])
shutil.rmtree(work, ignore_errors=True)
