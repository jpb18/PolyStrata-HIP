"""Speed of the experts' shared reads: forced windows of one size over the same tokens (so every arm does the same
work), arms alternated: the build before (--ref-exe), this build with STRATA_POLY_EXPERT_SHARE=0, and with it on.
Prints the engine's own decode tok/s of each run."""
import argparse, json, pathlib, re, shutil, subprocess, sys, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools" / "hip"))
from _rocm_env import engine_env  # noqa: E402


def run(exe, a, prompt, extra, share, work, tag):
    prof = work / f"{tag}-profile.txt"
    shutil.copyfile(a.profile, prof)
    cmd = [exe, "--model", a.model, "--max-context", a.max_context, "--threads", "6", "--expert-profile", str(prof),
           "--expert-swaps", "0", "--expert-cache-max", a.cache, "--tokens", "@" + str(prompt), *extra]
    env = engine_env(exe)
    env["STRATA_POLY_WARM"] = "1"
    if share is not None:
        env["STRATA_POLY_EXPERT_SHARE"] = share
    import os
    if os.environ.get("T_TRACE"):
        env["STRATA_POLY_TRACE"] = os.environ["T_TRACE"]
    if os.environ.get("T_SHOWERR"):
        env["T_SHOWERR"] = "1"
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise SystemExit(f"{tag}: exit {p.returncode}")
    run.last_err = p.stderr
    g = lambda pat: (re.search(pat, p.stderr) or [None, None])[1]
    if os.environ.get("T_SHOWERR"):
        sys.stderr.write(chr(10).join(l for l in p.stderr.splitlines() if "trace" in l.lower() or l.startswith("  "))[-6000:] + chr(10))
    return p, {"decode": g(r"(?:generated|forced) \d+ tokens ([\d.]+) tok/s"), "vram": g(r"experts from VRAM ([\d.]+)%"),
               "prefill": g(r"prompt \d+ tokens ([\d.]+) tok/s")}


ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True)
ap.add_argument("--ref-exe", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--profile", required=True)
ap.add_argument("--prompt", default="B-2k")
ap.add_argument("--max-new", type=int, default=128)
ap.add_argument("--windows", default="4")
ap.add_argument("--max-context", default="49152")
ap.add_argument("--cache", default="4000")
ap.add_argument("--pairs", type=int, default=3)
a = ap.parse_args()
work = pathlib.Path(tempfile.mkdtemp(prefix="share-ab-"))
sets = json.loads((ROOT / "bench/results/2026-10-10-hip-poly-vs-llama-vulkan/prompts.json").read_text())["sets"]
prompt = work / "prompt.txt"
prompt.write_text(",".join(map(str, sets[a.prompt])))
p, _ = run(a.ref_exe, a, prompt, ["--max-new", str(a.max_new), "--window", "1"], None, work, "ids")
force = work / "force.txt"
force.write_text(p.stdout.strip().splitlines()[-1])
win = ["--force-tokens", "@" + str(force), "--force-window", a.windows, "--window", "8"]
arms = [("ref", a.ref_exe, None), ("share-off", a.exe, "0"), ("share-on", a.exe, None)]
rows = {k: [] for k, _, _ in arms}
for i in range(a.pairs):
    for k, exe, share in (arms if i % 2 == 0 else arms[::-1]):
        _, r = run(exe, a, prompt, win, share, work, f"{k}-{i}")
        rows[k].append(r)
        print(k, i, json.dumps(r), flush=True)
for k, v in rows.items():
    d = sorted(float(r["decode"]) for r in v)
    print(k, "median decode", d[len(d) // 2], "tok/s")
shutil.rmtree(work, ignore_errors=True)
