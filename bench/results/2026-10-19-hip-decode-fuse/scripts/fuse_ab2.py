"""Deterministic A/B of STRATA_POLY_FUSE: --tokens runs with --expert-swaps 0 and a fresh copy of the same profile,
so every process caches the same experts; with the same bits in both arms the tokens and the guesses taken are the
same, and only the launches differ.  Pairs alternated in order; the engine's own 'generated N tokens X tok/s'."""
import json, os, pathlib, re, shutil, statistics, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools" / "hip"))
from _rocm_env import engine_env  # noqa: E402

EXE = str(ROOT / "build-hip-poly-fuse" / "strata-poly.exe")
MODEL = os.environ["POLY_MODEL"]  # the GGUF
PROFILE = os.environ["POLY_PROFILE"]  # its expert profile, copied for each process
OUT = pathlib.Path(sys.argv[1])
PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PROMPTS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["B-2k", "B-45k"]
WINDOWS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["default", "1"]
NEW = 256

work = pathlib.Path(tempfile.mkdtemp(prefix="fuse-ab2-"))
sets = json.loads((ROOT / "bench/results/2026-10-10-hip-poly-vs-llama-vulkan/prompts.json").read_text())["sets"]
results = []
for name in PROMPTS:
    pf = work / f"{name}.txt"
    pf.write_text(",".join(map(str, sets[name])))
    for p in range(PAIRS):
        for win in WINDOWS:
            for fuse in (["0", "1"] if p % 2 == 0 else ["1", "0"]):
                prof = work / "profile.txt"
                shutil.copyfile(PROFILE, prof)
                cmd = [EXE, "--model", MODEL, "--max-context", "49152", "--threads", "6", "--expert-profile", str(prof),
                       "--expert-swaps", "0", "--tokens", "@" + str(pf), "--max-new", str(NEW)]
                if win != "default":
                    cmd += ["--window", win]
                env = engine_env(EXE)
                env["STRATA_POLY_FUSE"] = fuse
                env["STRATA_POLY_WARM"] = "1"
                pr = subprocess.run(cmd, capture_output=True, text=True, env=env)
                m = re.search(r"prompt (\d+) tokens ([\d.]+) tok/s, generated (\d+) tokens ([\d.]+) tok/s.*?"
                              r"experts from VRAM ([\d.]+)%.*?(\d+) of (\d+) guessed", pr.stderr)
                row = {"prompt": name, "pair": p, "window": win, "fuse": fuse, "rc": pr.returncode}
                if m:
                    row.update(prefill=float(m[2]), decode=float(m[4]), vram=float(m[5]), right=int(m[6]),
                               guessed=int(m[7]), ids=pr.stdout.strip().splitlines()[-1] if pr.stdout.strip() else "")
                else:
                    row["err"] = pr.stderr[-800:]
                results.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "ids"}), flush=True)
                OUT.write_text(json.dumps(results, indent=1))

print("\nprompt  window   fuse0 median  fuse1 median  change   (runs)")
for name in PROMPTS:
    for win in WINDOWS:
        v = {f: [r["decode"] for r in results if r["prompt"] == name and r["window"] == win and r["fuse"] == f and "decode" in r]
             for f in "01"}
        same = {r["ids"] for r in results if r["prompt"] == name and r["window"] == win and "ids" in r}
        if v["0"] and v["1"]:
            a, b = statistics.median(v["0"]), statistics.median(v["1"])
            print(f"{name:6s}  {win:7s}  {a:8.2f}      {b:8.2f}     {100 * (b / a - 1):+5.1f}%   {v['0']} {v['1']}  "
                  f"same tokens in every run: {len(same) == 1}")
shutil.rmtree(work, ignore_errors=True)
