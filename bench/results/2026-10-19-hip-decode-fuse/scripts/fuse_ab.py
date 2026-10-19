"""A/B of STRATA_POLY_FUSE on one build: rounds of one process per arm, the arms' order alternated each round,
each process from a fresh copy of the same expert profile.  tools/hip/decode_trace.py does each process (a warm-up
on A-2k, then B-2k and B-45k, 128 greedy tokens each)."""
import json, os, pathlib, shutil, statistics, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
EXE = str(ROOT / "build-hip-poly-fuse" / "strata-poly.exe")
MODEL = os.environ["POLY_MODEL"]  # the GGUF
PROFILE = os.environ["POLY_PROFILE"]  # its expert profile, copied for each process
OUT = pathlib.Path(sys.argv[1])
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
WINDOWS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["default", "1"]

work = pathlib.Path(tempfile.mkdtemp(prefix="fuse-ab-"))
for r in range(ROUNDS):
    for win in WINDOWS:
        arms = ["0", "1"] if r % 2 == 0 else ["1", "0"]
        for fuse in arms:
            prof = work / f"profile-{r}-{win}-{fuse}.txt"
            shutil.copyfile(PROFILE, prof)
            args = f"--expert-profile {prof} --threads 6" + ("" if win == "default" else f" --window {win}")
            label = f"fuse{fuse}-w{win}"
            print(f"== round {r} {label}", flush=True)
            subprocess.run([sys.executable, str(ROOT / "tools/hip/decode_trace.py"), "--exe", EXE, "--model", MODEL,
                            "--args", args, "--env", f"STRATA_POLY_FUSE={fuse}", "--only", "B-2k,B-45k",
                            "--label", label, "--out", str(OUT)], check=True)

res = json.loads(OUT.read_text(encoding="utf-8"))
table = {}
for e in res:
    for run in e["runs"]:
        table.setdefault((e["label"], run["prompt"]), []).append(run["decode_tps"])
print("\nlabel           prompt   decode tok/s (median, runs)")
for (label, prompt), v in sorted(table.items()):
    print(f"{label:15s} {prompt:7s} {statistics.median(v):7.2f}  {v}")
shutil.rmtree(work, ignore_errors=True)
