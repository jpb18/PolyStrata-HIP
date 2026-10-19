"""Flash attention against the products it replaces, in the engine: the same engine, STRATA_POLY_FLASH=0 and 1, one
process a run, the Q8 launch's arguments (q8_0 KV, 262,144 positions), the prompt's tokens and a few greedy tokens.
Prints each run's own prefill and decode tok/s, and compares the logits of every step against the first run's:
the largest difference, whether the same token comes first, and the Kullback-Leibler divergence of the
distributions (nats).

    python flash_ab.py --prompt B-45k --order 0,1,0,1
    python flash_ab.py --prompt cat130k --order 0,1          (A-45k, B-45k, A-31k and B-8k one after the other)
    python flash_ab.py --prompt A-45k+A-31k --order 0,1      (any sets joined with +)
    python flash_ab.py --prompt cat130k --order 0,fp16:0 --ctx 140000   (an fp16 cache as the benign perturbation)
"""
import argparse, json, pathlib, re, shutil, subprocess, sys, tempfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools" / "hip"))
from _rocm_env import engine_env  # noqa: E402

N_VOCAB = 248320
MODEL = r"C:\Users\Joao_\models\Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"
PROFILE = ROOT / "tmp/hip-launch-q8/q8-ooc679i9/expert-profile.txt"

ap = argparse.ArgumentParser()
ap.add_argument("--exe", default=str(ROOT / "build-hip-poly-win" / "strata-poly.exe"))
ap.add_argument("--prompt", default="B-2k")
ap.add_argument("--order", default="0,1,0,1")
ap.add_argument("--new", type=int, default=24)
ap.add_argument("--ctx", default="262144")
a = ap.parse_args()

sets = json.loads((ROOT / "bench/results/2026-10-10-hip-poly-vs-llama-vulkan/prompts.json").read_text())["sets"]
ids = sum((sets[k] for k in (("A-45k", "B-45k", "A-31k", "B-8k") if a.prompt == "cat130k" else a.prompt.split("+"))), [])
work = pathlib.Path(tempfile.mkdtemp(prefix="flash-ab-"))
prompt = work / "prompt.txt"
prompt.write_text(",".join(map(str, ids)))
print(f"prompt {a.prompt}: {len(ids)} tokens", flush=True)

runs = []
for k, item in enumerate(a.order.split(",")):
    kv, flash = item.split(":") if ":" in item else ("q8_0", item)   # "fp16:0" is an fp16 cache, flash off
    prof = work / "profile.txt"
    shutil.copyfile(PROFILE, prof)
    lg = work / f"{k}.logits"
    cmd = [a.exe, "--model", MODEL, "--threads", "6", "--turn-token", "248045", "--max-context", a.ctx, "--kv", kv,
           "--expert-profile", str(prof), "--expert-swaps", "0", "--expert-cache-max", "3000", "--window", "1",
           "--tokens", "@" + str(prompt), "--max-new", str(a.new), "--logits-out", str(lg)]
    env = engine_env(a.exe)
    env["STRATA_POLY_WARM"] = "1"
    env["STRATA_POLY_FLASH"] = flash
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    g = lambda pat: (re.search(pat, p.stderr) or [None, None])[1]
    row = {"flash": f"{kv}:{flash}", "rc": p.returncode, "prefill": g(r"prompt \d+ tokens ([\d.]+) tok/s"),
           "decode": g(r"generated \d+ tokens ([\d.]+) tok/s")}
    if p.returncode == 0:
        row["ids"] = [int(x) for x in p.stdout.strip().splitlines()[-1].split(",") if x]
        row["logits"] = np.fromfile(lg, dtype=np.float32).reshape(-1, N_VOCAB)
    else:
        row["err"] = p.stderr[-1500:]
    runs.append(row)
    print({k2: v for k2, v in row.items() if k2 not in ("ids", "logits")}, flush=True)


def softmax(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


ok = [r for r in runs if r["rc"] == 0]
base = ok[0]
for r in ok[1:]:
    steps = min(len(base["logits"]), len(r["logits"]))
    d = np.abs(base["logits"][:steps] - r["logits"][:steps])
    pb, pr = softmax(base["logits"][:steps]), softmax(r["logits"][:steps])
    kl = (pb * (np.log(pb + 1e-300) - np.log(pr + 1e-300))).sum(axis=-1)
    top = (base["logits"][:steps].argmax(-1) == r["logits"][:steps].argmax(-1)).mean()
    print(f"flash={r['flash']} against the first run (flash={base['flash']}): {steps} steps, largest logit difference "
          f"{d.max():.4f}, mean {d.mean():.5f}, same first token {top * 100:.0f} %, KL mean {kl.mean():.2e} "
          f"max {kl.max():.2e} nats, tokens equal {r['ids'] == base['ids']}")
shutil.rmtree(work, ignore_errors=True)
