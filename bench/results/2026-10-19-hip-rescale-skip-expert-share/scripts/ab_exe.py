"""Two engine builds alternated (T_ORDER, "base" = build-hip-poly-win, "new" = build-hip-poly-fuse), each a process of its
own: the Q8 launch's arguments, T_PROMPT, T_CTX, T_NEW tokens, T_WINDOW (1, or "default" for guesses).  Prints the engine's
own prefill and decode tok/s, and compares every run's logits with the first run's, as bits."""
import json, pathlib, re, shutil, subprocess, sys, tempfile, time
import numpy as np

ROOT = pathlib.Path(r"C:\Users\Joao_\Documents\PolyStrata")
sys.path.insert(0, str(ROOT / "tools" / "hip"))
from _rocm_env import engine_env  # noqa: E402

EXES = {"base": str(ROOT / "build-hip-poly-win" / "strata-poly.exe"), "new": str(ROOT / "build-hip-poly-fuse" / "strata-poly.exe")}
import os
WINDOW = os.environ.get("T_WINDOW", "1")
ORDER = os.environ.get("T_ORDER", "base,new,base").split(",")
PROMPT = os.environ.get("T_PROMPT", "B-2k")
NEW = int(os.environ.get("T_NEW", "48"))
CTX = os.environ.get("T_CTX", "262144")
MODEL = r"C:\Users\Joao_\models\Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"
PROFILE = ROOT / "tmp/hip-launch-q8/q8-ooc679i9/expert-profile.txt"
TOK = pathlib.Path(r"C:\Users\Joao_\models\qwen36-q3-pack\tokenizer\tokenizer.json")
N_VOCAB = 248320


def detok(ids):
    vocab = json.loads((TOK.parent / "vocab.json").read_text(encoding="utf-8"))
    inv = {v: k for k, v in vocab.items()}
    sp = TOK.parent / "token_type.json"
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    byte = {chr(c): b for b, c in zip(bs, cs)}
    out = bytearray()
    for i in ids:
        s = inv.get(i)
        out += bytes(byte.get(ch, 63) for ch in s) if s is not None else b"<%d>" % i
    return out.decode("utf-8", errors="replace")


work = pathlib.Path(tempfile.mkdtemp(prefix="q8-262k-"))
sets = json.loads((ROOT / "bench/results/2026-10-10-hip-poly-vs-llama-vulkan/prompts.json").read_text())["sets"]
prompt = work / "prompt.txt"
prompt.write_text(",".join(map(str, sets[PROMPT])))
runs = []
for k, fuse in enumerate(ORDER):
    EXE = EXES[fuse]
    prof = work / "profile.txt"
    shutil.copyfile(PROFILE, prof)
    lg = work / f"{k}.logits"
    cmd = [EXE, "--model", MODEL, "--threads", "6", "--turn-token", "248045", "--max-context", CTX, "--kv", "q8_0",
           "--expert-profile", str(prof), "--expert-swaps", "0", "--expert-cache-max", "3000", *(["--window", WINDOW] if WINDOW != "default" else []),
           "--tokens", "@" + str(prompt), "--max-new", str(NEW), "--logits-out", str(lg)]
    env = engine_env(EXE)
    env.pop("STRATA_POLY_FUSE", None)
    if os.environ.get("T_STAT"): env["STRATA_POLY_SHARE_STAT"] = "1"
    env["STRATA_POLY_WARM"] = "1"
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.time() - t
    err = p.stderr
    row = {"fuse": fuse, "rc": p.returncode, "wall_s": round(wall, 1)}
    for pat, key in ((r"loaded in ([\d.]+) s", "load_s"), (r"prompt \d+ tokens ([\d.]+) tok/s", "prefill"),
                     (r"generated \d+ tokens ([\d.]+) tok/s", "decode"), (r"experts from VRAM ([\d.]+)%", "vram_hit"),
                     (r"(\d+ of \d+) guessed", "guesses"), (r"expert set ([0-9a-f]+)", "expert_set")):
        m = re.search(pat, err)
        row[key] = m[1] if m else None
    row["fuse_line"] = "STRATA_POLY_FUSE: on" in err
    row["share"] = [l for l in err.splitlines() if l.startswith("share-stat")][-3:]
    row["vram_line"] = next((l for l in err.splitlines() if l.startswith("strata-poly: VRAM MiB")), "")[:200]
    if p.returncode != 0:
        row["err"] = err[-1500:]
    else:
        row["ids"] = [int(x) for x in p.stdout.strip().splitlines()[-1].split(",") if x]
        row["logits"] = np.fromfile(lg, dtype=np.uint32).reshape(-1, N_VOCAB)
    runs.append(row)
    print(json.dumps({k: v for k, v in row.items() if k not in ("ids", "logits")}), flush=True)

ok = [r for r in runs if r["rc"] == 0]
base = ok[0]
for r in ok[1:]:
    steps = min(len(base["logits"]), len(r["logits"]))
    same = sum(bool(np.array_equal(base["logits"][i], r["logits"][i])) for i in range(steps))
    print(f"run fuse={r['fuse']} against the first (fuse={base['fuse']}): {same} of {steps} steps' logits the same bits, "
          f"tokens equal: {r['ids'] == base['ids']}")
print("\nOUTPUT (first run):\n" + detok(base["ids"]))
shutil.rmtree(work, ignore_errors=True)
