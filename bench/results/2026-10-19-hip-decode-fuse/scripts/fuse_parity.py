"""STRATA_POLY_FUSE=0 against =1 (and optionally another build): every step's logits, bit for bit.

Mode "decode": --tokens PROMPT --max-new N --window 1 (a token a call, routed on the card).
Mode "window": --force-tokens (the decode's tokens) --force-window SIZES: windows of guessed tokens.
Each run gets its own copy of the expert profile (the engine writes it back on exit), --expert-swaps 0 and a fixed
--expert-cache-max, so every run caches the same experts.
"""
import argparse, json, pathlib, shutil, subprocess, sys, tempfile, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools" / "hip"))
from _rocm_env import engine_env  # noqa: E402

N_VOCAB = 248320


def run(exe, model, profile, prompt_file, extra, fuse, work, tag):
    prof = work / f"{tag}-profile.txt"
    shutil.copyfile(profile, prof)
    out = work / f"{tag}.logits"
    cmd = [exe, "--model", model, "--max-context", "49152", "--threads", "6", "--expert-profile", str(prof),
           "--expert-swaps", "0", "--expert-cache-max", "4000", "--tokens", "@" + str(prompt_file),
           "--logits-out", str(out), *extra]
    env = engine_env(exe)
    env["STRATA_POLY_FUSE"] = "1" if fuse else "0"
    env["STRATA_POLY_WARM"] = "1"
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-3000:])
        raise SystemExit(f"{tag}: exit {p.returncode}")
    ids = [int(x) for x in p.stdout.strip().splitlines()[-1].split(",") if x]
    info = [l for l in p.stderr.splitlines() if "FUSE" in l or l.startswith(("prompt ", "steps", "forced"))]
    return {"ids": ids, "logits": np.fromfile(out, dtype=np.uint32).reshape(-1, N_VOCAB), "info": info,
            "s": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--ref-exe")
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--prompt", default="B-2k")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--windows", default="4,3,2,4/2,4/1")
    ap.add_argument("--json")
    a = ap.parse_args()
    work = pathlib.Path(tempfile.mkdtemp(prefix="fuse-parity-"))
    sets = json.loads((ROOT / "bench/results/2026-10-10-hip-poly-vs-llama-vulkan/prompts.json").read_text())["sets"]
    prompt = work / "prompt.txt"
    prompt.write_text(",".join(map(str, sets[a.prompt])))
    res = {"prompt": a.prompt, "tokens": len(sets[a.prompt])}

    def cmp(x, y):
        steps = min(len(x["logits"]), len(y["logits"]))
        same = sum(bool(np.array_equal(x["logits"][i], y["logits"][i])) for i in range(steps))
        return {"steps": steps, "same_bits": same, "ids_equal": x["ids"] == y["ids"]}

    dec = ["--max-new", str(a.max_new), "--window", "1"]
    off = run(a.exe, a.model, a.profile, prompt, dec, False, work, "dec-off")
    on = run(a.exe, a.model, a.profile, prompt, dec, True, work, "dec-on")
    res["decode_off_vs_on"] = cmp(off, on)
    res["decode_info_on"] = on["info"]
    if a.ref_exe:
        ref = run(a.ref_exe, a.model, a.profile, prompt, dec, False, work, "dec-ref")
        res["decode_ref_vs_off"] = cmp(ref, off)
    force = work / "force.txt"
    force.write_text(",".join(map(str, off["ids"])))
    win = ["--force-tokens", "@" + str(force), "--force-window", a.windows, "--window", "8"]
    woff = run(a.exe, a.model, a.profile, prompt, win, False, work, "win-off")
    won = run(a.exe, a.model, a.profile, prompt, win, True, work, "win-on")
    res["window_off_vs_on"] = cmp(woff, won)
    # a window's logits against the decode's (what Engine::window() promises), as a check of the check
    res["window_off_vs_decode_off"] = cmp(woff, off)
    print(json.dumps(res, indent=1))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
