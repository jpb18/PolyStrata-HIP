"""Every step's logits, bit for bit: the engine before the attention rescale skip and the experts' shared reads
(--ref-exe), against this build (--exe) with STRATA_POLY_EXPERT_SHARE=0 and with it on.  A decode (--window 1) and
windows of guessed tokens (--force-window, the decode's tokens) so that tokens of a window pick the same experts."""
import argparse, json, pathlib, shutil, subprocess, sys, tempfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools" / "hip"))
from _rocm_env import engine_env  # noqa: E402

N_VOCAB = 248320


def run(exe, model, profile, prompt_file, extra, share, work, tag, ctx, kv):
    prof = work / f"{tag}-profile.txt"
    shutil.copyfile(profile, prof)
    out = work / f"{tag}.logits"
    cmd = [exe, "--model", model, "--max-context", ctx, "--threads", "6", "--expert-profile", str(prof),
           "--expert-swaps", "0", "--expert-cache-max", "4000", "--tokens", "@" + str(prompt_file),
           "--logits-out", str(out), *extra]
    if kv:
        cmd += ["--kv", kv]
    env = engine_env(exe)
    env["STRATA_POLY_WARM"] = "1"
    if share is not None:
        env["STRATA_POLY_EXPERT_SHARE"] = share
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-3000:])
        raise SystemExit(f"{tag}: exit {p.returncode}")
    ids = [int(x) for x in p.stdout.strip().splitlines()[-1].split(",") if x]
    return {"ids": ids, "logits": np.fromfile(out, dtype=np.uint32).reshape(-1, N_VOCAB)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--ref-exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--prompt", default="B-2k")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--windows", default="4,3,2,4/2,4/1")
    ap.add_argument("--max-context", default="49152")
    ap.add_argument("--kv", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    work = pathlib.Path(tempfile.mkdtemp(prefix="share-parity-"))
    sets = json.loads((ROOT / "bench/results/2026-10-10-hip-poly-vs-llama-vulkan/prompts.json").read_text())["sets"]
    prompt = work / "prompt.txt"
    prompt.write_text(",".join(map(str, sets[a.prompt])))
    res = {"prompt": a.prompt, "tokens": len(sets[a.prompt])}

    def cmp(x, y):
        steps = min(len(x["logits"]), len(y["logits"]))
        same = sum(bool(np.array_equal(x["logits"][i], y["logits"][i])) for i in range(steps))
        return {"steps": steps, "same_bits": same, "ids_equal": x["ids"] == y["ids"]}

    def go(exe, extra, share, tag):
        return run(exe, a.model, a.profile, prompt, extra, share, work, tag, a.max_context, a.kv)

    dec = ["--max-new", str(a.max_new), "--window", "1"]
    ref = go(a.ref_exe, dec, None, "dec-ref")
    res["decode_ref_vs_new"] = cmp(ref, go(a.exe, dec, None, "dec-new"))
    force = work / "force.txt"
    force.write_text(",".join(map(str, ref["ids"])))
    win = ["--force-tokens", "@" + str(force), "--force-window", a.windows, "--window", "8"]
    wref = go(a.ref_exe, win, None, "win-ref")
    res["window_ref_vs_new_share_off"] = cmp(wref, go(a.exe, win, "0", "win-off"))
    res["window_ref_vs_new"] = cmp(wref, go(a.exe, win, None, "win-new"))
    res["window_ref_vs_decode_ref"] = cmp(wref, ref)
    print(json.dumps(res, indent=1))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    shutil.rmtree(work, ignore_errors=True)


main()
