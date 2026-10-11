"""The run config of one of START-POLY-HIP.bat's profiles (AMD_KV_STREAMING.md, section 12), written by
tools/hip/poly_config.py the first time and reused after that.

    python tools/hip/poly_profile.py 128k [--gguf C:/models/X.gguf | --base qwen3.6]
    python tools/hip/poly_profile.py --list

Profiles: 48k (context 49,152: the fastest for chats), 128k and 262k (the KV cache wholly in VRAM: the fastest for
long prompts on a 16 GB card), 262k-stream (context 262,144 with positions from 49,152 on in RAM: for a card whose
VRAM other programs hold).  The model is --gguf, else the one of a run config strata-<name>-hip.json already here
(with several, --base picks the one whose name has that text in it, else the first by name); the profiles share its
tokenizer folder and expert profile.  Prints the run config's file name.  --list prints the models set up here, a line
each: name|display name|GGUF.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILES = {   # name suffix, context, extra strata-poly arguments
    "48k": ("", 49152, []),
    "128k": ("-128k", 131072, []),
    "262k": ("-262k", 262144, []),
    "262k-stream": ("-262k-stream", 262144, ["--kv-resident", "49152"]),
}
SUFFIX = re.compile(r"-(128k|262k|262k-stream)$")


def arg(args: list[str], key: str) -> str | None:
    return args[args.index(key) + 1] if key in args and args.index(key) + 1 < len(args) else None


def cores() -> int:
    if os.name == "nt":
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"],
                             capture_output=True, text=True).stdout.strip()
        if out.isdigit():
            return int(out)
    return max(1, (os.cpu_count() or 2) // 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", nargs="?", choices=list(PROFILES))
    ap.add_argument("--gguf", help="the model (default: the one already set up here)")
    ap.add_argument("--base", help="of the models set up here, the one whose name has this text in it")
    ap.add_argument("--list", action="store_true", help="print the models set up here and stop")
    a = ap.parse_args()

    # the models' base configs: those without a profile's suffix
    bases = []
    for f in sorted(ROOT.glob("strata-*-hip*.json")):
        name = f.stem[len("strata-"):]
        if SUFFIX.search(name):
            continue
        bases.append((name, json.loads(f.read_text(encoding="utf-8"))))
    if a.list:
        for name, cfg in bases:
            print(f"{name}|{cfg.get('display_name', name)}|{arg(cfg.get('args', []), '--model') or ''}")
        return 0
    if not a.profile:
        ap.error("a profile is needed (or --list)")
    suffix, ctx, extra = PROFILES[a.profile]
    base = None
    for name, cfg in bases:
        model = arg(cfg.get("args", []), "--model")
        if a.gguf and model and pathlib.Path(model).resolve() != pathlib.Path(a.gguf).resolve():
            continue
        if not a.gguf and a.base and a.base.lower() not in name.lower():
            continue
        base = (name, cfg)
        break
    if a.base and not a.gguf and base is None:
        print(f"no model set up here has {a.base!r} in its name", file=sys.stderr)
        return 2
    if a.gguf:
        gguf = str(pathlib.Path(a.gguf).resolve())
        base_name = base[0] if base else re.sub(r"[^a-z0-9.]+", "-", pathlib.Path(gguf).stem.lower()).strip("-") + "-hip"
    elif base:
        gguf, base_name = arg(base[1]["args"], "--model"), base[0]
    else:
        print("no model is set up yet: give its .gguf", file=sys.stderr)
        return 2

    name = base_name + suffix
    out = ROOT / f"strata-{name}.json"
    if out.exists():
        args = json.loads(out.read_text(encoding="utf-8")).get("args", [])
        if arg(args, "--max-context") == str(ctx) and arg(args, "--model") and \
                pathlib.Path(arg(args, "--model")).resolve() == pathlib.Path(gguf).resolve():
            print(out.name)
            return 0

    cmd = [sys.executable, str(ROOT / "tools" / "hip" / "poly_config.py"), "--model", gguf, "--name", name,
           "--context", str(ctx)]
    threads = arg(base[1]["args"], "--threads") if base else None
    cmd += ["--threads", threads or str(cores())]
    if base and base[1].get("port"):   # a model's profiles share its port (Qwen3.5 8090, Qwen3.6 8091 here)
        cmd += ["--port", str(base[1]["port"])]
    # every profile shares the base config's tokenizer folder and expert profile
    if base and base[1].get("tokenizer"):
        cmd += ["--data", str(pathlib.Path(base[1]["tokenizer"]).parent)]
    else:
        cmd += ["--data", str(pathlib.Path(gguf).parent / (base_name + "-pack"))]
    cmd += extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not out.exists():
        sys.stderr.write(p.stdout + p.stderr)
        return 1
    print(out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
