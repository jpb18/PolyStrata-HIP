"""A run config for strata-poly built with HIP (STRATA_POLY_HIP, AMD_PORT_IMPLEMENTATION.md), beside setup.py, so
that the server, its web page, `polystrata.py --list-models` and `polystrata.py benchmark` take the model.  Setup
does not install strata-poly for AMD cards yet; this writes what it would.

    python tools/hip/poly_config.py --model C:/models/Qwen3.5-35B-A3B-Q3_K_M.gguf [--context 49152] [--threads 6]

It writes the model's tokenizer folder from the GGUF when there is none (tools/strata_tokenizer.py, with llama.cpp's
gguf-py: --gguf-py, else third_party/llama.cpp/gguf-py), and
strata-<name>.json in the repository's folder, with the engine from build-hip-poly-win and the ROCm SDK's libraries
(.rocm-win) on its PATH.  Then:

    python polystrata.py --model <name>              (a chat in the terminal; the server starts by itself)
    python -m serve.server --engine strata --config strata-<name>.json --port 8090   (the API and the web page)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the GGUF (the first file of a split one)")
    ap.add_argument("--name", help="the model's name in the server (default: from the file's name, with -hip)")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for the experts outside VRAM (0: the "
                    "engine's own count, every logical CPU; on a 6-core CPU 6 was faster than 12 here)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--build", default=str(ROOT / "build-hip-poly-win"))
    ap.add_argument("--rocm-venv", default=str(ROOT / ".rocm-win"))
    ap.add_argument("--data", help="where the tokenizer folder and the expert profile go (default: beside the GGUF)")
    ap.add_argument("--gguf-py", help="llama.cpp's gguf-py folder, for the tokenizer")
    ap.add_argument("--extra", default="", help="more strata-poly arguments, e.g. \"--vram-reserve-mib 1536\"")
    a = ap.parse_args()

    gguf = pathlib.Path(a.model).resolve()
    if not gguf.is_file():
        raise SystemExit(f"no such file: {gguf}")
    exe = pathlib.Path(a.build) / ("strata-poly.exe" if os.name == "nt" else "strata-poly")
    if not exe.is_file():
        raise SystemExit(f"no {exe}: build it first (tools\\hip\\build_windows.bat poly)")
    name = a.name or re.sub(r"[^a-z0-9.]+", "-", gguf.stem.lower()).strip("-") + "-hip"
    data = pathlib.Path(a.data) if a.data else gguf.parent / (name + "-pack")
    tok = data / "tokenizer"
    if not (tok / "vocab.json").exists():
        candidates = [a.gguf_py, ROOT / "third_party" / "llama.cpp" / "gguf-py"]
        gp = next((pathlib.Path(c) for c in candidates if c and (pathlib.Path(c) / "gguf").is_dir()), None)
        if gp is None:
            raise SystemExit("no gguf-py for the tokenizer: give --gguf-py <llama.cpp>/gguf-py")
        subprocess.run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(gguf),
                        "--out", str(data)], check=True, env=dict(os.environ, STRATA_GGUF_PY=str(gp)))
    rocm_sdk = pathlib.Path(a.rocm_venv) / "Scripts" / "rocm-sdk.exe"
    lib_dirs = []
    if rocm_sdk.is_file():
        root = subprocess.run([str(rocm_sdk), "path", "--root"], capture_output=True, text=True, check=True)
        lib_dirs.append(str(pathlib.Path(root.stdout.strip()) / "bin"))
    args = ["--model", str(gguf), "--max-context", str(a.context), "--expert-profile",
            str(data / "expert-profile.txt")]
    if a.threads > 0:
        args += ["--threads", str(a.threads)]
    args += a.extra.split()
    cfg = {"exe": str(exe), "args": args, "cwd": str(ROOT), "tokenizer": str(tok), "model_name": name,
           "display_name": f"{gguf.stem} (HIP, experimental)", "log": str(ROOT / f"strata-{name}.log"),
           "lib_dirs": lib_dirs, "backend": "hip", "port": a.port}
    out = ROOT / f"strata-{name}.json"
    out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"written: {out}\nmodel name: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
