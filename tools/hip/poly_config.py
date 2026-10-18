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
    ap.add_argument("--overwrite", action="store_true", help="explicitly replace an existing named config")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for the experts outside VRAM (0: the "
                    "engine's own count, every logical CPU; on a 6-core CPU 6 was faster than 12 here)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--no-switch-on-request", action="store_true",
                    help="leave out switch_on_request (by default a request that names another model set up here "
                         "switches the server to it, so that every model shares the port)")
    ap.add_argument("--build", default=str(ROOT / "build-hip-poly-win"))
    ap.add_argument("--rocm-venv", default=str(ROOT / ".rocm-win"))
    ap.add_argument("--data", help="where the tokenizer folder and the expert profile go (default: beside the GGUF)")
    ap.add_argument("--gguf-py", help="llama.cpp's gguf-py folder, for the tokenizer")
    ap.add_argument("--kv-resident", help="KV streaming (AMD_KV_STREAMING.md): positions of the attention caches kept "
                    "in VRAM, the rest in pinned RAM; auto: all in VRAM when they fit, else as many as fit (default: "
                    "the engine's, all in VRAM)")
    ap.add_argument("--kv-ram-mib", type=int, help="with --kv-resident: the most RAM the rest may pin")
    ap.add_argument("--extra", default="", help="more strata-poly arguments, e.g. \"--vram-reserve-mib 1536\"")
    ap.add_argument("--no-turn-token", action="store_true",
                    help="leave out --turn-token (by default the id of <|im_start|> when the tokenizer has it: a chat's "
                         "next turn then reads only what is new, AMD_PORT_PHASE2.md)")
    a = ap.parse_args()

    gguf = pathlib.Path(a.model).resolve()
    if not gguf.is_file():
        raise SystemExit(f"no such file: {gguf}")
    exe = pathlib.Path(a.build) / ("strata-poly.exe" if os.name == "nt" else "strata-poly")
    if not exe.is_file():
        raise SystemExit(f"no {exe}: build it first (tools\\hip\\build_windows.bat poly)")
    name = a.name or re.sub(r"[^a-z0-9.]+", "-", gguf.stem.lower()).strip("-") + "-hip"
    out = ROOT / f"strata-{name}.json"
    if out.exists() and not a.overwrite:
        raise SystemExit(f"config already exists: {out}; choose another --name or use --overwrite explicitly")
    data = pathlib.Path(a.data) if a.data else gguf.parent / (name + "-pack")
    tok = data / "tokenizer"
    if not (tok / "vocab.json").exists():
        candidates = [a.gguf_py, ROOT / "third_party" / "llama.cpp" / "gguf-py",
                      ROOT / "third_party" / "llama.cpp-3cf0325" / "gguf-py",
                      pathlib.Path(a.build) / "_deps" / "strata_llamacpp-src" / "gguf-py"]
        if os.environ.get("STRATA_GGML_DIR"):
            candidates.insert(1, pathlib.Path(os.environ["STRATA_GGML_DIR"]) / "gguf-py")
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
    if a.kv_resident:
        args += ["--kv-resident", a.kv_resident]
    if a.kv_ram_mib is not None:
        args += ["--kv-ram-mib", str(a.kv_ram_mib)]
    args += a.extra.split()
    if not a.no_turn_token and "--turn-token" not in args:
        try:
            turn = json.loads((tok / "vocab.json").read_text(encoding="utf-8")).get("<|im_start|>")
        except (OSError, ValueError):
            turn = None
        if isinstance(turn, int):
            args += ["--turn-token", str(turn)]
    cfg = {"exe": str(exe), "args": args, "cwd": str(ROOT), "tokenizer": str(tok), "model_name": name,
           "display_name": f"{gguf.stem} (HIP, experimental)", "log": str(ROOT / f"strata-{name}.log"),
           "lib_dirs": lib_dirs, "backend": "hip", "port": a.port}
    if not a.no_switch_on_request:
        cfg["switch_on_request"] = True
    with out.open("w" if a.overwrite else "x", encoding="utf-8") as stream:
        stream.write(json.dumps(cfg, indent=1))
    print(f"written: {out}\nmodel name: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
