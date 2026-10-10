"""The environment strata-poly's HIP build runs in on Windows: the program's folder and the ROCm SDK's libraries
(<repo>/.rocm-win, as tools/hip/build_windows.bat installs it) ahead on the PATH.  Elsewhere, or without that
folder, the environment as it is."""
from __future__ import annotations

import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]


def engine_env(engine: str, rocm_venv: str | None = None) -> dict:
    env = dict(os.environ)
    if os.name != "nt":
        return env
    dirs = [str(pathlib.Path(engine).resolve().parent)]
    sdk = pathlib.Path(rocm_venv or ROOT / ".rocm-win") / "Scripts" / "rocm-sdk.exe"
    if sdk.is_file():
        root = subprocess.run([str(sdk), "path", "--root"], capture_output=True, text=True)
        if root.returncode == 0 and root.stdout.strip():
            dirs.append(str(pathlib.Path(root.stdout.strip()) / "bin"))
    env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
    return env
