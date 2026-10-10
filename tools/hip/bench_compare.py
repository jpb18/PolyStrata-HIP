"""strata-poly against llama.cpp's llama-server (Vulkan or HIP builds) on the same GGUF, the same token ids and the
same context: prefill, decode, time to the first token, load time, VRAM and RAM, in a process just loaded and in
one that has answered already.

    python tools/hip/bench_compare.py prepare --llama-server <llama-server.exe> --model M.gguf --out prompts.json
    python tools/hip/bench_compare.py run --prompts prompts.json --label poly --engine poly \
        --exe build-hip-poly-win/strata-poly.exe --model M.gguf --ctx 49152 --args "--threads 6" --json out.json
    python tools/hip/bench_compare.py run --prompts prompts.json --label vulkan-ncmoe8 --engine llama \
        --exe <llama-server.exe> --model M.gguf --ctx 49152 --args "-ngl 99 --n-cpu-moe 8 -t 6 -fa on" --json out.json

`prepare` makes two sets of prompts of about 2K, 8K, 31K and 45K tokens (--sizes) from this repository's documents,
none sharing text with another, each a user message in the model's chat template, as token ids from llama-server's
own tokenizer (/apply-template, /tokenize): set A and set B.

`run` starts the engine, times its load (to strata-poly's READY or llama-server's /health), and sends, greedy and
`--max-new` tokens each: set A ("cold": the first prompts the process reads), set B ("warm": new prompts to a process
that has answered), and B's two shortest again ("cache": what each engine keeps of a prompt it has read).  For each
request: the time to the first token on this side of the pipe, the engine's own prompt and decode times, and the
tokens it wrote.  While it runs, the engine process's VRAM (Windows' GPU Process Memory counters: dedicated, and
shared, which is system memory the card uses when the dedicated is full) and RAM (private bytes and working set,
psutil) are sampled every second or so; the highest are reported.  Results are appended to --json.

What each engine does that the other does not is in the report this tool was written for (AMD_PORT_PHASE2.md):
strata-poly guesses tokens from repeats unless --window 1, and keeps an expert cache that follows the answer;
llama-server places experts once (--n-cpu-moe) and has no guesses unless told.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _rocm_env import engine_env  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ASK = "Read the following text, then list its five most important points, one line each.\n\n"


def post(url: str, body: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(url: str, proc: subprocess.Popen, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        if proc.poll() is not None:
            raise SystemExit(f"llama-server ended (exit {proc.returncode}) before it was ready")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.25)
    raise SystemExit("llama-server was not ready in time")


# ------------------------------------------------------------------------------------------------ prepare
def corpus() -> str:
    files = sorted((ROOT / "docs").glob("*.md")) + sorted((ROOT / "src" / "poly").glob("*.*")) + \
        sorted((ROOT / "serve").glob("*.py"))
    return "\n\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)


def prepare(a) -> int:
    port = 8189
    url = f"http://127.0.0.1:{port}"
    cmd = [a.llama_server, "-m", a.model, "--port", str(port), "-c", "4096", "-ngl", "0", "--no-webui"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_health(url, proc, 600)
        text = corpus()
        at = 0
        sets = {}
        for tag in ("A", "B"):
            for size in a.sizes:
                # grow the slice until its tokens reach the size, then cut to it
                chars = int(size * 3.2)
                while True:
                    chunk = text[at:at + chars]
                    if at + chars > len(text):
                        raise SystemExit(f"the corpus has too little text for set {tag} at {size} tokens")
                    templ = post(url + "/apply-template",
                                 {"messages": [{"role": "user", "content": ASK + chunk}]})["prompt"]
                    ids = post(url + "/tokenize", {"content": templ, "add_special": True,
                                                    "parse_special": True})["tokens"]
                    if len(ids) >= size:
                        break
                    chars = int(chars * size / max(len(ids), 1) * 1.02) + 64
                # cut the text by the same share, so the template's end stays
                while len(ids) > size * 1.01:
                    chars = int(chars * 0.995)
                    chunk = text[at:at + chars]
                    templ = post(url + "/apply-template",
                                 {"messages": [{"role": "user", "content": ASK + chunk}]})["prompt"]
                    ids = post(url + "/tokenize", {"content": templ, "add_special": True,
                                                    "parse_special": True})["tokens"]
                sets[f"{tag}-{size // 1000}k"] = ids
                print(f"{tag}-{size // 1000}k: {len(ids)} tokens from characters {at} to {at + chars}", flush=True)
                at += chars
    finally:
        proc.terminate()
        proc.wait(60)
    pathlib.Path(a.out).write_text(json.dumps({"model": a.model, "sets": sets}), encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------------ memory
def vram_in_use_mib() -> float | None:
    """Windows: the dedicated VRAM all processes hold, before the engine starts (the desktop, a browser): what the
    engine has not, and an engine that does not fit what is left spills into shared memory."""
    if os.name != "nt":
        return None
    ps = ("((Get-Counter -ErrorAction SilentlyContinue '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples"
          "|Measure-Object CookedValue -Sum).Sum")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
    try:
        return round(float(out) / 2**20)
    except ValueError:
        return None


class Sampler:
    """The highest VRAM (dedicated, shared) and RAM (private, working set) of one process, sampled."""

    def __init__(self, pid: int):
        self.pid = pid
        self.peak = {"vram_dedicated_mib": 0.0, "vram_shared_mib": 0.0, "ram_private_mib": 0.0, "ram_ws_mib": 0.0}
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.loop, daemon=True)

    def gpu(self) -> tuple[float, float]:
        ps = (f"$s=(Get-Counter -ErrorAction SilentlyContinue '\\GPU Process Memory(pid_{self.pid}_*)\\Dedicated Usage',"
              f"'\\GPU Process Memory(pid_{self.pid}_*)\\Shared Usage').CounterSamples;"
              "$d=($s|?{$_.Path -like '*dedicated*'}|Measure-Object CookedValue -Sum).Sum;"
              "$h=($s|?{$_.Path -like '*shared*'}|Measure-Object CookedValue -Sum).Sum;\"$d $h\"")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
        try:
            d, h = out.split()
            return float(d) / 2**20, float(h) / 2**20
        except ValueError:
            return 0.0, 0.0

    def loop(self) -> None:
        import psutil
        p = psutil.Process(self.pid)
        while not self.stop.is_set():
            try:
                m = p.memory_info()
                self.peak["ram_private_mib"] = max(self.peak["ram_private_mib"], getattr(m, "private", 0) / 2**20)
                self.peak["ram_ws_mib"] = max(self.peak["ram_ws_mib"], m.rss / 2**20)
            except Exception:
                return
            if os.name == "nt":
                d, h = self.gpu()
                self.peak["vram_dedicated_mib"] = max(self.peak["vram_dedicated_mib"], d)
                self.peak["vram_shared_mib"] = max(self.peak["vram_shared_mib"], h)
            self.stop.wait(0.5)

    def start(self):
        self.t.start()
        return self

    def done(self) -> dict:
        self.stop.set()
        self.t.join(30)
        return {k: round(v) for k, v in self.peak.items()}


# ------------------------------------------------------------------------------------------------ engines
class Poly:
    def __init__(self, exe, model, ctx, args, err=subprocess.DEVNULL):
        self.err = err
        self.cmd = [exe, "--model", model, "--max-context", str(ctx), "--serve", *args]
        self.env = engine_env(exe)

    def start(self) -> float:
        t = time.time()
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err,
                                  text=True, bufsize=1, env=self.env)
        for line in self.p.stdout:
            if line.startswith("READY"):
                return time.time() - t
        raise SystemExit("strata-poly ended before READY")

    def gen(self, ids, n) -> dict:
        t = time.time()
        first = None
        self.p.stdin.write(f"GEN {n} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        for line in self.p.stdout:
            if line.startswith("T ") and first is None:
                first = time.time() - t
            if line.startswith("ERR"):
                raise SystemExit("strata-poly: " + line.strip())
            if line.startswith("DONE"):
                f = line.split()
                produced, prompt, prompt_ms, decode_ms, read = int(f[1]), int(f[2]), float(f[3]), float(f[4]), int(f[14])
                # DONE produced n prompt_ms decode_ms finish guessed_right guessed resume hits lookups 0 0 0.0 read 0
                return {"ttft_s": round(first or 0, 3), "wall_s": round(time.time() - t, 3), "produced": produced,
                        "prompt_tokens": prompt, "read_tokens": read, "prompt_ms": prompt_ms, "decode_ms": decode_ms,
                        "finish": f[5], "guessed_right": int(f[6]), "guessed": int(f[7]),
                        "vram_expert_hits": round(int(f[9]) / max(int(f[10]), 1), 3)}
        raise SystemExit("strata-poly ended during a request")

    def stop(self):
        try:
            self.p.stdin.close()
            self.p.wait(120)
        except Exception:
            self.p.kill()


class Llama:
    def __init__(self, exe, model, ctx, args, err=subprocess.DEVNULL):
        self.err = err
        self.port = 8190
        self.url = f"http://127.0.0.1:{self.port}"
        self.cmd = [exe, "-m", model, "--port", str(self.port), "-c", str(ctx), "--no-webui", "-np", "1", *args]
        self.env = engine_env(exe)

    def start(self) -> float:
        t = time.time()
        self.p = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=self.err, env=self.env)
        wait_health(self.url, self.p, 900)
        return time.time() - t

    def gen(self, ids, n) -> dict:
        body = {"prompt": ids, "n_predict": n, "temperature": 0.0, "top_k": 1, "stream": True, "cache_prompt": True}
        req = urllib.request.Request(self.url + "/completion", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = time.time()
        first = None
        timings = {}
        stop = ""
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                d = json.loads(line[6:])
                if first is None and (d.get("content") or d.get("tokens")):
                    first = time.time() - t
                if d.get("stop"):
                    timings = d.get("timings", {})
                    stop = d.get("stop_type", "")
        return {"ttft_s": round(first or 0, 3), "wall_s": round(time.time() - t, 3),
                "produced": timings.get("predicted_n", 0), "prompt_tokens": len(ids),
                "read_tokens": timings.get("prompt_n", 0), "prompt_ms": timings.get("prompt_ms", 0.0),
                "decode_ms": timings.get("predicted_ms", 0.0), "finish": stop}

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(60)
        except subprocess.TimeoutExpired:
            self.p.kill()


def run(a) -> int:
    pr = json.loads(pathlib.Path(a.prompts).read_text())["sets"]
    args = shlex.split(a.args, posix=False)
    try:   # what the rest of the PC left: an engine short of RAM pages, and that is not its speed
        import psutil
        ram_free = round(psutil.virtual_memory().available / 2**20)
    except Exception:
        ram_free = None
    vram_others = vram_in_use_mib()
    err = open(a.stderr, "w", encoding="utf-8") if a.stderr else subprocess.DEVNULL
    eng = (Poly if a.engine == "poly" else Llama)(a.exe, a.model, a.ctx, args, err)
    load_s = eng.start()
    pid = eng.p.pid
    samp = Sampler(pid).start()
    print(f"[{a.label}] loaded in {load_s:.1f} s (RAM free before: {ram_free} MiB, VRAM held by other processes: {vram_others} MiB)", flush=True)
    rows = []
    try:
        plan = [("cold", k) for k in pr if k.startswith("A-")] + [("warm", k) for k in pr if k.startswith("B-")]
        plan += [("cache", k) for k in [k for k in pr if k.startswith("B-")][:2]]
        if a.only:   # a sweep's few prompts, e.g. "A-2k,A-8k"
            plan = [(ph, k) for ph, k in plan if k in a.only.split(",") and ph != "cache"]
        for phase, k in plan:
            r = eng.gen(pr[k], a.max_new)
            r.update(phase=phase, prompt=k)
            r["prefill_tok_s"] = round(r["read_tokens"] / (r["prompt_ms"] / 1000), 1) if r["prompt_ms"] else None
            r["decode_tok_s"] = round(r["produced"] / (r["decode_ms"] / 1000), 2) if r["decode_ms"] else None
            rows.append(r)
            print(f"[{a.label}] {phase:5s} {k:7s} {r['prompt_tokens']:6d} tok (read {r['read_tokens']:6d}): "
                  f"TTFT {r['ttft_s']:7.2f} s, prefill {r['prefill_tok_s']} tok/s, decode {r['decode_tok_s']} tok/s "
                  f"({r['produced']} tok, {r['finish']})", flush=True)
    finally:
        mem = samp.done()
        eng.stop()
    print(f"[{a.label}] peak: {mem}", flush=True)
    out = pathlib.Path(a.json)
    allr = json.loads(out.read_text()) if out.is_file() else []
    allr.append({"label": a.label, "engine": a.engine, "exe": a.exe, "args": a.args, "ctx": a.ctx,
                 "max_new": a.max_new, "load_s": round(load_s, 2), "memory": mem,
                 "ram_free_mib_before": ram_free, "vram_others_mib_before": vram_others, "rows": rows,
                 "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    out.write_text(json.dumps(allr, indent=1), encoding="utf-8")
    return 0


def summary(a) -> int:
    """The runs of a --json file by configuration (the label without its "-rN"): for each phase and prompt the
    median of the runs and their range, then load time and memory."""
    import statistics
    runs = json.loads(pathlib.Path(a.json).read_text())
    groups: dict[str, list] = {}
    for r in runs:
        groups.setdefault(re.sub(r"-r\d+$", "", r["label"]), []).append(r)

    def cell(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-"
        m = statistics.median(vals)
        fmt = (lambda v: f"{v:.2f}") if m < 10 else (lambda v: f"{v:.1f}") if m < 100 else (lambda v: f"{v:.0f}")
        if all(isinstance(v, int) for v in vals):
            fmt = lambda v: f"{v:.0f}"   # noqa: E731 - token counts
        return fmt(m) if len(vals) == 1 else f"{fmt(m)} ({fmt(min(vals))}-{fmt(max(vals))})"

    for name, rs in groups.items():
        print(f"\n### {name} ({len(rs)} runs): {rs[0]['engine']} {rs[0]['args']}\n")
        print("| Phase | Prompt | Tokens | Read | TTFT s | Prefill tok/s | Decode tok/s |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        keys = []
        for row in rs[0]["rows"]:
            keys.append((row["phase"], row["prompt"]))
        for ph, pr in keys:
            rows = [x for r in rs for x in r["rows"] if x["phase"] == ph and x["prompt"] == pr]
            print(f"| {ph} | {pr} | {rows[0]['prompt_tokens']} | {cell([x['read_tokens'] for x in rows])} | "
                  f"{cell([x['ttft_s'] for x in rows])} | {cell([x['prefill_tok_s'] for x in rows])} | "
                  f"{cell([x['decode_tok_s'] for x in rows])} |")
        mem = {k: cell([r["memory"].get(k) for r in rs]) for k in rs[0]["memory"]}
        print(f"\nload {cell([r['load_s'] for r in rs])} s; peak VRAM dedicated {mem.get('vram_dedicated_mib')} MiB, "
              f"shared {mem.get('vram_shared_mib')} MiB; RAM private {mem.get('ram_private_mib')} MiB, working set "
              f"{mem.get('ram_ws_mib')} MiB; before each run RAM free "
              f"{cell([r.get('ram_free_mib_before') for r in rs])} MiB, VRAM held by other processes "
              f"{cell([r.get('vram_others_mib_before') for r in rs])} MiB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--llama-server", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[2048, 8192, 31000, 45000])
    r = sub.add_parser("run")
    r.add_argument("--prompts", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--engine", choices=["poly", "llama"], required=True)
    r.add_argument("--exe", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--ctx", type=int, default=49152)
    r.add_argument("--max-new", type=int, default=128)
    r.add_argument("--args", default="")
    r.add_argument("--json", required=True)
    r.add_argument("--stderr", help="the engine's log goes here (default: dropped)")
    r.add_argument("--only", help="only these prompts (comma-separated names, no cache pass): for a sweep")
    m = sub.add_parser("summary")
    m.add_argument("--json", required=True)
    a = ap.parse_args()
    return prepare(a) if a.cmd == "prepare" else run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    sys.exit(main())
