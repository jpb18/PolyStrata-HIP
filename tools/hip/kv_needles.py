"""Can strata-poly still find what a long prompt said at its start, middle and end (AMD_KV_STREAMING.md)?

    python tools/hip/kv_needles.py --engine build-hip-poly-win/strata-poly.exe --model M.gguf \
        --tokenizer C:/models/qwen35-pack/tokenizer --lengths 32768,65536,131072 --ctx 131072 \
        --engine-args "--threads 6 --kv-resident 49152" --json out.json

For each length, a document of that many tokens from this repository's docs and sources, with three notes put in
at 5 %, 50 % and 95 % of it: each a made-up code that nothing else in the text gives.  One engine process (the
--serve protocol) is asked about each note in turn, greedy, without thinking, and once for all three.  The
questions share the document, so the first request reads it all and the others read it again only from the last
checkpoint before the question (the time to the first token is that reuse).  A note counts as found when its code
is in the answer.  Also per request: prompt and decode speed, and the process's peak VRAM, RAM and page faults;
and the system's committed memory and page file use before and after.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
from _rocm_env import engine_env  # noqa: E402
from bench_compare import Sampler  # noqa: E402

NEEDLES = [  # depth, the note, the question, the code the answer must hold
    (0.05, "Important note for the reader: the access code of the north archive is 4817-PLUM.",
     "What is the access code of the north archive?", "4817-PLUM"),
    (0.50, "Important note for the reader: the name of the blue team's backup server is HERON-62.",
     "What is the name of the blue team's backup server?", "HERON-62"),
    (0.95, "Important note for the reader: the meeting on Thursday is in room 3B-917.",
     "In which room is the meeting on Thursday?", "3B-917"),
]
HEAD = ("<|im_start|>user\nBelow is a long collection of documents and source files. Somewhere in it are a few "
        "notes for the reader. Read it, then answer the question at the end.\n\n")


def corpus() -> str:
    fs = sorted(ROOT.glob("docs/*.md")) + sorted(ROOT.glob("src/**/*.cpp")) + sorted(ROOT.glob("src/**/*.hpp")) + \
        sorted(ROOT.glob("src/**/*.cu")) + sorted(ROOT.glob("serve/*.py")) + sorted(ROOT.glob("tools/*.py"))
    return "\n\n".join(f.read_text(encoding="utf-8", errors="replace") for f in fs)


SUMMARY = "Summarize in detail what the documents above are about."


def build(tok, length: int, text_ids: list[int], summary: bool = False) -> dict:
    """The prompts for one length: the document, then each question."""
    head = tok.encode(HEAD, parse_special=True)
    def tail(q):
        return tok.encode(f"\n\nQuestion: {q} Answer with the code only.<|im_end|>\n<|im_start|>assistant\n"
                          "<think>\n\n</think>\n\n", parse_special=True)
    notes = [tok.encode("\n\n" + n + "\n\n") for _, n, _, _ in NEEDLES]
    longest_tail = max(len(tail(q)) for _, _, q, _ in NEEDLES + [(0, "", "List the access code of the north "
                       "archive, the name of the blue team's backup server and the room of the meeting on Thursday.", "")])
    n_doc = length - len(head) - longest_tail - sum(len(n) for n in notes)
    if n_doc > len(text_ids):
        raise SystemExit(f"the corpus has {len(text_ids)} tokens, {n_doc} are needed")
    doc = list(text_ids[:n_doc])
    for (depth, _, _, _), n in sorted(zip(NEEDLES, notes), key=lambda x: -x[0][0]):   # deepest first: places hold
        at = int(depth * n_doc)
        doc[at:at] = n
    asks = [(q, [code]) for _, _, q, code in NEEDLES]
    asks.append(("List the access code of the north archive, the name of the blue team's backup server and the "
                 "room of the meeting on Thursday.", [c for _, _, _, c in NEEDLES]))
    if summary:   # a long answer, for the decode speed: no code to find
        asks.append((SUMMARY, []))
    return {"doc": head + doc, "asks": [(q, codes, tail(q)) for q, codes in asks]}


def system_memory() -> dict:
    import psutil
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    out = {"ram_available_mib": vm.available >> 20, "pagefile_used_mib": sw.used >> 20}
    if os.name == "nt":   # the commit charge: what the system has promised, RAM and page file together
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = Status()
        st.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            out["commit_mib"] = (st.ullTotalPageFile - st.ullAvailPageFile) >> 20
            out["commit_limit_mib"] = st.ullTotalPageFile >> 20
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lengths", required=True, help="prompt lengths in tokens, comma-separated")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--summary", type=int, default=0, help="also ask for a summary of N tokens, for the decode speed")
    ap.add_argument("--engine-args", default="")
    ap.add_argument("--json")
    ap.add_argument("--stderr", help="the engine's log (default: dropped)")
    a = ap.parse_args()
    import strata_tokenizer as ST
    tok = ST.Tokenizer.from_dir(a.tokenizer)
    lengths = [int(x) for x in a.lengths.split(",")]
    t = time.time()
    text = corpus()
    chars = int(max(lengths) * 4.2)   # enough text for the longest, a few characters a token
    text_ids = tok.encode(text[:chars])
    print(f"corpus: {len(text_ids)} tokens from {chars} characters in {time.time() - t:.0f} s", flush=True)

    engine = str(pathlib.Path(a.engine).resolve())
    cmd = [engine, "--model", a.model, "--max-context", str(a.ctx), "--serve",
           *shlex.split(a.engine_args, posix=False)]
    before = system_memory()
    err = open(a.stderr, "w", encoding="utf-8") if a.stderr else subprocess.DEVNULL
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                         env=engine_env(engine))
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit("strata-poly ended before READY")
    load_s = time.time() - t0
    import psutil
    proc = psutil.Process(p.pid)
    samp = Sampler(p.pid).start()
    print(f"loaded in {load_s:.1f} s; system before: {before}", flush=True)
    rows = []
    failed = None
    try:
        for length in lengths:
            pr = build(tok, length, text_ids, a.summary > 0)
            for q, codes, tail in pr["asks"]:
                ids = pr["doc"] + tail
                faults0 = proc.memory_info().num_page_faults
                t = time.time()
                first = None
                out = []
                # an answer as long as the context leaves room for (the engine refuses more)
                want = min(a.summary if q == SUMMARY else a.max_new, a.ctx - len(ids))
                if want < 1:
                    raise SystemExit(f"a prompt of {len(ids)} tokens leaves no room in a context of {a.ctx}")
                p.stdin.write(f"GEN {want} {','.join(map(str, ids))}\n")
                p.stdin.flush()
                done = None
                for line in p.stdout:
                    if line.startswith("T "):
                        if first is None:
                            first = time.time() - t
                        out.append(int(line.split()[1]))
                    elif line.startswith("ERR"):
                        raise SystemExit("strata-poly: " + line.strip())
                    elif line.startswith("DONE"):
                        done = line.split()
                        break
                if done is None:
                    raise SystemExit("strata-poly ended during a request")
                answer = tok.decode([i for i in out if i < 248000])
                found = [c for c in codes if c in answer]
                prompt_ms, decode_ms, read = float(done[3]), float(done[4]), int(done[14])
                r = {"length": length, "prompt_tokens": len(ids), "question": q, "codes": codes, "found": found,
                     "answer": answer.strip(), "ttft_s": round(first or 0, 2), "read_tokens": read,
                     "prompt_tok_s": round(read / (prompt_ms / 1000), 1) if prompt_ms else None,
                     "produced": int(done[1]),
                     "decode_tok_s": round((int(done[1]) - 1) / (decode_ms / 1000), 2) if decode_ms else None,
                     "page_faults": proc.memory_info().num_page_faults - faults0,
                     "experts_vram": round(int(done[9]) / max(int(done[10]), 1), 3)}
                rows.append(r)
                print(f"{length:7d} {'OK ' if len(found) == len(codes) else 'MISS'} {len(found)}/{len(codes)} "
                      f"({r['produced']} tokens) "
                      f"TTFT {r['ttft_s']:7.1f} s (read {read:6d}, {r['prompt_tok_s']} tok/s), decode "
                      f"{r['decode_tok_s']} tok/s, page faults {r['page_faults']}: {q[:40]!r} -> {r['answer'][:70]!r}",
                      flush=True)
    except SystemExit as e:   # what was measured is still written
        failed = str(e)
        print(f"FAILED: {failed}", flush=True)
    finally:
        try:
            p.stdin.close()
            p.wait(300)
        except Exception:
            p.kill()
        peak = samp.done()
    after = system_memory()
    print(f"peak of the process: {peak}; system after: {after}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"engine_args": a.engine_args, "ctx": a.ctx, "load_s": load_s,
                                                    "system_before": before, "system_after": after, "peak": peak,
                                                    "failed": failed, "rows": rows}, indent=1), encoding="utf-8")
    return 1 if failed else 0 if all(len(r["found"]) == len(r["codes"]) for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
