"""The tables of AMD_KV_STREAMING.md from tools/hip/kv_needles.py's JSON files: one run a file, named
<config>-<repetition>.json; the median over a configuration's repetitions, the range in brackets.

    python tools/hip/kv_summary.py bench/results/2026-10-11-hip-poly-kv-streaming/matrix/*.json
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import statistics
import sys


def cell(vals, fmt="{:.1f}"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = statistics.median(vals)
    if len(vals) == 1:
        return fmt.format(m)
    return f"{fmt.format(m)} ({fmt.format(min(vals))}-{fmt.format(max(vals))})"


def main() -> int:
    runs = collections.defaultdict(list)
    for f in sys.argv[1:]:
        name = pathlib.Path(f).stem
        cfg = re.sub(r"-\d+$", "", name)
        runs[cfg].append(json.loads(pathlib.Path(f).read_text(encoding="utf-8")))
    print("| Configuration | Prompt | Found | TTFT s | Prefill tok/s | Decode tok/s (128) | Reuse TTFT s | Reps |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for cfg in sorted(runs):
        # per length, each run's requests in order: the first read the document, the others reused it
        by_len = collections.defaultdict(list)
        for r in runs[cfg]:
            per = collections.defaultdict(list)
            for row in r["rows"]:
                per[row["length"]].append(row)
            for length, rows in per.items():
                by_len[length].append(rows)
        for length in sorted(by_len):
            reps = by_len[length]
            n_rep = len(reps)
            first = [rows[0] for rows in reps]
            later = [row for rows in reps for row in rows[1:]]
            asks = [row for rows in reps for row in rows if row["codes"]]
            found = sum(len(row["found"]) for row in asks)
            want = sum(len(row["codes"]) for row in asks)
            summ = [row["decode_tok_s"] for row in later if not row["codes"]]
            reuse = [row["ttft_s"] for row in later]
            print(f"| {cfg} | {length} | {found}/{want} | {cell([r['ttft_s'] for r in first])} | "
                  f"{cell([r['prompt_tok_s'] for r in first], '{:.0f}')} | {cell(summ, '{:.2f}')} | "
                  f"{cell(reuse)} | {n_rep} |")
    print()
    print("| Configuration | Load s | Peak VRAM dedicated / shared MiB | Peak RAM private / working set MiB | "
          "RAM available before / after MiB | Commit before / after MiB | Page file before / after MiB |")
    print("| --- | ---: | --- | --- | --- | --- | --- |")
    for cfg in sorted(runs):
        rs = runs[cfg]
        pk = lambda k: cell([r["peak"][k] for r in rs], "{:.0f}")
        sb = lambda k: cell([r["system_before"].get(k) for r in rs], "{:.0f}")
        sa = lambda k: cell([r["system_after"].get(k) for r in rs], "{:.0f}")
        print(f"| {cfg} | {cell([r['load_s'] for r in rs])} | {pk('vram_dedicated_mib')} / {pk('vram_shared_mib')} | "
              f"{pk('ram_private_mib')} / {pk('ram_ws_mib')} | {sb('ram_available_mib')} / {sa('ram_available_mib')} | "
              f"{sb('commit_mib')} / {sa('commit_mib')} | {sb('pagefile_used_mib')} / {sa('pagefile_used_mib')} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
