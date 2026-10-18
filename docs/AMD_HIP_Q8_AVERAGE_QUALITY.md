# Paired average-quality extension

Policy: `KV-Q8-average-v1-2026-10-17`. Opt-in extension of the existing functional
harness; no operational model/server settings or engine kernels change.

The question is the measured quality loss from changing **KV cache FP16 to Q8**
while retaining the same Q3 model weights. This is not a comparison of FP16
weights against Q8 weights. R05 mitigation and historical T4/T5 approvals remain
separate; diagnostic variants are not pooled into this estimate.

## Casebook and controls

The plan freezes 200 cases and 400 main generations, one execution per case/arm:

| Area | Cases | Evaluation |
|---|---:|---|
| Code | 40 | Execute public and hidden tests in the existing restricted Python worker; successful agent completion also required |
| Reasoning | 40 | Typed JSON oracle: arithmetic, change, sets, units, counting, weighted sums, logic, ordering, probability and shortest path |
| Extraction/instructions | 40 | Typed JSON oracle: filtering, aggregation, sorting, missing/null fields, projection and transformations |
| Document context | 40 | Controlled archive lookup, joins, revisions, exceptions, totals, ordering, missing facts, authority and multiple hops |
| Open Portuguese answers | 40 | Blind human rubric and preference, without showing KV identity |

These are **80 families**, not 200 independent problem types: 40 distinct code
contracts and ten families of four related cases in each other area. Code includes
the existing 20 contracts plus 20 additional functions. Historical answers are not
reused or pooled; new IDs prevent cross-suite reuse. The existing 20 contracts
are known diagnostic material, not an unseen external benchmark. Numeric and
wording variants are explicitly clustered rather than counted as independent evidence.

This is a synthetic casebook, not a representative sample of all user requests.
Document contexts contain controlled distractor records, not real-world document
diversity. The four lengths are 128/384/768/1280 archive rows, with facts at
different positions. Preparation measures actual prompt tokens with the real
tokenizer/template and reserves output budget plus a 256-token margin within
49152 tokens. Actual API prompt-token counts remain in transcripts. No claim of
128K-context validation is made. Code remains pure Python microprojects, not
large repository editing.

Both arms share executable, model bytes, frozen expert profile, context capacity,
prompt and final validator. Existing resource admission, exclusive GPU lock,
sequential ownership checks and cleanup apply. Order is balanced within area
using the existing four-block schedule. Thinking is off and temperature is zero;
there are no mitigation prompts, retries of completed answers or model switching.
Non-code cases have one turn and 2048 output tokens; code retains the existing
ten-turn/2048-token-per-turn contract. Truncations and incorrect outputs remain
failures; infrastructure errors never become model failures.

## Mean and uncertainty

Default area weights are equal (20% each). Supply a five-key JSON object with
nonnegative weights when creating the plan to reflect a different workload.
Weights are normalized and frozen **before inference**. Changing them later
requires another campaign; the reporter does not silently choose favorable weights.

For every area and for the weighted complete casebook, report:

- FP16 and Q8 task accuracy.
- Q8-only loss frequency: FP16 passes, Q8 fails.
- Q8-only gain frequency: Q8 passes, FP16 fails.
- Net delta in percentage points, Q8 minus FP16; average loss uses the opposite sign.
- Relative accuracy loss `(accuracy_fp16 - accuracy_q8) / accuracy_fp16`, null when
  baseline accuracy is zero. This is not relative model-weight precision error.
- Shared failures, missing scores and review coverage separately.

Gains do not erase the reported loss frequency. The mean is a descriptive balance
of gains and losses, not automatic acceptance of every regression. All-shared
failures imply zero observed differential loss but provide no evidence of useful
task quality. The paired comparison retains `BASELINE_LIMITED` where applicable.

An empirical 95% percentile interval uses 4000 deterministic bootstrap draws:
resample whole families within each area, keep FP16/Q8 outcomes paired, and apply
the frozen area weights. The interval describes this limited casebook under an
assumption of independent families; it does not establish a population-wide bound.
Replicating the same family does not narrow it artificially. Degenerate zero-width
intervals, especially with no observed discordance, do not prove zero future risk.

Missing required pairs or reviews leave the overall mean unavailable. Per-area
completed-only counts remain explicitly descriptive. A renormalized objective-only
subset can be reported once those areas are complete, clearly excluding open answers.
Configuration/integrity errors suppress the weighted estimates. No acceptable-loss
threshold, equivalence claim or Q8 promotion is inferred.

## Blind open-answer review

Complete open-answer generations receive `REVIEW_PENDING`, **not PASS**. They are
terminal generation records: resuming does not regenerate them. A truncated or
empty generation remains FAIL regardless of later rubric marks.

`export-blind` produces `BLIND_PACKET.json` with random item IDs, prompts, rubrics
and A/B answers. A/B order is randomized independently. It contains no engine
settings, timings, KV labels or result filenames. Keep `BLIND_KEY.private.json`
away from the reviewer. The actual text is not rewritten or stripped to conceal
any model-generated references.

The reviewer marks every `content`, `instruction` and `clarity` criterion true or
false for each answer; all three are required to pass. They also provide A/B/TIE
preference and a rationale. Preference counts are separate from binary task accuracy.
Use a copy of `REVIEW_TEMPLATE.json`, retaining its packet hash, and fill in reviewer
identity. Empty marks are not accepted. The implementation supplies no automatic
judge and does not invent human reviews. One final adjudication per item is supported;
multi-reviewer agreement is not measured.

Imports bind to packet, freeze and inference-record hashes. Unknown/duplicate/stale
items or incomplete criteria are rejected before appending. Partial review batches
are allowed; missing items keep the overall result incomplete. Accepted decisions
are append-only in `REVIEWS.jsonl`, with their source hash; reimport cannot overwrite
an existing adjudication. Raw inference records and their hashes stay unchanged.

## Commands

From the repository root:

```powershell
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.average_quality plan
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.average_quality prepare
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.average_quality preflight
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.average_quality run
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.average_quality export-blind
# Human reviewer fills a separate copy of REVIEW_TEMPLATE.json.
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.average_quality import-reviews --reviews PATH_TO_COMPLETED_REVIEW.json
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.average_quality report
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.average_quality audit
```

Use `plan --weights PATH.json` to choose workload weights, and `--campaign PATH`
for another output directory. Default output is
`bench/results/2026-10-17-hip-kv-q8-average-quality`.
`plan` is CPU-only; `prepare` runs reference/estimator/review checks, verifies prompt
capacity, and freezes actual model/runtime inputs before any scored inference.
Source and records remain auditable. A later source edit deliberately invalidates
a live freeze; do not rewrite the manifest to hide drift.

Results are `AVERAGE_RESULTS.md` and `AVERAGE_SUMMARY.json`. The generic campaign
report now dispatches to the appropriate R05/average reporter, avoiding assumptions
about a tool-use category in these extensions. Original-suite reporting stays intact.
