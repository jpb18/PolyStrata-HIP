# R05: model mitigation experiment

Status: prepared as an opt-in experiment; no new inference results yet.

Question: does a verification instruction, or that instruction plus enabled reasoning,
reduce observed Q8-only scheduling failures relative to FP16 KV? Model weights remain
Qwen3.6-35B-A3B UD-Q3_K_M in both arms. This does not compare Q3 versus higher precision weights.

The historical R05 returned 15–18 with FP16 and 4–7 with Q8 for reservations
[2,5), [7,9), [11,15), earliest start 4 and duration 3. It is reference evidence,
not a new repetition and not included in the new experiment's score.

## Frozen design

- 12 cases: original, translated, scaled, touching either boundary, starting inside a
  reservation, adjacent and nested reservations, unsorted input, after all reservations,
  empty reservations and an exactly fitting gap.
- Three conditions: `direct`, `check` (verification instruction, thinking off),
  `thinking_check` (same verification instruction, thinking on, effort high).
- Three executions of each condition/case in each KV format: 108 pairs,
  216 generations. Temperature 0 throughout. Repetition measures execution stability;
  repetitions and related variants are not independent statistical samples.
- All conditions have a 2048-token output budget, one turn, identical final JSON
  validation. The historical 1024-token run is not a strict repeat of this design.
  Thinking consumes output budget; truncations remain failures and are reported.
- Exact JSON integer keys `start,end`; correct duration, no overlap and earliest valid
  start are encoded in the oracle. No validator feedback, retries after a completed
  failure, tools or automatic FP16 fallback are supplied to the evaluated model.
- Existing campaign admission, GPU lock, owned-process cleanup, fixed experts,
  pinned executable/profile/model, append-only records and input/transcript hashes apply.
  The four blocks counterbalance FP16/Q8 order. Operational profiles stay unchanged.

The old harness explicitly disabled thinking. Absence of an explanation does not
by itself prove absence of reasoning; the new request settings and returned reasoning
channel must be inspected. Requesting thinking is not proof that it ran: the report
counts records with returned reasoning content. If it is absent, treat that condition
as unverified rather than claiming a reasoning benefit.

## Commands

From the repository root, using the existing Python environment:

```powershell
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.r05_experiment plan
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.r05_experiment prepare
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.r05_experiment preflight
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.r05_experiment run
.venv/Scripts/python.exe -B -m tools.hip.functional_quality.r05_experiment report
```

Default output: `bench/results/2026-10-17-hip-kv-q8-r05-experiment`.
`plan` creates a reviewable CPU-only design. `prepare` runs CPU checks and freezes
actual model/runtime inputs before inference. Preparation hashes the full GGUF.
No external download is needed. Use a new `--campaign PATH` for a different experiment.

## Interpretation

Read `R05_RESULTS.md` and `R05_SUMMARY.json` alongside the standard campaign report.
Compare Q8-only failures separately within each condition, and direct-to-mitigated
recoveries AND new losses within each arm (recorded by case and repetition).
Correct JSON alone is not sufficient; wrong schedules and generation truncations fail.
Inspect timing, token usage and reasoning-channel presence in the captured requests.

Do not accept a mitigation merely because aggregate accuracy rises, or because Q8
gains cancel losses elsewhere. Consistent recovery of original R05 and reduced
discordance across variants, without newly introduced failures, supports further
validation. Mixed results remain review-required. This small related-case study cannot
prove general equivalence, causal attribution, or justify Q8 promotion by itself.
The thinking condition changes effort and thinking settings together: it estimates
that package, not their separate effects. Changes to KV precision or model weight
precision require a separate experiment.
