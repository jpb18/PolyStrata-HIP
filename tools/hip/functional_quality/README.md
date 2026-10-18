# Paired Q8 versus FP16 functional harness

Per-task comparison separates shared task failures from Q8-only failures and
successes. Q8-only failures require review and cannot be cancelled by gains on
other tasks. Absolute validators and scores remain available. T5 trajectory
similarity and T7 acceptance are diagnostics, not quality oracles.
See [the comparison policy](../../../docs/AMD_HIP_KV_Q8_PAIRED_TESTS.md).

The opt-in [R05 mitigation experiment](../../../docs/AMD_HIP_R05_EXPERIMENT.md)
compares direct answers, verification instructions and enabled reasoning in
both KV formats. Its separate CLI is `python -m tools.hip.functional_quality.r05_experiment`.

The [average-quality extension](../../../docs/AMD_HIP_Q8_AVERAGE_QUALITY.md) adds
200 cases in five areas, weighted paired loss/gain estimates, family-bootstrap
intervals and blind Portuguese-answer review. Use
`python -m tools.hip.functional_quality.average_quality plan` to inspect the design.

Run from the repository root using its existing Python environment. No engine
source, launchers or operational profiles are modified. CPU tests use loopback
HTTP and disposable restricted Python workers; they do not load a model.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_functional_quality.py -v
.venv/Scripts/python.exe -m tools.hip.functional_quality.campaign prepare
.venv/Scripts/python.exe -m tools.hip.functional_quality.campaign preflight
.venv/Scripts/python.exe -m tools.hip.functional_quality.campaign run --suite pilot
.venv/Scripts/python.exe -m tools.hip.functional_quality.campaign run --suite original
.venv/Scripts/python.exe -m tools.hip.functional_quality.campaign run --suite api
.venv/Scripts/python.exe -m tools.hip.functional_quality.campaign report
.venv/Scripts/python.exe -m tools.hip.functional_quality.campaign audit
```

The default is `bench/results/2026-10-17-hip-kv-q8-paired-functional`, a new campaign.
The old revision-2 campaign stays frozen; do not run its report using changed live
sources. Read-only retrospective comparison is available through
`tools/hip/kv_q8_replay.py --out NEW_DIRECTORY`, preserving the old results.
Use `--campaign PATH` for a separate campaign. `prepare` runs CPU checks before
creating task files and a manifest; an existing freeze is verified, never reset.
It hashes the full GGUF and runtime inputs, which takes time. A partially written
preparation is refused for inspection rather than overwritten.

`run` is sequential and refuses unknown/insufficient resources or another
inference process. A refusal is not an approval request and is not a model
failure: resolve the recorded condition, then run the same command. It skips
all completed PASS **and FAIL** records. `--retry-infra` explicitly admits only
interrupted/infrastructure attempts; these remain in the append-only history.
Never delete failures to get another main answer. A damaged JSONL journal or
changed frozen input stops integrity checking; preserve it and investigate.

The original suite reuses completed C01..C14 from the pilot (or conversely) and
generates only missing main answers. API15 invokes the existing poly_api_check.py.
Pilot, original code and API records remain separate. `original_t8` now reports
the paired original-code assessment; API/tool JSON have a separate integration
status. `legacy_original_t8` retains the old complete-component rule as a
diagnostic. No field approves Q8 promotion. Timing distributions are descriptive and missing native metrics
remain null. This short-context microproject pilot does not test large repository
editing, arbitrary shell programs, long-context quality or general agent autonomy.

Modules: `tasks` freezes inputs/oracles; `code_worker` and `code` execute restricted
projects; `mock` owns the REST fixtures; `client` implements the actual SSE API;
`evaluate` drives tool cycles; `server` owns experimental processes; `resources`
checks admission and locking; `campaign` freezes, resumes, audits and reports.

See PROTOCOL.md for scoring, budgets, resource limits and symmetric validator
repair policy. Model-facing code tools never expose the campaign, oracle source
or hidden tests. Restricted workers intentionally support pure Python only.
