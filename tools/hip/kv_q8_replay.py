"""Read-only replay of frozen T5/T6/T7/T8 evidence under paired Q8 assessment.

Writes a NEW --out directory only; never runs a model or audits current sources
against an old freeze. Validates frozen task/experiment and transcript hashes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.hip.functional_quality.common import digest, rows, sha, strict_json, write_json
from tools.hip.functional_quality.campaign import registry
from tools.hip.functional_quality.comparison import POLICY_ID, compare_tasks, mtp_diagnostic, text_diagnostic


def frozen_functional(campaign):
    manifest = strict_json((campaign / 'MANIFEST.json').read_text(encoding='utf-8'))
    if digest({k: v for k, v in manifest.items() if k != 'freeze_id'}) != manifest['freeze_id']:
        raise ValueError('historical freeze identity mismatch')
    for name in ('TASKS.jsonl', 'ORIGINAL_CODE.jsonl', 'EXPERIMENT.json'):
        p = campaign / name
        expected = manifest['inputs'].get(str(p.resolve()))
        if not expected or sha(p) != expected:
            raise ValueError('frozen task/experiment changed: ' + str(p))
    result, transcripts = {}, set()
    for suite, taskfile in (('pilot', 'TASKS.jsonl'), ('original', 'ORIGINAL_CODE.jsonl')):
        records = registry(campaign / f'results-{suite}.jsonl', manifest['freeze_id'])
        result[suite] = compare_tasks(rows(campaign / taskfile), records)
        transcripts.update(Path(r['event_file']) for r in records.values() if r.get('event_file'))
    api = registry(campaign / 'results-api.jsonl', manifest['freeze_id'])
    result['api_integration'] = compare_tasks([{'id': 'api15', 'category': 'api'}], api)
    transcripts.update(Path(r['event_file']) for r in api.values() if r.get('event_file'))
    result['provenance'] = {'freeze_id': manifest['freeze_id'], 'task_and_experiment_hashes': 'VERIFIED',
                            'record_and_transcript_hashes': 'VERIFIED', 'transcripts': len(transcripts),
                            'current_source_audit': 'NOT_APPLICABLE_TO_HISTORICAL_REPLAY',
                            'shared_C01_C14': 'Same observations reused; suites must not be pooled as independent'}
    return result, transcripts


def needles(records, manifest):
    tasks, paired, logs = {}, {}, set()
    latest = {r['tag']: r for r in records if r['tag'].startswith('cp2-needle-') and '-v2-' in r['tag']}
    expected = manifest['needles']
    # Define the denominator from the frozen plan, not from successful records.
    # An entire missing FP16/Q8 pair must remain INCOMPLETE.
    known_tags = {f'cp2-needle-{group}-{arm}-fit{fit}' for group in expected
                  for arm in ('fp16', 'q8') for fit in (0, 1)}
    if set(latest) - known_tags:
        raise ValueError('unplanned needle run identity')
    for group, spec in expected.items():
        for fit in (0, 1):
            for request in spec['requests']:
                task = request['prompt'] + '-fit' + str(fit)
                category = 'primary' if request['single'] and not group.endswith('-extra') else 'extra' if request['single'] else 'joint'
                tasks[task] = {'id': task, 'category': category}
    for r in latest.values():
        source = r['key_src']
        if r['kv'] not in ('fp16', 'q8') or source['kv'] != r['kv']:
            raise ValueError('needle arm identity mismatch')
        if source['ctx'] != r['ctx'] or source['fit'] != r['fit']:
            raise ValueError('needle control identity mismatch')
        group = r['tag'][len('cp2-needle-'):].rsplit('-' + r['kv'] + '-fit', 1)[0]
        if source['requests'] != expected[group]['requests'] or r['ctx'] != expected[group]['ctx']:
            raise ValueError('needle request differs from frozen plan')
        if r.get('rc') == 0 and not r.get('expert_set'):
            raise ValueError('missing actual needle expert-set identity')
        controls = {k: v for k, v in source.items() if k != 'kv'}
        frozen = digest(controls)
        indexed = {v['label']: v for v in r.get('rows', [])}
        if len(indexed) != len(r.get('rows', [])):
            raise ValueError('duplicate needle request outcome')
        if r.get('log') and r.get('log_sha256'):
            log = Path(r['log'])
            if sha(log) != r['log_sha256']:
                raise ValueError('needle execution log changed')
            logs.add(log)
        for request in source['requests']:
            task = request['prompt'] + '-fit' + str(int(r['fit']))
            row = indexed.get(request['prompt'])
            status = 'INFRA_ERROR' if r.get('rc') != 0 or r.get('process_rc') != 0 else 'NOT_RUN' if row is None else 'PASS' if row['found'] == request['codes'] else 'FAIL'
            if row is not None and row['codes'] != request['codes']:
                raise ValueError('needle expected output identity mismatch')
            config = {'exe': source['exe'], 'frozen_controls': controls, 'expert_set': r.get('expert_set'),
                      'args': ['--kv', 'q8_0'] if r['kv'] == 'q8' else []}
            key = task, r['kv']
            if key in paired:
                raise ValueError('duplicate needle pair identity')
            paired[key] = {'task_id': task, 'arm': r['kv'], 'status': status,
                           'freeze_id': frozen, 'config': config}
    result = compare_tasks(list(tasks.values()), paired)
    result['corpus'] = 'Validated v2 only; invalidated v1 excluded'
    return result, logs


def replay(functional, native, reviews, out):
    functional, native, reviews, out = (Path(p).resolve() for p in (functional, native, reviews, out))
    for source in (functional, native, reviews.parent):
        if out.is_relative_to(source):
            raise ValueError('--out must be outside every historical source directory')
    if out.exists():
        raise ValueError('--out already exists; never overwrite a replay or old report')
    inputs = {functional / n for n in ('MANIFEST.json', 'TASKS.jsonl', 'ORIGINAL_CODE.jsonl', 'EXPERIMENT.json',
                                       'results-pilot.jsonl', 'results-original.jsonl', 'results-api.jsonl')}
    inputs.update(native / n for n in ('runs.jsonl', 'cp2-mtp.jsonl', 'cp2-t5-verdicts.jsonl', 'cp2-runs.jsonl',
                                      'cp2-prompts-manifest.json'))
    inputs.add(reviews)
    function, transcripts = frozen_functional(functional)
    inputs.update(transcripts)
    entries = strict_json(reviews.read_text(encoding='utf-8'))['entries']
    review_by_fp = {json.dumps(r['fingerprint'], sort_keys=True): r for r in entries}
    if len(review_by_fp) != len(entries):
        raise ValueError('duplicate review fingerprint')
    plan = strict_json((native / 'cp2-prompts-manifest.json').read_text(encoding='utf-8'))
    verdicts = [v for v in rows(native / 'cp2-t5-verdicts.jsonl') if v.get('arm') == 'q8']
    indexed = {(v['case'], v['protocol']['fit']): v for v in verdicts}
    expected_t5 = {(r['prompt'], fit) for r in plan['t5'] for fit in (0, 1)}
    if len(indexed) != len(verdicts) or set(indexed) - expected_t5:
        raise ValueError('duplicate or unplanned T5 comparison')
    t5 = []
    for key in sorted(expected_t5):
        v = indexed.get(key, {'result': 'NOT_RUN'})
        review = review_by_fp.get(json.dumps(v.get('fingerprint'), sort_keys=True))
        t5.append(text_diagnostic(v, review))
    t6, logs = needles(rows(native / 'cp2-runs.jsonl'), plan)
    inputs.update(logs)
    runs = {r['tag']: r for r in rows(native / 'runs.jsonl')}
    t7 = [dict(mtp_diagnostic(runs.get(m['runs']['fp16'], {}), runs.get(m['runs']['q8'], {})),
               prompt=m['prompt'], fit=m['fit'], historical_passed=m['passed']) for m in rows(native / 'cp2-mtp.jsonl')]
    result = {'assessment_policy': POLICY_ID, 'method': 'Retrospective comparison; no new model outputs',
              'new_inference_processes': 0,
              'T5': {'status': 'ERROR' if any(v['status'] in ('ERROR', 'INCOMPLETE') for v in t5) else 'DIAGNOSTIC_ONLY',
                     'comparisons': len(t5), 'pending_outlier_reviews': sum(v.get('outlier_review_pending', False) for v in t5),
                     'functional_quality_not_decided_by_token_similarity': True, 'rows': t5},
              'T6': t6, 'T7': {'quality_result': 'NOT_EVALUATED', 'performance_verdict': 'INCONCLUSIVE', 'pairs': t7},
              'T8': function, 'T4': 'Existing quantitative comparator and manual approval remain separate',
              'T9_T10_T11': 'System integrity/operational checks remain separate; no reclassification or new run',
              'q8_promotion': 'NOT_YET_APPROVED'}
    hashes = {str(p): sha(p) for p in sorted(inputs)}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'INPUTS.json', hashes)
    write_json(out / 'COMPARISON.json', result)
    lines = ['# Paired Q8 versus FP16 — replay', '',
             'Retrospective replay of existing observations. No new GPU inference or automatic promotion.', '',
             f"T5: {result['T5']['status']}; {len(t5)} Q8 comparisons; pending outlier reviews: {result['T5']['pending_outlier_reviews']}.",
             f"T6: {t6['status']}; paired checks: {len(t6['pairs'])}; counts: {json.dumps(t6['counts'])}.",
             'T7: acceptance is efficiency, not answer quality. Calibrated performance remains INCONCLUSIVE.', '',
             '| Functional suite | Paired assessment | Both pass | Shared failures | Q8-only failure | Q8-only success |',
             '|---|---|---:|---:|---|---|']
    for suite in ('pilot', 'original', 'api_integration'):
        v = function[suite]
        lines.append(f"| {suite} | {v['status']} | {v['counts'].get('BOTH_PASS', 0)} | {v['counts'].get('SHARED_TASK_FAILURE', 0)} | "
                     f"{', '.join(v['q8_only_failures']) or 'none'} | {', '.join(v['q8_only_successes']) or 'none'} |")
    lines += ['', 'A shared failure is not attributed to Q8; its cause remains unknown. Q8-only failures are observed candidates requiring matched repetition/review, not established quantization bugs.',
              'Equal aggregate scores do not cancel discordances. Pilot and original share C01..C14 and are not independent replications.',
              'API/integration, native integrity and missing data retain their own checks. T4 manual acceptance, T9 and T11 history are unchanged.', '',
              f"T7 controls valid for {sum(r['status'] == 'EFFICIENCY_DIAGNOSTIC' for r in t7)}/{len(t7)} pairs; "
              f"legacy acceptance below -2pp: {sum(r.get('legacy_minus_2pp_satisfied') is False for r in t7)}. "
              'No functional FAIL is inferred from these acceptance gaps.', '', 'Details: COMPARISON.json; source/transcript/log hashes: INPUTS.json.', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    changed = [str(p) for p in inputs if sha(p) != hashes[str(p)]]
    write_json(out / 'PRESERVATION.json', {'checked': len(inputs), 'changed': changed, 'preserved': not changed})
    if changed:
        raise RuntimeError('input changed during replay')
    print(json.dumps({'T5': result['T5']['status'], 'T6': t6['status'],
                      'T7_valid_controls': sum(r['status'] == 'EFFICIENCY_DIAGNOSTIC' for r in t7),
                      'T8_pilot': function['pilot']['status'], 'T8_original': function['original']['status'],
                      'historical_inputs_preserved': len(inputs)}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--functional-campaign', type=Path, default=ROOT / 'bench/results/2026-10-16-hip-kv-q8-functional-t8/revision-2')
    parser.add_argument('--native-campaign', type=Path, default=ROOT / 'bench/results/2026-10-15-hip-kv-q8-phase4b')
    parser.add_argument('--reviews', type=Path, default=ROOT / 'bench/results/2026-10-17-hip-kv-q8-t4-t5-closure/t5-review-registry.json')
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    try:
        replay(a.functional_campaign, a.native_campaign, a.reviews, a.out)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'Invalid comparison: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
