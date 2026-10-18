"""Opt-in R05 mitigation experiment; never changes operational model settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, rows, write_json
from .comparison import compare_tasks

DEFAULT = ROOT / 'bench/results/2026-10-17-hip-kv-q8-r05-experiment'
CASES = [
    ('original', [(2, 5), (7, 9), (11, 15)], 4, 3),
    ('translated', [(22, 25), (27, 29), (31, 35)], 24, 3),
    ('scaled', [(4, 10), (14, 18), (22, 30)], 8, 6),
    ('touch_end', [(2, 5), (8, 11)], 5, 3),
    ('touch_start', [(7, 9)], 4, 3),
    ('inside', [(2, 5)], 4, 3),
    ('adjacent', [(2, 5), (5, 8)], 4, 3),
    ('nested', [(2, 10), (4, 6), (12, 15)], 4, 3),
    ('unsorted', [(11, 15), (2, 5), (7, 9)], 4, 3),
    ('after_all', [(2, 5), (7, 9)], 10, 3),
    ('empty', [], 4, 3),
    ('exact_gap', [(2, 5), (8, 12)], 4, 3),
]
CHECK = (' Before answering, verify the duration, check for overlap with every existing '
         'reservation using half-open boundaries, and verify that no earlier feasible start exists. '
         'Return only the final JSON object, without explanation.')


def earliest(intervals, minimum, duration):
    start = minimum
    for left, right in sorted(intervals):
        if right <= start:
            continue
        if start + duration <= left:
            break
        start = right
    return {'start': start, 'end': start + duration}


def tasks():
    out = []
    for index, (name, intervals, minimum, duration) in enumerate(CASES):
        occupied = ','.join(f'[{a},{b})' for a, b in intervals) or 'no intervals'
        prompt = (f'An exclusive resource is held on half-open intervals {occupied}. '
                  f'A new job needs {duration} consecutive units, cannot start before {minimum}. '
                  'Return earliest start and end as JSON integer keys start,end.')
        for mode in ('direct', 'check', 'thinking_check'):
            for repeat in range(1, 4):
                out.append({'id': f'R05V{index:02}-{mode}-{repeat}', 'category': 'reasoning',
                            'case': name, 'condition': mode, 'repeat': repeat,
                            'intervals': intervals, 'minimum': minimum, 'duration': duration,
                            'system': 'Follow the task exactly. No markdown or commentary.',
                            'prompt': prompt + (CHECK if mode != 'direct' else ''),
                            'expected': earliest(intervals, minimum, duration), 'validator': 'exact',
                            'max_turns': 1, 'max_tokens': 2048,
                            'reasoning_enabled': mode == 'thinking_check'})
    return out


def report(directory):
    from .campaign import audit, registry
    manifest = audit(directory)
    frozen = rows(directory / 'TASKS.jsonl')
    records = registry(directory / 'results-pilot.jsonl', manifest['freeze_id'])
    result = {'status': 'INCOMPLETE', 'causal_attribution': 'NOT_ESTABLISHED', 'conditions': {},
              'thinking_observation': {}, 'within_arm_changes': []}
    lines = ['# R05 mitigation experiment', '',
             'Repeated deterministic executions measure stability, not independent samples.', '',
             '| Condition | FP16 pass / planned | Q8 pass / planned | Q8-only failures | Shared failures |',
             '|---|---:|---:|---:|---:|']
    for mode in ('direct', 'check', 'thinking_check'):
        selected = [t for t in frozen if t['condition'] == mode]
        paired = compare_tasks(selected, records)
        result['conditions'][mode] = paired
        scores = {arm: sum(records.get((t['id'], arm), {}).get('status') == 'PASS' for t in selected)
                  for arm in ('fp16', 'q8')}
        failures = [p['task_id'] for p in paired['pairs']
                    if p['classification'] == 'Q8_ONLY_FAILURE_OBSERVED']
        shared = sum(p['classification'] == 'SHARED_TASK_FAILURE' for p in paired['pairs'])
        lines.append(f'| {mode} | {scores["fp16"]}/{len(selected)} | {scores["q8"]}/{len(selected)} | {len(failures)} | {shared} |')
        if failures:
            lines.extend(['', 'Q8-only failures: ' + ', '.join(failures), ''])
    for arm in ('fp16', 'q8'):
        thinking = [records.get((t['id'], arm), {}) for t in frozen if t['reasoning_enabled']]
        observed = [r for r in thinking if r.get('status') in ('PASS', 'FAIL')]
        result['thinking_observation'][arm] = {
            'completed_records': len(observed),
            'with_reasoning_content': sum(any(q.get('message', {}).get('reasoning_content')
                                            for q in r.get('requests', [])) for r in observed)}
        for name, *_ in CASES:
            for repeat in range(1, 4):
                selected = [t for t in frozen if t['case'] == name and t['repeat'] == repeat]
                statuses = {t['condition']: records.get((t['id'], arm), {}).get('status', 'NOT_RUN')
                            for t in selected}
                result['within_arm_changes'].append({'arm': arm, 'case': name, 'repeat': repeat,
                                                     'statuses': statuses})
    statuses = [v['status'] for v in result['conditions'].values()]
    result['status'] = ('ERROR' if 'ERROR' in statuses else 'INCOMPLETE' if 'INCOMPLETE' in statuses
                        else 'COMPLETED_FOR_REVIEW')
    lines += ['', 'No mitigation is automatically accepted. Inspect losses as well as recoveries,',
              'JSON failures, truncation, reasoning-channel presence and latency in the transcripts.',
              'The historical R05 run is reference evidence, not an additional repetition.']
    write_json(directory / 'R05_SUMMARY.json', result)
    (directory / 'R05_RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'report': str(directory / 'R05_RESULTS.md')}))


def main():
    from . import campaign
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('plan', 'prepare', 'preflight', 'run', 'report', 'audit'))
    ap.add_argument('--campaign', type=Path, default=DEFAULT)
    args = ap.parse_args()
    directory = args.campaign.resolve()
    if args.command == 'plan':
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / 'PLAN.json'
        if target.exists():
            raise RuntimeError('plan already exists; preserve it')
        write_json(target, {'status': 'PLANNED_NOT_RUN', 'tasks': tasks(),
                            'task_count': len(tasks()), 'planned_generations': 2 * len(tasks())})
    elif args.command == 'prepare':
        campaign.prepare(directory, task_set='r05')
    elif args.command == 'run':
        campaign.run(directory, 'pilot')
        report(directory)
    elif args.command == 'report':
        report(directory)
    elif args.command == 'audit':
        campaign.audit(directory)
    else:
        from .resources import admission, snapshot
        state = snapshot()
        print(json.dumps({'memory': state, 'refused': admission(state)}, indent=2))


if __name__ == '__main__':
    main()
