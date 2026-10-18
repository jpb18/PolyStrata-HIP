from __future__ import annotations

import argparse
import collections
import json
import importlib.metadata
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .common import DEFAULT_CAMPAIGN, ROOT, append, digest, now, rows, sha, strict_json, write_json
from .evaluate import evaluate
from .comparison import POLICY_ID, compare_tasks
from .resources import ExclusiveGPU, MIN_RUNNING_FREE_RAM_MIB, admission, snapshot
from .server import EXE_SHA, PROFILE_SHA, ExperimentalServer
from .tasks import code_tasks, pilot

COMPLETED = ('PASS', 'FAIL', 'REVIEW_PENDING')


def schedule(tasks):
    groups = [[], []]
    counts = collections.Counter()
    for task in tasks:
        group = counts[task['category']] % 2
        counts[task['category']] += 1
        groups[group].append(task['id'])
    return [{'block': n, 'arm': arm, 'task_ids': groups[g]} for n, (g, arm) in
            enumerate([(0, 'fp16'), (0, 'q8'), (1, 'q8'), (1, 'fp16')])]


def source_inputs():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [Path(__file__).with_name('PROTOCOL.md'), ROOT / 'tests/test_functional_quality.py']
    files += [ROOT / 'tests/test_kv_q8_comparison.py']
    files += [ROOT / 'tests/test_r05_experiment.py', ROOT / 'docs/AMD_HIP_R05_EXPERIMENT.md']
    files += [ROOT / 'tests/test_average_quality.py', ROOT / 'docs/AMD_HIP_Q8_AVERAGE_QUALITY.md']
    files += list((ROOT / 'serve').rglob('*.py'))
    files += list((ROOT / 'serve').rglob('*.jinja'))
    files += [Path(sys.executable), Path(sys._base_executable)]
    files += [ROOT / 'tools/hip/_rocm_env.py', ROOT / 'tools/hip/poly_api_check.py', ROOT / 'tools/strata_tokenizer.py']
    return files


def audit(campaign):
    manifest = strict_json((campaign / 'MANIFEST.json').read_text(encoding='utf-8'))
    if digest({k: v for k, v in manifest.items() if k != 'freeze_id'}) != manifest['freeze_id']:
        raise RuntimeError('freeze manifest identity mismatch')
    bad = [p for p, h in manifest['inputs'].items() if not Path(p).is_file() or sha(p) != h]
    if bad:
        raise RuntimeError('frozen inputs changed: ' + ', '.join(bad))
    return manifest


def prepare(campaign, task_set='standard'):
    if task_set not in ('standard', 'r05', 'average'):
        raise ValueError('unknown task set')
    if task_set == 'r05':
        from .r05_experiment import tasks as r05_tasks
        selected, original = r05_tasks(), []
    elif task_set == 'average':
        from .average_tasks import tasks as average_tasks
        selected, original = average_tasks(), []
        from .average_quality import validate_plan
        validate_plan(strict_json((campaign / 'PLAN.json').read_text(encoding='utf-8')), selected)
        if digest(rows(campaign / 'CASEBOOK.jsonl')) != digest(selected):
            raise ValueError('reviewable casebook differs from planned tasks')
    else:
        selected, original = pilot(), code_tasks()
    campaign.mkdir(parents=True, exist_ok=True)
    if (campaign / 'MANIFEST.json').exists():
        existing = audit(campaign)
        saved = strict_json((campaign / 'EXPERIMENT.json').read_text(encoding='utf-8'))
        if saved.get('task_set', 'standard') != task_set:
            raise RuntimeError('existing campaign has a different task set; use a new directory')
        print('Already frozen: ' + existing['freeze_id'])
        return
    if any(campaign.glob('results-*.jsonl')):
        raise RuntimeError('refusing to freeze over existing results')
    for name in ('TASKS.jsonl', 'ORIGINAL_CODE.jsonl', 'PROTOCOL.md', 'EXPERIMENT.json'):
        if (campaign / name).exists():
            raise RuntimeError('partial preparation found; inspect before writing: ' + name)
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                           '-p', 'test_functional_quality.py', '-v'], cwd=ROOT, capture_output=True, text=True)
    output = test.stdout + test.stderr
    if test.returncode:
        (campaign / 'cpu-tests.txt').write_text(output, encoding='utf-8')
        raise RuntimeError('CPU validation failed; see cpu-tests.txt')
    paired_test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                                 '-p', 'test_kv_q8_comparison.py', '-v'], cwd=ROOT, capture_output=True, text=True)
    (campaign / 'cpu-tests.txt').write_text(output + paired_test.stdout + paired_test.stderr, encoding='utf-8')
    if paired_test.returncode:
        raise RuntimeError('Paired comparison validation failed; see cpu-tests.txt')
    if task_set == 'r05':
        r05_test = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                                  '-p', 'test_r05_experiment.py', '-v'], cwd=ROOT, capture_output=True, text=True)
        with (campaign / 'cpu-tests.txt').open('a', encoding='utf-8') as log:
            log.write(r05_test.stdout + r05_test.stderr)
        if r05_test.returncode:
            raise RuntimeError('R05 validation failed; see cpu-tests.txt')
    if task_set == 'average':
        check = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                                '-p', 'test_average_quality.py', '-v'], cwd=ROOT, capture_output=True, text=True)
        with (campaign / 'cpu-tests.txt').open('a', encoding='utf-8') as log:
            log.write(check.stdout + check.stderr)
        if check.returncode:
            raise RuntimeError('Average-quality validation failed; see cpu-tests.txt')
    exe = ROOT / 'build-hip-poly-kvq8p3-win/strata-poly-kvq8p3.exe'
    profile = ROOT / 'bench/results/2026-10-15-hip-kv-q8-phase4b/provenance/expert-profile-campaign.txt'
    if sha(exe) != EXE_SHA or sha(profile) != PROFILE_SHA:
        raise RuntimeError('candidate/profile pin mismatch')
    base = strict_json((ROOT / 'strata-qwen3.6-35b-a3b-ud-q3_k_m-hip.json').read_text(encoding='utf-8'))
    model = Path(base['args'][base['args'].index('--model') + 1])
    tokenizer = Path(base['tokenizer'])
    if not model.is_file() or not tokenizer.is_dir() or not any(tokenizer.iterdir()):
        raise RuntimeError('real model/tokenizer missing; no byte tokenizer fallback allowed')
    if task_set == 'average':
        from .average_quality import context_admission
        write_json(campaign / 'CONTEXT_TOKENS.json', context_admission(selected, tokenizer))
    experiment = {'assessment_policy': POLICY_ID, 'task_set': task_set,
                  'exe': str(exe), 'model': str(model), 'profile': str(profile), 'tokenizer': str(tokenizer),
                  'lib_dirs': base.get('lib_dirs', []), 'pilot_order': schedule(selected),
                  'original_order': schedule(original), 'port': 8197,
                  'env': {'STRATA_POLY_ATTN_FIT': '1', 'STRATA_POLY_CACHE_HASH': '1',
                          'STRATA_POLY_WARM': '1', 'STRATA_POLY_TRACE': '1'},
                  'common_args': ['--max-context', '49152', '--threads', '6', '--turn-token', '248045',
                                  '--expert-cache-max', '5000', '--expert-swaps', '0',
                                  '--conversation-cache-mib', '0']}
    if task_set == 'average':
        settings = strict_json((campaign / 'PLAN.json').read_text(encoding='utf-8'))
        from .average_quality import validate_weights, validate_plan
        validate_plan(settings, selected)
        experiment['average_policy'] = {'weights': validate_weights(settings['weights']),
                                        'bootstrap_seed': 170026, 'bootstrap_draws': 4000}
    for name, tasks in [('TASKS.jsonl', selected), ('ORIGINAL_CODE.jsonl', original)]:
        (campaign / name).write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks), encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('PROTOCOL.md'), campaign / 'PROTOCOL.md')
    write_json(campaign / 'EXPERIMENT.json', experiment)
    # Preserve starting local work without altering, resetting or staging anything.
    for command, out in [(['git', 'status', '--short', '--untracked-files=all'], 'initial-status.txt'),
                         (['git', 'diff', '--binary'], 'initial-worktree.patch')]:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True)
        if proc.returncode:
            raise RuntimeError('cannot capture repository state')
        (campaign / out).write_bytes(proc.stdout)
    evidence = [ROOT / 'docs/AMD_HIP_KV_Q8.md']
    for folder, names in [
        ('2026-10-14-hip-kv-q8-phase4', ['PROMOTION.md', 'cp2-results.md', 'p4_t5_policy.py']),
        ('2026-10-15-hip-kv-q8-phase4b', ['PROMOTION.md', 'CHECKPOINT.md', 'provenance/campaign-profile.json']),
        ('2026-10-16-hip-mtp-t7-investigation', ['HANDOFF.md', 'FINDINGS.md', 'PROTOCOL.md', 't7_run.py']),
        ('2026-10-16-hip-mtp-t8-window-fixed', ['RESULTS.md', 'PROTOCOL.md', 't8_run.py'])]:
        evidence += [ROOT / 'bench/results' / folder / name for name in names]
    protect = list(ROOT.glob('strata-*.json'))
    # Freeze DLLs used beside the executable and SDK libraries without loading a GPU.
    dlls = list(exe.parent.glob('*.dll'))
    for directory in experiment['lib_dirs']:
        dlls += list(Path(directory).glob('*.dll'))
    files = source_inputs() + list(tokenizer.rglob('*')) + [exe, model, profile] + evidence + protect + dlls
    files += [campaign / n for n in ('TASKS.jsonl', 'ORIGINAL_CODE.jsonl', 'PROTOCOL.md', 'EXPERIMENT.json', 'cpu-tests.txt')]
    if task_set == 'average':
        files += [campaign / 'PLAN.json', campaign / 'CASEBOOK.jsonl', campaign / 'CONTEXT_TOKENS.json']
    print('Hashing frozen inputs, including GGUF bytes...', flush=True)
    manifest = {'version': 2, 'assessment_policy': POLICY_ID, 'at': now(), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                'runtime': {'python': sys.version, 'packages': {name: importlib.metadata.version(name) for name in ('psutil', 'jinja2')}},
                'cpu_validation': 'PASS', 'inputs': {str(p.resolve()): sha(p) for p in files if p.is_file()}}
    manifest['freeze_id'] = digest(manifest)
    write_json(campaign / 'MANIFEST.json', manifest)
    write_json(campaign / 'FREEZE_CHECKPOINT.json', {'at': now(), 'freeze_id': manifest['freeze_id'],
                                                   'model_requests_before_freeze': 0, 'cpu': 'PASS'})
    print('Frozen: ' + manifest['freeze_id'])


def registry(path, freeze_id):
    result = {}
    for rec in rows(path):
        if rec['freeze_id'] != freeze_id:
            raise RuntimeError('result belongs to another freeze')
        if rec['result_sha'] != digest({k: v for k, v in rec.items() if k != 'result_sha'}):
            raise RuntimeError('result record integrity failure')
        if rec.get('event_file') and sha(rec['event_file']) != rec['event_sha']:
            raise RuntimeError('transcript integrity failure')
        key = rec['task_id'], rec['arm']
        if key in result and result[key]['status'] in COMPLETED:
            raise RuntimeError('duplicate completed main execution')
        result[key] = rec
    return result


def report(campaign):
    experiment_path = campaign / 'EXPERIMENT.json'
    if experiment_path.exists():
        kind = strict_json(experiment_path.read_text(encoding='utf-8')).get('task_set', 'standard')
        if kind == 'average':
            from .average_quality import report as average_report
            return average_report(campaign)
        if kind == 'r05':
            from .r05_experiment import report as r05_report
            return r05_report(campaign)
    manifest = audit(campaign)
    summary = {'at': now(), 'freeze_id': manifest['freeze_id'], 'assessment_policy': POLICY_ID,
               'original_t8': 'INCOMPLETE', 'legacy_original_t8': 'NOT_RUN',
               'promotion': 'NOT YET APPROVED', 'suites': {}}
    lines = ['# Paired Q8 versus FP16 results', '',
             'Absolute model scores are descriptive; shared task failures are not KV Q8 regressions.',
             'Q8-only failures require review even when improvements elsewhere cancel the score difference.', '',
             '| Suite/category | FP16 pass/completed | Q8 pass/completed | Pairs | FP16 only | Q8 only | Shared failures |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for suite, taskfile in [('pilot', 'TASKS.jsonl'), ('original', 'ORIGINAL_CODE.jsonl')]:
        tasks = rows(campaign / taskfile)
        reg = registry(campaign / f'results-{suite}.jsonl', manifest['freeze_id'])
        categories = {}
        for category in sorted({t['category'] for t in tasks}):
            selected = [t for t in tasks if t['category'] == category]
            cat = {'tasks': len(selected), 'arms': {}, 'paired': 0, 'both_pass': 0, 'fp16_only': 0, 'q8_only': 0, 'both_fail': 0}
            for arm in ('fp16', 'q8'):
                entries = [reg.get((t['id'], arm), {'status': 'NOT_RUN'}) for t in selected]
                counts = collections.Counter(r['status'] for r in entries)
                toolcalls = [c for r in entries for c in r.get('calls', [])]
                cat['arms'][arm] = {'counts': dict(counts), 'tool_json_calls': len(toolcalls),
                                    'tool_json_valid': sum(c['json_valid'] for c in toolcalls)}
            for task in selected:
                a, b = (reg.get((task['id'], arm), {}).get('status') for arm in ('fp16', 'q8'))
                if a in ('PASS', 'FAIL') and b in ('PASS', 'FAIL'):
                    cat['paired'] += 1
                    label = 'both_pass' if a == b == 'PASS' else 'both_fail' if a == b else 'fp16_only' if a == 'PASS' else 'q8_only'
                    cat[label] += 1
            categories[category] = cat
            scores = []
            for arm in ('fp16', 'q8'):
                counts = cat['arms'][arm]['counts']
                scores.append(f"{counts.get('PASS', 0)}/{counts.get('PASS', 0) + counts.get('FAIL', 0)}")
            lines += [f"| {suite}/{category} | {scores[0]} | {scores[1]} | {cat['paired']} | {cat['fp16_only']} | {cat['q8_only']} | {cat['both_fail']} |"]
        complete = all(reg.get((t['id'], arm), {}).get('status') in ('PASS', 'FAIL') for t in tasks for arm in ('fp16', 'q8'))
        code_scores = {a: sum(reg.get((t['id'], a), {}).get('status') == 'PASS' for t in tasks if t['category'] == 'code') for a in ('fp16', 'q8')}
        metrics = {}
        for arm in ('fp16', 'q8'):
            records = [r for (task_id, a), r in reg.items() if a == arm and r['status'] in ('PASS', 'FAIL')]
            requests = [q for r in records for q in r.get('requests', [])]
            metrics[arm] = {key: distribution(values) for key, values in {
                'task_wall_s': [r.get('wall_s') for r in records],
                'request_ttft_s': [q.get('ttft_s') for q in requests],
                'prefill_tps': [(q.get('timings') or {}).get('prompt_per_second') for q in requests],
                'decode_tps': [(q.get('timings') or {}).get('predicted_per_second') for q in requests]}.items()}
            metrics[arm]['prompt_tokens'] = sum(q['usage']['prompt_tokens'] for q in requests) if requests else None
            metrics[arm]['completion_tokens'] = sum(q['usage']['completion_tokens'] for q in requests) if requests else None
        summary['suites'][suite] = {'complete': complete, 'categories': categories, 'code_scores': code_scores,
                                   'metrics': metrics,
                                   'comparison': compare_tasks(tasks, reg),
                                   'code_delta': code_scores['q8'] - code_scores['fp16'] if complete else None}
    api = registry(campaign / 'results-api.jsonl', manifest['freeze_id'])
    summary['api'] = {arm: api.get(('api15', arm), {}).get('status', 'NOT_RUN') for arm in ('fp16', 'q8')}
    original = summary['suites']['original']
    toolcat = summary['suites']['pilot']['categories']['tools']
    json_complete = toolcat['paired'] == 12
    json_ok = json_complete and all(v['tool_json_calls'] > 0 and v['tool_json_valid'] == v['tool_json_calls'] for v in toolcat['arms'].values())
    if original['complete'] and all(s in ('PASS', 'FAIL') for s in summary['api'].values()) and json_complete:
        summary['legacy_original_t8'] = 'PASS' if original['code_delta'] >= -1 and all(s == 'PASS' for s in summary['api'].values()) and json_ok else 'FAIL'
    summary['original_t8'] = original['comparison']['status']
    api_complete = all(s in ('PASS', 'FAIL') for s in summary['api'].values())
    integration_status = ('ERROR' if any(s not in ('PASS', 'FAIL', 'NOT_RUN') for s in summary['api'].values()) else
                          'INCOMPLETE' if not api_complete or not json_complete else
                          'PASS' if all(s == 'PASS' for s in summary['api'].values()) and json_ok else 'BLOCKING')
    summary['integration_checks'] = {'status': integration_status,
                                     'api15': summary['api'], 'all_tool_json_valid': json_ok,
                                     'complete_tool_pairs': json_complete,
                                     'meaning': 'Protocol/integration checks, separate from model task quality'}
    lines += ['', f"Paired original-code assessment: {summary['original_t8']}",
              f"Paired pilot assessment: {summary['suites']['pilot']['comparison']['status']}",
              f"Legacy original T8 criterion (descriptive): {summary['legacy_original_t8']}", '',
              f"Protocol/integration checks: {integration_status}", '',
              'Q8-only failures: ' + json.dumps({s: v['comparison']['q8_only_failures'] for s, v in summary['suites'].items()}),
              'Q8-only successes: ' + json.dumps({s: v['comparison']['q8_only_successes'] for s, v in summary['suites'].items()}), '',
              'API15: ' + json.dumps(summary['api']), '',
              'Pilot tool JSON calls: ' + json.dumps({a: {'valid': v['tool_json_valid'], 'total': v['tool_json_calls']} for a, v in toolcat['arms'].items()}), '',
              'Per-arm token totals and timing distributions are in SUMMARY.json.', '',
              'Historical Phase 4 / Phase 4b gates were not rewritten.',
              'No superiority or promotion conclusion is drawn from these pairs. Missing metrics stay null.',
              'Original and pilot share C01..C14: do not pool them as independent replications.']
    write_json(campaign / 'SUMMARY.json', summary)
    (campaign / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


def distribution(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return {'n': len(values), 'min': min(values), 'median': statistics.median(values), 'max': max(values)} if values else None


def reuse_shared_code(campaign, suite, regpath, freeze_id):
    """C01..C14 are the same frozen tasks; never generate a second main answer."""
    if suite == 'api':
        return
    other = 'original' if suite == 'pilot' else 'pilot'
    source = registry(campaign / f'results-{other}.jsonl', freeze_id)
    target = registry(regpath, freeze_id)
    for (task_id, arm), rec in source.items():
        if task_id not in {f'C{n:02}' for n in range(1, 15)} or rec['status'] not in ('PASS', 'FAIL'):
            continue
        if (task_id, arm) in target:
            continue
        reused = {k: v for k, v in rec.items() if k != 'result_sha'}
        reused.update(suite=suite, reused_from={'suite': other, 'result_sha': rec['result_sha']}, reused_at=now())
        reused['result_sha'] = digest(reused)
        append(regpath, reused)


def block_config(experiment, work, arm):
    profile = work / 'expert-profile.txt'
    shutil.copyfile(experiment['profile'], profile)
    return {'exe': experiment['exe'], 'cwd': str(work), 'args': ['--model', experiment['model'],
            '--expert-profile', str(profile), *experiment['common_args'], *(['--kv', 'q8_0'] if arm == 'q8' else [])],
            'tokenizer': experiment['tokenizer'], 'model_name': 'functional-t8-experimental', 'backend': 'hip',
            'lib_dirs': experiment['lib_dirs'], 'env': experiment['env'], 'log': str(work / 'engine.log'),
            'host': '127.0.0.1', 'switch_on_request': False, 'open_browser': False,
            'tool_call_recovery': False, 'reasoning_loop_recovery': False, 'repeat_stop_tokens': 0,
            'sampling': {'temperature': 0}}


def run(campaign, suite, retry_infra=False, *, final_report=True):
    manifest = audit(campaign)
    freeze_id = manifest['freeze_id']
    experiment = strict_json((campaign / 'EXPERIMENT.json').read_text(encoding='utf-8'))
    tasks = {t['id']: t for t in rows(campaign / ('TASKS.jsonl' if suite == 'pilot' else 'ORIGINAL_CODE.jsonl'))}
    regpath, events = campaign / f'results-{suite}.jsonl', campaign / 'events.jsonl'
    plan = experiment['pilot_order' if suite == 'pilot' else 'original_order']
    if suite == 'api':
        plan = [{'block': 0, 'arm': 'fp16', 'task_ids': ['api15']}, {'block': 1, 'arm': 'q8', 'task_ids': ['api15']}]
    with ExclusiveGPU(ROOT / 'tmp/functional-quality/gpu.lock'):
        reuse_shared_code(campaign, suite, regpath, freeze_id)
        reg = registry(regpath, freeze_id)
        started = {(r['task_id'], r['arm']) for r in rows(events) if r.get('kind') == 'task_started' and r.get('suite') == suite}
        for key in started:
            if (key not in reg or reg[key]['status'] == 'INFRA_ERROR') and not retry_infra:
                raise RuntimeError('interrupted/infrastructure attempt requires --retry-infra: ' + str(key))
        for block in plan:
            pending = [i for i in block['task_ids'] if reg.get((i, block['arm']), {}).get('status') not in COMPLETED]
            if not pending:
                continue
            state = snapshot()
            reasons = admission(state)
            append(events, {'at': now(), 'kind': 'admission', 'suite': suite, 'block': block, 'memory': state, 'refused': reasons})
            write_json(campaign / 'PREFLIGHT.json', {'at': now(), 'memory': state, 'refused': reasons})
            if reasons:
                raise RuntimeError('GPU launch refused: ' + '; '.join(reasons))
            work = campaign / 'runtime' / f'{suite}-{block["block"]}-{time.time_ns()}'
            work.mkdir(parents=True)
            config = block_config(experiment, work, block['arm'])
            config_path = work / 'config.json'
            write_json(config_path, config)
            server = ExperimentalServer(config_path, config, experiment['port'], events)
            try:
                client = server.start()
                warm = client.complete([{'role': 'user', 'content': 'Reply only READY.'}], [], 16)
                append(events, {'at': now(), 'kind': 'warmup', 'suite': suite, 'block': block, 'response': warm})
                for task_id in pending:
                    attempt = work / f'{task_id}-{block["arm"]}'
                    attempt.mkdir()
                    event_file = attempt / 'transcript.jsonl'
                    before = snapshot(server.owned)
                    append(events, {'at': now(), 'kind': 'task_started', 'suite': suite, 'task_id': task_id,
                                    'arm': block['arm'], 'attempt': str(attempt), 'freeze_id': freeze_id})
                    rec = {'at': now(), 'suite': suite, 'task_id': task_id, 'arm': block['arm'],
                           'freeze_id': freeze_id, 'block': block['block'], 'memory_before': before,
                           'config': config, 'config_sha': sha(config_path), 'event_file': str(event_file)}
                    try:
                        if before['ram_available_mib'] < MIN_RUNNING_FREE_RAM_MIB or before['other_inference']:
                            raise RuntimeError('resource pressure or competing inference during block')
                        # No unobserved engine recovery or process replacement is admitted.
                        verify_owned(server)
                        if suite == 'api':
                            rec.update(run_api(client, attempt, event_file))
                        else:
                            rec.update(evaluate(tasks[task_id], client, attempt / 'project', event_file))
                        verify_owned(server)
                        check_engine_log(config['log'])
                        rec['memory_after'] = snapshot(server.owned)
                        if rec['memory_after']['ram_available_mib'] < MIN_RUNNING_FREE_RAM_MIB:
                            raise RuntimeError('low RAM after task')
                    except Exception as exc:
                        rec.update(status='INFRA_ERROR', error=f'{type(exc).__name__}: {exc}')
                    if not event_file.exists():
                        append(event_file, {'at': now(), 'error': rec.get('error')})
                    rec['event_sha'] = sha(event_file)
                    rec['result_sha'] = digest(rec)
                    append(regpath, rec)
                    reg[(task_id, block['arm'])] = rec
                    write_json(campaign / 'RUN_CHECKPOINT.json', {'at': now(), 'freeze_id': freeze_id,
                               'suite': suite, 'last': [task_id, block['arm']], 'status': rec['status'],
                               'results_sha': sha(regpath)})
                    print(f'{suite} {task_id} {block["arm"]}: {rec["status"]}', flush=True)
                    if rec['status'] == 'INFRA_ERROR':
                        raise RuntimeError(rec['error'])
            finally:
                server.close()
            # Detect any frozen or operational input drift before the next block.
            audit(campaign)
    if final_report:
        task_set = experiment.get('task_set', 'standard')
        if task_set == 'average':
            from .average_quality import report as average_report
            average_report(campaign)
        elif task_set == 'r05':
            from .r05_experiment import report as r05_report
            r05_report(campaign)
        else:
            report(campaign)


def check_engine_log(path):
    log = Path(path).read_text(encoding='utf-8', errors='replace')
    for marker in ('GPU_FATAL', 'out of memory', 'nonfinite', 'hipErrorOutOfMemory'):
        if marker.lower() in log.lower():
            raise RuntimeError('engine failure marker: ' + marker)


def verify_owned(server):
    import psutil
    if server.proc.poll() is not None:
        raise RuntimeError('experimental server died')
    for pid, identity in server.owned.items():
        p = psutil.Process(pid)
        if p.create_time() != identity['created'] or p.exe() != identity['exe'] or p.cmdline() != identity['cmd']:
            raise RuntimeError('owned process identity changed')
    children = psutil.Process(server.proc.pid).children(recursive=True)
    current = {p.pid for p in children}
    if current != set(server.owned) - {server.proc.pid}:
        unknown = [(p.pid, p.name(), p.cmdline()) for p in children if p.pid not in server.owned]
        raise RuntimeError(f'engine was restarted or unexpected child appeared: unknown={unknown}, current={current}, owned={server.owned}')


def run_api(client, attempt, event_file):
    command = [sys.executable, str(ROOT / 'tools/hip/poly_api_check.py'), '--base', client.base,
               '--json', str(attempt / 'api15.json')]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    (attempt / 'api15.txt').write_text(proc.stdout + proc.stderr, encoding='utf-8')
    data = strict_json((attempt / 'api15.json').read_text(encoding='utf-8'))
    checks = {k: v for k, v in data.items() if isinstance(v, dict) and 'ok' in v}
    if len(checks) != 15:
        raise RuntimeError('poly_api_check did not return all 15 checks')
    append(event_file, {'at': now(), 'command': command, 'result': data,
                        'output': proc.stdout + proc.stderr, 'returncode': proc.returncode})
    return {'status': 'PASS' if proc.returncode == 0 and all(c['ok'] for c in checks.values()) else 'FAIL',
            'api_checks': data, 'wall_s': time.perf_counter() - started}


def main():
    ap = argparse.ArgumentParser(description='Frozen opt-in T8 functional evaluation')
    ap.add_argument('command', choices=['prepare', 'audit', 'preflight', 'run', 'report'])
    ap.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    ap.add_argument('--suite', choices=['pilot', 'original', 'api'], default='pilot')
    ap.add_argument('--retry-infra', action='store_true')
    args = ap.parse_args()
    campaign = args.campaign.resolve()
    try:
        if args.command == 'prepare':
            prepare(campaign)
        elif args.command == 'audit':
            freeze_id = audit(campaign)['freeze_id']
            print('Integrity PASS: ' + freeze_id)
            for suite in ('pilot', 'original', 'api'):
                registry(campaign / f'results-{suite}.jsonl', freeze_id)
        elif args.command == 'preflight':
            state = snapshot()
            value = {'at': now(), 'memory': state, 'refused': admission(state)}
            write_json(campaign / 'PREFLIGHT.json', value)
            print(json.dumps(value, indent=2))
            return 2 if value['refused'] else 0
        elif args.command == 'run':
            run(campaign, args.suite, args.retry_infra)
        else:
            report(campaign)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
