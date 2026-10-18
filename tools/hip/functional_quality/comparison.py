"""Paired Q8/FP16 assessment; absolute task scores stay unchanged.

Shared task failures are not attributed to KV quantization. Discordant outcomes
require review; improvements never cancel Q8-only failures. No causal or
statistical equivalence claim is made from a single paired observation.
"""
from __future__ import annotations

from collections import Counter
import json
import math

POLICY_ID = 'KV-Q8-paired-v1-2026-10-17'
TERMINAL = frozenset({'PASS', 'FAIL'})


def normalized_args(args, arm, *, strip_profile=True):
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError('arguments must be a list of strings')
    result, formats = [], []
    i = 0
    while i < len(args):
        flag = args[i]
        if flag in ('--kv', '--expert-profile') and (flag == '--kv' or strip_profile):
            if i + 1 >= len(args) or args[i + 1].startswith('--'):
                raise ValueError('missing value for ' + flag)
            if flag == '--kv':
                formats.append(args[i + 1])
            i += 2
            continue
        if flag.startswith(('--kv=', '--kv-k', '--kv-v')):
            raise ValueError('unsupported or mixed KV override')
        result.append(flag)
        i += 1
    expected = 'q8_0' if arm == 'q8' else 'fp16'
    if len(formats) > 1 or formats and formats != [expected] or arm == 'q8' and not formats:
        raise ValueError('KV flags do not match the recorded arm')
    return result


def configuration(record, arm):
    cfg = record.get('config')
    if not isinstance(cfg, dict) or not cfg.get('exe') or not record.get('freeze_id'):
        raise ValueError('missing frozen executable/configuration identity')
    # These paths are per-process outputs. Profile input bytes are pinned by
    # the common freeze and checked by server admission before each block.
    value = {k: v for k, v in cfg.items() if k not in ('cwd', 'log', 'args')}
    value['args'] = normalized_args(cfg.get('args'), arm)
    value['freeze_id'] = record['freeze_id']
    return value


def compare_task(task_id, fp16, q8):
    out = {'task_id': task_id, 'fp16': (fp16 or {}).get('status', 'NOT_RUN'),
           'q8': (q8 or {}).get('status', 'NOT_RUN'), 'policy': POLICY_ID}
    if any(s not in TERMINAL for s in (out['fp16'], out['q8'])):
        out['classification'] = ('INFRA_ERROR' if any(s not in TERMINAL | {'NOT_RUN'}
                                                    for s in (out['fp16'], out['q8'])) else 'INCOMPLETE')
        return out
    try:
        if fp16.get('arm') != 'fp16' or q8.get('arm') != 'q8':
            raise ValueError('wrong arm identity')
        if fp16.get('task_id') != task_id or q8.get('task_id') != task_id:
            raise ValueError('wrong task identity')
        if configuration(fp16, 'fp16') != configuration(q8, 'q8'):
            raise ValueError('different frozen inputs or runtime settings')
        if any(r.get(k) for r in (fp16, q8) for k in ('integrity_error', 'nonfinite', 'config_mismatch')):
            raise ValueError('integrity failure cannot become a shared model failure')
    except (ValueError, TypeError, KeyError) as exc:
        out.update(classification='CONFIG_OR_INTEGRITY_ERROR', reason=str(exc))
        return out
    out['classification'] = ('BOTH_PASS' if out['fp16'] == out['q8'] == 'PASS' else
                             'SHARED_TASK_FAILURE' if out['fp16'] == out['q8'] == 'FAIL' else
                             'Q8_ONLY_FAILURE_OBSERVED' if out['fp16'] == 'PASS' else 'Q8_ONLY_SUCCESS_OBSERVED')
    out['causal_attribution'] = 'NOT_ESTABLISHED'
    return out


def compare_tasks(tasks, records):
    """Records are keyed by (task id, arm), after transcript/hash validation."""
    ids = [t['id'] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate task identity')
    pairs = [dict(compare_task(t['id'], records.get((t['id'], 'fp16')), records.get((t['id'], 'q8'))),
                  category=t.get('category', 'unknown')) for t in tasks]
    counts = Counter(p['classification'] for p in pairs)
    if counts['CONFIG_OR_INTEGRITY_ERROR'] or counts['INFRA_ERROR']:
        status = 'ERROR'
    elif not pairs or counts['INCOMPLETE']:
        status = 'INCOMPLETE'
    elif counts['Q8_ONLY_FAILURE_OBSERVED']:
        status = 'REVIEW_REQUIRED'
    elif counts['BOTH_PASS'] + counts['Q8_ONLY_SUCCESS_OBSERVED'] == 0:
        status = 'BASELINE_LIMITED'
    else:
        status = 'NO_Q8_REGRESSION_OBSERVED'
    return {'policy': POLICY_ID, 'status': status, 'counts': dict(counts), 'pairs': pairs,
            'q8_only_failures': [p['task_id'] for p in pairs if p['classification'] == 'Q8_ONLY_FAILURE_OBSERVED'],
            'q8_only_successes': [p['task_id'] for p in pairs if p['classification'] == 'Q8_ONLY_SUCCESS_OBSERVED'],
            'shared_task_failures': [p['task_id'] for p in pairs if p['classification'] == 'SHARED_TASK_FAILURE'],
            'general_equivalence_proven': False, 'q8_promotion': 'NOT_YET_APPROVED'}


def text_diagnostic(verdict, review=None):
    """T5 trajectory metrics are diagnostics, never a functional correctness oracle."""
    if any(verdict.get('integrity', {}).values()) or verdict.get('classification') in (
            'NONFINITE', 'INTEGRITY_ERROR', 'EXEC_ERROR', 'CONFIG_MISMATCH'):
        return {'status': 'ERROR', 'policy': POLICY_ID, 'functional_result': 'NOT_EVALUATED'}
    if verdict.get('result') in ('NOT_RUN', 'ERROR'):
        return {'status': 'INCOMPLETE', 'policy': POLICY_ID, 'functional_result': 'NOT_EVALUATED'}
    if verdict.get('result') not in TERMINAL or not verdict.get('fingerprint'):
        return {'status': 'ERROR', 'policy': POLICY_ID, 'functional_result': 'NOT_EVALUATED'}
    reviewed = False
    if review is not None:
        if json.dumps(review.get('fingerprint'), sort_keys=True) != json.dumps(verdict['fingerprint'], sort_keys=True):
            return {'status': 'ERROR', 'policy': POLICY_ID, 'reason': 'review fingerprint mismatch'}
        reviewed = (review.get('review_state') == 'CLOSED' and
                    review.get('decision') == 'ACCEPT_REVIEWED_RESIDUAL_DIVERGENCE')
    return {'status': 'DIAGNOSTIC_ONLY', 'policy': POLICY_ID, 'fingerprint': verdict['fingerprint'],
            'historical_strict_result': verdict['result'], 'observed': verdict.get('observed', {}),
            'functional_result': 'REVIEWED_RESIDUAL_DIVERGENCE' if reviewed else 'NOT_EVALUATED',
            'outlier_review_pending': verdict['result'] == 'FAIL' and not reviewed}


def mtp_diagnostic(fp16, q8):
    """Report matched efficiency/performance separately; acceptance is not quality."""
    out = {'policy': POLICY_ID, 'status': 'EFFICIENCY_DIAGNOSTIC', 'quality_result': 'NOT_EVALUATED',
           'performance_verdict': 'INCONCLUSIVE', 'general_equivalence_proven': False}
    try:
        required = ('ctx', 'cap', 'exe_sha', 'prompt_sha', 'expert_set', 'flags')
        if any(k not in r or r[k] is None for r in (fp16, q8) for k in required):
            raise ValueError('missing control metadata')
        if any(fp16[k] != q8[k] for k in required):
            raise ValueError('different model/prompt/experts/runtime controls')
        if fp16.get('rc') != 0 or q8.get('rc') != 0:
            raise ValueError('execution failed')
        if normalized_args(fp16.get('args'), 'fp16') != normalized_args(q8.get('args'), 'q8'):
            raise ValueError('different generation controls')
        rates = {}
        for arm, r in (('fp16', fp16), ('q8', q8)):
            right, offered = r.get('right'), r.get('guessed')
            if type(right) is not int or type(offered) is not int or not 0 <= right <= offered or offered == 0:
                raise ValueError('missing or invalid MTP counters')
            rates[arm] = 100 * right / offered
        out.update(acceptance_pct=rates, delta_acceptance_pp=rates['q8'] - rates['fp16'],
                   legacy_minus_2pp_satisfied=rates['q8'] >= rates['fp16'] - 2)
        same_ids = bool(fp16.get('ids')) and fp16['ids'] == q8.get('ids')
        out['same_generated_tokens'] = same_ids
        # Even matching output does not establish calibrated timing, repeated
        # measurements or equal speculative schedules. No automatic perf PASS.
        out['decode_tps_descriptive'] = {a: r.get('decode_tps') for a, r in (('fp16', fp16), ('q8', q8))}
        if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                                 or not math.isfinite(v) or v <= 0)
               for v in out['decode_tps_descriptive'].values()):
            raise ValueError('invalid timing metric')
        out['performance_limitation'] = ('No calibrated repeated paired timings/schedule control; '
                                        'different text also changes the workload' if not same_ids else
                                        'No calibrated repeated paired timings/schedule control')
    except (ValueError, TypeError, KeyError) as exc:
        out.update(status='INCOMPARABLE', reason=str(exc))
    return out
