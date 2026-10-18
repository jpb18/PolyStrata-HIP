"""CPU regression checks for attribution, paired outcomes and diagnostic gates."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tools.hip.functional_quality.comparison import (
    compare_task, compare_tasks, mtp_diagnostic, normalized_args, text_diagnostic,
)


def record(arm, status='PASS', task='A'):
    return {'task_id': task, 'arm': arm, 'status': status, 'freeze_id': 'same-inputs',
            'config': {'exe': 'same-engine', 'tokenizer': 'same-tokenizer',
                       'args': ['--model', 'same-model', '--max-context', '49152',
                                '--expert-profile', arm + '-private-profile', '--expert-cache-max', '5000']
                               + (['--kv', 'q8_0'] if arm == 'q8' else []),
                       'env': {'STRATA_POLY_ATTN_FIT': '1'}, 'cwd': arm, 'log': arm + '.log'}}


def native(arm, right=90, offered=100):
    return {'rc': 0, 'ctx': 49152, 'cap': 5000, 'exe_sha': 'exe', 'prompt_sha': 'prompt',
            'expert_set': 'same-set', 'flags': {'STRATA_POLY_WARM': '1'},
            'args': ['--window', '4'] + (['--kv', 'q8_0'] if arm == 'q8' else []),
            'right': right, 'guessed': offered, 'ids': [1, 2, 3], 'decode_tps': 50.0}


class PairedTests(unittest.TestCase):
    def test_same_failure_is_not_a_q8_regression_or_an_automatic_pass(self):
        tasks = [{'id': 'A'}]
        records = {('A', a): record(a, 'FAIL') for a in ('fp16', 'q8')}
        r = compare_tasks(tasks, records)
        self.assertEqual(r['status'], 'BASELINE_LIMITED')
        self.assertEqual(r['shared_task_failures'], ['A'])
        self.assertEqual(r['q8_only_failures'], [])
        self.assertFalse(r['general_equivalence_proven'])

    def test_all_four_paired_outcomes(self):
        expected = {('PASS', 'PASS'): 'BOTH_PASS', ('FAIL', 'FAIL'): 'SHARED_TASK_FAILURE',
                    ('PASS', 'FAIL'): 'Q8_ONLY_FAILURE_OBSERVED', ('FAIL', 'PASS'): 'Q8_ONLY_SUCCESS_OBSERVED'}
        for (fp, q8), classification in expected.items():
            with self.subTest(fp=fp, q8=q8):
                self.assertEqual(compare_task('A', record('fp16', fp), record('q8', q8))['classification'], classification)

    def test_equal_scores_cannot_cancel_regression_candidates(self):
        tasks = [{'id': 'A'}, {'id': 'B'}]
        records = {('A', 'fp16'): record('fp16'), ('A', 'q8'): record('q8', 'FAIL'),
                   ('B', 'fp16'): record('fp16', 'FAIL', 'B'), ('B', 'q8'): record('q8', 'PASS', 'B')}
        r = compare_tasks(tasks, records)
        self.assertEqual(r['status'], 'REVIEW_REQUIRED')
        self.assertEqual(r['q8_only_failures'], ['A'])
        self.assertEqual(r['q8_only_successes'], ['B'])

    def test_shared_failures_do_not_block_informative_matched_passes(self):
        tasks = [{'id': 'A'}, {'id': 'B'}]
        records = {(t, a): record(a, 'PASS' if t == 'A' else 'FAIL', t)
                   for t in ('A', 'B') for a in ('fp16', 'q8')}
        self.assertEqual(compare_tasks(tasks, records)['status'], 'NO_Q8_REGRESSION_OBSERVED')

    def test_different_controls_are_not_quantization_results(self):
        for mutation in ('freeze', 'experts', 'env', 'model', 'kv'):
            fp, q8 = record('fp16'), record('q8')
            if mutation == 'freeze': q8['freeze_id'] = 'other-inputs'
            if mutation == 'experts': q8['config']['args'][7] = '4000'
            if mutation == 'env': q8['config']['env']['STRATA_POLY_ATTN_FIT'] = '0'
            if mutation == 'model': q8['config']['args'][1] = 'other-model'
            if mutation == 'kv': q8['config']['args'][-1] = 'fp16'
            with self.subTest(mutation=mutation):
                self.assertEqual(compare_task('A', fp, q8)['classification'], 'CONFIG_OR_INTEGRITY_ERROR')

    def test_infrastructure_and_missing_results_never_become_shared_failures(self):
        tasks = [{'id': 'A'}]
        for state in ('INFRA_ERROR', 'NOT_RUN'):
            records = {('A', 'fp16'): record('fp16', state), ('A', 'q8'): record('q8', state)}
            with self.subTest(state=state):
                r = compare_tasks(tasks, records)
                self.assertEqual(r['status'], 'ERROR' if state == 'INFRA_ERROR' else 'INCOMPLETE')
        self.assertEqual(compare_tasks(tasks, {})['status'], 'INCOMPLETE')

    def test_integrity_failure_not_masked_by_both_task_failures(self):
        fp, q8 = record('fp16', 'FAIL'), record('q8', 'FAIL')
        q8['nonfinite'] = True
        self.assertEqual(compare_task('A', fp, q8)['classification'], 'CONFIG_OR_INTEGRITY_ERROR')

    def test_wrong_task_and_duplicate_tasks_rejected(self):
        self.assertEqual(compare_task('A', record('fp16', task='B'), record('q8'))['classification'],
                         'CONFIG_OR_INTEGRITY_ERROR')
        with self.assertRaises(ValueError):
            compare_tasks([{'id': 'A'}, {'id': 'A'}], {})

    def test_malformed_or_mixed_kv_flags_rejected(self):
        for args in (['--kv'], ['--kv', 'q8_0', '--kv', 'q8_0'], ['--kv-k', 'q8_0'], ['--kv=q8_0'], []):
            with self.subTest(args=args), self.assertRaises(ValueError):
                normalized_args(args, 'q8')

    def test_text_difference_is_not_a_functional_failure(self):
        v = {'result': 'FAIL', 'classification': 'NEW_DIVERGENCE', 'fingerprint': {'case': 'A'},
             'observed': {'ref_gap': .36}, 'integrity': {}}
        r = text_diagnostic(v)
        self.assertEqual(r['status'], 'DIAGNOSTIC_ONLY')
        self.assertEqual(r['functional_result'], 'NOT_EVALUATED')
        self.assertTrue(r['outlier_review_pending'])
        review = {'fingerprint': v['fingerprint'], 'review_state': 'CLOSED',
                  'decision': 'ACCEPT_REVIEWED_RESIDUAL_DIVERGENCE'}
        r = text_diagnostic(v, review)
        self.assertFalse(r['outlier_review_pending'])
        self.assertEqual(r['functional_result'], 'REVIEWED_RESIDUAL_DIVERGENCE')

    def test_text_integrity_and_wrong_review_cannot_be_approved(self):
        v = {'result': 'FAIL', 'fingerprint': {'case': 'A'}, 'integrity': {'nonfinite': True}}
        self.assertEqual(text_diagnostic(v)['status'], 'ERROR')
        v['integrity'] = {}
        self.assertEqual(text_diagnostic(v, {'fingerprint': {'case': 'B'}})['status'], 'ERROR')
        self.assertEqual(text_diagnostic({'result': 'NOT_RUN'})['status'], 'INCOMPLETE')

    def test_mtp_acceptance_drop_is_efficiency_not_quality_failure(self):
        r = mtp_diagnostic(native('fp16'), native('q8', 70))
        self.assertEqual(r['delta_acceptance_pp'], -20)
        self.assertFalse(r['legacy_minus_2pp_satisfied'])
        self.assertEqual(r['status'], 'EFFICIENCY_DIAGNOSTIC')
        self.assertEqual(r['quality_result'], 'NOT_EVALUATED')
        self.assertEqual(r['performance_verdict'], 'INCONCLUSIVE')

    def test_mtp_missing_mismatched_or_nonfinite_data_never_pass(self):
        for mutation in ('counters', 'timing', 'window', 'prompt'):
            q = native('q8')
            if mutation == 'counters': q['guessed'] = 0
            if mutation == 'timing': q['decode_tps'] = float('nan')
            if mutation == 'window': q['args'][1] = '8'
            if mutation == 'prompt': q['prompt_sha'] = 'different'
            with self.subTest(mutation=mutation):
                self.assertEqual(mtp_diagnostic(native('fp16'), q)['status'], 'INCOMPARABLE')

    def test_mtp_different_text_does_not_establish_performance_regression(self):
        q = native('q8')
        q['ids'] = [9, 8, 7]
        r = mtp_diagnostic(native('fp16'), q)
        self.assertFalse(r['same_generated_tokens'])
        self.assertEqual(r['performance_verdict'], 'INCONCLUSIVE')


class ReportAndCLITests(unittest.TestCase):
    def test_needle_actual_experts_and_process_exit_are_checked(self):
        from tools.hip.kv_q8_replay import needles
        request = {'prompt': 'note-A', 'single': True, 'codes': ['A']}
        manifest = {'needles': {'n8192-v2': {'ctx': 49152, 'requests': [request]}}}
        records = []
        for fit in (0, 1):
            for arm in ('fp16', 'q8'):
                records.append({'tag': f'cp2-needle-n8192-v2-{arm}-fit{fit}', 'rc': 0, 'process_rc': 0,
                                'ctx': 49152, 'fit': bool(fit), 'kv': arm, 'expert_set': 'same-set',
                                'key_src': {'exe': 'same-exe', 'ctx': 49152, 'fit': bool(fit),
                                            'kv': arm, 'requests': [request]},
                                'rows': [{'label': 'note-A', 'codes': ['A'], 'found': ['A']}]})
        self.assertEqual(needles(records, manifest)[0]['status'], 'NO_Q8_REGRESSION_OBSERVED')
        records[1]['expert_set'] = 'other-set'
        self.assertEqual(needles(records, manifest)[0]['status'], 'ERROR')
        records[1]['expert_set'] = 'same-set'
        records[1]['process_rc'] = 70
        self.assertEqual(needles(records, manifest)[0]['status'], 'ERROR')

    def test_missing_both_needle_arms_still_has_a_denominator(self):
        from tools.hip.kv_q8_replay import needles
        manifest = {'needles': {'n8192-v2': {'ctx': 49152, 'requests': [
            {'prompt': 'note-A', 'single': True, 'codes': ['A']}]}}}
        result, logs = needles([], manifest)
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertEqual(len(result['pairs']), 2)  # the two planned FIT conditions
        self.assertEqual(logs, set())

    def test_replay_refuses_writes_inside_history(self):
        from tools.hip.kv_q8_replay import replay
        from tools.hip.functional_quality.common import scratch
        with scratch() as directory:
            source = Path(directory) / 'historic'
            with self.assertRaises(ValueError):
                replay(source, Path(directory) / 'native', Path(directory) / 'reviews.json', source / 'new-report')
            self.assertFalse(source.exists())

    def test_live_campaign_report_uses_paired_assessment(self):
        from tools.hip.functional_quality import campaign
        from tools.hip.functional_quality.common import append, digest, scratch
        with scratch() as directory:
            path = Path(directory)
            code = [{'id': t, 'category': 'code'} for t in ('A', 'B')]
            tasks = code + [{'id': 'T', 'category': 'tools'}]
            for filename, selected in (('TASKS.jsonl', tasks), ('ORIGINAL_CODE.jsonl', code)):
                for task in selected:
                    append(path / filename, task)
            for suite, selected in (('pilot', tasks), ('original', code)):
                for task in selected:
                    for arm in ('fp16', 'q8'):
                        status = 'FAIL' if task['id'] == 'T' or (task['id'] == 'A' and arm == 'q8') or (task['id'] == 'B' and arm == 'fp16') else 'PASS'
                        r = record(arm, status, task['id'])
                        r['result_sha'] = digest(r)
                        append(path / f'results-{suite}.jsonl', r)
            for arm in ('fp16', 'q8'):
                r = record(arm, task='api15')
                r['result_sha'] = digest(r)
                append(path / 'results-api.jsonl', r)
            with patch.object(campaign, 'audit', return_value={'freeze_id': 'same-inputs'}), contextlib.redirect_stdout(io.StringIO()):
                campaign.report(path)
            result = json.loads((path / 'SUMMARY.json').read_text())
            self.assertEqual(result['original_t8'], 'REVIEW_REQUIRED')
            self.assertEqual(result['integration_checks']['status'], 'INCOMPLETE')
            paired = result['suites']['pilot']['comparison']
            self.assertEqual(paired['q8_only_failures'], ['A'])
            self.assertEqual(paired['q8_only_successes'], ['B'])
            self.assertEqual(paired['shared_task_failures'], ['T'])
            self.assertIn('Shared failures', (path / 'RESULTS.md').read_text())

    def test_future_freezes_include_the_comparator_and_its_tests(self):
        from tools.hip.functional_quality.campaign import source_inputs
        from tools.hip.functional_quality.common import DEFAULT_CAMPAIGN
        names = {str(p).replace('\\', '/') for p in source_inputs()}
        self.assertTrue(any(n.endswith('/functional_quality/comparison.py') for n in names))
        self.assertTrue(any(n.endswith('/tests/test_kv_q8_comparison.py') for n in names))
        self.assertEqual(DEFAULT_CAMPAIGN.name, '2026-10-17-hip-kv-q8-paired-functional')

    def test_parity_and_q8_cli_modes_have_different_meaning(self):
        import numpy as np
        from tools.hip import kv_parity
        from tools.hip.functional_quality.common import scratch
        with scratch() as directory:
            prompt = Path(directory) / 'prompt.txt'
            prompt.write_text('1,2', encoding='utf-8')
            for mode, expected_exit in (('parity', 1), ('q8-relative', 0)):
                def run(args, extra, tag, wd, engine=None):
                    q8 = tag != 'ref'
                    logits = np.array([[0, 5, -1] if q8 else [5, 0, -1]] * 2, dtype=np.float32).ravel()
                    return {'args': extra, 'ids': [1, 1] if q8 else [0, 0], 'logits': logits,
                            'info': [], 'expert_set': 'fixed-set', 'wall_s': 1}
                argv = ['kv_parity', '--engine', 'fixture', '--model', 'fixture', '--prompt', str(prompt),
                        '--ctx', '49152', '--ref', '--kv fp16', '--test', '--kv q8_0',
                        '--args', '--expert-profile fixture --expert-cache-max 5000 --expert-swaps 0',
                        '--comparison-mode', mode]
                output = io.StringIO()
                with patch.object(sys, 'argv', argv), patch.object(kv_parity, 'run', side_effect=run), contextlib.redirect_stdout(output):
                    self.assertEqual(kv_parity.main(), expected_exit)
                if mode == 'q8-relative':
                    self.assertIn('DIAGNOSTIC_ONLY', output.getvalue())

    def test_parity_nonfinite_after_first_divergence_is_not_hidden(self):
        import numpy as np
        from tools.hip.kv_parity import compare
        ref = {'ids': [0, 0], 'logits': np.array([[5, 0], [5, 0]], dtype=np.float32)}
        q8 = {'args': '', 'ids': [1, 1], 'logits': np.array([[0, 5], [float('nan'), 0]], dtype=np.float32)}
        with self.assertRaises(ValueError):
            compare(ref, q8, 2)

    def test_parity_missing_capture_and_different_lengths_are_visible(self):
        import numpy as np
        from tools.hip.kv_parity import compare
        ref = {'ids': [0, 0], 'logits': np.array([[5, 0], [5, 0]], dtype=np.float32)}
        q8 = {'args': '', 'ids': [0], 'logits': np.array([[5, 0]], dtype=np.float32), 'info': [], 'wall_s': 1}
        self.assertTrue(compare(ref, q8, 2)['length_mismatch'])
        q8['ids'] = [0, 0]
        with self.assertRaises(ValueError):
            compare(ref, q8, 2)


if __name__ == '__main__':
    unittest.main()
