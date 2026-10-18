"""CPU-only checks for scheduling oracle, experiment balance and request settings."""
import io
import json
import unittest
from unittest.mock import patch

from tools.hip.functional_quality.client import APIClient
from tools.hip.functional_quality.r05_experiment import CASES, earliest, tasks
from tools.hip.functional_quality.campaign import schedule
from tools.hip.functional_quality.common import scratch
from tools.hip.functional_quality.r05_experiment import report
from pathlib import Path


class R05ExperimentTests(unittest.TestCase):
    def test_oracle_against_exhaustive_search(self):
        intervals = [(a, b) for a in range(5) for b in range(a + 1, 7)]
        for first in intervals:
            for second in intervals:
                for minimum in range(5):
                    for duration in (1, 2, 3):
                        occupied = [first, second]
                        expected = next(s for s in range(minimum, 12)
                                        if all(not (s < b and s + duration > a) for a, b in occupied))
                        self.assertEqual(earliest(occupied, minimum, duration),
                                         {'start': expected, 'end': expected + duration})

    def test_original_and_boundaries(self):
        self.assertEqual(earliest(CASES[0][1], 4, 3), {'start': 15, 'end': 18})
        self.assertEqual(earliest([(2, 5), (8, 11)], 5, 3), {'start': 5, 'end': 8})
        self.assertEqual(earliest([], 4, 3), {'start': 4, 'end': 7})

    def test_factorial_and_paired_schedule(self):
        generated = tasks()
        self.assertEqual(len(generated), 108)
        self.assertEqual(len({t['id'] for t in generated}), 108)
        plan = schedule(generated)
        for arm in ('fp16', 'q8'):
            ids = [i for b in plan if b['arm'] == arm for i in b['task_ids']]
            self.assertCountEqual(ids, [t['id'] for t in generated])
        for name, *_ in CASES:
            selected = [t for t in generated if t['case'] == name]
            self.assertEqual(len(selected), 9)
            self.assertEqual(len({json.dumps(t['expected']) for t in selected}), 1)
            self.assertEqual({t['max_tokens'] for t in selected}, {2048})
        self.assertEqual({t['reasoning_enabled'] for t in generated if t['condition'] == 'direct'}, {False})
        self.assertEqual({t['reasoning_enabled'] for t in generated if t['condition'] == 'check'}, {False})
        self.assertEqual({t['reasoning_enabled'] for t in generated if t['condition'] == 'thinking_check'}, {True})

    def test_request_settings_default_preserved_and_thinking_opt_in(self):
        event = {'choices': [{'delta': {'content': '{}'}, 'finish_reason': 'stop'}],
                 'usage': {'completion_tokens': 1}}
        data = ('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n').encode()
        for enabled in (False, True):
            with patch('urllib.request.urlopen', return_value=io.BytesIO(data)) as opened:
                APIClient('http://127.0.0.1:8197').complete([], [], 2048, reasoning=enabled)
                payload = json.loads(opened.call_args.args[0].data)
                self.assertEqual(payload['temperature'], 0)
                self.assertEqual(payload['max_tokens'], 2048)
                self.assertEqual(payload['reasoning_effort'], 'high' if enabled else 'none')
                self.assertEqual(payload['chat_template_kwargs']['enable_thinking'], enabled)

    def test_unrun_report_keeps_missing_pairs_incomplete(self):
        with scratch() as folder:
            directory = Path(folder)
            (directory / 'TASKS.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks()), encoding='utf-8')
            with patch('tools.hip.functional_quality.campaign.audit', return_value={'freeze_id': 'fixture'}):
                report(directory)
            result = json.loads((directory / 'R05_SUMMARY.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertTrue(all(v['counts']['INCOMPLETE'] == 36 for v in result['conditions'].values()))
            self.assertEqual(result['thinking_observation']['q8']['completed_records'], 0)


if __name__ == '__main__':
    unittest.main()
