"""CPU-only checks. No model, engine, GPU, downloads or operational profile writes."""
from __future__ import annotations

import collections
import copy
import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.hip.functional_quality import campaign as campaign_module
from tools.hip.functional_quality.campaign import registry, reuse_shared_code, schedule
from tools.hip.functional_quality.client import APIClient
from tools.hip.functional_quality.code import CodeProject, run_code
from tools.hip.functional_quality.common import ROOT, append, digest, rows, schema, scratch, sha, strict_json, write_json
from tools.hip.functional_quality.evaluate import evaluate, validate_answer
from tools.hip.functional_quality.mock import MockREST, Store, TOOLS, validate_tools
from tools.hip.functional_quality.resources import ExclusiveGPU, admission
from tools.hip.functional_quality.tasks import (case, code_task, code_tasks, initial_store,
                                                instruction_tasks, pilot, reasoning_tasks, tool_tasks)


class ScriptedAPI:
    """Actual HTTP/SSE fixture with disposable scripted responses, never an evaluated model."""
    def __init__(self, messages):
        self.messages, self.seen = iter(messages), []
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                outer.seen.append(strict_json(self.rfile.read(int(self.headers['Content-Length'])).decode()))
                message, finish = next(outer.messages)
                events = [{'choices': [{'delta': {'role': 'assistant'}, 'finish_reason': None}]}]
                if message.get('content'):
                    events += [{'choices': [{'delta': {'content': message['content']}, 'finish_reason': None}]}]
                for i, call in enumerate(message.get('tool_calls', [])):
                    f = call['function']
                    events += [{'choices': [{'delta': {'tool_calls': [{'index': i, 'id': call['id'],
                              'function': {'name': f['name'], 'arguments': ''}}]}, 'finish_reason': None}]}]
                    args = f['arguments']
                    for chunk in (args[:len(args)//2], args[len(args)//2:]):
                        events += [{'choices': [{'delta': {'tool_calls': [{'index': i, 'function': {'arguments': chunk}}]},
                                                 'finish_reason': None}]}]
                events += [{'choices': [{'delta': {}, 'finish_reason': finish}],
                            'usage': {'prompt_tokens': 10, 'completion_tokens': 2},
                            'timings': {'prompt_per_second': 100, 'predicted_per_second': 20}}]
                data = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events) + 'data: [DONE]\n\n'
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(data.encode())))
                self.end_headers()
                self.wfile.write(data.encode())

            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = APIClient(f'http://127.0.0.1:{self.server.server_port}')

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)


def call(id_, name, args):
    return {'id': id_, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


def calling(*calls):
    return ({'tool_calls': list(calls)}, 'tool_calls')


class FunctionalTests(unittest.TestCase):
    def test_catalog_and_balance(self):
        tasks = pilot()
        self.assertEqual(collections.Counter(t['category'] for t in tasks),
                         {'code': 14, 'tools': 12, 'reasoning': 6, 'instructions': 4})
        self.assertEqual(len({t['id'] for t in tasks}), 36)
        self.assertEqual(len(code_tasks()), 20)
        order = schedule(tasks)
        self.assertEqual([b['arm'] for b in order], ['fp16', 'q8', 'q8', 'fp16'])
        self.assertEqual([len(b['task_ids']) for b in order], [18] * 4)
        self.assertEqual(order[0]['task_ids'], order[1]['task_ids'])
        self.assertEqual(order[2]['task_ids'], order[3]['task_ids'])
        self.assertFalse(set(order[0]['task_ids']) & set(order[2]['task_ids']))
        for group in (order[0], order[2]):
            self.assertEqual(collections.Counter(t['category'] for t in tasks if t['id'] in group['task_ids']),
                             {'code': 7, 'tools': 6, 'reasoning': 3, 'instructions': 2})

    def test_all_code_reference_solutions_execute(self):
        for task in code_tasks():
            with self.subTest(task=task['id']):
                self.assertTrue(run_code(task['reference'], task['function'], task['public'] + task['hidden'])['ok'])
                wrong = 'def ' + task['function'] + '(*args):\n    return None\n'
                self.assertFalse(run_code(wrong, task['function'], task['public'] + task['hidden'])['ok'])

    def test_hidden_boundaries_reject_plausible_mutants(self):
        mutants = {1: 'def clamp(x,low,high): return min(max(x,low),high)',
                   3: 'def unique(values): return list(set(values))',
                   4: 'def pairs(text): return dict(x.strip().split("=") for x in text.split(";"))',
                   5: 'def merge(intervals): return [[min(x[0] for x in intervals),max(x[1] for x in intervals)]]',
                   8: 'def transpose(rows): return [list(x) for x in zip(*rows)]',
                   10: 'def rolling(values,k): return [sum(values[i:i+k]) for i in range(len(values)-k+1)]',
                   14: 'def apply(stock,changes):\n    for k,v in changes: stock[k]=stock.get(k,0)+v\n    return stock\n'}
        for n, source in mutants.items():
            task = code_tasks()[n-1]
            with self.subTest(task=task['id']):
                self.assertFalse(run_code(source, task['function'], task['hidden'])['ok'])

    def test_worker_rejects_host_access_and_mutation(self):
        for source in ['import os\ndef f(x): return x',
                       'def f(x): return open("secret").read()',
                       'def f(x): return x.__class__',
                       'def f(x): return "{0.__class__}".format(x)',
                       'def f(x): return globals()',
                       'def f(x): return __builtins__',
                       'def f(x):\n    x.append(3)\n    return x\n']:
            with self.subTest(source=source):
                self.assertFalse(run_code(source, 'f', [case([[1]], [1, 3])])['ok'])

    def test_worker_timeout(self):
        result = run_code('def f():\n    while True: pass\n', 'f', [case([], 0)])
        self.assertFalse(result['ok'])
        self.assertIn(result.get('error'), ('worker_timeout', 'worker_exit'))

    def test_project_paths_hidden_tests_and_isolation(self):
        task = code_tasks()[0]
        with scratch() as wd:
            a = CodeProject(task, Path(wd) / 'a')
            b = CodeProject(task, Path(wd) / 'b')
            for path in ('../MANIFEST.json', 'C:/Windows/win.ini', 'hidden_tests.json', 'reference'):
                with self.assertRaises(ValueError):
                    a.execute('read_file', {'path': path})
            with self.assertRaises(ValueError):
                a.execute('write_file', {'path': 'public_tests.json', 'content': '[]'})
            a.execute('write_file', {'path': 'solution.py', 'content': task['reference']})
            self.assertTrue(a.execute('run_tests', {})['ok'])
            self.assertTrue(a.validate()['ok'])
            self.assertFalse(b.validate()['ok'])
            self.assertEqual(task['files']['solution.py'], b.files['solution.py'])

    def test_mock_all_tasks_over_real_rest_and_reset(self):
        pristine = initial_store()
        for task in tool_tasks():
            with self.subTest(task=task['id']):
                mock = MockREST(task['initial'])
                try:
                    mock.execute('list_items', {})
                    for name, args in task['mutations']:
                        mock.execute(name, args)
                    verdict = validate_tools(task, mock.store, json.dumps(task['answer']))
                    self.assertTrue(verdict['ok'], verdict)
                    self.assertFalse(validate_tools(task, mock.store, '{}')['ok'])
                finally:
                    mock.close()
                self.assertEqual(task['initial'], pristine)
                self.assertEqual(Store(task['initial']).state, pristine)

    def test_mock_sequence_argument_and_atomic_rejection(self):
        task = tool_tasks()[2]
        store = Store(task['initial'])
        store.execute('list_items', {})
        store.execute('reserve', {'item': 'B', 'qty': 2, 'order': 'o2'})
        store.execute('cancel', {'order': 'o2'})
        # Same final stock, wrong call arguments must fail.
        self.assertFalse(validate_tools(task, store, json.dumps(task['answer']))['ok'])
        before = copy.deepcopy(store.state)
        for name, args in [('reserve', {'item': 'A', 'qty': 99, 'order': 'bad'}),
                           ('restock', {'item': 'A', 'qty': True}), ('reset', {}),
                           ('restock', {'item': 'A', 'qty': -1}), ('get_item', {'item': 'Z'})]:
            with self.assertRaises(ValueError):
                store.execute(name, args)
            self.assertEqual(store.state, before)

    def test_strict_json_and_schema(self):
        for text in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '```json\n{}\n```', '{} trailing'):
            with self.assertRaises(ValueError):
                strict_json(text)
        spec = TOOLS[2]['function']['parameters']
        for args in ({'item': 'A', 'qty': True, 'order': 'z'}, {'item': 'A', 'qty': 2},
                     {'item': 'A', 'qty': 1, 'order': 'z', 'extra': 1}):
            with self.assertRaises(ValueError):
                schema(args, spec)

    def test_reasoning_and_instruction_validators(self):
        for task in reasoning_tasks() + instruction_tasks():
            text = task['expected'] if task['validator'] == 'lines' else json.dumps(task['expected'])
            with self.subTest(task=task['id']):
                self.assertTrue(validate_answer(task, text)['ok'])
                self.assertFalse(validate_answer(task, text + '\nextra')['ok'])
                self.assertFalse(validate_answer(task, '{}')['ok'])
        self.assertEqual(11 * 2 * 2 * 256 * 49152 * 2, reasoning_tasks()[0]['expected']['fp16_bytes'])
        self.assertEqual(11 * 2 * 2 * (256 + 8 * 2) * 49152, reasoning_tasks()[0]['expected']['q8_bytes'])
        boolean_task = reasoning_tasks()[-1]
        self.assertFalse(validate_answer(boolean_task, '{"A":0,"B":0,"C":1}')['ok'])
        self.assertFalse(validate_answer(instruction_tasks()[1], 'alpha=3\nbeta=1\ngamma=2\n\n')['ok'])

    def test_disposable_code_agent_cycle_over_sse(self):
        task = code_task(99, 'Implement increment(x): return x+1.', 'increment(x)',
                         'def increment(x): return x+1\n', [case([2], 3)], [case([-8], -7)])
        fixture = ScriptedAPI([calling(call('r', 'read_file', {'path': 'solution.py'})),
                               calling(call('w', 'write_file', {'path': 'solution.py', 'content': task['reference']})),
                               calling(call('t', 'run_tests', {})), ({'content': 'Done.'}, 'stop')])
        try:
            with scratch() as wd:
                result = evaluate(task, fixture.client, Path(wd) / 'project', Path(wd) / 'events.jsonl')
                self.assertEqual(result['status'], 'PASS')
                self.assertEqual(len(result['requests']), 4)
                self.assertEqual(len(result['calls']), 3)
                self.assertGreater(result['requests'][0]['ttft_s'], 0)
                self.assertEqual(len(rows(Path(wd) / 'events.jsonl')), 7)
                self.assertTrue(all(m['temperature'] == 0 and m['max_tokens'] == 2048 for m in fixture.seen))
                self.assertIn('tool', [m['role'] for m in fixture.seen[-1]['messages']])
                self.assertNotIn('hidden', json.dumps(fixture.seen))
        finally:
            fixture.close()

    def test_disposable_tool_agent_cycle_over_sse(self):
        task = {'id': 'SMOKE-REST', 'category': 'tools', 'system': 'Use inventory tools.', 'prompt': 'Reserve one Z.',
                'initial': {'items': {'Z': {'name': 'smoke', 'stock': 2, 'price': 1}}, 'orders': {}},
                'mutations': [('reserve', {'item': 'Z', 'qty': 1, 'order': 'smoke'})],
                'answer': {'remaining': 1}, 'max_turns': 4, 'max_tokens': 128}
        fixture = ScriptedAPI([calling(call('a', 'list_items', {})),
                               calling(call('b', 'reserve', {'item': 'Z', 'qty': 1, 'order': 'smoke'})),
                               ({'content': '{"remaining":1}'}, 'stop')])
        try:
            with scratch() as wd:
                result = evaluate(task, fixture.client, Path(wd) / 'project', Path(wd) / 'events.jsonl')
                self.assertEqual(result['status'], 'PASS', result)
                self.assertTrue(all(c['json_valid'] for c in result['calls']))
        finally:
            fixture.close()

    def test_bad_call_stays_functional_failure_after_recovery(self):
        task = {'id': 'SMOKE-FAIL', 'category': 'instructions', 'system': 'Only JSON.',
                'prompt': 'Return {"x":1}', 'expected': {'x': 1}, 'validator': 'exact', 'max_turns': 2, 'max_tokens': 64}
        bad = {'id': 'bad', 'function': {'name': 'anything', 'arguments': '{bad'}}
        fixture = ScriptedAPI([calling(bad), ({'content': '{"x":1}'}, 'stop')])
        try:
            with scratch() as wd:
                result = evaluate(task, fixture.client, Path(wd) / 'p', Path(wd) / 'e.jsonl')
                self.assertEqual(result['status'], 'FAIL')
                self.assertFalse(result['calls'][0]['json_valid'])
        finally:
            fixture.close()

    def test_truncation_is_functional_fail(self):
        task = instruction_tasks()[0]
        fixture = ScriptedAPI([({'content': json.dumps(task['expected'])}, 'length')])
        try:
            with scratch() as wd:
                self.assertEqual(evaluate(task, fixture.client, Path(wd) / 'p', Path(wd) / 'e.jsonl')['status'], 'FAIL')
        finally:
            fixture.close()

    def test_registry_integrity_and_terminal_fail_reuse(self):
        with scratch() as wd:
            path, event = Path(wd) / 'r.jsonl', Path(wd) / 'e.jsonl'
            append(event, {'kind': 'fixture'})
            rec = {'task_id': 'smoke', 'arm': 'fp16', 'freeze_id': 'f', 'status': 'FAIL',
                   'event_file': str(event), 'event_sha': sha(event)}
            rec['result_sha'] = digest(rec)
            append(path, rec)
            self.assertEqual(registry(path, 'f')[('smoke', 'fp16')]['status'], 'FAIL')
            with self.assertRaises(RuntimeError):
                registry(path, 'other')
            append(event, {'tamper': True})
            with self.assertRaises(RuntimeError):
                registry(path, 'f')

    def test_registry_retries_only_infrastructure(self):
        with scratch() as wd:
            path = Path(wd) / 'r.jsonl'
            for status in ('INFRA_ERROR', 'FAIL'):
                rec = {'task_id': 'smoke', 'arm': 'q8', 'freeze_id': 'f', 'status': status}
                rec['result_sha'] = digest(rec)
                append(path, rec)
            self.assertEqual(registry(path, 'f')[('smoke', 'q8')]['status'], 'FAIL')
            append(path, rec)
            with self.assertRaises(RuntimeError):
                registry(path, 'f')

    def test_shared_code_reuses_hashed_main_result(self):
        with scratch() as wd:
            campaign = Path(wd)
            rec = {'task_id': 'C01', 'arm': 'fp16', 'suite': 'pilot', 'freeze_id': 'f', 'status': 'FAIL'}
            rec['result_sha'] = digest(rec)
            append(campaign / 'results-pilot.jsonl', rec)
            target = campaign / 'results-original.jsonl'
            reuse_shared_code(campaign, 'original', target, 'f')
            reuse_shared_code(campaign, 'original', target, 'f')
            result = registry(target, 'f')
            self.assertEqual(len(rows(target)), 1)
            self.assertEqual(result[('C01', 'fp16')]['reused_from']['result_sha'], rec['result_sha'])
            self.assertEqual(result[('C01', 'fp16')]['status'], 'FAIL')

    def test_actual_polystrata_http_client_with_mock_engine(self):
        from serve.server import ByteTokenizer, ChatTemplate, MockEngine, Service, serve
        tok = ByteTokenizer()
        scripts = ['<tool_call>\n<function=get_item>\n<parameter=item>\nZ\n</parameter>\n</function>\n</tool_call>',
                   '{"stock":2}']
        svc = Service(MockEngine(tok, scripts), tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
        http = serve(svc, port=0)
        try:
            client = APIClient(f'http://127.0.0.1:{http.server_address[1]}')
            task = {'id': 'SMOKE-REAL-SERVER', 'category': 'tools', 'system': 'Use inventory tools.', 'prompt': 'Read Z.',
                    'initial': {'items': {'Z': {'name': 'fixture', 'stock': 2, 'price': 1}}, 'orders': {}},
                    'mutations': [], 'answer': {'stock': 2}, 'max_turns': 3, 'max_tokens': 256}
            with scratch() as wd:
                result = evaluate(task, client, Path(wd) / 'p', Path(wd) / 'e.jsonl')
            self.assertEqual(result['status'], 'PASS', result)
            self.assertEqual(result['calls'][0]['name'], 'get_item')
            self.assertIsNone(result['requests'][0]['timings'])
        finally:
            http.shutdown()
            http.server_close()

    def test_runner_sequential_blocks_and_resume_without_regeneration(self):
        fixture = ScriptedAPI([({'content': 'READY'}, 'stop'), ({'content': '7'}, 'stop'),
                               ({'content': 'READY'}, 'stop'), ({'content': '7'}, 'stop')])
        started, stopped = [], []
        class FakeServer:
            def __init__(self, path, config, port, events):
                self.owned = {}
                self.arm = config['args'][-1] if '--kv' in config['args'] else 'fp16'
            def start(self):
                started.append(self.arm)
                return fixture.client
            def close(self):
                stopped.append(self.arm)
        try:
            with scratch() as wd:
                campaign = Path(wd)
                profile = campaign / 'frozen-profile'
                profile.write_text('fixture')
                task = {'id': 'SMOKE-RUN', 'category': 'instructions', 'system': 'Only JSON.', 'prompt': 'Return integer 7.',
                        'expected': 7, 'validator': 'exact', 'max_turns': 1, 'max_tokens': 64}
                append(campaign / 'TASKS.jsonl', task)
                experiment = {'profile': str(profile), 'exe': 'fixture', 'model': 'fixture', 'tokenizer': 'fixture',
                              'lib_dirs': [], 'env': {}, 'common_args': [], 'port': 0,
                              'pilot_order': [{'block': n, 'arm': arm, 'task_ids': [task['id']]} for n, arm in enumerate(('fp16', 'q8'))]}
                write_json(campaign / 'EXPERIMENT.json', experiment)
                state = {'ram_available_mib': 20000, 'vram_used_mib': 1000, 'process_scan_complete': True, 'other_inference': []}
                with patch.object(campaign_module, 'audit', return_value={'freeze_id': 'f'}), \
                     patch.object(campaign_module, 'ROOT', campaign), \
                     patch.object(campaign_module, 'snapshot', return_value=state), \
                     patch.object(campaign_module, 'ExperimentalServer', FakeServer), \
                     patch.object(campaign_module, 'verify_owned'), patch.object(campaign_module, 'check_engine_log'), \
                     patch.object(campaign_module, 'report'):
                    campaign_module.run(campaign, 'pilot')
                    campaign_module.run(campaign, 'pilot')
                self.assertEqual(started, ['fp16', 'q8_0'])
                self.assertEqual(stopped, started)
                self.assertEqual(len(rows(campaign / 'results-pilot.jsonl')), 2)
        finally:
            fixture.close()

    def test_server_cleanup_refuses_changed_process_identity(self):
        from tools.hip.functional_quality.server import ExperimentalServer
        server = ExperimentalServer('fixture', {}, 0, 'fixture')
        server.proc = MagicMock()
        server.proc.pid = 123
        server.proc.poll.return_value = 0
        server.owned = {123: {'created': 1, 'exe': 'old', 'cmd': []}}
        process = MagicMock()
        process.create_time.return_value = 2
        with patch('psutil.Process', return_value=process):
            with self.assertRaises(RuntimeError):
                server.close()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_real_venv_listener_descendant_identity_and_cleanup(self):
        from tools.hip.functional_quality.server import ExperimentalServer
        with scratch() as wd:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            server = ExperimentalServer('fixture', {'exe': str(Path(wd) / 'unused-engine.exe')}, port, Path(wd) / 'e.jsonl')
            server.command = [sys.executable, '-m', 'serve.server', '--engine', 'mock', '--port', str(port)]
            server.console = open(os.devnull, 'w')
            server.proc = subprocess.Popen(server.command, cwd=ROOT, stdout=server.console, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                for step in range(80):
                    server.remember()
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    time.sleep(.05)
                else:
                    self.fail('mock server did not start')
                server.remember()
                listener = server.listener_owner()
                self.assertIn(listener, server.owned)
                self.assertIn(server.owned[listener]['role'], ('server', 'launcher'))
                campaign_module.verify_owned(server)
                from types import SimpleNamespace
                bad_listener = SimpleNamespace(pid=-1, status='LISTEN', laddr=SimpleNamespace(port=port))
                with patch('psutil.net_connections', return_value=[bad_listener]):
                    with self.assertRaises(RuntimeError):
                        server.listener_owner()
            finally:
                server.close()
            self.assertIsNotNone(server.proc.poll())

    def test_gpu_admission_and_exclusive_lock(self):
        state = {'ram_available_mib': 20000, 'vram_used_mib': 1000,
                 'process_scan_complete': True, 'other_inference': []}
        self.assertEqual(admission(state), [])
        for key, value in [('ram_available_mib', 10000), ('vram_used_mib', None), ('vram_used_mib', 4096),
                           ('process_scan_complete', False), ('other_inference', [{'pid': 1}])]:
            self.assertTrue(admission({**state, key: value}))
        with scratch() as wd:
            path = Path(wd) / 'gpu.lock'
            with ExclusiveGPU(path):
                with self.assertRaises(RuntimeError):
                    with ExclusiveGPU(path):
                        pass
            with ExclusiveGPU(path):
                pass


if __name__ == '__main__':
    unittest.main()
