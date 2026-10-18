from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .common import schema, scratch, strict_json, tool

CODE_TOOLS = [tool('read_file', 'Read solution.py or public_tests.json.', {'path': {'type': 'string'}}),
              tool('write_file', 'Replace solution.py with pure Python source.',
                   {'path': {'type': 'string'}, 'content': {'type': 'string', 'maxLength': 32768}}),
              tool('run_tests', 'Execute the immutable public tests against solution.py.', {})]


def run_code(source, function, cases):
    with scratch() as wd:
        try:
            proc = subprocess.run([sys.executable, '-I', '-S', str(Path(__file__).with_name('code_worker.py'))],
                                  input=json.dumps({'source': source, 'function': function, 'cases': cases}),
                                  text=True, capture_output=True, cwd=wd, timeout=5)
        except subprocess.TimeoutExpired:
            return {'ok': False, 'error': 'worker_timeout'}
        if proc.returncode:
            return {'ok': False, 'error': 'worker_exit', 'returncode': proc.returncode}
        try:
            result = strict_json(proc.stdout)
        except (ValueError, TypeError):
            return {'ok': False, 'error': 'worker_protocol'}
        # OS job setup failure is infrastructure, not a model answer.
        if result.get('error', '').startswith('OSError:'):
            raise RuntimeError('worker isolation unavailable: ' + result['error'])
        return result


class CodeProject:
    tools = CODE_TOOLS

    def __init__(self, task, directory):
        self.task = task
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.files = dict(task['files'])
        self.files['public_tests.json'] = json.dumps(task['public'], indent=2)
        self.writes = 0
        self.test_runs = 0
        for name, content in self.files.items():
            (self.directory / name).write_text(content, encoding='utf-8')

    def execute(self, name, args):
        definition = next((t for t in self.tools if t['function']['name'] == name), None)
        if definition is None:
            raise ValueError('unknown tool')
        schema(args, definition['function']['parameters'])
        if name == 'read_file':
            if args['path'] not in self.files:
                raise ValueError('file unavailable')
            return {'content': self.files[args['path']]}
        if name == 'write_file':
            if args['path'] != 'solution.py':
                raise ValueError('only solution.py is writable')
            self.files['solution.py'] = args['content']
            (self.directory / 'solution.py').write_text(args['content'], encoding='utf-8')
            self.writes += 1
            return {'written': 'solution.py'}
        self.test_runs += 1
        return run_code(self.files['solution.py'], self.task['function'], self.task['public'])

    def validate(self):
        result = run_code(self.files['solution.py'], self.task['function'], self.task['hidden'])
        public = run_code(self.files['solution.py'], self.task['function'], self.task['public'])
        return {'ok': result['ok'] and public['ok'] and self.writes > 0 and self.test_runs > 0,
                'hidden': result, 'public': public, 'writes': self.writes, 'test_runs': self.test_runs}
