"""Actual OpenAI SSE client, with no silent request retries."""
from __future__ import annotations

import json
import time
import urllib.request

from .common import strict_json


class APIClient:
    def __init__(self, base, timeout=300):
        if not base.startswith('http://127.0.0.1:'):
            raise ValueError('only a dedicated loopback endpoint is supported')
        self.base, self.timeout = base.rstrip('/'), timeout

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as response:
            return strict_json(response.read().decode())

    def complete(self, messages, tools, max_tokens, *, reasoning=False):
        payload = {'model': 'default', 'messages': messages, 'temperature': 0,
                   'max_tokens': max_tokens, 'stream': True, 'stream_options': {'include_usage': True},
                   'reasoning_effort': 'none', 'chat_template_kwargs': {'enable_thinking': False},
                   'strata_checkpoint': False}
        if reasoning:
            payload['reasoning_effort'] = 'high'
            payload['chat_template_kwargs'] = {'enable_thinking': True}
        if tools:
            payload['tools'] = tools
        request = urllib.request.Request(self.base + '/v1/chat/completions', json.dumps(payload).encode(),
                                         {'Content-Type': 'application/json'})
        start = time.perf_counter()
        content, reasoning, calls, events = '', '', {}, []
        first, first_answer, usage, timings, finish, done = None, None, None, None, None, False
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw in response:
                if time.perf_counter() - start > self.timeout:
                    raise TimeoutError('request exceeded wall time limit')
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                value = line[5:].strip()
                if value == '[DONE]':
                    done = True
                    break
                event = strict_json(value)
                events.append(event)
                if 'error' in event:
                    raise RuntimeError('API error: ' + str(event['error']))
                usage = event.get('usage') or usage
                timings = event.get('timings') or timings
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    if delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls'):
                        if first is None:
                            first = time.perf_counter() - start
                    if delta.get('content'):
                        if first_answer is None:
                            first_answer = time.perf_counter() - start
                        content += delta['content']
                    reasoning += delta.get('reasoning_content') or ''
                    for call in delta.get('tool_calls') or []:
                        slot = calls.setdefault(call['index'], {'id': '', 'type': 'function',
                                                                'function': {'name': '', 'arguments': ''}})
                        if call.get('id'):
                            slot['id'] = call['id']
                        f = call.get('function') or {}
                        slot['function']['name'] += f.get('name') or ''
                        slot['function']['arguments'] += f.get('arguments') or ''
                    finish = choice.get('finish_reason') or finish
        if not done or finish is None or usage is None:
            raise RuntimeError('incomplete SSE response (DONE, finish or usage missing)')
        message = {'role': 'assistant', 'content': content or None}
        if calls:
            message['tool_calls'] = [calls[i] for i in sorted(calls)]
        if reasoning:
            message['reasoning_content'] = reasoning
        return {'message': message, 'finish_reason': finish, 'usage': usage, 'timings': timings,
                'ttft_s': first, 'first_answer_s': first_answer, 'wall_s': time.perf_counter() - start,
                'request': payload, 'events': events}
