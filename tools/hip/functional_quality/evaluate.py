from __future__ import annotations

import json
import time

from .code import CodeProject
from .common import append, now, strict_json
from .mock import MockREST, strict_equal, validate_tools


def validate_answer(task, text):
    try:
        # A terminal LF/CRLF is allowed for exact line tasks; no spaces or extra blank lines.
        if task['validator'] == 'lines':
            value = text.removesuffix('\r\n') if text.endswith('\r\n') else text.removesuffix('\n')
            ok = value == task['expected']
        else:
            ok = strict_equal(strict_json(text), task['expected'])
    except (ValueError, TypeError):
        ok = False
    return {'ok': ok}


def evaluate(task, client, directory, event_path):
    start = time.perf_counter()
    project, mock = None, None
    if task['category'] == 'code':
        project = CodeProject(task, directory)
    if task['category'] == 'tools':
        mock = MockREST(task['initial'])
    executor = project or mock
    messages = [{'role': 'system', 'content': task['system']}, {'role': 'user', 'content': task['prompt']}]
    calls, requests, errors, final = [], [], [], None
    try:
        for turn in range(task['max_turns']):
            options = {'reasoning': True} if task.get('reasoning_enabled') else {}
            response = client.complete(messages, executor.tools if executor else [], task['max_tokens'], **options)
            requests.append({k: v for k, v in response.items() if k not in ('events', 'request')})
            append(event_path, {'at': now(), 'kind': 'request', 'turn': turn, **response})
            message = response['message']
            messages.append(message)
            pending = message.get('tool_calls') or []
            if not pending:
                if response['finish_reason'] == 'tool_calls':
                    errors.append('tool_calls finish without calls')
                final = message.get('content') or ''
                break
            if len(pending) > 8:
                errors.append('more than 8 calls in one turn')
                break
            for call in pending:
                rec = {'id': call.get('id'), 'name': (call.get('function') or {}).get('name'),
                       'arguments': (call.get('function') or {}).get('arguments'), 'json_valid': False}
                try:
                    args = strict_json(rec['arguments'])
                    rec['json_valid'] = True
                    if not executor:
                        raise ValueError('tools unavailable for this task')
                    if not rec['id'] or rec['id'] in {c['id'] for c in calls}:
                        raise ValueError('missing or duplicate call id')
                    result = executor.execute(rec['name'], args)
                    rec['executed'] = True
                except (ValueError, TypeError, KeyError) as exc:
                    result = {'error': str(exc)}
                    rec['error'] = str(exc)
                    errors.append(str(exc))
                rec['result'] = result
                calls.append(rec)
                append(event_path, {'at': now(), 'kind': 'tool', 'turn': turn, **rec})
                messages.append({'role': 'tool', 'tool_call_id': call.get('id') or 'invalid',
                                 'content': json.dumps(result)})
        if project:
            verdict = project.validate()
        elif mock:
            verdict = validate_tools(task, mock.store, final or '')
        elif task['category'] == 'open':
            verdict = {'ok': False, 'quality': 'AWAITING_BLIND_REVIEW'}
        else:
            verdict = validate_answer(task, final or '')
        # Budget exhaustion and generation truncation are functional failures, not retries.
        completed = final is not None and requests[-1]['finish_reason'] == 'stop'
        verdict['ok'] = verdict['ok'] and completed and not errors
        verdict['completed'] = completed
        status = 'PASS' if verdict['ok'] else 'FAIL'
        if task['category'] == 'open' and completed and final.strip() and not errors:
            status = 'REVIEW_PENDING'
        return {'status': status, 'validator': verdict,
                'calls': calls, 'requests': requests, 'errors': errors, 'final': final,
                'wall_s': time.perf_counter() - start}
    finally:
        if mock:
            mock.close()
