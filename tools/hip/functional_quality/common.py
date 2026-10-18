from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMPAIGN = ROOT / 'bench/results/2026-10-17-hip-kv-q8-paired-functional'


def scratch():
    # Windows AppContainer temp roots may allow creation but deny cleanup/children.
    # Every recursively cleaned target is created under this explicit workspace root.
    base = ROOT / 'tmp/functional-quality'
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='fixture-', dir=base)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def rows(path):
    path = Path(path)
    return [strict_json(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()] if path.exists() else []


def schema(value, spec):
    """Small, closed JSON Schema subset used by this harness's tool definitions."""
    kind = spec['type']
    types = {'object': dict, 'array': list, 'string': str, 'integer': int, 'boolean': bool}
    if type(value) is not types[kind]:
        raise ValueError('expected ' + kind)
    if kind == 'object':
        props = spec['properties']
        if not set(spec.get('required', props)).issubset(value) or set(value) - set(props):
            raise ValueError('missing or additional properties')
        for key, child in value.items():
            schema(child, props[key])
    if kind == 'array':
        for child in value:
            schema(child, spec['items'])
    if kind == 'integer' and value < spec.get('minimum', -10**30):
        raise ValueError('below minimum')
    if kind == 'string' and len(value) > spec.get('maxLength', 32768):
        raise ValueError('string too long')


def tool(name, description, properties):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': list(properties), 'additionalProperties': False}}}
