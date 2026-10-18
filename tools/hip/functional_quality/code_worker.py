"""Restricted pure-Python worker. Launched with -I -S in a disposable directory.

This is a deliberately small language, not a general shell or Python security sandbox.
No host modules, imports, file/network primitives, reflection or private attributes are
exposed. OS process limits bound memory, CPU and descendants before evaluated code runs.
"""
from __future__ import annotations

import ast
import builtins
import copy
import json
import os
import sys

BUILTINS = ('abs all any bool dict enumerate float int isinstance len list max min range '
            'reversed round set sorted str sum tuple zip ValueError TypeError').split()
METHODS = set(('append extend insert pop remove clear copy count index reverse sort get items keys values '
               'update setdefault add discard union intersection difference split rsplit join strip lstrip '
               'rstrip lower upper startswith endswith replace find rfind isdigit isalpha isalnum').split())


def process_limits():
    if os.name == 'nt':
        import ctypes as c
        from ctypes import wintypes as w
        class Basic(c.Structure):
            _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64), ('flags', w.DWORD),
                        ('min_ws', c.c_size_t), ('max_ws', c.c_size_t), ('active', w.DWORD),
                        ('affinity', c.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]
        class Io(c.Structure):
            _fields_ = [(n, c.c_uint64) for n in ('ro', 'wo', 'oo', 'rt', 'wt', 'ot')]
        class Extended(c.Structure):
            _fields_ = [('basic', Basic), ('io', Io), ('process_memory', c.c_size_t),
                        ('job_memory', c.c_size_t), ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]
        k = c.WinDLL('kernel32', use_last_error=True)
        k.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        k.CreateJobObjectW.restype = w.HANDLE
        k.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        k.GetCurrentProcess.restype = w.HANDLE
        job = k.CreateJobObjectW(None, None)
        info = Extended()
        # Per-process memory, active process count, CPU time, kill descendants on handle close.
        info.basic.flags = 0x100 | 0x8 | 0x2 | 0x2000
        info.basic.active = 1
        info.basic.process_time = 3 * 10_000_000
        info.process_memory = 256 * 2**20
        if not job or not k.SetInformationJobObject(job, 9, c.byref(info), c.sizeof(info)):
            raise OSError(c.get_last_error(), 'could not set worker job limits')
        if not k.AssignProcessToJobObject(job, k.GetCurrentProcess()):
            raise OSError(c.get_last_error(), 'could not assign worker to limited job')
        # Keep the handle alive until process exit.
        return job
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (256 * 2**20, 256 * 2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    return None


def check_source(source):
    if len(source.encode()) > 32768:
        raise ValueError('source exceeds 32 KiB')
    tree = ast.parse(source, filename='solution.py')
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef,
                             ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith)):
            raise ValueError('unsupported Python construct: ' + type(node).__name__)
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('private names are unavailable')
        if isinstance(node, ast.arg) and node.arg.startswith('_'):
            raise ValueError('private parameters are unavailable')
        if isinstance(node, ast.FunctionDef) and (node.name.startswith('_') or node.decorator_list):
            raise ValueError('private functions/decorators are unavailable')
        if isinstance(node, ast.Attribute) and node.attr not in METHODS:
            raise ValueError('unsupported attribute: ' + node.attr)
    return tree


def same(a, b):
    if type(a) is not type(b):
        # Numeric floats are allowed where the mathematical result is integral.
        return type(a) in (int, float) and type(b) in (int, float) and a == b
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def evaluate(source, function, cases):
    tree = check_source(source)
    code = compile(tree, 'solution.py', 'exec')
    results = []
    for item in cases:
        ns = {'__builtins__': {n: getattr(builtins, n) for n in BUILTINS}}
        args = copy.deepcopy(item['args'])
        before = copy.deepcopy(args)
        ok, error = False, None
        try:
            exec(code, ns)
            value = ns[function](*args)
            ok = 'raises' not in item and same(value, item.get('expected'))
        except (MemoryError, RecursionError):
            raise
        except Exception as exc:
            error = type(exc).__name__
            ok = error == item.get('raises')
        ok = ok and same(args, before)
        results.append({'ok': ok, 'exception_type': error})
    return {'ok': all(r['ok'] for r in results), 'passed': sum(r['ok'] for r in results),
            'total': len(results), 'cases': results}


def main():
    try:
        job_handle = process_limits()
        data = json.loads(sys.stdin.read(1_000_000))
        result = evaluate(data['source'], data['function'], data['cases'])
    except Exception as exc:
        result = {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)[:200]}
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
