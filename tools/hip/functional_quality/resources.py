from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .common import now, strict_json

# Conservative admission limits, frozen before any model output is observed.
# Prior candidate logs: ~10894 MiB VRAM and ~1 GiB pinned RAM, plus warm GGUF pages.
MIN_FREE_RAM_MIB = 16384
MAX_BACKGROUND_VRAM_MIB = 3072
MIN_RUNNING_FREE_RAM_MIB = 2048


def snapshot(owned=()):
    import psutil
    vm = psutil.virtual_memory()
    result = {'at': now(), 'ram_available_mib': vm.available / 2**20,
              'ram_total_mib': vm.total / 2**20, 'swap': dict(psutil.swap_memory()._asdict()),
              'other_inference': [], 'process_scan_complete': True}
    for p in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
        try:
            name = (p.info['name'] or '').lower()
            command = ' '.join(p.info['cmdline'] or []).lower()
            inference = any(s in name for s in ('strata', 'llama', 'ollama', 'llmster', 'vllm'))
            inference |= ('python' in name and any(s in command for s in ('serve.server', 'llama_cpp.server', 'vllm.entrypoints')))
            if p.pid not in owned and inference:
                result['other_inference'].append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            result['process_scan_complete'] = False
    if os.name == 'nt':
        script = '''$ErrorActionPreference='Stop'
$gpu=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | Select-Object Name,DedicatedUsage,SharedUsage)
$mem=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory | Select-Object AvailableMBytes,PagesInputPersec,PageReadsPersec
@{gpu=$gpu; memory=$mem} | ConvertTo-Json -Depth 4 -Compress'''
        try:
            proc = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                                  capture_output=True, text=True, timeout=20)
            if proc.returncode:
                raise RuntimeError(proc.stderr.strip()[:400])
            result['windows'] = strict_json(proc.stdout)
            gpus = result['windows']['gpu']
            result['vram_used_mib'] = sum(int(g['DedicatedUsage']) for g in gpus) / 2**20 if gpus else None
            result['vram_shared_mib'] = sum(int(g['SharedUsage']) for g in gpus) / 2**20 if gpus else None
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            result['resource_error'] = str(exc)
            result['vram_used_mib'] = None
    else:
        result['resource_error'] = 'GPU admission telemetry is currently implemented for Windows only'
        result['vram_used_mib'] = None
    return result


def admission(state):
    reasons = []
    if state['ram_available_mib'] < MIN_FREE_RAM_MIB:
        reasons.append(f'free RAM below {MIN_FREE_RAM_MIB} MiB')
    if state.get('vram_used_mib') is None:
        reasons.append('VRAM telemetry unavailable')
    elif state['vram_used_mib'] > MAX_BACKGROUND_VRAM_MIB:
        reasons.append(f'background VRAM above {MAX_BACKGROUND_VRAM_MIB} MiB')
    if not state['process_scan_complete']:
        reasons.append('process inventory incomplete')
    if state['other_inference']:
        reasons.append('another inference process is running')
    return reasons


class ExclusiveGPU:
    """OS lock survives file leftovers and releases automatically on process exit."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError('another functional campaign owns the GPU lock') from None
        return self

    def __exit__(self, *args):
        self.file.close()
