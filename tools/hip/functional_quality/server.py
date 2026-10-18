"""Start only dedicated experimental servers; stop only verified children we own."""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from .client import APIClient
from .common import ROOT, append, now, sha

EXE_SHA = '309aab05003e794a6f4975e1499afc77f8d26030aa85a453f2535b90e108f3c5'
PROFILE_SHA = '1bc70f7edf706578a24aca5a6e8beee4f308e192142ae88339968f5da1924a9d'
EXPERT_SET = 'dea737b94e80ddcd'


class ExperimentalServer:
    def __init__(self, config_path, config, port, events):
        self.config_path, self.config, self.port = Path(config_path), config, port
        self.events = events
        self.proc = None
        self.owned = {}

    def remember(self):
        import psutil
        if self.proc is None:
            return
        parent = psutil.Process(self.proc.pid)
        # Windows venv python.exe can redirect to a child base interpreter. Accept
        # it only as a descendant with the exact same server command and a known
        # interpreter executable, never merely because it listens on our port.
        server_command = getattr(self, 'command', None)
        interpreters = {Path(sys.executable).resolve(), Path(sys._base_executable).resolve()}
        for p in [parent] + parent.children(recursive=True):
            try:
                exe, cmd = p.exe(), p.cmdline()
                role = None
                if p.pid == self.proc.pid:
                    role = 'launcher'
                elif Path(exe).resolve() == Path(self.config['exe']).resolve():
                    role = 'engine'
                elif Path(exe).resolve() in interpreters and cmd == server_command:
                    role = 'server'
                elif (os.name == 'nt' and Path(exe).resolve() ==
                      (Path(os.environ['SystemRoot']) / 'System32/conhost.exe').resolve()
                      and len(cmd) == 2 and cmd[1] == '0x4'):
                    # CREATE_NO_WINDOW's own console helper, inside this subtree.
                    role = 'console'
                if role:
                    self.owned.setdefault(p.pid, {'created': p.create_time(), 'exe': exe, 'cmd': cmd, 'role': role})
            except psutil.NoSuchProcess:
                pass

    def listener_owner(self):
        import psutil
        owners = {c.pid for c in psutil.net_connections(kind='tcp') if c.status == 'LISTEN' and c.laddr.port == self.port}
        allowed = {pid for pid, identity in self.owned.items() if identity.get('role') in ('launcher', 'server')}
        if len(owners) != 1 or not owners.issubset(allowed):
            raise RuntimeError('experimental listener ownership mismatch')
        return next(iter(owners))

    def start(self):
        if sha(self.config['exe']) != EXE_SHA:
            raise RuntimeError('candidate binary hash mismatch')
        args = self.config['args']
        profile = args[args.index('--expert-profile') + 1]
        if sha(profile) != PROFILE_SHA:
            raise RuntimeError('per-block expert profile hash mismatch')
        with socket.socket() as sock:
            if os.name == 'nt':
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind(('127.0.0.1', self.port))
        sys.path.insert(0, str(ROOT / 'tools/hip'))
        from _rocm_env import engine_env
        env = engine_env(self.config['exe'])
        for key in list(env):
            if key.startswith(('STRATA_POLY_', 'ROCBLAS_', 'HIPBLASLT_', 'TENSILE_')):
                env.pop(key)
        env.update(self.config['env'])
        env.pop('STRATA_API_KEY', None)
        self.console = self.config_path.with_suffix('.console.log').open('w', encoding='utf-8')
        command = [sys.executable, '-m', 'serve.server', '--engine', 'strata', '--config', str(self.config_path),
                   '--host', '127.0.0.1', '--port', str(self.port)]
        self.command = command
        self.proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=self.console, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        append(self.events, {'at': now(), 'kind': 'server_started', 'pid': self.proc.pid, 'command': command})
        client = APIClient(f'http://127.0.0.1:{self.port}')
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError('experimental server exited during startup')
            self.remember()
            try:
                health = client.get('/health')
                if health.get('status') == 'ok':
                    break
            except (OSError, ValueError):
                pass
            time.sleep(.5)
        else:
            raise TimeoutError('experimental server did not become ready')
        log = Path(self.config['log']).read_text(encoding='utf-8', errors='replace')
        sets = re.findall(r'expert set ([0-9a-f]+) \((\d+) experts', log)
        if not sets or sets[-1] != (EXPERT_SET, '5000'):
            raise RuntimeError('resident expert set/count not confirmed')
        # Query the service plus identify its engine child. Do not infer identity from a port alone.
        listener_pid = self.listener_owner()
        expected_kv = 'q8_0' if '--kv' in args else 'fp16'
        if f'({49152} positions, {expected_kv})' not in log:
            raise RuntimeError('KV format/context not confirmed in engine startup log')
        if sum(i['role'] == 'engine' for i in self.owned.values()) != 1:
            raise RuntimeError('expected exactly one owned engine')
        append(self.events, {'at': now(), 'kind': 'server_ready', 'owned': self.owned,
                             'listener_pid': listener_pid, 'expert_set': EXPERT_SET, 'kv': expected_kv,
                             'status': client.get('/v1/status')})
        return client

    def close(self):
        if self.proc is None:
            return
        import psutil
        if self.proc.poll() is None:
            self.remember()
        # Engine first, server last. Every identity includes PID, creation time, exe and command.
        for pid, identity in sorted(self.owned.items(), key=lambda item: (
                {'engine': 0, 'server': 1, 'console': 2, 'launcher': 3}.get(item[1].get('role'), int(item[0] == self.proc.pid)))):
            try:
                p = psutil.Process(pid)
                if (p.create_time() != identity['created'] or p.exe() != identity['exe']
                        or p.cmdline() != identity['cmd']):
                    raise RuntimeError('refusing to terminate process with changed identity')
                p.terminate()
                try:
                    p.wait(10)
                except psutil.TimeoutExpired:
                    # Recheck identity again before a forced stop.
                    if p.create_time() != identity['created'] or p.exe() != identity['exe']:
                        raise RuntimeError('identity changed while stopping')
                    p.kill()
                    p.wait(10)
                append(self.events, {'at': now(), 'kind': 'owned_process_stopped', 'pid': pid, **identity})
            except psutil.NoSuchProcess:
                pass
        self.proc.wait(timeout=10)
        self.console.close()
