"""Small START_HIP_POLY adapter: isolated, voluntary Q8, with the measured expert cache ceiling or none; never
reload an occupied port."""
from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.hip.poly_profile import PROFILES
from tools.hip.functional_quality.resources import (ExclusiveGPU, admission, snapshot,
                                                     MIN_RUNNING_FREE_RAM_MIB)

PROFILE = ROOT / "data/hip/qwen36-q3-q8-expert-profile.txt"
PROFILE_SHA = "1bc70f7edf706578a24aca5a6e8beee4f308e192142ae88339968f5da1924a9d"
EXPERT_SET = "dea737b94e80ddcd"
BASE = "qwen3.6-35b-a3b-ud-q3_k_m-hip"
VISION_BASE = BASE + "-vision"
# Reuse the FP16 profile geometry without changing its menus, files or generation path.
Q8_CONTEXTS = {"32k": 32768, **{name: PROFILES[name][1] for name in ("48k", "128k", "262k")}}
# Same frozen Phase 4b profile, with its measured cap/set at each context.
Q8_EXPERTS = {32768: (5000, EXPERT_SET), 49152: (5000, EXPERT_SET),
              131072: (3500, "25cc61a5c70b01e1"), 262144: (2000, "a1f73bb14a36142e")}
# The cache ceiling the menus offer: that measured value, or none ("uncapped": the engine sizes the cache to the free
# VRAM it finds - AMD_HIP_VRAM_FLOOR.md, where the cap below the 10.75 GiB step cost hit rate and tokens/s at 262K).
Q8_CAPS = ("capped", "uncapped")


def choose_menu(title, options, default, timeout=20):
    print(f"\n  {title}\n", file=sys.stderr)
    for number, (_, label) in enumerate(options, 1):
        print(f"   {number}  {label}", file=sys.stderr)
    while True:
        print(f"  Escolhe 1-{len(options)} [Enter ou 20 s = {default}]: ", end="", file=sys.stderr, flush=True)
        if os.name == "nt" and sys.stdin.isatty():
            import msvcrt
            chars = []
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not msvcrt.kbhit():
                    time.sleep(0.05)
                    continue
                char = msvcrt.getwch()
                if char in ("\r", "\n"):
                    break
                if char == "\x03":
                    raise KeyboardInterrupt
                if char == "\b":
                    if chars:
                        chars.pop()
                        print("\b \b", end="", file=sys.stderr, flush=True)
                elif char in ("\x00", "\xe0"):
                    msvcrt.getwch()  # Ignore both bytes of a function/arrow key.
                else:
                    chars.append(char)
                    print(char, end="", file=sys.stderr, flush=True)
            else:
                chars = []  # An incomplete choice must never opt into Q8 on timeout.
            choice = "".join(chars).strip()
            print(file=sys.stderr)
        else:
            choice = sys.stdin.readline().strip()
        if not choice:
            return options[0][0]
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1][0]
        print(f"  Opcao invalida. Usa 1-{len(options)} ou Enter.", file=sys.stderr)


def choose_kv(timeout=20):
    return choose_menu("PolyStrata HIP - formato KV",
                       [("fp16", "FP16 - Estavel / Padrao"),
                        ("q8_0", "Q8_0 (sem parking)")], "FP16", timeout)


def choose_context(timeout=20):
    return choose_menu("Q8_0 - contexto",
                       [(name, f"{Q8_CONTEXTS[name]} tokens - " +
                         ("262K" if name == "262k" else name.upper()))
                        for name in Q8_CONTEXTS], "32K", timeout)


def choose_caps(timeout=20):
    measured = ", ".join(f"{name.upper()} {Q8_EXPERTS[context][0]}" for name, context in Q8_CONTEXTS.items())
    return choose_menu("Q8_0 - tecto da cache de experts",
                       [("capped", f"Com tecto - o valor medido de cada contexto ({measured} experts)"),
                        ("uncapped", "Sem tecto - a cache enche a VRAM livre (medido a 262K: 4 330 a 5 097 "
                                     "experts, contra 2 000 com o tecto)")], "Com tecto", timeout)


def choose_model(kv="fp16", mode="server"):
    import subprocess
    result = subprocess.run([sys.executable, str(ROOT / "tools/hip/poly_profile.py"), "--list"],
                            capture_output=True, text=True, check=True)
    models = []
    for line in result.stdout.splitlines():
        fields = line.split("|", 2)
        if len(fields) >= 2 and (kv != "q8_0" or fields[0] in (BASE, VISION_BASE)):
            models.append((fields[0], fields[1]))
    if not models:
        raise ValueError("Nenhum modelo configurado para o formato KV selecionado.")
    previous = ROOT / "build-hip-poly-win/last-model.txt"
    preferred = previous.read_text().strip() if previous.exists() else ""
    models.sort(key=lambda item: item[0] != preferred)
    if mode == "check":
        return models[0][0]
    return choose_menu("PolyStrata HIP - modelo", models, models[0][1])


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def arg(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args[:-1] else default


def checked_file(path, expected=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Ficheiro em falta: {path}")
    digest = sha(path)
    if expected and digest != expected:
        raise ValueError(f"SHA-256 inesperado: {path}: {digest}; esperado {expected}")
    return digest


def prepare_q8(config=None, base=None, gguf=None, profile="32k", caps="capped"):
    if profile not in Q8_CONTEXTS:
        raise ValueError(f"Contexto Q8 nao suportado: {profile}; escolhe 32k, 48k, 128k ou 262k (sem streaming).")
    if caps not in Q8_CAPS:
        raise ValueError(f"Tecto Q8 desconhecido: {caps}; escolhe {' ou '.join(Q8_CAPS)}.")
    context = Q8_CONTEXTS[profile]
    cap, _ = Q8_EXPERTS[context]
    if caps == "uncapped":
        # No --expert-cache-max at all: the engine sizes the cache to the free VRAM it finds minus its reserve.
        cap = 0
    # Keep this config outside the installed models: an FP16 server must not discover Q8 as a hot-swap target.
    if base and not any(base.lower() in name for name in (BASE, VISION_BASE)):
        raise ValueError("Q8 disponivel apenas para Qwen3.6-35B-A3B UD-Q3_K_M.")
    selected = VISION_BASE if base and "vision" in base.lower() else BASE
    source = Path(config) if config else ROOT / f"strata-{selected}.json"
    checked_file(source)
    cfg = json.loads(source.read_text(encoding="utf-8-sig"))
    if cfg.get("backend") != "hip" or cfg.get("model_name") not in (BASE, VISION_BASE):
        raise ValueError("O perfil Q8 requer o perfil HIP Qwen3.6-35B-A3B UD-Q3_K_M.")
    model = Path(arg(cfg.get("args", []), "--model", ""))
    if gguf and Path(gguf).resolve() != model.resolve():
        raise ValueError("O GGUF indicado nao corresponde ao modelo validado para Q8.")
    if model.name.lower() != "qwen3.6-35b-a3b-ud-q3_k_m.gguf" or not model.is_file():
        raise ValueError(f"Modelo Q8 esperado em falta ou diferente: {model}")
    vision = cfg.get("model_name") == VISION_BASE
    if vision:
        if not isinstance(cfg.get("vision"), dict):
            raise ValueError("Perfil Vision sem encoder configurado.")
        for key in ("exe", "mmproj", "model"):
            checked_file(cfg["vision"].get(key, ""))
        if Path(cfg["vision"]["model"]).resolve() != model.resolve():
            raise ValueError("O modelo do encoder Vision difere do modelo de texto.")
    checked_file(cfg.get("exe", ""))   # the one engine, the config's (build-hip-poly-win)
    checked_file(PROFILE, PROFILE_SHA)
    # Replace only the experimental settings; keep tokenizer, sampling, threads, turn token, port and MTP.
    cfg = copy.deepcopy(cfg)
    replaced = {"--max-context", "--kv", "--kv-k", "--kv-v", "--kv-resident", "--kv-ram-mib",
                "--expert-profile", "--expert-cache-max", "--expert-swaps", "--conversation-cache-mib",
                "--conversation-cache-slots", "--image-min-tokens"}
    # The textual base currently carries this Vision-only option; the pinned Phase 3 binary rejects it.
    # The Poly image protocol uses encoder embeddings without this CLI option.
    args, old, i = [], cfg["args"], 0
    while i < len(old):
        if old[i] in replaced:
            if i + 1 >= len(old):
                raise ValueError(f"Argumento sem valor: {old[i]}")
            i += 2
        else:
            args.append(old[i])
            i += 1
    runs = ROOT / "tmp/hip-launch-q8"
    runs.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="q8-", dir=runs))
    expert = run / "expert-profile.txt"
    expert.write_bytes(PROFILE.read_bytes())
    args += ["--max-context", str(context), "--kv", "q8_0", "--expert-profile", str(expert)]
    if cap:
        args += ["--expert-cache-max", str(cap)]
    args += ["--expert-swaps", "0", "--conversation-cache-mib", "0", "--conversation-cache-slots", "0"]
    ceiling = "" if cap else " sem tecto"
    cfg.update(args=args, cwd=str(ROOT), log=str(run / "engine-q8_0.log"),
               switch_on_request=False, lazy_load=False,
               display_name=f"Qwen3.6-35B-A3B Q8_0 {profile.upper()}{ceiling}" + (" + Vision" if vision else ""))
    # Preserve the selected Vision encoder; slot restore and external unload hooks stay disabled.
    for key in (("slot_save_path", "before_load") if vision else ("vision", "slot_save_path", "before_load")):
        cfg.pop(key, None)
    cfg["env"] = dict(cfg.get("env") or {}, STRATA_POLY_CACHE_HASH="1", STRATA_POLY_WARM="1")
    out = run / f"strata-{BASE}.json"
    out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return out


def q8_policy(cfg):
    """(context, cap, expert set).  A cap of 0 with set None is the profile with no ceiling: the engine sizes the
    cache to the free VRAM, so nothing pins the set and verify_ready checks the count instead."""
    args = cfg["args"]
    context = int(arg(args, "--max-context", "0"))
    if context not in Q8_EXPERTS:
        raise ValueError(f"Contexto Q8 nao suportado: {context}")
    measured, expert_set = Q8_EXPERTS[context]
    ceiling = arg(args, "--expert-cache-max")
    if ceiling == str(measured):
        cap = measured
    elif ceiling is None:
        cap, expert_set = 0, None
    else:
        raise ValueError(f"Perfil Q8 invalido: --expert-cache-max {ceiling} nao e o medido a este contexto "
                         f"({measured}) nem a ausencia de tecto.")
    required = {"--kv": "q8_0", "--expert-swaps": "0",
                "--conversation-cache-mib": "0", "--conversation-cache-slots": "0"}
    if any(arg(args, key) != value or args.count(key) != 1 for key, value in required.items()) or \
            args.count("--max-context") != 1 or args.count("--expert-cache-max") > 1 or \
            any(flag in args for flag in ("--kv-k", "--kv-v", "--kv-resident")):
        raise ValueError("Perfil Q8 invalido: KV, contexto, experts ou parking nao correspondem ao perfil selecionado.")
    if cfg.get("vision") and (cfg.get("model_name") != VISION_BASE or not isinstance(cfg["vision"], dict)):
        raise ValueError("Perfil Vision Q8 invalido.")
    if cfg.get("model_name") == VISION_BASE and not cfg.get("vision"):
        raise ValueError("Perfil Vision sem encoder configurado.")
    if cfg.get("backend") != "hip" or cfg.get("slot_save_path") or cfg.get("switch_on_request"):
        raise ValueError("Perfil Q8 invalido: requer HIP, sem parking/restore ou troca automatica.")
    return context, cap, expert_set


def launcher_pids():
    """The executing interpreter and, on Windows, its verified venv redirector only."""
    import psutil
    owned = {os.getpid()}
    if os.name != "nt":
        return owned
    try:
        current = psutil.Process()
        parent = current.parent()
        if parent is not None and parent.name().lower() in ("python.exe", "pythonw.exe") and \
                parent.cmdline() == current.cmdline() and \
                Path(parent.exe()).resolve() == Path(sys.executable).resolve() and \
                Path(parent.exe()).resolve() != Path(current.exe()).resolve():
            owned.add(parent.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        # If identity cannot be proved, leave it in the inventory and retain the refusal.
        pass
    return owned


def check_resources(cfg):
    context, _, _ = q8_policy(cfg)
    owned = launcher_pids()
    state = snapshot(owned=owned)
    # The existing campaign scanner recognises module-style servers; include script-style launchers as well.
    import psutil
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.pid in owned:
                continue
            command = " ".join(process.info.get("cmdline") or []).lower().replace("\\", "/")
            if "python" in (process.info.get("name") or "").lower() and any(
                    marker in command for marker in ("serve/server.py", "hip/poly_launch.py")):
                if process.pid not in {p["pid"] for p in state["other_inference"]}:
                    state["other_inference"].append(process.info)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            state["process_scan_complete"] = False
    reasons = admission(state)
    print(f"  Guarda Q8 {context}: RAM livre={state['ram_available_mib']:.0f} MiB; "
          f"VRAM ocupada={state.get('vram_used_mib')} MiB", flush=True)
    if reasons:
        for process in state["other_inference"]:
            # Do not print command arguments, which can contain API keys.
            print(f"  Processo bloqueante: PID={process.get('pid')} nome={process.get('name')} "
                  f"executavel={process.get('exe') or 'nao disponivel'}", flush=True)
        raise ValueError("Arranque Q8 recusado pelas guardas existentes: " + "; ".join(reasons))
    return state


def health_at(port):
    # Local control traffic must not follow a user's HTTP proxy settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
        return json.load(response)


def port_free(port):
    # A failed bind includes a listener that has no HTTP health endpoint. No POST/load, stop or kill here.
    with socket.socket() as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            who = "outro processo (ou porta indisponivel)"
            try:
                health = health_at(port)
                if health.get("service") == "strata":
                    who = f"PolyStrata, modelo {health.get('model')}, loaded={health.get('loaded')}"
            except (OSError, ValueError):
                pass
            raise ValueError(f"Porta {port} ocupada: {who}. Nada foi parado ou trocado. "
                             "Encerra normalmente a sessao anterior e volta a executar o START.") from None


def describe(cfg, kv):
    args = cfg["args"]
    print("\n  PolyStrata HIP", flush=True)
    print(f"  Motor: {'Q8_0' if kv == 'q8_0' else 'FP16 - Estavel / Padrao'}")
    print(f"  Executavel: {cfg['exe']}\n  SHA-256: {sha(cfg['exe'])}")
    print(f"  Modelo: {arg(args, '--model')}\n  KV Cache: {kv} (configurado)")
    print(f"  Contexto: {arg(args, '--max-context')}\n  Porta: {cfg.get('port', 8090)}")
    window = arg(args, "--window", "auto (motor)")
    mtp = "Desativado" if "--no-draft" in args or window == "1" else "Ativo (configurado)"
    print(f"  MTP: {mtp}; window={window}\n  Perfil: {arg(args, '--expert-profile')}")
    if kv == "q8_0":
        context, cap, expert_set = q8_policy(cfg)
        if cap:
            print(f"  Experts: cap={cap}, swaps=0; set esperado {expert_set}; parking=0")
        else:
            print("  Experts: sem tecto (a cache enche a VRAM livre), swaps=0; parking=0")
        if context == Q8_CONTEXTS["262k"]:
            print("  Contexto 262K: 262K")
        print(f"  Logs: {cfg['log']}")
    print("  Estado: configuracao verificada; INFO/READY por confirmar", flush=True)


def verify_ready(engine, cfg):
    context, cap, expert_set = q8_policy(cfg)
    info = engine.info
    if info.get("kv") != "q8_0" or info.get("context") != context or engine.max_context != context:
        raise ValueError(f"Q8 recusado: INFO/READY inesperado: {info}, READY {engine.max_context}")
    if info.get("conversation_cache_mib") != 0 or (cap and info.get("expert_slots") != cap):
        raise ValueError(f"Q8 recusado: parking ou expert cache inesperado: {info}")
    if Path(engine.spawn[0]).resolve() != Path(cfg["exe"]).resolve():
        raise ValueError("Q8 recusado: executavel efetivo diferente do configurado.")
    log = Path(cfg["log"]).read_text(encoding="utf-8", errors="replace")
    # With no ceiling the set is whatever the free VRAM held, so its size is not pinned to a measurement; the log's
    # own set line must at least confirm the count INFO reports, and the cache must not be empty.
    if cap:
        if f"expert set {expert_set} ({cap} experts in VRAM)" not in log:
            raise ValueError("Q8 recusado: o log nao confirma o conjunto de experts congelado.")
    elif not isinstance(info.get("expert_slots"), int) or info["expert_slots"] <= 0 or \
            f"({info['expert_slots']} experts in VRAM)" not in log:
        raise ValueError(f"Q8 recusado: sem tecto, mas o log nao confirma o conjunto: {info}")
    # Keep stdout protocol evidence in the launcher's own stream. On Windows, opening the child's stderr
    # file a second time can leave its inherited file offset behind and overwrite this appended record.
    print(f"[launcher] INFO efetivo {json.dumps(info, sort_keys=True)}; READY {engine.max_context}", flush=True)
    experts = expert_set if cap else f"sem tecto: {info.get('expert_slots')}"
    print(f"  INFO confirmado: kv=q8_0 context={context} experts={experts}; READY {context}", flush=True)


def check_loaded_resources(engine, cfg):
    import psutil
    context, _, _ = q8_policy(cfg)
    floor = 4096 if context >= 131072 else MIN_RUNNING_FREE_RAM_MIB
    free_ram = psutil.virtual_memory().available / 2**20
    free_vram = engine.info.get("vram_free_mib")
    short = engine.info.get("vram_reserve_short_mib")
    problems = []
    if free_ram < floor:
        shortage = math.ceil(floor - free_ram)
        problems.append(f"faltam {shortage} MiB de RAM disponivel para a margem de seguranca; "
                        "fecha programas para libertar memoria e volta a executar o START")
    if not isinstance(free_vram, int):
        problems.append("o motor nao reportou VRAM livre; nao e possivel confirmar a margem")
    elif free_vram < 512:
        problems.append(f"faltam {512 - free_vram} MiB de VRAM livre")
    if isinstance(short, int) and short > 0:
        problems.append(f"a cache minima deixou a reserva curta em {short} MiB (--expert-cache-min): "
                        "baixa o minimo de experts ou a reserva")
    if problems:
        raise ValueError(f"Q8 recusado apos carga: RAM livre {free_ram:.0f} MiB (minimo {floor}); "
                         f"VRAM livre {free_vram} MiB (minimo 512). " + "; ".join(problems) +
                         ". INFO/READY confirmado, mas o servidor HTTP ainda nao foi aberto.")
    print(f"  Guarda apos carga: RAM livre={free_ram:.0f} MiB; VRAM livre={free_vram} MiB", flush=True)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def run_server(config, cfg, kv, chat=False, open_browser=False):
    if kv == "q8_0":
        check_resources(cfg)
    sys.path.insert(0, str(ROOT))
    from serve import server
    original_start, original_serve = server.start_model, server.serve

    def start_checked(actual, tok, lazy=False):
        result = original_start(actual, tok, lazy)
        if kv == "q8_0":
            try:
                verify_ready(result[0], actual)
                check_loaded_resources(result[0], actual)
            except Exception:
                result[0].close()  # Only the engine just created by this invocation, before HTTP is served.
                raise
        elif not lazy:
            engine = result[0]
            reported = engine.info.get("kv", "nao reportado pelo binario operacional; FP16 por omissao")
            print(f"  Motor iniciado: {engine.spawn[0]}; PID={engine.proc.pid}; "
                  f"READY {engine.max_context}; KV efetivo: {reported}", flush=True)
        return result

    def serve_checked(svc, host="127.0.0.1", port=8095):
        httpd = original_serve(svc, host=host, port=port)
        try:
            health = health_at(port)
            if kv == "q8_0" and (health.get("service") != "strata" or health.get("status") != "ok"
                                   or not health.get("loaded") or health.get("model") != cfg["model_name"]
                                   or health.get("max_context") != q8_policy(cfg)[0]):
                raise ValueError(f"Q8: health inesperado: {health}")
            print(f"  Health confirmado em 127.0.0.1:{port}: {health}", flush=True)
        except Exception:
            httpd.shutdown()
            svc.engine.close()
            raise
        if chat:
            def terminal():
                import _thread
                import chat as client
                try:
                    client.chat(f"http://127.0.0.1:{port}", "medium", 4096,
                                cfg.get("api_key") or os.environ.get("STRATA_API_KEY", ""))
                finally:
                    _thread.interrupt_main()
            threading.Thread(target=terminal, daemon=True).start()
        return httpd

    server.start_model, server.serve = start_checked, serve_checked
    argv = sys.argv
    sys.argv = ["serve/server.py", "--engine", "strata", "--config", str(config),
                "--port", str(cfg.get("port", 8090)), "--host", "127.0.0.1"]
    if open_browser and not chat:
        sys.argv += ["--open"]
    try:
        return server.main()
    finally:
        sys.argv = argv
        server.start_model, server.serve = original_start, original_serve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--choose-kv", action="store_true")
    ap.add_argument("--choose-context", action="store_true")
    ap.add_argument("--choose-caps", action="store_true")
    ap.add_argument("--choose-model", action="store_true")
    ap.add_argument("--prepare-q8", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--base")
    ap.add_argument("--gguf")
    ap.add_argument("--profile", default="32k")
    ap.add_argument("--caps", choices=list(Q8_CAPS), default="capped")
    ap.add_argument("--kv", choices=["fp16", "q8_0"], default="fp16")
    ap.add_argument("--mode", choices=["server", "chat", "check", "preflight"], default="server")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    try:
        if a.choose_kv:
            print(choose_kv())
            return 0
        if a.choose_model:
            print(choose_model(a.kv, a.mode))
            return 0
        if a.choose_context:
            print(choose_context())
            return 0
        if a.choose_caps:
            print(choose_caps())
            return 0
        if a.prepare_q8:
            print(prepare_q8(a.config, a.base, a.gguf, a.profile, a.caps))
            return 0
        if not a.config:
            raise ValueError("Falta --config.")
        checked_file(a.config)
        cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
        checked_file(cfg["exe"])
        args = cfg["args"]
        if a.kv == "q8_0":
            checked_file(arg(args, "--expert-profile", ""), PROFILE_SHA)
            q8_policy(cfg)
        elif any(arg(args, k, "fp16") != "fp16" for k in ("--kv", "--kv-k", "--kv-v")):
            raise ValueError("FP16 selecionado mas o perfil pede outro formato KV.")
        describe(cfg, a.kv)
        port_free(int(cfg.get("port", 8090)))
        if a.mode == "check":
            print(f"  CHECK: nao iniciado; args={json.dumps(args)}")
            return 0
        if a.mode == "preflight":
            if a.kv != "q8_0":
                raise ValueError("Preflight de recursos disponivel apenas para Q8.")
            with ExclusiveGPU(ROOT / "tmp/functional-quality/gpu.lock"):
                check_resources(cfg)
            print("  PREFLIGHT aprovado; nenhum motor iniciado.", flush=True)
            return 0
        if a.kv == "q8_0":
            with Path(cfg["log"]).open("a", encoding="utf-8") as stream:
                stream.write(f"[launcher] Q8 kv=q8_0 exe={cfg['exe']} sha256={sha(cfg['exe'])}\n"
                             f"[launcher] config={a.config} port={cfg.get('port', 8090)} args={json.dumps(args)}\n")
        if a.kv == "q8_0":
            # Keep server diagnostics as well as raw engine stderr (including startup failures/GPU_FATAL).
            with Path(cfg["log"]).with_name("launcher-q8_0.log").open("a", encoding="utf-8") as log:
                with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                    try:
                        with ExclusiveGPU(ROOT / "tmp/functional-quality/gpu.lock"):
                            return run_server(a.config, cfg, a.kv, a.mode == "chat", a.open)
                    except BaseException as error:
                        print(f"[launcher] Q8 terminou: {type(error).__name__}: {error}", file=sys.stderr)
                        raise
        return run_server(a.config, cfg, a.kv, a.mode == "chat", a.open)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"[START HIP] ERRO: {error}", file=sys.stderr, flush=True)
        print("Sem fallback automatico. FP16 operacional preservado.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
