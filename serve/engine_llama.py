"""serve/engine_llama.py - Strata's engine boundary over llama.cpp's `llama-server`.

The Strata engine (`strata --serve`) is CUDA: sm_80+ tensor-core kernels, cudaGraph, cuBLAS. On a GPU without
CUDA - the Intel Arc Pro B70 this was written for - the same GGUF weights run through llama.cpp's SYCL backend.
This class puts that engine behind the same boundary the server already speaks:

    Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of token ids   (None = heartbeat)

so everything above it - the OpenAI and Anthropic endpoints, streaming, tool calls, MCP, the web app - is
unchanged.  It talks to llama-server's `/completion` with the prompt AS TOKEN IDS (the server's own tokenizer
produced them, from the same GGUF, so the ids agree) and `return_tokens`, which streams ids back one chunk
at a time with the timings Strata reports.

Two ways to get an engine:

    LlamaEngine.attach("http://127.0.0.1:8081")            a llama-server someone else runs (a container, say)
    LlamaEngine.spawn([exe, *args], log=..., cwd=...)      start `llama-server` ourselves and own its lifetime

Both settle `max_context` from `/props` and expose the same `info` the Monitor tab reads.

What it does not do (yet): images.  Strata's `Vision` encodes images with `strata-vision` into embeddings the
CUDA engine reads (`GENI`); llama-server has its own multimodal path (`--mmproj`) that takes the image itself.
Until that is wired, run the server with no `vision` in the config and it refuses image requests cleanly.
"""
from __future__ import annotations

import http.client
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

HEARTBEAT_S = 10                 # yield None this often while the server is quiet (reading a long prompt)
READY_TIMEOUT_S = 20 * 60        # a 30 GB model from a cold SSD can take this long to load
STOP_TYPES = {"eos": "stop", "word": "stop", "limit": "length"}   # llama-server's stop_type -> Strata's finish


class EngineDied(RuntimeError):
    """The engine went away in the middle of a request."""


class LlamaEngine:
    def __init__(self, url: str, spawn: list[str] | None = None, cwd: str | None = None, log: str | None = None,
                 env: dict | None = None, model_path: str = "", name: str = "llama.cpp"):
        self.url = url.rstrip("/")
        self.spawn_cmd, self.cwd, self.log_path, self.env = spawn, cwd, log, env
        self.model_path = model_path
        self.proc: subprocess.Popen | None = None
        self.log = None
        self.ended = False
        self.max_context = 0
        self.can_stop = True          # closing the HTTP response stops llama-server's generation
        self.last: dict = {}
        self.info: dict = {"engine": name}
        self.progress = None          # llama-server reports no prompt-reading progress on this path
        self._lock = threading.Lock()
        if spawn:
            self._start()
        self._wait_ready()

    # ------------------------------------------------------------------------------------ construction
    @classmethod
    def attach(cls, url: str, model_path: str = "") -> "LlamaEngine":
        """Use a llama-server that is already running (started by hand, by a container, by the render front end)."""
        return cls(url, model_path=model_path)

    @classmethod
    def spawn(cls, cmd: list[str], url: str, cwd: str | None = None, log: str | None = None,
              env: dict | None = None, model_path: str = "") -> "LlamaEngine":
        """Start `llama-server` ourselves.  `cmd` is the whole command line, `url` where it will listen."""
        return cls(url, spawn=cmd, cwd=cwd, log=log, env=env, model_path=model_path)

    def _start(self):
        self.log = open(self.log_path, "a", encoding="utf-8") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.spawn_cmd, cwd=self.cwd, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=self.env)
        self.ended = False

    def _wait_ready(self):
        """`/health` says 503 until the model is loaded; `/props` answers once it is and carries n_ctx."""
        t0 = time.time()
        while True:
            if self.proc is not None and self.proc.poll() is not None:
                raise EngineDied(f"llama-server exited during load (code {self.proc.returncode})"
                                 + (f"; see {self.log_path}" if self.log_path else ""))
            try:
                props = self._get("/props", timeout=5)
                gen = props.get("default_generation_settings") or {}
                self.max_context = int(gen.get("n_ctx") or 0)
                if self.max_context:
                    self.info.update(
                        version=(props.get("build_info") or "").strip() or None,
                        n_ctx=self.max_context,
                        model=os.path.basename(props.get("model_path") or self.model_path or ""),
                        modalities=",".join(k for k, v in (props.get("modalities") or {}).items() if v) or "text",
                    )
                    if not self.model_path:
                        self.model_path = props.get("model_path") or ""
                    return
            except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
                pass
            if time.time() - t0 > READY_TIMEOUT_S:
                raise EngineDied(f"llama-server at {self.url} did not become ready in {READY_TIMEOUT_S // 60} min")
            elapsed = int(time.time() - t0)
            if elapsed and elapsed % 30 == 0:
                print(f"[strata] still loading ({elapsed} s) - a 30 GB model takes a few minutes ...", flush=True)
            time.sleep(1)

    # ------------------------------------------------------------------------------------------ http
    def _get(self, path: str, timeout: float = 30):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post_stream(self, path: str, body: dict, timeout: float):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"content-type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    # -------------------------------------------------------------------------------------- generate
    @staticmethod
    def sampling_body(sampling: dict) -> dict:
        """Strata's request sampling -> llama-server's field names.  Absent temperature means greedy, as the
        CUDA engine does; a client's own values always win over the server's defaults."""
        out: dict = {}
        t = sampling.get("temperature")
        out["temperature"] = float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) and t > 0 else 0.0
        for src, dst, lo in (("top_p", "top_p", 0.0), ("min_p", "min_p", 0.0),
                             ("presence_penalty", "presence_penalty", None),
                             ("frequency_penalty", "frequency_penalty", None),
                             ("repetition_penalty", "repeat_penalty", None)):
            v = sampling.get(src)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[dst] = float(v)
        tk = sampling.get("top_k")
        if isinstance(tk, int) and not isinstance(tk, bool) and tk >= 0:
            out["top_k"] = tk
        n = sampling.get("penalty_last_n")
        if isinstance(n, int) and not isinstance(n, bool):
            out["repeat_last_n"] = n
        seed = sampling.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            out["seed"] = seed
        return out

    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event,
                 embeddings=None) -> Iterator[int | None]:
        """Yields token ids as llama-server streams them, and None every HEARTBEAT_S while it is quiet.
        A consumer that stops early, or `cancel`, closes the response, which stops the generation server-side."""
        if embeddings:
            raise ValueError("images are not supported on the llama.cpp engine yet (start without vision)")
        body = {"prompt": [int(t) for t in ids], "n_predict": int(max_new), "stream": True,
                "return_tokens": True, "cache_prompt": True, **self.sampling_body(sampling or {})}
        self.last = {}
        self.progress = None
        try:
            resp = self._post_stream("/completion", body, timeout=HEARTBEAT_S)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise ValueError(f"llama-server refused the request ({e.code}): {detail}") from None
        except (urllib.error.URLError, OSError) as e:
            raise EngineDied(f"llama-server is not answering: {e}") from None
        finished = False
        try:
            buf = b""
            while True:
                if cancel.is_set():
                    return
                try:
                    chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                except TimeoutError:
                    yield None                                # quiet: reading a long prompt
                    continue
                except (http.client.IncompleteRead, OSError) as e:
                    raise EngineDied(f"the connection to llama-server broke mid-answer: {e}") from None
                if not chunk:
                    raise EngineDied("llama-server closed the stream without finishing")
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data: "):
                        continue
                    ev = json.loads(line[6:])
                    if "error" in ev:
                        raise ValueError(str(ev["error"]))
                    for t in ev.get("tokens") or []:
                        if cancel.is_set():
                            return
                        yield int(t)
                    if ev.get("stop"):
                        tm = ev.get("timings") or {}
                        self.last = {"generated": int(tm.get("predicted_n") or ev.get("tokens_predicted") or 0),
                                     "prompt_tokens": int(tm.get("prompt_n") or ev.get("tokens_evaluated") or 0),
                                     "prompt_ms": float(tm.get("prompt_ms") or 0.0),
                                     "decode_ms": float(tm.get("predicted_ms") or 0.0),
                                     "finish": STOP_TYPES.get(ev.get("stop_type"), "stop"),
                                     "reused": int(tm.get("cache_n") or 0)}
                        finished = True
                        return
        finally:
            try:
                resp.close()                                  # early stop: dropping the connection stops the server
            except Exception:
                pass
            if not finished and not self.last:
                self.last = {"finish": "cancelled"}

    # ------------------------------------------------------------------------------------- lifecycle
    def alive(self) -> bool:
        if self.ended:
            return False
        if self.proc is not None:
            return self.proc.poll() is None
        try:
            self._get("/health", timeout=3)
            return True
        except Exception:
            return False

    def exit_code(self):
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    def death_note(self) -> str:
        tail = ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", "replace")
        except (OSError, TypeError):
            pass
        low = tail.lower()
        if "out of memory" in low or "sycl_out_of_memory" in low or "failed to allocate" in low:
            return "the GPU ran out of memory (a smaller context, or fewer layers on the card)"
        if "sigsegv" in low or "segmentation fault" in low:
            return "llama-server crashed (if SYCL_CACHE_PERSISTENT is set, unset it: the JIT cache segfaults on Xe2)"
        if self.proc is None:
            return f"the llama-server at {self.url} stopped answering (it is not managed by this server)"
        return "llama-server stopped; the end of its log:\n" + tail[-1200:]

    def restart(self):
        if self.proc is None:
            self.ended = False
            self._wait_ready()                              # attached: wait for whoever runs it to bring it back
            return
        try:
            self.proc.kill()
        except OSError:
            pass
        info = dict(self.info)
        self._start()
        self._wait_ready()
        self.info = {**info, **self.info}

    def close(self):
        self.ended = True
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=60)      # a container unloading 30 GB takes a moment
        except Exception:
            try:
                self.proc.kill()
            except OSError:
                pass


def engine_from_config(cfg: dict, env: dict | None = None) -> LlamaEngine:
    """Build the engine the run config describes.

        "llama": {"url": "http://127.0.0.1:8081"}                       attach to a running llama-server
        "llama": {"exe": ".../llama-server", "args": [...], "port": 8081}  spawn it (args after -m/--port are ours)

    `model` (the GGUF) is optional and only labels the Monitor tab when attached."""
    L = cfg.get("llama") or {}
    model = L.get("model") or cfg.get("model") or ""
    if L.get("exe"):
        port = int(L.get("port") or 8081)
        host = L.get("host") or "127.0.0.1"
        cmd = [L["exe"], *[str(a) for a in (L.get("args") or [])], "--host", host, "--port", str(port)]
        if model and "-m" not in cmd and "--model" not in cmd:
            cmd += ["-m", model]
        return LlamaEngine.spawn(cmd, f"http://{host}:{port}", cwd=cfg.get("cwd"), log=cfg.get("log"), env=env,
                                 model_path=model)
    url = L.get("url")
    if not url:
        raise ValueError('the config\'s "llama" block needs either "url" (attach) or "exe" (spawn)')
    start = L.get("start")
    if start and not _answering(url):
        # setup wrote a start script (a container that owns the GPU): run it as our child, so the run script is
        # one click and closing the server stops the container. If something already answers at the url, use it.
        print(f"[strata] starting llama-server: {start}", flush=True)
        return LlamaEngine.spawn([start], url, cwd=cfg.get("cwd"), log=cfg.get("log"), env=env, model_path=model)
    return LlamaEngine.attach(url, model_path=model)


def _answering(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3) as r:
            return r.status < 500
    except urllib.error.HTTPError as e:
        return e.code < 500                 # 503 = loading: someone is already bringing it up
    except Exception:
        return False
