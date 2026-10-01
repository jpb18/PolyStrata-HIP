"""serve/server.py for an Intel Arc: the same server, with the Intel pieces added from outside.

    python sycl/serve/server_intel.py --engine strata --config strata-<model>.json --port 8080 [...]

Every argument is serve/server.py's.  Nothing in serve/ is edited (the SYCL port keeps out of the shared files so
upstream merges stay clean); this wrapper adds, at run time:

  - the Monitor tab's GPU readings on an Intel Arc (sycl/serve/xe_telemetry.py) when no NVIDIA card answers;
  - a model menu in the web app's header, for a host that swaps models on one GPU: the config names the RPC that
    does the swap as "model_switcher" (it takes {"mode": m} and answers {"mode", "up", "starting", "choices",
    "urls"}); the menu offers the models served on this server's port.  Without that key nothing changes.
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import serve.server as S  # noqa: E402
import serve.telemetry as T  # noqa: E402
from xe_telemetry import _XeGpu  # noqa: E402


def install_xe_reader():
    """telemetry.gpu_reader -> the xe reader when NVML has no card and an Intel Arc is there."""
    orig = T.gpu_reader

    def gpu_reader(index=0, amd=False):
        g = orig(index, amd)
        if g.ok():
            return g
        xe = _XeGpu(index)
        return xe if xe.ok() else g

    T.gpu_reader = gpu_reader


def model_switcher(url: str, port, model, mode: str = "") -> dict:
    """The host's model swapper: the models it serves on THIS port, which one holds the card, whether one is loading.
    With `mode`, asks for that one first; this server is then usually the one stopped, so the web app waits for
    /health to name the new model."""
    try:
        body = json.dumps({"mode": mode} if mode else {}).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            res = json.load(r)
        st = res.get("result", res)
    except (OSError, ValueError) as e:
        return {"enabled": True, "error": f"the model switcher did not answer ({e})"}
    urls = st.get("urls") or {}
    mine = {k: v for k, v in (st.get("choices") or {}).items()
            if not urls or str(urllib.parse.urlsplit(str(urls.get(k, ""))).port) == str(port)}
    return {"enabled": True, "mode": st.get("mode"), "up": st.get("up"), "starting": st.get("starting"),
            "choices": mine, "model": model}


def install_switcher(argv):
    """Wraps make_handler: GET/POST /switcher, the menu's script and style, and the index page that loads them."""
    def arg(name, default=None):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default

    try:
        cfg = json.loads(Path(arg("--config")).read_text(encoding="utf-8-sig")) if arg("--config") else {}
    except (OSError, ValueError):
        cfg = {}
    url = cfg.get("model_switcher")
    if not url:
        return
    port = arg("--port", "8080")
    web = HERE / "web"
    types = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
    make_handler = S.make_handler

    def make(svc):
        base = make_handler(svc)

        class Handler(base):
            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0].rstrip("/")
                if path == "/switcher":
                    if self._authorized():
                        self._json(200, model_switcher(url, port, svc.model))
                    return
                if path.startswith("/sycl-web/"):
                    f = web / path[len("/sycl-web/"):]
                    if f.parent != web or f.suffix not in types or not f.is_file():
                        self._json(404, {"error": {"message": "not found"}})
                        return
                    self._send(f.read_bytes(), types[f.suffix])
                    return
                if path == "":
                    page = (ROOT / "serve" / "web" / "index.html").read_text(encoding="utf-8")
                    page = page.replace("</head>", '<link rel="stylesheet" href="sycl-web/switcher.css">\n</head>', 1)
                    page = page.replace("</body>", '<script src="sycl-web/switcher.js"></script>\n</body>', 1)
                    self._send(page.encode("utf-8"), "text/html; charset=utf-8")
                    return
                super().do_GET()

            def do_POST(self):
                path = self.path.split("?")[0].rstrip("/")
                if path == "/switcher":
                    if not self._authorized():
                        return
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    self._json(200, model_switcher(url, port, svc.model, str(req.get("mode") or "")))
                    return
                super().do_POST()

        return Handler

    S.make_handler = make


if __name__ == "__main__":
    install_xe_reader()
    install_switcher(sys.argv[1:])
    sys.exit(S.main())
