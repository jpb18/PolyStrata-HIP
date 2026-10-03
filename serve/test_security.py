"""serve/test_security.py - the Host check (DNS rebinding), against the mock engine (no GPU, no pack).

    python -m unittest serve.test_security -v
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, MockEngine, Service, allowed_hosts_of, host_allowed,  # noqa: E402
                          host_name, host_names_for, serve)

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {"localhost", "127.0.0.1", "::1"}


class HostNames(unittest.TestCase):
    def test_host_name(self):
        self.assertEqual(host_name("Example.COM:8080"), "example.com")
        self.assertEqual(host_name("localhost"), "localhost")
        self.assertEqual(host_name("[::1]:8095"), "::1")
        self.assertEqual(host_name("[::1]"), "::1")
        self.assertEqual(host_name("strata.example.com."), "strata.example.com")
        for bad in ("", "a:b", "[::1", "[::1]x", "a b", "evil.com/x"):
            self.assertEqual(host_name(bad), "", bad)

    def test_allowed_hosts_of(self):
        self.assertEqual(allowed_hosts_of(None), [])
        self.assertEqual(allowed_hosts_of("Strata.Example.com"), ["strata.example.com"])
        self.assertEqual(allowed_hosts_of(["https://a.example.com:8443/x", ".example.org", "*"], "box, nas.lan "),
                         ["a.example.com", ".example.org", "*", "box", "nas.lan"])
        for bad in (["bad name"], 5, [3], "http://"):
            with self.assertRaises(ValueError):
                allowed_hosts_of(bad)

    def test_names_from_the_bind_address(self):
        with mock.patch("serve.server.lan_addresses", return_value=["192.168.1.20"]):
            self.assertEqual(host_names_for("127.0.0.1"), LOCAL)                 # this PC only: nothing else
            self.assertEqual(host_names_for("localhost"), LOCAL)
            names = host_names_for("0.0.0.0")                                    # other devices too
            self.assertIn(socket.gethostname().lower(), names)
            self.assertIn(socket.gethostname().lower() + ".local", names)
            self.assertIn("192.168.1.20", names)
            self.assertIn("10.0.0.7", host_names_for("10.0.0.7"))
            names = host_names_for("127.0.0.1", ["strata.example.com", "*"], ["https://chat.example.net"])
            self.assertTrue({"strata.example.com", "chat.example.net"} <= names)
            self.assertNotIn("*", names)


class HostCheck(unittest.TestCase):
    def test_accepted(self):
        for host in ("localhost:8095", "LOCALHOST", "127.0.0.1:1", "[::1]:8095", "192.168.1.20:8095", "[fe80::1]",
                     "app.localhost:8095", None, "", "box:8095", "a.example.org", "example.org:443"):
            self.assertTrue(host_allowed(host, LOCAL | {"box", ".example.org"}), host)

    def test_refused(self):
        # a rebinding page sends its own name; lookalikes of allowed names stay out
        for host in ("evil.example.com", "evil.example.com:8095", "localhost.evil.com", "127.0.0.1.nip.io",
                     "box.evil.com", "xexample.org", "bad host", "a:b:c:d:e:f:g:h:i"):
            self.assertFalse(host_allowed(host, LOCAL | {"box", ".example.org"}), host)

    def test_any_host(self):
        self.assertTrue(host_allowed("evil.example.com", LOCAL, any_host=True))


class OverHttp(unittest.TestCase):
    """The checks in the running server, as a browser and as curl/the SDKs send their requests."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = None

    def start(self, **attrs):
        for k, v in attrs.items():
            setattr(self.svc, k, v)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def req(self, method, path, body=None, headers=None, host=None):
        """(status, JSON, what the server printed); `host` replaces the Host header http.client sends."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        h = dict(headers or {})
        if host is not None:
            h["Host"] = host
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                c.request(method, path, body=data, headers=h)
                r = c.getresponse()
                raw = r.read()
            return r.status, json.loads(raw) if raw[:1] in (b"{", b"[") else raw, out.getvalue()
        finally:
            c.close()

    def chat_body(self):
        return {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}

    # --- Host
    def test_rebinding_name_is_refused_everywhere(self):
        self.start()
        for method, path in (("GET", "/status"), ("GET", "/metrics"), ("GET", "/"), ("GET", "/health"),
                             ("POST", "/settings"), ("POST", "/v1/chat/completions"), ("OPTIONS", "/v1/models")):
            code, body, log = self.req(method, path, {} if method == "POST" else None,
                                       {"Content-Type": "application/json"}, host=f"evil.example.com:{self.port}")
            self.assertEqual(code, 403, path)
            if method != "OPTIONS":
                self.assertIn("allowed_hosts", body["error"]["message"])
            self.assertEqual(log.count("\n"), 1, path)                          # one log line each
            self.assertIn("evil.example.com", log)

    def test_local_names_and_ips_pass(self):
        self.start()
        for host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}",
                     f"192.168.1.20:{self.port}"):
            self.assertEqual(self.req("GET", "/status", host=host)[0], 200, host)

    def test_no_host_header_passes(self):
        self.start()
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as s:      # an HTTP/1.0 client
            s.sendall(b"GET /health HTTP/1.0\r\n\r\n")
            raw = b""
            while chunk := s.recv(65536):
                raw += chunk
        self.assertTrue(raw.startswith(b"HTTP/1.0 200"), raw[:40])

    def test_allowed_hosts_and_wildcard(self):
        self.start(allowed_hosts=["strata.example.com", ".home.arpa"])
        self.assertEqual(self.req("GET", "/status", host="strata.example.com")[0], 200)
        self.assertEqual(self.req("GET", "/status", host="nas.home.arpa:8095")[0], 200)
        self.assertEqual(self.req("GET", "/status", host="evil.example.com")[0], 403)
        self.svc.allowed_hosts = ["*"]
        self.assertEqual(self.req("GET", "/status", host="evil.example.com")[0], 200)

    def test_a_trusted_origin_s_host_is_allowed(self):
        self.start(trusted_origins=["https://strata.example.com"])
        self.assertEqual(self.req("GET", "/status", host="strata.example.com")[0], 200)


if __name__ == "__main__":
    unittest.main()
