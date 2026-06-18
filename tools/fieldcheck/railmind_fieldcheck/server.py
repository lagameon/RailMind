"""Stdlib HTTP server for the standalone Field Input Check. Zero external dependencies
(Python standard library only).

  GET  /        -> single-page UI
  GET  /health  -> {"ok": true}
  POST /run     -> body = JSON field-check config; returns the report JSON
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .page import PAGE
from .report import run_field_check


def _json_default(o):
    import numpy as np
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/run":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            cfg = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, json.dumps(run_field_check(cfg), default=_json_default))
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def log_message(self, fmt, *args):
        pass


def serve(host: str = "127.0.0.1", port: int = 8090) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _Handler)
