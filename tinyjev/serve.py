"""Local HTTP server for a tinyjev checkpoint.

Two routes:
  POST /predict        NanoJev's native schema, {"states": [...]}
  POST /v1/systemone   TypeSafe System One shape, so existing clients work unchanged

Binds loopback and has no authentication. Do not expose it.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

MAX_BODY = 2_000_000


def _reject_nonfinite(value):
    raise ValueError("non-finite JSON literals are not accepted")


def make_handler(agent):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):  # quieter default logging
            pass

        def _send(self, code: int, data: dict):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError(f"request body must be 1..{MAX_BODY} bytes")
            origin = self.headers.get("Origin")
            if origin:
                raise ValueError("cross-origin requests are disabled")
            return json.loads(self.rfile.read(length), parse_constant=_reject_nonfinite)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ready": True, "model": agent.name, "family": agent.manifest["family"],
                                 "backend": agent.backend})
            elif self.path == "/v1/models":
                self._send(200, {"data": [{"id": agent.name, "family": agent.manifest["family"],
                                           "backend": agent.backend,
                                           "upstream": agent.manifest.get("upstream", {})}]})
            else:
                self._send(404, {"error": "unknown endpoint"})

        def do_POST(self):
            try:
                if self.path == "/predict":
                    self._send(200, agent.predict(self._body()))
                elif self.path == "/v1/systemone":
                    self._send(200, agent.systemone(self._body()))
                else:
                    self._send(404, {"error": "unknown endpoint"})
            except ValueError as exc:
                self._send(422, {"error": str(exc)})
            except Exception as exc:  # never leak a traceback to the socket
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(agent, host: str = "127.0.0.1", port: int = 8077):
    server = HTTPServer((host, port), make_handler(agent))
    print(f"tinyjev [{agent.name} on {agent.backend}] listening on http://{host}:{port}  "
          f"(POST /predict, POST /v1/systemone)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
