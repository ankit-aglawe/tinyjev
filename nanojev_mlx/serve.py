"""Local HTTP server for a converted NanoJev checkpoint.

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
TYPE_ALIASES = {"noul": "boolean", "boolean": "boolean", "choice": "choice", "score": "score"}


def _reject_nonfinite(value):
    raise ValueError("non-finite JSON literals are not accepted")


def systemone_to_native(body: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a System One request into NanoJev's native states payload."""
    if not isinstance(body, dict) or "state" not in body or "questions" not in body:
        raise ValueError('System One requests need "state" and "questions"')
    questions = {}
    for qid, q in body["questions"].items():
        if not isinstance(q, dict):
            raise ValueError(f"{qid}: question must be an object")
        typ = TYPE_ALIASES.get(q.get("type"))
        if typ is None:
            raise ValueError(f"{qid}: unsupported question type {q.get('type')!r}")
        out = {"type": typ, "instructions": q.get("instructions")}
        if "criteria" in q and q["criteria"] is not None:
            criteria = q["criteria"]
            if typ == "choice" and isinstance(criteria, dict):
                # System One allows a null description; NanoJev requires text.
                criteria = {k: (v if isinstance(v, str) and v.strip() else k)
                            for k, v in criteria.items()}
            out["criteria"] = criteria
        questions[qid] = out
    return {"states": [{"id": body.get("id") or "request", "state": body["state"],
                        "questions": questions}]}


def native_to_systemone(result: Dict[str, Any]) -> Dict[str, Any]:
    answers = result["states"][0]["answers"] if result["states"] else {}
    converted = {}
    for qid, a in answers.items():
        item = {"type": "noul" if a["type"] == "boolean" else a["type"],
                "probabilities": a["probabilities"]}
        if a["type"] == "boolean":
            item["noul"] = a["p_true"]
        elif a["type"] == "choice":
            item["choice"] = a["choice"]
        else:
            item.update(score=a["score"], legend=a.get("legend"))
        converted[qid] = item
    return {"model": "nanojev-mlx", "answers": converted,
            "latency_ms": result["execution"]["model_ms"]}


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
                self._send(200, {"ready": True, "backend": "mlx",
                                 "body_dtype": agent.config.get("body_dtype")})
            elif self.path == "/v1/models":
                self._send(200, {"data": [{"id": "nanojev-mlx",
                                           "base_model": agent.config.get("base_model"),
                                           "set_head": agent.config.get("set_head")}]})
            else:
                self._send(404, {"error": "unknown endpoint"})

        def do_POST(self):
            try:
                if self.path == "/predict":
                    self._send(200, agent.predict(self._body()))
                elif self.path == "/v1/systemone":
                    body = self._body()
                    self._send(200, native_to_systemone(
                        agent.predict(systemone_to_native(body))))
                else:
                    self._send(404, {"error": "unknown endpoint"})
            except ValueError as exc:
                self._send(422, {"error": str(exc)})
            except Exception as exc:  # never leak a traceback to the socket
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(agent, host: str = "127.0.0.1", port: int = 8077):
    server = HTTPServer((host, port), make_handler(agent))
    print(f"nanojev-mlx listening on http://{host}:{port}  "
          f"(POST /predict, POST /v1/systemone)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
