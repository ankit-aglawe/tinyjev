"""TinyJev-4B v2 routes to a local Ollama; the old pointer model stays pinned to the Hub tag v1."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import tinyjev
from tinyjev.registry import resolve

REPLY = {"model": "x", "answers": {
    "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "returns": 0.1}, "confidence": 0.8},
    "refund": {"type": "noul", "noul": 0.97},
    "anger": {"type": "score", "score": 1.4, "legend": {"0": "calm", "1": "cross", "2": "furious"},
              "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}, "confidence": 0.3}}}


def test_v2_goes_through_ollama_and_v1_is_pinned():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"], seen["body"] = self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps(REPLY).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        agent = tinyjev.load("TinyJev-4B")
        agent.host = f"http://127.0.0.1:{srv.server_port}"
        out = agent.predict({"state": "charged twice", "questions": {
            "team": {"type": "choice", "instructions": "Team?", "criteria": {"billing": None, "returns": None}},
            "refund": {"type": "noul", "instructions": "Refund?"},
            "anger": {"type": "score", "instructions": "Anger?", "criteria": ["calm", "cross", "furious"]}}})
    finally:
        srv.shutdown()
    assert seen["path"] == "/v1/systemone" and seen["body"]["model"] == agent.model
    assert seen["body"]["questions"]["refund"]["type"] == "noul"           # boolean folded back for Ollama
    a = out["states"][0]["answers"]
    assert a["team"]["choice"] == "billing" and a["refund"]["p_true"] == 0.97 and a["refund"]["value"] is True
    assert a["anger"]["level"] == 2 and out["model"]["backend"] == "ollama"
    assert resolve("TinyJev-4B-v1") == ("AnkitAI/TinyJev-4B", None, "v1")
    assert resolve("some/repo") == ("some/repo", None, None)
