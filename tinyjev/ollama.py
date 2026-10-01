"""TinyJev v2 runs inside Ollama (>= 0.35.1, `CAPABILITY decision`). Ollama renders the prompt and reads the
answer letters' probabilities itself, so this agent is a thin client of its local /v1/systemone endpoint and
returns the same result shape as the in-process models."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict

from .agent import SCHEMA_VERSION, normalize_request


def _answer(a: dict) -> dict:
    if a["type"] == "noul":
        p = float(a["noul"])
        return {"type": "noul", "noul": round(p, 2), "value": p >= 0.5, "p_true": p,
                "probabilities": {"false": 1 - p, "true": p}}
    if a["type"] == "choice":
        return {**a, "value": a["choice"]}
    probs = a.get("probabilities") or {}
    return {**a, "value": a["score"], "level": int(max(probs, key=probs.get)) if probs else None}


class OllamaAgent:
    backend = "ollama"
    quantize = 0

    def __init__(self, model: str, host: str | None = None):
        self.model = model
        host = host or os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434"
        self.host = (host if host.startswith("http") else "http://" + host).rstrip("/")
        self.manifest = {"family": "ollama", "name": model}

    @property
    def name(self) -> str:
        return self.model

    def systemone(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Ollama's own System One response for a System One request."""
        data = json.dumps({**body, "model": self.model}).encode()
        req = urllib.request.Request(self.host + "/v1/systemone", data, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama {e.code}: {e.read().decode(errors='replace')[:300]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"no Ollama at {self.host} ({e.reason}). Install Ollama 0.35.1 or newer, then "
                               f"`ollama pull {self.model}`") from e

    def predict(self, payload: Dict[str, Any], temperature: float = 1.0) -> Dict[str, Any]:
        if temperature != 1.0:
            raise ValueError("Ollama decision models score at temperature 1.0")
        records, states, n, t = normalize_request(payload), [], 0, time.perf_counter()
        for rec in records:
            qs = {q["id"]: {**{k: v for k, v in q.items() if k != "id"},
                            "type": "noul" if q["type"] == "boolean" else q["type"]} for q in rec["questions"]}
            res = self.systemone({"state": rec["state"], "questions": qs})
            states.append({"id": rec["id"], "answers": {k: _answer(a) for k, a in res["answers"].items()}})
            n += len(qs)
        return {"schema_version": SCHEMA_VERSION,
                "model": {"name": self.model, "family": "ollama", "backend": "ollama", "host": self.host},
                "temperature": {"value": 1.0},
                "execution": {"states": len(records), "questions": n, "autoregressive_decode_steps": 0,
                              "model_ms": round((time.perf_counter() - t) * 1000, 2)},
                "states": states}
