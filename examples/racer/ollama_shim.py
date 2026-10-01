"""Run Jev Grand Prix on a local Ollama decision model (Ollama >= 0.35 /v1/systemone) instead of TypeSafe's API.

    OLLAMA_MODEL=tinyjev OLLAMA_URL=http://127.0.0.1:11434 python3 server.py

Same request shape as typesafe_sdk (state + typed questions), so server.py keeps its questions unchanged.
"""
import json, os, urllib.request
from types import SimpleNamespace as NS

URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/") + "/v1/systemone"
MODEL = os.environ["OLLAMA_MODEL"]


class TypeSafeError(Exception):
    pass


def Choice(instructions, criteria):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def Noul(instructions, criteria=None):
    return {"type": "noul", "instructions": instructions, **({"criteria": criteria} if criteria else {})}


class TypeSafeClient:
    def __init__(self, timeout=10.0):
        self.timeout, self.models = timeout, NS(list=lambda: self.system_one("warm-up", {"ok": Noul("Ready?")}))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def system_one(self, state, questions):
        req = urllib.request.Request(URL, json.dumps({"model": MODEL, "state": state, "questions": questions}).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.load(r)
        except Exception as e:
            raise TypeSafeError(str(e)) from e
        choices, nouls = {}, {}
        for name, a in d["answers"].items():
            if a["type"] == "noul":
                nouls[name] = NS(noul=a.get("noul", a.get("probability")))
            else:
                choices[name] = NS(choice=a.get("choice"), probabilities=a.get("probabilities"), confidence=a.get("confidence", 0.0))
        return NS(choices=choices, nouls=nouls, model=d.get("model", MODEL),
                  usage=NS(input_tokens=(d.get("usage") or {}).get("input_tokens", 0)))
