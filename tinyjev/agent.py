"""One loaded model: a family (prompt + head) on a backend (MLX or torch)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import backends, families
from .families import softmax

SCHEMA_VERSION = "tinyjev-v1"
TYPE_ALIASES = {"noul": "boolean", "boolean": "boolean", "choice": "choice", "score": "score"}


def _resolve(model_path, subfolder: Optional[str] = None) -> Path:
    from .registry import resolve
    repo, sub = resolve(str(model_path))
    sub = subfolder or sub
    p = Path(repo).expanduser()
    if p.exists():
        root = p.resolve()
    else:
        from huggingface_hub import snapshot_download
        patterns = [f"{sub}/*"] if sub else ["*"]
        root = Path(snapshot_download(repo, allow_patterns=patterns))
    return root / sub if sub else root


def normalize_request(payload: Dict[str, Any]) -> List[dict]:
    """Accept NanoJev's {"states": [...]} or a System One {"state", "questions"} body.

    Returns records: {"id", "state", "questions": [{"id","type","instructions","criteria"?}]}
    with `noul` folded into `boolean`."""
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    if "states" in payload:
        if set(payload) != {"states"} or not isinstance(payload["states"], list) or not payload["states"]:
            raise ValueError('native request must be exactly {"states": [...]} with at least one state')
        raw = payload["states"]
    elif "state" in payload and "questions" in payload:
        raw = [{"id": payload.get("id") or "request", "state": payload["state"],
                "questions": payload["questions"]}]
    else:
        raise ValueError('request needs either {"states": [...]} or {"state", "questions"}')

    records, seen = [], set()
    for s in raw:
        if not isinstance(s, dict) or not {"id", "state", "questions"} <= set(s):
            raise ValueError("each state needs id, state and questions")
        sid = s["id"]
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError("state id must be a unique non-empty string")
        seen.add(sid)
        qs = s["questions"]
        if not isinstance(qs, dict) or not qs:
            raise ValueError(f"{sid}: questions must be a non-empty object")
        out = []
        for qid, q in qs.items():
            if not isinstance(qid, str) or not qid.strip() or not isinstance(q, dict):
                raise ValueError(f"{sid}: question ids must be non-empty strings mapping to objects")
            typ = TYPE_ALIASES.get(q.get("type"))
            if typ is None:
                raise ValueError(f"{sid}:{qid}: unsupported question type {q.get('type')!r}")
            item = {"id": qid, "type": typ, "instructions": q.get("instructions")}
            if "criteria" in q and q["criteria"] is not None:
                item["criteria"] = q["criteria"]
            out.append(item)
        records.append({"id": sid, "state": s["state"], "questions": out})
    return records


class Agent:
    def __init__(self, model_path, backend: Optional[str] = None, device: Optional[str] = None,
                 subfolder: Optional[str] = None, quantize: int = 0):
        root = _resolve(model_path, subfolder)
        manifest_path = root / "tinyjev.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{root} is not a tinyjev checkpoint (no tinyjev.json); "
                                    f"run `tinyjev convert` first")
        self.manifest = json.loads(manifest_path.read_text())
        self.root = root
        self.family = families.make(self.manifest["family"], root, self.manifest)
        name = backend or backends.default_backend()
        kw = {"device": device} if (device and name == "torch") else {}
        if name == "torch":
            from .backends.torch_backend import Qwen3Backbone
            self.backbone = Qwen3Backbone(self.manifest["backbone_config"], str(root / "weights.safetensors"), **kw)
        else:
            self.backbone = backends.make(name, self.manifest["backbone_config"], str(root / "weights.safetensors"),
                                          quantize=quantize)
        self.backend = self.backbone.name
        self.quantize = int(quantize)

    @property
    def name(self) -> str:
        return self.manifest.get("name", self.manifest["family"])

    def _run(self, records: List[dict], temperature: float):
        results, paths, elapsed = {}, 0, 0.0
        for rec in records:
            enc = self.family.encode(rec)
            t = time.perf_counter()
            hs = self.backbone.hidden_rows(enc.prefix, enc.rows, self.family.pad_token_id)
            logits = self.family.logits(hs, enc)
            elapsed += time.perf_counter() - t
            paths += len(enc.rows)
            for q in enc.questions:
                z = logits[q["id"]]
                probs = softmax(z / temperature).tolist()
                results[f"{rec['id']}:{q['id']}"] = {
                    "state_id": rec["id"], "qid": q["id"], "type": q["type"], "keys": q["keys"],
                    "logits": [float(v) for v in z], "probabilities": probs,
                    "answer": self.family.answer(q, probs),
                    "path_token_counts": [len(enc.prefix) + len(enc.rows[i]) for i in
                                          (q["rows"] if "rows" in q else [q["row"]])]}
        return results, paths, elapsed * 1000.0

    def logits(self, payload: Dict[str, Any]) -> Dict[str, dict]:
        """Raw per-candidate logits/probabilities keyed by `<state id>:<question id>`."""
        results, _, _ = self._run(normalize_request(payload), 1.0)
        return {k: {kk: v[kk] for kk in ("type", "logits", "probabilities", "answer", "path_token_counts")}
                | {"candidate_ids": v["keys"]} for k, v in results.items()}

    def predict(self, payload: Dict[str, Any], temperature: float = 1.0) -> Dict[str, Any]:
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
                or temperature <= 0 or temperature != temperature:
            raise ValueError("temperature must be a finite positive number")
        records = normalize_request(payload)
        results, paths, ms = self._run(records, float(temperature))
        states = {r["id"]: {"id": r["id"], "answers": {}} for r in records}
        for v in results.values():
            states[v["state_id"]]["answers"][v["qid"]] = v["answer"]
        return {
            "schema_version": SCHEMA_VERSION,
            "model": {"name": self.name, "family": self.manifest["family"], "backend": self.backend,
                      "quantize_bits": self.quantize or None,
                      "directory": str(self.root), "upstream": self.manifest.get("upstream", {})},
            "temperature": {"value": float(temperature)},
            "execution": {"states": len(records), "questions": len(results), "candidate_paths": paths,
                          "autoregressive_decode_steps": 0, "model_ms": round(ms, 2)},
            "states": list(states.values()),
        }

    def systemone(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """TypeSafe System One response for a System One request."""
        res = self.predict(body)
        answers = {}
        for qid, a in res["states"][0]["answers"].items():
            if a["type"] in ("boolean", "noul"):
                answers[qid] = {"type": "noul", "noul": round(float(a["p_true"]), 4)}
            elif a["type"] == "choice":
                answers[qid] = {"type": "choice", "choice": a["choice"],
                                "confidence": a.get("confidence"), "probabilities": a["probabilities"]}
            else:
                answers[qid] = {"type": "score", "score": round(float(a["score"]), 4),
                                "legend": a.get("legend"), "probabilities": a["probabilities"],
                                "confidence": a.get("confidence")}
        return {"model": body.get("model") or self.name, "answers": answers,
                "latency_ms": res["execution"]["model_ms"]}


def load(model_path, backend: Optional[str] = None, device: Optional[str] = None,
         subfolder: Optional[str] = None, quantize: int = 0) -> Agent:
    """`load("kev-0.6b")` (alias), `load("/path/to/dir")`, or `load("org/repo", subfolder="name")`.
    quantize=8 or 4 quantizes the backbone's Linear layers at load time (mlx backend)."""
    return Agent(model_path, backend=backend, device=device, subfolder=subfolder, quantize=quantize)
