"""NanoJev: one row per candidate, EOS pooling, scalar scorer + set-attention over choice options.

Prompt bytes and the head follow upstream `predict_toy_decisions` / `train_toy_decisions`
(MIT, OpenJev contributors). Parity is checked against upstream's own CUDA predictions.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List

import numpy as np

from . import Encoded, load_head, softmax

QUESTION_TYPES = {"boolean", "choice", "score"}
SET_DIM, SET_HEADS = 128, 4


def _nonempty(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


def validate_question(where: str, q: dict):
    if not isinstance(q, dict) or set(q) - {"id", "type", "instructions", "criteria"}:
        raise ValueError(f"{where}: question may only hold type, instructions, criteria")
    typ = q.get("type")
    if typ not in QUESTION_TYPES or not _nonempty(q.get("instructions")):
        raise ValueError(f"{where}: invalid type or instructions")
    c = q.get("criteria")
    if typ == "boolean":
        if "criteria" in q and (not isinstance(c, dict) or set(c) - {"false", "true"}
                                or not all(_nonempty(v) for v in c.values())):
            raise ValueError(f"{where}: boolean criteria may only hold non-empty false/true")
    elif typ == "choice":
        if not isinstance(c, dict) or not 2 <= len(c) <= 255 or not all(
                _nonempty(k) and _nonempty(v) for k, v in c.items()):
            raise ValueError(f"{where}: choice criteria must be 2-255 non-empty id: description pairs")
    else:
        if not isinstance(c, list) or not 2 <= len(c) <= 10 or not all(_nonempty(v) for v in c):
            raise ValueError(f"{where}: score criteria must be an ordered list of 2-10 non-empty levels")


def candidates(q: dict):
    typ = q["type"]
    if typ == "boolean":
        return ["false", "true"], ["The proposition is true."]
    if typ == "choice":
        ids = list(q["criteria"])
        return ids, [f"{k}: {q['criteria'][k]}" for k in ids]
    return [str(i) for i in range(len(q["criteria"]))], list(q["criteria"])


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


class NanoJevFamily:
    name = "nanojev"

    def __init__(self, root, manifest: dict):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        tk = manifest["tokenizer"]
        self.eos_token_id = int(tk["eos_token_id"])
        self.pad_token_id = int(tk.get("pad_token_id", tk["eos_token_id"]))
        self.max_length = int(manifest.get("max_length", 8192))
        self.set_head = manifest["head"].get("set_head", "attention")
        self.w = load_head(root)

    def _enc(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    # ---- prompt ----
    def encode(self, record: dict) -> Encoded:
        state = record["state"]
        if not isinstance(state, (str, dict, list)) or not state:
            raise ValueError(f"{record['id']}: state must be a non-empty string, object or array")
        prefix = self._enc(f"State:\n{state}\n")          # str() of dict/list, as upstream trained
        rows, questions = [], []
        for q in record["questions"]:
            where = f"{record['id']}:{q['id']}"
            validate_question(where, q)
            typ = q["type"]
            ids, texts = candidates(q)
            head = f"Question type: {typ}\nQuestion:\n{q['instructions']}\n"
            if typ == "boolean" and "criteria" in q:
                for key, label in (("false", "False"), ("true", "True")):
                    if key in q["criteria"]:
                        head += f"{label} criterion: {q['criteria'][key]}\n"
            head_ids = self._enc(head)
            start = len(rows)
            for t in texts:
                rows.append(head_ids + self._enc(f"Candidate:\n{t}\nDecision:") + [self.eos_token_id])
            longest = len(prefix) + max(len(r) for r in rows[start:])
            if longest > self.max_length:
                raise ValueError(f"{where}: candidate path is {longest} tokens, over max_length={self.max_length}")
            questions.append({"id": q["id"], "type": typ, "keys": ids,
                              "rows": list(range(start, len(rows)))})
        return Encoded(prefix=prefix, rows=rows, questions=questions)

    # ---- head ----
    def _set_attention(self, u: np.ndarray) -> np.ndarray:
        w = self.w
        k_, d = u.shape
        qkv = u @ w["set_attention.in_proj_weight"].T + w["set_attention.in_proj_bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        hd = d // SET_HEADS
        heads = lambda t: t.reshape(k_, SET_HEADS, hd).transpose(1, 0, 2)   # [H,K,hd]
        q, k, v = heads(q), heads(k), heads(v)
        s = (q @ k.transpose(0, 2, 1)) / math.sqrt(hd)
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s); a /= a.sum(-1, keepdims=True)
        mixed = (a @ v).transpose(1, 0, 2).reshape(k_, d)
        return mixed @ w["set_attention.out_proj.weight"].T + w["set_attention.out_proj.bias"]

    def logits(self, hidden_rows: List[np.ndarray], enc: Encoded) -> Dict[str, np.ndarray]:
        w, out = self.w, {}
        for q in enc.questions:
            leaves = np.stack([hidden_rows[i][-1] for i in q["rows"]]).astype(np.float32)
            h = layer_norm(leaves, w["norm.weight"], w["norm.bias"])
            z = (h @ w["scalar.weight"].T + w["scalar.bias"]).reshape(-1)
            if q["type"] == "choice" and self.set_head == "attention":
                log_k = np.full((h.shape[0], 1), math.log(h.shape[0]), dtype=np.float32)
                u = np.concatenate([h, log_k], -1) @ w["set_project.weight"].T + w["set_project.bias"]
                delta = (np.tanh(u + self._set_attention(u)) @ w["set_output.weight"].T
                         + w["set_output.bias"]).reshape(-1)
                z = z + delta
            if q["type"] == "boolean":
                z = np.array([0.0, float(z[0])], dtype=np.float32)
            out[q["id"]] = z.astype(np.float32)
        return out

    # ---- answers ----
    @staticmethod
    def answer(q: dict, probs: List[float]) -> dict:
        ids = q["keys"]
        if len(probs) != len(ids) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probs) \
                or abs(math.fsum(probs) - 1.0) > 1e-5:
            raise ValueError("model produced invalid probabilities")
        best = max(range(len(ids)), key=probs.__getitem__)
        a = {"type": q["type"], "probabilities": dict(zip(ids, probs))}
        if q["type"] == "boolean":
            a.update(p_true=probs[1], value=bool(best))
        elif q["type"] == "choice":
            a.update(choice=ids[best], value=ids[best])
        else:
            s = math.fsum(i * p for i, p in enumerate(probs))
            a.update(score=s, level=best, value=s)
        return a
