"""Kev: one row per question, delimiter tokens, pointer head (<decide> against each </opt>).

Prompt layout, delimiter choice and the head follow `kev.model` / `kev.api` (Apache-2.0,
Jared Palmer). Each question runs as its own causal row continuing from the state, which
kev.model documents as exactly equivalent to its packed block-causal form.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List

import numpy as np

from . import Encoded, load_head

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
MAX_OPTIONS = 255


def render(v, indent: int = 0) -> str:
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list))
                     else f"{pad}{k}: {render(x)}" for k, x in v.items())


def option_text(name: str, desc) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


class PointerFamily:
    name = "pointer"

    def __init__(self, root, manifest: dict):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        ids = [self.tok.token_to_id(t) for t in SPECIAL]
        if any(i is None for i in ids):
            raise ValueError("tokenizer lacks the Qwen delimiter tokens Kev relies on")
        self.state_id, self.q_id, self.o_id, self.c_id, self.d_id = ids
        tk = manifest["tokenizer"]
        self.pad_token_id = int(tk.get("pad_token_id", 0))
        self.max_state = int(manifest.get("max_state", 8192))
        self.max_branch = int(manifest.get("max_branch", 8192))
        self.temperature = float(manifest["head"].get("temperature", 1.0))
        self.dp = int(manifest["head"].get("head_dim", 256))
        self.w = load_head(root)

    def user_tokens(self, text: str) -> List[int]:
        # caller text can never forge a delimiter: <|name|> becomes <¦name¦> before tokenizing
        return self.tok.encode(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).ids

    # ---- prompt ----
    def encode(self, record: dict) -> Encoded:
        state_tokens = self.user_tokens(render(record["state"]))
        if len(state_tokens) + 1 > self.max_state:
            raise ValueError(f"{record['id']}: state exceeds {self.max_state} tokens")
        prefix = [self.state_id] + state_tokens
        rows, questions = [], []
        for q in record["questions"]:
            where = f"{record['id']}:{q['id']}"
            typ = q["type"]
            c = q.get("criteria")
            if typ == "boolean":
                c = c or {}
                if not isinstance(c, dict) or set(c) - {"false", "true"}:
                    raise ValueError(f"{where}: boolean criteria may only hold false/true")
                keys, opts = ["false", "true"], [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
            elif typ == "choice":
                if not isinstance(c, dict) or not 1 <= len(c) <= MAX_OPTIONS:
                    raise ValueError(f"{where}: choice criteria must hold 1..{MAX_OPTIONS} options")
                keys, opts = list(c), [option_text(k, v) for k, v in c.items()]
            elif typ == "score":
                if not isinstance(c, list) or not 2 <= len(c) <= MAX_OPTIONS:
                    raise ValueError(f"{where}: score criteria must be a list of 2..{MAX_OPTIONS} levels")
                keys, opts = [str(i) for i in range(len(c))], [render(x) for x in c]
            else:
                raise ValueError(f"{where}: unsupported question type {typ!r}")
            instr = [self.q_id] + self.user_tokens(render(q["instructions"]))
            spans = [[self.o_id] + self.user_tokens(o) + [self.c_id] for o in opts]
            branch = instr + [t for sp in spans for t in sp] + [self.d_id]
            if len(branch) > self.max_branch - len(prefix):
                raise ValueError(f"{where}: branch too long ({len(branch)} tokens)")
            ends, cursor = [], len(instr)
            for sp in spans:
                cursor += len(sp)
                ends.append(cursor - 1)
            rows.append(branch)
            questions.append({"id": q["id"], "type": typ, "keys": keys, "row": len(rows) - 1,
                              "decide": len(prefix) + len(branch) - 1,
                              "opts": [len(prefix) + e for e in ends],
                              "legend": dict(zip(keys, opts)) if typ == "score" else None})
        return Encoded(prefix=prefix, rows=rows, questions=questions)

    # ---- head ----
    def logits(self, hidden_rows: List[np.ndarray], enc: Encoded) -> Dict[str, np.ndarray]:
        w, out = self.w, {}
        for q in enc.questions:
            h = hidden_rows[q["row"]]
            qv = h[q["decide"]] @ w["q.weight"].T + w["q.bias"]
            kv = h[q["opts"]] @ w["k.weight"].T + w["k.bias"]
            z = (kv @ qv) / math.sqrt(self.dp)
            out[q["id"]] = (z / self.temperature).astype(np.float32)
        return out

    # ---- answers (System One shape, as kev.api.to_answers) ----
    @staticmethod
    def answer(q: dict, probs: List[float]) -> dict:
        r2 = lambda x: round(float(x), 2)
        if q["type"] == "boolean":
            return {"type": "noul", "noul": r2(probs[1]), "value": bool(probs[1] >= 0.5),
                    "p_true": probs[1], "probabilities": dict(zip(q["keys"], probs))}
        if q["type"] == "choice":
            K = len(probs)
            best = max(range(K), key=probs.__getitem__)
            conf = 1.0 if K == 1 else (max(probs) - 1 / K) / (1 - 1 / K)
            return {"type": "choice", "choice": q["keys"][best], "value": q["keys"][best],
                    "confidence": r2(conf), "probabilities": dict(zip(q["keys"], probs))}
        L = len(probs)
        mode = max(range(L), key=probs.__getitem__)
        score = sum(i * p for i, p in enumerate(probs))
        conf = 1.0 - sum(p * abs(i - mode) for i, p in enumerate(probs)) / (L - 1)
        return {"type": "score", "score": r2(score), "value": score, "level": mode, "legend": q["legend"],
                "probabilities": {str(i): v for i, v in enumerate(probs)}, "confidence": r2(conf)}
