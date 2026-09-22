"""Model families. A family owns its prompt format and its decision head.

    enc = family.encode(record)                 # -> Encoded(prefix, rows, questions)
    hs  = backbone.hidden_rows(enc.prefix, enc.rows, family.pad_token_id)
    out = family.logits(hs, enc)                # -> {question id: np.ndarray logits}

Heads run in numpy (fp32); they are tiny and this keeps every backend numerically identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Encoded:
    prefix: List[int]                       # tokens shared by every row (the state)
    rows: List[List[int]]                   # per-row suffix tokens, appended to the prefix
    questions: List[Dict[str, Any]] = field(default_factory=list)   # family-specific readout info


def make(name: str, root, manifest: dict):
    if name == "nanojev":
        from .nanojev import NanoJevFamily as F
    elif name == "kev":
        from .kev import KevFamily as F
    else:
        raise ValueError(f"unknown family {name!r}")
    return F(root, manifest)


def load_head(weights_path: str) -> Dict[str, Any]:
    """Only the fp32 `head.*` tensors, as numpy, without touching the backbone."""
    import numpy as np
    from safetensors import safe_open
    out = {}
    with safe_open(weights_path, framework="numpy") as f:
        for key in f.keys():
            if key.startswith("head."):
                out[key[len("head."):]] = np.asarray(f.get_tensor(key), dtype=np.float32)
    return out


def softmax(z):
    import numpy as np
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float64)
