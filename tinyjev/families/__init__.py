"""Model families: a prompt format plus the decision head that reads it.

    pointer  one branch per question ending in a decide token, scored against each option's end token
    marker   one sequence per candidate, pooled at its end token, with attention across the candidate set


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
    aliases = {"nanojev": "marker", "kev": "pointer"}      # layouts written before the rename
    name = aliases.get(name, name)
    if name == "marker":
        from .marker import MarkerFamily as F
    elif name == "pointer":
        from .pointer import PointerFamily as F
    else:
        raise ValueError(f"unknown family {name!r}; expected 'pointer' or 'marker'")
    return F(root, manifest)


def load_head(root) -> Dict[str, Any]:
    """The fp32 decision head from head.safetensors, as numpy; the backbone is never touched."""
    import numpy as np
    from pathlib import Path
    from safetensors import safe_open
    out = {}
    with safe_open(str(Path(root) / "head.safetensors"), framework="numpy") as f:
        for key in f.keys():
            out[key] = np.asarray(f.get_tensor(key), dtype=np.float32)
    return out


def softmax(z):
    import numpy as np
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float64)
