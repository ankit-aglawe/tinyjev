"""Backbone backends. Each exposes `Qwen3Backbone(config, weights_path)` with

    hidden_rows(prefix: list[int], suffixes: list[list[int]], pad_token: int) -> list[np.ndarray]

returning the fp32 hidden states [len(prefix)+len(suffix_i), d] of every row prefix+suffix_i.
Attention is causal, so a shared prefix can be run once and its KV broadcast to every suffix.
"""
from __future__ import annotations

import platform


def available() -> list[str]:
    out = []
    try:
        import mlx.core  # noqa: F401
        out.append("mlx")
    except Exception:
        pass
    try:
        import torch  # noqa: F401
        out.append("torch")
    except Exception:
        pass
    return out


def default_backend() -> str:
    have = available()
    if "mlx" in have and platform.machine() == "arm64" and platform.system() == "Darwin":
        return "mlx"
    if have:
        return have[0]
    raise RuntimeError("install either mlx (Apple Silicon) or torch")


def make(name: str, config: dict, weights_path: str, prefix_min_tokens: int = 96):
    if name == "mlx":
        from .mlx_backend import Qwen3Backbone
    elif name == "torch":
        from .torch_backend import Qwen3Backbone
    else:
        raise ValueError(f"unknown backend {name!r}; choose mlx or torch")
    return Qwen3Backbone(config, weights_path, prefix_min_tokens=prefix_min_tokens)
