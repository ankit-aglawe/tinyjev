"""Qwen3 backbone on MLX with shared-prefix KV reuse."""
from __future__ import annotations

from typing import List, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model


def qwen3_args(cfg: dict) -> ModelArgs:
    rope = cfg.get("rope_parameters") or {}
    return ModelArgs(
        model_type="qwen3", hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"], intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"], rms_norm_eps=cfg["rms_norm_eps"],
        vocab_size=cfg["vocab_size"], num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], max_position_embeddings=cfg.get("max_position_embeddings", 40960),
        rope_theta=rope.get("rope_theta", cfg.get("rope_theta", 1000000)),
        tie_word_embeddings=cfg.get("tie_word_embeddings", True))


class Qwen3Backbone:
    name = "mlx"

    def __init__(self, config: dict, weights_path: str, prefix_min_tokens: int = 96,
                 quantize: int = 0, group_size: int = 64):
        self.model = Qwen3Model(qwen3_args(config))
        weights = mx.load(weights_path)          # model.safetensors: standard Qwen3Model keys
        self.model.load_weights(list(weights.items()))
        if quantize:
            # Backbone Linear layers only (the decision head runs fp32 in numpy). Embeddings stay
            # unquantized: they are gathered, not multiplied, and small models lose most at 4-bit.
            nn.quantize(self.model, group_size=group_size, bits=int(quantize),
                        class_predicate=lambda _p, m: isinstance(m, nn.Linear))
        self.quantize = int(quantize)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.prefix_min_tokens = prefix_min_tokens

    def _rows_plain(self, rows: Sequence[Sequence[int]], pad: int) -> List[np.ndarray]:
        lengths = [len(r) for r in rows]
        width = max(lengths)
        tokens = mx.array([list(r) + [pad] * (width - len(r)) for r in rows])
        h = self.model(tokens).astype(mx.float32)
        mx.eval(h)
        arr = np.array(h)
        return [arr[i, :n] for i, n in enumerate(lengths)]

    def _rows_shared(self, prefix: Sequence[int], suffixes: Sequence[Sequence[int]], pad: int):
        from mlx_lm.models.cache import KVCache
        cache = [KVCache() for _ in self.model.layers]
        h_prefix = self.model(mx.array([list(prefix)]), cache=cache).astype(mx.float32)
        k = len(suffixes)
        shared = []
        for c in cache:
            keys, values = c.state
            bc = KVCache()
            bc.keys, bc.values = mx.repeat(keys, k, axis=0), mx.repeat(values, k, axis=0)
            bc.offset = keys.shape[2]
            shared.append(bc)
        lengths = [len(s) for s in suffixes]
        width = max(lengths)
        tokens = mx.array([list(s) + [pad] * (width - len(s)) for s in suffixes])
        h = self.model(tokens, cache=shared).astype(mx.float32)
        mx.eval(h_prefix, h)
        hp, hs = np.array(h_prefix)[0], np.array(h)
        return [np.concatenate([hp, hs[i, :n]], axis=0) for i, n in enumerate(lengths)]

    def hidden_rows(self, prefix: Sequence[int], suffixes: Sequence[Sequence[int]], pad: int):
        if len(suffixes) >= 2 and len(prefix) >= self.prefix_min_tokens:
            return self._rows_shared(prefix, suffixes, pad)
        return self._rows_plain([list(prefix) + list(s) for s in suffixes], pad)
