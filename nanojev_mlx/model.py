"""MLX port of NanoJev's DecisionModel.

Mirrors upstream `train_toy_decisions.DecisionModel` exactly:
a Qwen3 backbone, EOS pooling, a scalar scorer, and an attention set-head that
lets candidates in a `choice` question see each other before scoring.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import ModelArgs as Qwen3Args, Qwen3Model

SET_DIM = 128
SET_HEADS = 4
NEG_INF = -1e9


def qwen3_args(backbone_config: dict) -> Qwen3Args:
    """Build mlx-lm Qwen3 args from the checkpoint's backbone_config/config.json."""
    rope = backbone_config.get("rope_parameters") or {}
    return Qwen3Args(
        model_type="qwen3",
        hidden_size=backbone_config["hidden_size"],
        num_hidden_layers=backbone_config["num_hidden_layers"],
        intermediate_size=backbone_config["intermediate_size"],
        num_attention_heads=backbone_config["num_attention_heads"],
        rms_norm_eps=backbone_config["rms_norm_eps"],
        vocab_size=backbone_config["vocab_size"],
        num_key_value_heads=backbone_config["num_key_value_heads"],
        head_dim=backbone_config["head_dim"],
        max_position_embeddings=backbone_config.get("max_position_embeddings", 40960),
        rope_theta=rope.get("rope_theta", backbone_config.get("rope_theta", 1000000)),
        tie_word_embeddings=backbone_config.get("tie_word_embeddings", True),
    )


class SetAttention(nn.Module):
    """Self-attention over a question's candidate set.

    Hand-rolled rather than `nn.MultiHeadAttention` so the packed `in_proj_weight`
    from the PyTorch checkpoint maps with no renaming, and so the key-padding mask
    has exactly torch's semantics.
    """

    def __init__(self, dim: int = SET_DIM, heads: int = SET_HEADS):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.in_proj_weight = mx.zeros((3 * dim, dim))
        self.in_proj_bias = mx.zeros((3 * dim,))
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, valid: mx.array) -> mx.array:
        b, k, d = x.shape
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias
        q, key, v = mx.split(qkv, 3, axis=-1)

        def heads(t):
            return t.reshape(b, k, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        q, key, v = heads(q), heads(key), heads(v)
        scores = (q @ key.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        # torch's key_padding_mask blocks padded *keys* for every query.
        keymask = mx.where(valid, 0.0, NEG_INF).reshape(b, 1, 1, k)
        scores = scores + keymask.astype(scores.dtype)
        mixed = mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype) @ v
        mixed = mixed.transpose(0, 2, 1, 3).reshape(b, k, d)
        return self.out_proj(mixed)


class DecisionModel(nn.Module):
    def __init__(self, backbone_config: dict, set_head: str = "attention",
                 shared_prefix: bool = True):
        super().__init__()
        if set_head not in {"none", "attention"}:
            raise ValueError(f"unsupported set_head: {set_head!r}")
        hidden = backbone_config["hidden_size"]
        self.set_head = set_head
        # Every candidate path of a state row starts with the same state tokens. With
        # shared_prefix the backbone runs those once and the K suffixes attend to a
        # broadcast KV cache; the maths is identical, the work is not.
        self.shared_prefix = shared_prefix
        # Below this many prefix tokens the second forward call costs more than it saves.
        self.shared_prefix_min_tokens = 96
        self.backbone = Qwen3Model(qwen3_args(backbone_config))
        self.norm = nn.LayerNorm(hidden)
        self.scalar = nn.Linear(hidden, 1)
        if set_head == "attention":
            self.set_project = nn.Linear(hidden + 1, SET_DIM)
            self.set_attention = SetAttention()
            self.set_output = nn.Linear(SET_DIM, 1)

    def pool(self, paths: Sequence[Sequence[int]], pad_token: int) -> mx.array:
        """Run the backbone over padded candidate paths, return the EOS hidden state.

        Right padding is safe: attention is causal, so the final real token never
        attends to the padding that follows it.
        """
        lengths = [len(p) for p in paths]
        width = max(lengths)
        tokens = mx.array([list(p) + [pad_token] * (width - len(p)) for p in paths])
        hidden = self.backbone(tokens)
        rows = mx.arange(len(paths))
        return hidden[rows, mx.array([n - 1 for n in lengths])]

    def pool_shared(self, prefix: Sequence[int], suffixes: Sequence[Sequence[int]],
                    pad_token: int) -> mx.array:
        """EOS hidden state of prefix+suffix_i for every i, running the prefix once."""
        from mlx_lm.models.cache import KVCache

        cache = [KVCache() for _ in self.backbone.layers]
        self.backbone(mx.array([list(prefix)]), cache=cache)

        k_paths = len(suffixes)
        shared = []
        for layer_cache in cache:
            keys, values = layer_cache.state
            bc = KVCache()
            bc.keys = mx.repeat(keys, k_paths, axis=0)
            bc.values = mx.repeat(values, k_paths, axis=0)
            bc.offset = keys.shape[2]
            shared.append(bc)

        lengths = [len(s) for s in suffixes]
        width = max(lengths)
        tokens = mx.array([list(s) + [pad_token] * (width - len(s)) for s in suffixes])
        hidden = self.backbone(tokens, cache=shared)
        return hidden[mx.arange(k_paths), mx.array([n - 1 for n in lengths])]

    def _leaves(self, examples: List[dict], pad_token: int) -> mx.array:
        if not self.shared_prefix or not all("state_tokens" in ex for ex in examples):
            return self.pool([ids for ex in examples for ids in ex["leaf_tokens"]], pad_token)
        # Group consecutive examples that share a state segment (one state row's
        # questions arrive together) and run each group with one prefix pass.
        chunks, i = [], 0
        while i < len(examples):
            j, state = i, examples[i]["state_tokens"]
            while j < len(examples) and examples[j]["state_tokens"] == state:
                j += 1
            suffixes = [sfx for ex in examples[i:j] for sfx in ex["suffix_tokens"]]
            if len(suffixes) >= 2 and len(state) >= self.shared_prefix_min_tokens:
                chunks.append(self.pool_shared(state, suffixes, pad_token))
            else:
                chunks.append(self.pool([list(state) + list(sfx) for sfx in suffixes], pad_token))
            i = j
        return chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)

    def __call__(self, examples: List[dict], pad_token: int):
        """Return (logits [n_examples, kmax], valid [n_examples, kmax])."""
        leaves = self._leaves(examples, pad_token)

        kmax = max(len(ex["candidate_ids"]) for ex in examples)
        rows, valid_rows, offset = [], [], 0
        for ex in examples:
            n = len(ex["leaf_tokens"])
            k = len(ex["candidate_ids"])
            block = leaves[offset:offset + n]
            if n < kmax:
                block = mx.concatenate(
                    [block, mx.zeros((kmax - n, leaves.shape[-1]), dtype=leaves.dtype)])
            rows.append(block)
            valid_rows.append([i < k for i in range(kmax)])
            offset += n
        h = mx.stack(rows)
        valid = mx.array(valid_rows)

        h = self.norm(h)
        z = self.scalar(h).squeeze(-1).astype(mx.float32)

        picks = [i for i, ex in enumerate(examples) if ex["type"] == "choice"]
        if self.set_head == "attention" and picks:
            idx = mx.array(picks)
            hc, vc = h[idx], valid[idx]
            log_k = mx.log(vc.sum(-1).astype(mx.float32))
            log_k = mx.broadcast_to(log_k.reshape(-1, 1, 1), (len(picks), kmax, 1))
            u = self.set_project(mx.concatenate([hc, log_k.astype(hc.dtype)], axis=-1))
            delta = self.set_output(mx.tanh(u + self.set_attention(u, vc)))
            z = z.at[idx].add(delta.squeeze(-1).astype(mx.float32))

        out = []
        for i, ex in enumerate(examples):
            if ex["type"] == "boolean":
                # One semantic path, one scalar, read as logits [0, z].
                row = mx.concatenate(
                    [mx.zeros((1,)), z[i, :1], mx.zeros((kmax - 2,))]) if kmax > 2 \
                    else mx.concatenate([mx.zeros((1,)), z[i, :1]])
                out.append(row)
            else:
                out.append(z[i])
        logits = mx.stack(out)
        return mx.where(valid, logits, NEG_INF), valid
