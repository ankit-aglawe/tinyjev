"""Qwen3 backbone on PyTorch (CPU, CUDA or MPS) with shared-prefix KV reuse."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


class Qwen3Backbone:
    name = "torch"

    def __init__(self, config: dict, weights_path: str, prefix_min_tokens: int = 96, device=None):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else
                                              "mps" if torch.backends.mps.is_available() else "cpu"))
        cfg = AutoConfig.for_model(config.get("model_type", "qwen3"), **{
            k: v for k, v in config.items()
            if k not in ("model_type", "architectures", "transformers_version")})
        # transformers 4.x ignores `rope_parameters`; force the declared base explicitly
        rope = config.get("rope_parameters") or {}
        cfg.rope_theta = rope.get("rope_theta", config.get("rope_theta", cfg.rope_theta))
        cfg.use_cache = True
        self.model = AutoModel.from_config(cfg, attn_implementation="sdpa")
        weights = load_file(weights_path)
        state = {k[len("backbone."):]: v for k, v in weights.items() if k.startswith("backbone.")}
        self.model.load_state_dict(state, strict=True)
        dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        self.model.to(self.device, dtype=dtype).eval()
        self.prefix_min_tokens = prefix_min_tokens

    def _pad(self, rows, pad):
        torch = self.torch
        lengths = [len(r) for r in rows]
        width = max(lengths)
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        att = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
            att[i, :len(r)] = 1
        return ids.to(self.device), att.to(self.device), lengths

    def _rows_plain(self, rows, pad) -> List[np.ndarray]:
        torch = self.torch
        ids, att, lengths = self._pad(rows, pad)
        with torch.inference_mode():
            h = self.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state.float().cpu().numpy()
        return [h[i, :n] for i, n in enumerate(lengths)]

    def _rows_shared(self, prefix, suffixes, pad):
        torch = self.torch
        from transformers import DynamicCache
        k = len(suffixes)
        with torch.inference_mode():
            p = torch.tensor([list(prefix)], device=self.device)
            out = self.model(input_ids=p, past_key_values=DynamicCache(), use_cache=True)
            hp = out.last_hidden_state[0].float().cpu().numpy()
            cache = out.past_key_values
            cache.reorder_cache(torch.zeros(k, dtype=torch.long, device=self.device))
            ids, att, lengths = self._pad(suffixes, pad)
            full_att = torch.cat([torch.ones((k, len(prefix)), dtype=torch.long, device=self.device), att], 1)
            pos = torch.arange(len(prefix), len(prefix) + ids.shape[1], device=self.device)[None].expand(k, -1)
            h = self.model(input_ids=ids, attention_mask=full_att, position_ids=pos,
                           past_key_values=cache, use_cache=True).last_hidden_state.float().cpu().numpy()
        return [np.concatenate([hp, h[i, :n]], axis=0) for i, n in enumerate(lengths)]

    def hidden_rows(self, prefix, suffixes, pad):
        if len(suffixes) >= 2 and len(prefix) >= self.prefix_min_tokens:
            return self._rows_shared(prefix, suffixes, pad)
        return self._rows_plain([list(prefix) + list(s) for s in suffixes], pad)
