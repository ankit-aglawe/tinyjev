"""NanoJev on Apple Silicon.

Upstream NanoJev refuses to run without CUDA. This package runs the same checkpoint
locally through MLX.

    import nanojev_mlx
    agent = nanojev_mlx.load("ankit/nanojev-mlx")
    agent.predict({"states": [...]})
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx

from .convert import convert
from .decide import answer_from_probabilities, prepare_examples, validate_request
from .model import DecisionModel

__all__ = ["Agent", "load", "convert", "__version__"]
__version__ = "0.1.0"

SCHEMA_VERSION = "nanojev-mlx-v1"


def _resolve(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser()
    if path.exists():
        return path.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(model_path)))


class Agent:
    """A loaded NanoJev checkpoint. Construct once; `predict` is cheap after that."""

    def __init__(self, model_path: str | Path, max_length: Optional[int] = None):
        from tokenizers import Tokenizer

        root = _resolve(model_path)
        cfg_path = root / "nanojev_mlx.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{root} is not a converted checkpoint (no nanojev_mlx.json). "
                f"Run `nanojev-mlx convert <upstream-checkpoint> {root}` first.")
        self.config = json.loads(cfg_path.read_text())
        backbone_config = json.loads((root / "backbone_config.json").read_text())

        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer" / "tokenizer.json"))
        self.eos_token_id = self._eos(root, backbone_config)
        self.pad_token_id = self.eos_token_id
        self.max_length = max_length or self.config.get("max_length", 8192)

        self.model = DecisionModel(backbone_config, self.config.get("set_head", "attention"))
        self.model.load_weights(str(root / "weights.safetensors"))
        self.model.eval()
        mx.eval(self.model.parameters())
        self.root = root

    def _eos(self, root: Path, backbone_config: dict) -> int:
        tok_cfg = root / "tokenizer" / "tokenizer_config.json"
        if tok_cfg.exists():
            token = json.loads(tok_cfg.read_text()).get("eos_token")
            if isinstance(token, dict):
                token = token.get("content")
            if isinstance(token, str):
                found = self.tokenizer.token_to_id(token)
                if found is not None:
                    return int(found)
        eos = backbone_config.get("eos_token_id")
        if not isinstance(eos, int):
            raise ValueError("checkpoint does not declare an eos token")
        return eos

    def _encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def logits(self, payload: Dict[str, Any]) -> Dict[str, dict]:
        """Raw per-candidate logits and probabilities, keyed by `<state id>:<question id>`."""
        examples = prepare_examples(payload, self._encode, self.eos_token_id, self.max_length)
        raw, _ = self.model(examples, self.pad_token_id)
        mx.eval(raw)
        out = {}
        for example, values in zip(examples, raw):
            k = len(example["candidate_ids"])
            scores = values[:k].astype(mx.float32)
            probs = mx.softmax(scores, axis=-1).tolist()
            out[example["id"]] = {
                "type": example["type"],
                "candidate_ids": example["candidate_ids"],
                "logits": scores.tolist(),
                "probabilities": probs,
                "answer": answer_from_probabilities(example, probs),
                "path_token_counts": [len(t) for t in example["leaf_tokens"]],
            }
        return out

    def predict(self, payload: Dict[str, Any], temperature: float = 1.0) -> Dict[str, Any]:
        """Answer every question in the request. One forward pass over all candidates."""
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
                or temperature <= 0 or temperature != temperature:
            raise ValueError("temperature must be a finite positive number")
        states = validate_request(payload)
        examples = prepare_examples(payload, self._encode, self.eos_token_id, self.max_length)

        started = time.perf_counter()
        raw, _ = self.model(examples, self.pad_token_id)
        mx.eval(raw)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        outputs = {s["id"]: {"id": s["id"], "answers": {}} for s in states}
        for example, values in zip(examples, raw):
            k = len(example["candidate_ids"])
            scores = (values[:k].astype(mx.float32) / temperature)
            probs = mx.softmax(scores, axis=-1).tolist()
            outputs[example["state_id"]]["answers"][example["qid"]] = \
                answer_from_probabilities(example, probs)

        return {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": {"directory": str(self.root), "base_model": self.config.get("base_model"),
                           "base_revision": self.config.get("base_revision"),
                           "set_head": self.config.get("set_head")},
            "temperature": {"value": float(temperature), "fitted_by_this_command": False},
            "execution": {"backend": "mlx", "body_dtype": self.config.get("body_dtype"),
                          "head_dtype": self.config.get("head_dtype"),
                          "states": len(states), "questions": len(examples),
                          "candidate_paths": sum(len(e["leaf_tokens"]) for e in examples),
                          "forward_passes": 1, "autoregressive_decode_steps": 0,
                          "model_ms": round(elapsed_ms, 2)},
            "states": list(outputs.values()),
        }


def load(model_path: str | Path, max_length: Optional[int] = None) -> Agent:
    """Load a converted NanoJev checkpoint from a local directory or the Hugging Face Hub."""
    return Agent(model_path, max_length=max_length)
