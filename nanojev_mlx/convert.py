"""Convert an upstream NanoJev checkpoint to MLX.

Streams tensors one at a time so a 2.4 GB fp32 checkpoint converts on a 16 GB machine.
The backbone is cast to fp16; the 12 head tensors stay fp32 because they are tiny and
sit downstream of every numerical decision.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx

HEAD_PREFIXES = ("norm.", "scalar.", "set_project.", "set_attention.", "set_output.")
REQUIRED = ("config.json", "backbone_config/config.json", "tokenizer/tokenizer.json",
            "best.safetensors")


def normalize_tokenizer_config(cfg: dict) -> tuple[dict, bool]:
    """Repair tokenizer fields that newer transformers/tokenizers refuse to load.

    NanoJev stores `extra_special_tokens` as a list; loaders expect a mapping and fail
    with "'list' object has no attribute 'keys'".
    """
    changed = False
    extra = cfg.get("extra_special_tokens")
    if isinstance(extra, list):
        cfg["extra_special_tokens"] = {f"extra_{i}": t for i, t in enumerate(extra)}
        changed = True
    if cfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg.pop("backend", None)
        cfg.pop("is_local", None)
        changed = True
    return cfg, changed


def convert(source: str | Path, dest: str | Path, dtype: str = "float16") -> Path:
    from safetensors import safe_open

    src, dst = Path(source).expanduser().resolve(strict=True), Path(dest).expanduser()
    for rel in REQUIRED:
        if not (src / rel).exists():
            raise FileNotFoundError(f"checkpoint is missing {rel}: {src / rel}")
    dst.mkdir(parents=True, exist_ok=True)

    body = mx.float16 if dtype == "float16" else mx.float32
    weights, n_body, n_head = {}, 0, 0
    with safe_open(str(src / "best.safetensors"), framework="numpy") as f:
        for key in f.keys():
            arr = mx.array(f.get_tensor(key))
            if key.startswith(HEAD_PREFIXES):
                weights[key] = arr.astype(mx.float32)
                n_head += 1
            else:
                weights[key] = arr.astype(body)
                n_body += 1
    mx.save_safetensors(str(dst / "weights.safetensors"), weights)
    del weights

    (dst / "backbone_config.json").write_text((src / "backbone_config/config.json").read_text())
    run_config = json.loads((src / "config.json").read_text())
    (dst / "nanojev_mlx.json").write_text(json.dumps({
        "format": "nanojev-mlx-v1",
        "set_head": run_config.get("set_head", "attention"),
        "max_length": run_config.get("max_length", 8192),
        "base_model": run_config.get("model"),
        "base_revision": run_config.get("resolved_model_revision"),
        "body_dtype": dtype,
        "head_dtype": "float32",
    }, indent=2))

    tok_dst = dst / "tokenizer"
    tok_dst.mkdir(exist_ok=True)
    for item in (src / "tokenizer").iterdir():
        shutil.copy2(item, tok_dst / item.name)
    cfg_path = tok_dst / "tokenizer_config.json"
    if cfg_path.exists():
        cfg, changed = normalize_tokenizer_config(json.loads(cfg_path.read_text()))
        if changed:
            cfg_path.write_text(json.dumps(cfg, indent=2))

    print(f"converted {n_body} backbone tensors to {dtype}, {n_head} head tensors to float32")
    print(f"-> {dst}")
    return dst
