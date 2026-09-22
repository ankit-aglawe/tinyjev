"""Convert upstream checkpoints into the tinyjev layout.

    <dest>/weights.safetensors   backbone.* (fp16 by default) + head.* (fp32)
    <dest>/tinyjev.json          family, backbone config, head config, tokenizer ids, upstream provenance
    <dest>/tokenizer/            tokenizer.json (+ config), repaired for current loaders

Both backends load exactly this. Conversion streams tensor by tensor so a 2.4 GB fp32
checkpoint converts on a 16 GB machine.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

import numpy as np

NANOJEV_HEAD_PREFIXES = ("norm.", "scalar.", "set_project.", "set_attention.", "set_output.")


def _save(dest: Path, weights: Dict[str, np.ndarray]):
    from safetensors.numpy import save_file
    save_file(weights, str(dest / "weights.safetensors"))


def _cast(arr: np.ndarray, dtype: str) -> np.ndarray:
    return arr.astype(np.float16 if dtype == "float16" else np.float32)


def normalize_tokenizer_config(cfg: dict):
    """Repair fields newer loaders refuse: NanoJev stores extra_special_tokens as a list."""
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


def _copy_tokenizer(src_dir: Path, dest: Path):
    tok = dest / "tokenizer"
    tok.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        if (src_dir / name).exists():
            shutil.copy2(src_dir / name, tok / name)
    cfg_path = tok / "tokenizer_config.json"
    if cfg_path.exists():
        cfg, changed = normalize_tokenizer_config(json.loads(cfg_path.read_text()))
        if changed:
            cfg_path.write_text(json.dumps(cfg, indent=2))
    return tok


def _eos_pad_from_tokenizer(tok_dir: Path, backbone_cfg: dict):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    eos = backbone_cfg.get("eos_token_id")
    cfg_path = tok_dir / "tokenizer_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        for key in ("eos_token", "pad_token"):
            v = cfg.get(key)
            if isinstance(v, dict):
                cfg[key] = v.get("content")
        if isinstance(cfg.get("eos_token"), str) and t.token_to_id(cfg["eos_token"]) is not None:
            eos = t.token_to_id(cfg["eos_token"])
        pad = t.token_to_id(cfg["pad_token"]) if isinstance(cfg.get("pad_token"), str) else None
    else:
        pad = None
    return int(eos), int(pad if pad is not None else eos)


# ---------------------------------------------------------------- NanoJev
def convert_nanojev(source, dest, dtype: str = "float16") -> Path:
    from safetensors import safe_open
    src, dst = Path(source).expanduser().resolve(strict=True), Path(dest).expanduser()
    for rel in ("config.json", "backbone_config/config.json", "tokenizer/tokenizer.json", "best.safetensors"):
        if not (src / rel).exists():
            raise FileNotFoundError(f"NanoJev checkpoint is missing {rel}")
    dst.mkdir(parents=True, exist_ok=True)
    run = json.loads((src / "config.json").read_text())
    backbone_cfg = json.loads((src / "backbone_config/config.json").read_text())

    weights, n_body, n_head = {}, 0, 0
    with safe_open(str(src / "best.safetensors"), framework="numpy") as f:
        for key in f.keys():
            arr = f.get_tensor(key)
            if key.startswith(NANOJEV_HEAD_PREFIXES):
                weights["head." + key] = _cast(arr, "float32"); n_head += 1
            elif key.startswith("backbone."):
                weights[key] = _cast(arr, dtype); n_body += 1
            else:
                raise ValueError(f"unexpected tensor {key}")
    _save(dst, weights); del weights
    tok = _copy_tokenizer(src / "tokenizer", dst)
    eos, pad = _eos_pad_from_tokenizer(tok, backbone_cfg)
    (dst / "tinyjev.json").write_text(json.dumps({
        "format": "tinyjev-v1", "family": "nanojev", "name": "nanojev",
        "backbone_config": backbone_cfg,
        "head": {"set_head": run.get("set_head", "attention")},
        "tokenizer": {"eos_token_id": eos, "pad_token_id": eos},
        "max_length": run.get("max_length", 8192),
        "dtypes": {"backbone": dtype, "head": "float32"},
        "upstream": {"repo": "C-Tianyu/NanoJev", "base_model": run.get("model"),
                     "base_revision": run.get("resolved_model_revision"),
                     "schema": run.get("schema_version")},
    }, indent=2))
    print(f"nanojev: {n_body} backbone tensors -> {dtype}, {n_head} head tensors -> float32\n-> {dst}")
    return dst


# ---------------------------------------------------------------- Kev
def convert_kev(adapter_dir, base_dir, dest, dtype: str = "float16", name: str = "kev") -> Path:
    """Merge the rank-r LoRA adapter into the Qwen3 base, add the pointer head from head.pt."""
    from safetensors import safe_open
    import torch

    adapter, base, dst = (Path(adapter_dir).expanduser().resolve(strict=True),
                          Path(base_dir).expanduser().resolve(strict=True), Path(dest).expanduser())
    dst.mkdir(parents=True, exist_ok=True)
    acfg = json.loads((adapter / "adapter_config.json").read_text())
    if acfg.get("trainable_token_indices"):
        raise ValueError("adapters with trained token embeddings are not supported by this converter")
    scale = float(acfg["lora_alpha"]) / float(acfg["r"])
    if acfg.get("use_rslora"):
        scale = float(acfg["lora_alpha"]) / float(acfg["r"]) ** 0.5
    backbone_cfg = json.loads((base / "config.json").read_text())

    # LoRA deltas keyed by the base tensor they modify
    deltas: Dict[str, np.ndarray] = {}
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="numpy") as f:
        keys = list(f.keys())
        a_keys = [k for k in keys if k.endswith("lora_A.weight")]
        for ka in a_keys:
            kb = ka.replace("lora_A.weight", "lora_B.weight")
            A, B = f.get_tensor(ka).astype(np.float32), f.get_tensor(kb).astype(np.float32)
            # peft key: base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight
            target = ka.split("base_model.model.")[-1].replace(".lora_A.weight", ".weight")
            if target.startswith("model."):
                target = target[len("model."):]
            deltas[target] = (B @ A) * scale
    merged_count = 0
    weights: Dict[str, np.ndarray] = {}
    base_files = sorted(base.glob("*.safetensors"))
    for bf in base_files:
        with safe_open(str(bf), framework="pt") as f:      # base is bf16; numpy cannot read it
            for key in f.keys():
                if key.startswith("lm_head."):
                    continue                                # never generates text
                t = f.get_tensor(key).to(torch.float32).numpy()
                short = key[len("model."):] if key.startswith("model.") else key
                if short in deltas:
                    t = t + deltas.pop(short); merged_count += 1
                weights["backbone." + short] = _cast(t, dtype)
    if deltas:
        raise ValueError(f"{len(deltas)} adapter tensors matched no base weight, e.g. {list(deltas)[:3]}")

    meta = torch.load(str(adapter / "head.pt"), map_location="cpu")
    head = meta["head"]
    for k in ("q.weight", "q.bias", "k.weight", "k.bias"):
        weights["head." + k] = head[k].to(torch.float32).numpy()
    _save(dst, weights); del weights
    tok = _copy_tokenizer(adapter, dst)
    eos, pad = _eos_pad_from_tokenizer(tok, backbone_cfg)
    (dst / "tinyjev.json").write_text(json.dumps({
        "format": "tinyjev-v1", "family": "kev", "name": name,
        "backbone_config": backbone_cfg,
        "head": {"head_dim": int(meta.get("head_dim", 256)), "temperature": float(meta.get("temperature", 1.0)),
                 "option_isolation": bool(meta.get("option_isolation", False))},
        "tokenizer": {"eos_token_id": eos, "pad_token_id": pad},
        "max_state": 8192, "max_branch": 8192,
        "dtypes": {"backbone": dtype, "head": "float32"},
        "upstream": {"repo": str(adapter_dir), "base_model": meta.get("base"),
                     "base_revision": meta.get("base_revision"), "lora_rank": int(acfg["r"]),
                     "lora_alpha": acfg["lora_alpha"], "merged_tensors": merged_count},
    }, indent=2))
    print(f"kev: merged {merged_count} LoRA deltas into the base, backbone -> {dtype}, head -> float32\n-> {dst}")
    return dst


def convert(family: str, dest, dtype: str = "float16", **kw) -> Path:
    if family == "nanojev":
        return convert_nanojev(kw["source"], dest, dtype)
    if family == "kev":
        return convert_kev(kw["adapter"], kw["base"], dest, dtype, name=kw.get("name", "kev"))
    raise ValueError(f"unknown family {family!r}")
