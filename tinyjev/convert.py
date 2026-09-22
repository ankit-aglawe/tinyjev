"""Convert upstream checkpoints into the tinyjev layout (v2, transformers-compatible).

    <dest>/config.json           standard Qwen3Model config (AutoModel.from_pretrained loads the backbone)
    <dest>/model.safetensors     backbone with standard Qwen3Model keys (fp16 by default)
    <dest>/head.safetensors      the decision head (fp32), keys without prefix
    <dest>/tinyjev.json          family, head config, tokenizer ids, upstream provenance
    <dest>/tokenizer.json ...    tokenizer files at the root, repaired for current loaders

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
    """Split into model.safetensors (backbone, standard keys) and head.safetensors (head, unprefixed keys)."""
    from safetensors.numpy import save_file
    backbone = {k[len("backbone."):]: v for k, v in weights.items() if k.startswith("backbone.")}
    head = {k[len("head."):]: v for k, v in weights.items() if k.startswith("head.")}
    save_file(backbone, str(dest / "model.safetensors"), metadata={"format": "pt"})
    save_file(head, str(dest / "head.safetensors"), metadata={"format": "pt"})


def _write_backbone_config(dest: Path, backbone_cfg: dict, dtype: str):
    """A config.json transformers 4.x and 5.x both read correctly: rope base at the top level AND under
    rope_parameters (4.x ignores the latter and would silently default to 10000 — the NanoJev bug)."""
    cfg = dict(backbone_cfg)
    rope = cfg.get("rope_parameters") or {}
    theta = rope.get("rope_theta", cfg.get("rope_theta"))
    if theta is not None:
        cfg["rope_theta"] = theta
        cfg["rope_parameters"] = {**rope, "rope_theta": theta, "rope_type": rope.get("rope_type", "default")}
    cfg["architectures"] = ["Qwen3Model"]
    cfg["model_type"] = cfg.get("model_type", "qwen3")
    cfg["dtype"] = dtype
    cfg["torch_dtype"] = dtype
    cfg.pop("transformers_version", None)
    (dest / "config.json").write_text(json.dumps(cfg, indent=2))


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
    tok = dest
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
    _write_backbone_config(dst, backbone_cfg, dtype)
    tok = _copy_tokenizer(src / "tokenizer", dst)
    eos, pad = _eos_pad_from_tokenizer(tok, backbone_cfg)
    (dst / "tinyjev.json").write_text(json.dumps({
        "format": "tinyjev-v2", "family": "nanojev", "name": "nanojev",
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
def convert_kev(adapter_dir, base_dir=None, dest=None, dtype: str = "float16", name: str = "kev") -> Path:
    """Convert a Kev run: either a LoRA run (adapter + head.pt, merged into `base_dir`) or a
    full fine-tune run from `kev.train --lora 0` (the run dir holds the whole backbone via
    save_pretrained, plus head.pt; `base_dir` is then ignored)."""
    from safetensors import safe_open
    try:
        import torch
    except ImportError as exc:
        raise ImportError("converting a Kev checkpoint needs torch: pip install 'tinyjev[convert]'") from exc

    adapter = Path(adapter_dir).expanduser().resolve(strict=True)
    dst = Path(dest).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    full_ft = not (adapter / "adapter_model.safetensors").exists()
    if full_ft:
        base = adapter                      # the run dir IS the backbone
        scale, acfg = 0.0, {"r": 0, "lora_alpha": 0}
    else:
        if base_dir is None:
            raise ValueError("a LoRA run needs base_dir (the Qwen3 base the adapter was trained on)")
        base = Path(base_dir).expanduser().resolve(strict=True)
        acfg = json.loads((adapter / "adapter_config.json").read_text())
        if acfg.get("trainable_token_indices"):
            raise ValueError("adapters with trained token embeddings are not supported by this converter")
        scale = float(acfg["lora_alpha"]) / float(acfg["r"])
        if acfg.get("use_rslora"):
            scale = float(acfg["lora_alpha"]) / float(acfg["r"]) ** 0.5
    backbone_cfg = json.loads((base / "config.json").read_text())

    # LoRA deltas keyed by the base tensor they modify (empty for a full fine-tune)
    deltas: Dict[str, np.ndarray] = {}
    adapter_file = adapter / "adapter_model.safetensors"
    with (safe_open(str(adapter_file), framework="numpy") if not full_ft else _NoTensors()) as f:
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
    base_files = sorted(p for p in base.glob("*.safetensors") if p.name != "adapter_model.safetensors")
    if not base_files:
        raise FileNotFoundError(f"no backbone safetensors found in {base}")
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
    _write_backbone_config(dst, backbone_cfg, dtype)
    tok = _copy_tokenizer(adapter, dst)
    eos, pad = _eos_pad_from_tokenizer(tok, backbone_cfg)
    (dst / "tinyjev.json").write_text(json.dumps({
        "format": "tinyjev-v2", "family": "kev", "name": name,
        "head": {"head_dim": int(meta.get("head_dim", 256)), "temperature": float(meta.get("temperature", 1.0)),
                 "option_isolation": bool(meta.get("option_isolation", False))},
        "tokenizer": {"eos_token_id": eos, "pad_token_id": pad},
        "max_state": 8192, "max_branch": 8192,
        "dtypes": {"backbone": dtype, "head": "float32"},
        "upstream": {"repo": str(adapter_dir), "base_model": meta.get("base"),
                     "base_revision": meta.get("base_revision"), "lora_rank": int(acfg["r"]),
                     "lora_alpha": acfg["lora_alpha"], "merged_tensors": merged_count,
                     "full_finetune": full_ft},
    }, indent=2))
    print(f"kev: merged {merged_count} LoRA deltas into the base, backbone -> {dtype}, head -> float32\n-> {dst}")
    return dst


class _NoTensors:
    """Stand-in for safe_open when a run has no adapter file."""
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def keys(self):
        return []


def convert(family: str, dest, dtype: str = "float16", **kw) -> Path:
    if family == "nanojev":
        return convert_nanojev(kw["source"], dest, dtype)
    if family == "kev":
        return convert_kev(kw["adapter"], kw.get("base"), dest, dtype, name=kw.get("name", "kev"))
    raise ValueError(f"unknown family {family!r}")
