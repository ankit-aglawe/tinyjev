"""Known checkpoints. Our model has its own Hub repo (a standard transformers layout plus
head.safetensors and tinyjev.json). Baseline conversions of other projects' models live together
in one repo, one subfolder each; `tinyjev.load(alias)` downloads only what it needs."""
MODELS = {
    "tinyjev-0.6b": {"repo": "AnkitAI/tinyjev-0.6b", "subfolder": None, "family": "kev", "params": "0.6B",
                     "what": "ours: Qwen3-0.6B-Base + pointer head, LoRA r16 lr 5e-5 on Kev decision-v7"},
    "nanojev":      {"repo": "AnkitAI/tinyjev-baselines", "subfolder": "nanojev", "family": "nanojev", "params": "0.6B",
                     "what": "baseline: C-Tianyu/NanoJev converted"},
    "kev-0.6b":     {"repo": "AnkitAI/tinyjev-baselines", "subfolder": "kev-0.6b", "family": "kev", "params": "0.6B",
                     "what": "baseline: jaredpalmer/kev-0.6b converted, LoRA merged"},
}


def resolve(name: str):
    """alias -> (repo, subfolder); anything else -> (name, None): a local path or a Hub repo id."""
    if name in MODELS:
        return MODELS[name]["repo"], MODELS[name]["subfolder"]
    return name, None
