"""Known checkpoints. One entry: our model. Other projects' models are converted locally with
`tinyjev convert` — we do not republish their weights."""
MODELS = {
    "tinyjev-0.6b": {"repo": "AnkitAI/tinyjev-0.6b", "family": "kev", "params": "0.6B",
                     "what": "Qwen3-0.6B-Base + pointer head, LoRA r16 lr 5e-5 on Kev decision-v7"},
}


def resolve(name: str):
    """alias -> (repo, None); anything else -> (name, None): a local path or a Hub repo id."""
    if name in MODELS:
        return MODELS[name]["repo"], None
    return name, None
