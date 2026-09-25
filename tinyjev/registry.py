"""Known checkpoints: our models. Other projects' models are converted locally with
`tinyjev convert` — we do not republish their weights."""
MODELS = {
    "TinyJev-0.6B": {"repo": "AnkitAI/TinyJev-0.6B", "family": "pointer", "params": "0.6B",
                     "what": "Qwen3-0.6B-Base + pointer head, 596M"},
    "TinyJev-4B": {"repo": "AnkitAI/TinyJev-4B", "family": "pointer", "params": "4B",
                   "what": "Qwen3-4B-Base + pointer head, 4.0B; 8 GB fp16, 4.5 GB at quantize=8"},
}
_BY_LOWER = {k.lower(): k for k in MODELS}


def resolve(name: str):
    """alias -> (repo, None); anything else -> (name, None): a local path or a Hub repo id."""
    key = _BY_LOWER.get(name.lower())          # "TinyJev-0.6B" and "TinyJev-0.6B" are the same model
    if key:
        return MODELS[key]["repo"], None
    return name, None
