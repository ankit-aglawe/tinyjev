"""Known checkpoints. All live in one Hub repo, one subfolder per model, so
`tinyjev.load("kev-0.6b")` downloads only that folder."""
HUB_REPO = "AnkitAI/tinyjev"

MODELS = {
    "tinyjev-0.6b": {"subfolder": "tinyjev-0.6b", "family": "kev", "params": "0.6B",
                     "upstream": "ours: Qwen3-0.6B-Base + pointer head, LoRA r16 lr 5e-5 on Kev decision-v7"},
    "nanojev":  {"subfolder": "nanojev",  "family": "nanojev", "params": "0.6B", "upstream": "C-Tianyu/NanoJev"},
    "kev-0.6b": {"subfolder": "kev-0.6b", "family": "kev",     "params": "0.6B", "upstream": "jaredpalmer/kev-0.6b"},
    "kev-4b":   {"subfolder": "kev-4b",   "family": "kev",     "params": "4B",   "upstream": "jaredpalmer/kev-4b@qwen3"},
}


def resolve(name: str):
    """alias -> (repo, subfolder); anything else -> (name, None) and the caller treats it as a path or repo."""
    if name in MODELS:
        return HUB_REPO, MODELS[name]["subfolder"]
    return name, None
