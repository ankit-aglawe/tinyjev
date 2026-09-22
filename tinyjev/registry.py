"""Known checkpoints. `tinyjev.load(alias)` resolves an alias to its Hub repo."""
MODELS = {
    "nanojev":  {"repo": "AnkitAI/tinyjev-nanojev",  "family": "nanojev", "params": "0.6B",
                 "upstream": "C-Tianyu/NanoJev"},
    "kev-0.6b": {"repo": "AnkitAI/tinyjev-kev-0.6b", "family": "kev", "params": "0.6B",
                 "upstream": "jaredpalmer/kev-0.6b"},
    "kev-4b":   {"repo": "AnkitAI/tinyjev-kev-4b",   "family": "kev", "params": "4B",
                 "upstream": "jaredpalmer/kev-4b@qwen3"},
}


def resolve(name: str) -> str:
    return MODELS[name]["repo"] if name in MODELS else name
