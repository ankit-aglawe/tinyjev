"""Known checkpoints: our models. Other projects' models are converted locally with
`tinyjev convert` — we do not republish their weights."""
OLLAMA_4B = "parable/tinyjev"   # the ollama.com name of TinyJev-4B v2
MODELS = {
    "TinyJev-0.6B": {"repo": "AnkitAI/TinyJev-0.6B", "family": "pointer", "params": "0.6B",
                     "what": "Qwen3-0.6B-Base + pointer head, 596M"},
    "TinyJev-4B": {"repo": "AnkitAI/TinyJev-4B", "family": "ollama", "ollama": OLLAMA_4B, "params": "4B",
                   "what": "v2: Qwen3.5-4B, letter readout inside Ollama 0.35.1+; 4.5 GB at Q8_0"},
    "TinyJev-4B-v1": {"repo": "AnkitAI/TinyJev-4B", "revision": "v1", "family": "pointer", "params": "4B",
                      "what": "v1: Qwen3-4B-Base + pointer head, 4.0B; 8 GB fp16, 4.5 GB at quantize=8"},
}
_BY_LOWER = {k.lower(): k for k in MODELS}


def entry(name: str):
    return MODELS.get(_BY_LOWER.get(str(name).lower(), ""))


def resolve(name: str):
    """alias -> (repo, None, revision); anything else -> (name, None, None): a local path or a Hub repo id."""
    e = entry(name)                             # "tinyjev-0.6b" and "TinyJev-0.6B" are the same model
    if e:
        return e["repo"], None, e.get("revision")
    return name, None, None
