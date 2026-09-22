"""tinyjev: run tiny Jev-style decision models locally, on any machine.

    import tinyjev
    agent = tinyjev.load("AnkitAI/tinyjev-kev-0.6b")          # MLX on Apple Silicon, torch elsewhere
    agent.predict({"state": "...", "questions": {...}})       # System One request shape
    agent.predict({"states": [...]})                          # NanoJev's native shape
"""
from .agent import Agent, load, normalize_request
from .convert import convert
from .registry import MODELS

__all__ = ["Agent", "load", "convert", "normalize_request", "MODELS", "__version__"]
__version__ = "0.1.0"
