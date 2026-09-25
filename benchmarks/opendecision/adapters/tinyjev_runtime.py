"""Any tinyjev-v2 checkpoint through tinyjev's own runtime: tinyjev-0.6b, and the
kev-0.6b and nanojev heads converted to the same layout. One backbone family,
three heads, identical inputs.

    python adapters/tinyjev_runtime.py --model tinyjev-0.6b
    python adapters/tinyjev_runtime.py --model ~/.cache/tinyjev/v2/tinyjev-kev-0.6b --name kev-0.6b
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common import load_cases, Writer, timed
import tinyjev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyJev-0.6B")
    ap.add_argument("--name", default="")
    ap.add_argument("--quantize", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    agent = tinyjev.load(a.model, quantize=a.quantize)
    name = a.name or agent.name + (f"-int{a.quantize}" if a.quantize else "")
    T = agent.family.temperature if hasattr(agent.family, "temperature") else None
    w = Writer(name, {"runtime": "tinyjev " + tinyjev.__version__, "backend": agent.backend,
                      "quantize": a.quantize or None, "temperature": T,
                      "checkpoint": str(agent.root)})
    cases = load_cases()[: a.limit or None]
    for c in cases:
        q = {"type": "choice", "instructions": c["instructions"], "criteria": c["criteria"]}
        r, ms = timed(lambda: agent.predict({"state": c["state"], "questions": {"q": q}}))
        ans = r["states"][0]["answers"]["q"]
        w.row(c, ans["probabilities"], ms)
    w.close()


if __name__ == "__main__":
    main()
