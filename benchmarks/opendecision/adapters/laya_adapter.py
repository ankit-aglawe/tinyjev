"""Laya (convaiinnovations/laya, English checkpoint, ModernBERT-large 421M) via `pip install laya`.
Run inside a venv with laya installed:  <venv>/bin/python adapters/laya_adapter.py
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_cases, Writer, timed
import laya


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--subfolder", default=""); a = ap.parse_args()
    agent = laya.load("convaiinnovations/laya", subfolder=a.subfolder or None)
    w = Writer("laya-" + (a.subfolder or "english"), {"runtime": f"laya {getattr(laya, '__version__', '?')}", "checkpoint": "convaiinnovations/laya" + (f"/{a.subfolder}" if a.subfolder else ""),
                                "note": "English ModernBERT-large checkpoint, default device"})
    for c in load_cases()[: a.limit or None]:
        q = {"q": {"type": "choice", "instructions": c["instructions"], "criteria": c["criteria"]}}
        r, ms = timed(lambda: agent.predict(c["state"], q))
        ans = r["answers"]["q"] if "answers" in r else r["q"]
        probs = ans.get("probabilities") or ans.get("probs")
        w.row(c, probs, ms)
    w.close()


if __name__ == "__main__":
    main()
