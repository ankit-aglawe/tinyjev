"""OpenDecision's own engine (MoritzLaurer/ModernBERT-large-zeroshot-v2.0 as a zero-shot NLI
scorer) via `pip install opendecision`, on its own suite. The suite owner's row."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_cases, Writer, timed
from opendecision.engine import OpenDecisionEngine


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    eng = OpenDecisionEngine()
    w = Writer("opendecision-engine", {"runtime": "opendecision 0.1.1", "model": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0", "profile": "default"})
    for c in load_cases()[: a.limit or None]:
        r, ms = timed(lambda: eng.choice(state=c["state"], instructions=c["instructions"], criteria=c["criteria"]))
        probs = r.get("probabilities") or r.get("probs") or r.get("scores")
        w.row(c, probs, ms)
    w.close()


if __name__ == "__main__":
    main()
