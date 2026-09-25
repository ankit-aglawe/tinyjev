"""von (wfzyx/von, ModernBERT-Large 395M) via `pip install von-sdk`."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_cases, Writer, timed
import von


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    w = Writer("von-1.2", {"runtime": "von-sdk", "model": "von-latest"})
    for c in load_cases()[: a.limit or None]:
        r, ms = timed(lambda: von.decide(c["state"], c["criteria"], c["instructions"]))
        w.row(c, r.probabilities, ms)
    w.close()


if __name__ == "__main__":
    main()
