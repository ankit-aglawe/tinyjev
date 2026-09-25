"""lostargon/Tiny-Jev (the name collision): a different 0.6B Qwen3 decision model with
its own custom head, loaded through its bundled modeling code."""
import argparse, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_cases, Writer, timed
from transformers import AutoModel, AutoTokenizer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    rid = "lostargon/Tiny-Jev"
    tok = AutoTokenizer.from_pretrained(rid)
    model = AutoModel.from_pretrained(rid, trust_remote_code=True).eval()
    w = Writer("lostargon-tiny-jev", {"runtime": "transformers", "checkpoint": rid, "device": "cpu"})
    for c in load_cases()[: a.limit or None]:
        with torch.no_grad():
            r, ms = timed(lambda: model.choice(tok, c["state"], c["instructions"], c["criteria"]))
        w.row(c, r["probabilities"], ms)
    w.close()


if __name__ == "__main__":
    main()
