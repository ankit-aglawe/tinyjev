"""The 25-line Jev: the untrained Qwen3-0.6B-Base backbone, options lettered A..J,
next-token logits over the letters, softmax. No head, no training. This is the
row that answers "is the trained head doing anything?" on identical weights.

    python adapters/raw_qwen_readout.py
"""
import argparse, string, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_cases, Writer, timed
import mlx.core as mx
from mlx_lm import load

LETTERS = string.ascii_uppercase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    model, tok = load(a.model)
    # token id of " A", " B", ... as they follow "Answer:"
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]
    w = Writer("qwen3-0.6b-base-logit-readout", {"runtime": "mlx-lm", "backbone": a.model,
               "recipe": "options lettered A..J, prompt ends 'Answer:', softmax over next-token logits of the letter tokens"})
    for c in load_cases()[: a.limit or None]:
        opts = list(c["criteria"])
        lines = [f"{LETTERS[i]}. {k}: {c['criteria'][k]}" for i, k in enumerate(opts)]
        prompt = (f"Text:\n{c['state']}\n\nQuestion: {c['instructions']}\n\nOptions:\n" +
                  "\n".join(lines) + "\n\nAnswer with the letter of the best option.\nAnswer:")
        ids = mx.array(tok.encode(prompt))[None]

        def fwd():
            logits = model(ids)[0, -1]
            z = logits[mx.array(letter_ids[: len(opts)])]
            mx.eval(z)
            return np.array(z.astype(mx.float32))
        z, ms = timed(fwd)
        z = z - z.max()
        p = np.exp(z) / np.exp(z).sum()
        w.row(c, dict(zip(opts, p.tolist())), ms)
    w.close()


if __name__ == "__main__":
    main()
