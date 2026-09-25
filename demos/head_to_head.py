"""Same weights, two heads. Left lane: Qwen3-0.6B-Base as shipped, options lettered,
next-token logits over the letters (the "Jev in 25 lines" recipe). Right lane: the same
backbone with the tinyjev head. Same never-seen case fed to both, live, every frame.

    pip install 'tinyjev[mlx,demo]' mlx-lm
    python demos/head_to_head.py --gif demo_headtohead.gif
"""
from __future__ import annotations

import argparse
import json
import string
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _render as R  # noqa: E402
import tinyjev  # noqa: E402
from gate_desk import draw_cases, state_text, ROOT, SEED  # noqa: E402

LETTERS = string.ascii_uppercase


class RawReadout:
    """The untrained backbone, read through next-token letter logits."""

    def __init__(self, model_id="Qwen/Qwen3-0.6B-Base"):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.model, self.tok = load(model_id)
        self.letter_ids = [self.tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]

    def predict(self, case):
        mx = self.mx
        opts = list(case["criteria"])
        lines = [f"{LETTERS[i]}. {k}: {case['criteria'][k]}" for i, k in enumerate(opts)]
        prompt = (f"Text:\n{state_text(case['state'])}\n\nQuestion: {case['instructions']}\n\nOptions:\n" +
                  "\n".join(lines) + "\n\nAnswer with the letter of the best option.\nAnswer:")
        ids = mx.array(self.tok.encode(prompt))[None]
        t0 = time.perf_counter()
        z = self.model(ids)[0, -1][mx.array(self.letter_ids[: len(opts)])]
        mx.eval(z)
        ms = (time.perf_counter() - t0) * 1000
        z = np.array(z.astype(mx.float32)); z -= z.max(); p = np.exp(z) / np.exp(z).sum()
        probs = dict(zip(opts, p.tolist()))
        top = max(probs, key=probs.get)
        return {"choice": top, "confidence": probs[top], "probabilities": probs}, ms


def lane(f, x, w, title, sub, ans, ms, case, tally, accent):
    f.text((x, 84), title, R.F_MODE, R.INK)
    f.text((x, 104), sub, R.F_TINY, R.MUTED)
    right = ans["choice"] == case["expected"]
    f.text((x, 132), "PICKS", R.F_TINY, R.MUTED)
    f.text((x, 146), ans["choice"].replace("_", " "), R.F_MODE, accent if right else R.MOSS)
    f.text((x, 170), ("correct" if right else "wrong") + f" · confidence {ans['confidence']:.2f}", R.F_LABEL, R.MUTED)
    f.bar(x, 190, w, 8, ans["confidence"], accent if right else R.BAR_OFF)
    probs = ans["probabilities"]
    top = sorted(probs, key=probs.get, reverse=True)[:4]
    R.options_ledger(f, x, 208, w, top, probs, ans["choice"], title="", label_w=110, row=20)
    f.rule(x, 316, x + w)
    seen, ok, ms_sum = tally
    f.stat(x, 326, "right so far", f"{ok} / {seen}")
    f.stat(x + 150, 326, "mean", f"{ms_sum / seen:.0f} ms")
    f.text((x + w, 348), f"{ms:.0f} ms", R.F_VALUE, R.ACCENT_TEXT, anchor="ra")


def frame_for(case, n, total, raw_ans, raw_ms, tj_ans, tj_ms, raw_tally, tj_tally):
    f = R.Frame("Same weights, two heads",
                f"Qwen3-0.6B, one never-seen case at a time · seed {SEED} · both lanes live")
    f.text((32, 380), f"CASE {n} OF {total}  ·  {case['domain'].replace('_', ' ').upper()}", R.F_TINY, R.MUTED)
    y = 396
    for line in textwrap.wrap(state_text(case["state"]), 118)[:2]:
        f.text((32, y), line, R.F_BODY, R.INK); y += 18
    f.text((32, y + 2), textwrap.shorten(case["instructions"], 110) + f"   ·   expected: {case['expected'].replace('_', ' ')}",
           R.F_LABEL, R.MUTED)
    lane(f, 32, 370, "the 25-line readout", "same backbone · letter logits · no training", raw_ans, raw_ms, case, raw_tally, R.ACCENT_TEXT)
    f.d.line([R.px(420), R.px(84), R.px(420), R.px(364)], fill=R.RULE, width=max(1, R.S // 2))
    lane(f, 438, 370, "the trained head", "same backbone · tinyjev head · one forward pass", tj_ans, tj_ms, case, tj_tally, R.ACCENT_TEXT)
    return f.finish()


def closing(raw_tally, tj_tally):
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    raw, tj = s.get("qwen3-0.6b-base-logit-readout"), s.get("TinyJev-0.6B")
    f = R.Frame("Same weights, two heads", "the full 500, every case logged")
    f.text((32, 96), f"this recording: readout {raw_tally[1]}/{raw_tally[0]} · trained head {tj_tally[1]}/{tj_tally[0]}", R.F_LABEL, R.MUTED)
    y = 140
    if raw and tj:
        f.text((32, y), f"500 never-seen decisions, 25 domains, identical Qwen3-0.6B weights", R.F_MODE, R.INK); y += 40
        f.text((32, y), f"letter readout, no training:   {raw['correct']} / {raw['n']}   ({raw['accuracy']:.1%})", R.F_MODE, R.MOSS); y += 30
        f.text((32, y), f"tinyjev head:                          {tj['correct']} / {tj['n']}   ({tj['accuracy']:.1%})", R.F_MODE, R.ACCENT_TEXT); y += 40
        f.text((32, y), f"share it can handle at ≤2% error:   readout {raw['coverage_at_err']['2%']:.0%} · trained head {tj['coverage_at_err']['2%']:.0%}", R.F_LABEL, R.INK); y += 24
        f.text((32, y), f"{R.MACHINE} · readout {raw['mean_ms']:.0f} ms · head {tj['mean_ms']:.0f} ms", R.F_LABEL, R.MUTED); y += 40
        f.text((32, y), "benchmarks/opendecision in the repo: both runs, all probabilities, coverage curves.", R.F_BODY, R.MUTED)
    return f.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--gif", default="")
    ap.add_argument("--ms", type=int, default=2600)
    ap.add_argument("--colors", type=int, default=32)
    args = ap.parse_args()
    cases = draw_cases(args.n)
    raw = RawReadout()
    tj = tinyjev.load("TinyJev-0.6B")
    frames, rt, tt = [], [0, 0, 0.0], [0, 0, 0.0]
    for n, c in enumerate(cases, 1):
        ra, rms = raw.predict(c)
        q = {"q": {"type": "choice", "instructions": c["instructions"], "criteria": c["criteria"]}}
        t0 = time.perf_counter(); res = tj.predict({"state": c["state"], "questions": q}); tms = (time.perf_counter() - t0) * 1000
        ta = res["states"][0]["answers"]["q"]
        rt[0] += 1; rt[1] += ra["choice"] == c["expected"]; rt[2] += rms
        tt[0] += 1; tt[1] += ta["choice"] == c["expected"]; tt[2] += tms
        frames.append(frame_for(c, n, len(cases), ra, rms, ta, tms, rt, tt))
        print(f"{n:2d} {c['domain'][:18]:18s} readout {ra['choice'][:14]:14s} {ra['confidence']:.2f} {'ok' if ra['choice']==c['expected'] else '--'} | "
              f"head {ta['choice'][:14]:14s} {ta['confidence']:.2f} {'ok' if ta['choice']==c['expected'] else '--'}", flush=True)
    frames.append(closing(rt, tt))
    print(f"\nreadout {rt[1]}/{rt[0]} · head {tt[1]}/{tt[0]}")
    if args.gif:
        out = Path(args.gif); out.parent.mkdir(parents=True, exist_ok=True)
        kb = R.save_gif(frames, args.ms, out, hold_ms=3600, colors=args.colors)
        print(f"{out} · {len(frames)} frames · {kb} KB")


if __name__ == "__main__":
    main()
