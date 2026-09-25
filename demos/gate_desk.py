"""The gate desk: never-seen decisions stream past, and the confidence gate sorts them
into "handled" and "ask a person".

Cases are a seeded random draw from the OpenDecision Original Choice 500 holdout split,
a suite from 25 domains that was not in the training data. The seed is printed on every
frame, the draw is uniform over the holdout (not picked by domain), and a wrong answer
stays on screen: the point of the demo is that the gate, not the model, decides what
runs unattended.

    pip install 'tinyjev[mlx,demo]'
    python demos/gate_desk.py --gif demo_gate.gif
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _render as R  # noqa: E402
import tinyjev  # noqa: E402

GATE = 0.85
SEED = 20260918          # the suite's own split seed; nothing about this draw is hand-picked
ROOT = Path(__file__).resolve().parent.parent / "benchmarks" / "opendecision"


def state_text(state) -> str:
    """Flatten a dict/list state to one line for the frame; the model gets the raw state."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "  ".join(f"{k}: {v}" for k, v in state.items())
    return "  ".join(str(v) for v in state)


def draw_cases(n: int, split: str = "holdout"):
    rows = [json.loads(l) for l in (ROOT / "suite" / f"{split}.jsonl").read_text().splitlines() if l.strip()]
    idx = np.random.default_rng(SEED).choice(len(rows), size=n, replace=False)
    return [rows[i] for i in idx]


def full_run_numbers():
    """The closing frame quotes the full 500-case run from results/summary.json,
    so the headline numbers are the logged ones, not this 12-case sample."""
    s = json.loads((ROOT / "results" / "summary.json").read_text()).get("TinyJev-0.6B")
    if not s:
        return None
    g = s["gate_0.85"]
    return dict(n=s["n"], correct=s["correct"], gate_n=g["n"], gate_cov=g["coverage"],
                gate_acc=g["accuracy"], ms=s["mean_ms"])


def frame_for(case, ans, ms, n, total, tally):
    f = R.Frame("TinyJev gate desk",
                f"decisions from 25 domains it never trained on · seed {SEED} · one forward pass each")
    x, w = 32, 430
    f.text((x, 84), f"CASE {n} OF {total}  ·  {case['domain'].replace('_', ' ').upper()}", R.F_TINY, R.MUTED)
    f.d.rounded_rectangle([R.px(x), R.px(102), R.px(x + w), R.px(214)], radius=R.px(6), fill=R.PAPER_2)
    y = 116
    for line in textwrap.wrap(state_text(case["state"]), 60)[:4]:
        f.text((x + 18, y), line, R.F_BODY, R.INK)
        y += 19
    f.text((x, 224), textwrap.shorten(case["instructions"], 68), R.F_LABEL, R.MUTED)

    conf, chosen = ans["confidence"], ans["choice"]
    handled = conf >= GATE
    right = chosen == case["expected"]
    f.text((x, 252), "VERDICT", R.F_TINY, R.MUTED)
    if handled:
        f.text((x, 268), f"handled: {chosen.replace('_', ' ')}", R.F_MODE, R.ACCENT_TEXT)
        f.text((x, 292), f"confidence {conf:.2f}, at or above the {GATE:.2f} gate", R.F_LABEL, R.MUTED)
    else:
        f.text((x, 268), "ask a person", R.F_MODE, R.MOSS)
        f.text((x, 292), f"confidence {conf:.2f}, below the {GATE:.2f} gate · leaning {chosen.replace('_', ' ')}",
               R.F_LABEL, R.MUTED)
    f.text((x, 314), ("correct" if right else "wrong") + f" · expected {case['expected'].replace('_', ' ')}",
           R.F_LABEL, R.INK if right else R.MOSS)

    px_, pw = 496, 312
    probs = ans["probabilities"]
    top = sorted(probs, key=probs.get, reverse=True)[:7]
    R.options_ledger(f, px_, 84, pw, top, probs, chosen, title="THE MODEL'S PROBABILITIES", label_w=118, row=21)
    if len(probs) > len(top):
        f.text((px_, 84 + 16 + 21 * len(top)), f"+{len(probs) - len(top)} more options below 0.02", R.F_TINY, R.MUTED)
    f.rule(px_, 340, px_ + pw)
    f.text((px_, 348), "1 question, 1 forward pass", R.F_LABEL, R.MUTED)
    f.text((px_ + pw, 348), f"{ms:.0f} ms", R.F_VALUE, R.ACCENT_TEXT, anchor="ra")

    seen, hand, hand_ok, ok, ms_sum = tally
    f.stat(32, 452, "cases", str(seen))
    f.stat(130, 452, "handled", f"{hand} / {seen}" + (f" at {hand_ok / hand:.0%}" if hand else ""))
    f.stat(300, 452, "to a person", str(seen - hand))
    f.stat(430, 452, "mean", f"{ms_sum / seen:.0f} ms")
    return f.finish()


def closing_frame(sample_tally, full):
    f = R.Frame("TinyJev gate desk", "the full run, every case logged")
    seen, hand, hand_ok, ok, ms_sum = sample_tally
    f.text((32, 96), f"this recording: {seen} cases, {hand} handled at "
                     f"{(hand_ok / hand if hand else 0):.0%}, {seen - hand} to a person", R.F_LABEL, R.MUTED)
    if full:
        y = 140
        f.text((32, y), f"{full['n']} never-seen decisions · 25 domains", R.F_MODE, R.INK); y += 34
        f.text((32, y), f"{full['correct']} / {full['n']} correct overall", R.F_MODE, R.INK); y += 34
        f.text((32, y), f"handled on its own: {full['gate_n']} / {full['n']} ({full['gate_cov']:.0%}) at {full['gate_acc']:.1%}",
               R.F_MODE, R.ACCENT_TEXT); y += 34
        f.text((32, y), f"the other {full['n'] - full['gate_n']} handed to a person", R.F_MODE, R.MOSS); y += 34
        f.text((32, y), f"596M parameters · {R.MACHINE} · {full['ms']:.0f} ms a case", R.F_LABEL, R.MUTED); y += 40
        f.text((32, y), "benchmarks/opendecision in the repo has all 500 cases, the probabilities,", R.F_BODY, R.MUTED); y += 18
        f.text((32, y), "the coverage curve, and the same-input baselines it loses to.", R.F_BODY, R.MUTED)
    return f.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyJev-0.6B")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--gif", default="")
    ap.add_argument("--ms", type=int, default=2400)
    ap.add_argument("--colors", type=int, default=32)
    args = ap.parse_args()

    cases = draw_cases(args.n)
    agent = tinyjev.load(args.model)
    frames, tally = [], [0, 0, 0, 0, 0.0]   # seen, handled, handled-and-correct, correct, ms_sum
    for n, c in enumerate(cases, 1):
        q = {"q": {"type": "choice", "instructions": c["instructions"], "criteria": c["criteria"]}}
        t0 = time.perf_counter()
        res = agent.predict({"state": c["state"], "questions": q})
        ms = (time.perf_counter() - t0) * 1000
        ans = res["states"][0]["answers"]["q"]
        handled = ans["confidence"] >= GATE
        right = ans["choice"] == c["expected"]
        tally[0] += 1; tally[1] += handled; tally[2] += handled and right; tally[3] += right; tally[4] += ms
        frames.append(frame_for(c, ans, ms, n, len(cases), tally))
        print(f"{n:2d} {c['domain'][:20]:20s} {('HANDLED' if handled else 'person '):8s} "
              f"{ans['confidence']:.2f}  {'ok' if right else 'WRONG'}  {ms:.0f} ms", flush=True)
    full = full_run_numbers()
    frames.append(closing_frame(tally, full))
    seen, hand, hand_ok, ok, ms_sum = tally
    print(f"\n{seen} cases · handled {hand} ({hand_ok} right) · {seen - hand} to a person · {ok}/{seen} correct · {ms_sum / seen:.0f} ms mean")
    if args.gif:
        out = Path(args.gif); out.parent.mkdir(parents=True, exist_ok=True)
        kb = R.save_gif(frames, args.ms, out, hold_ms=3600, colors=args.colors)
        print(f"{out} · {len(frames)} frames · {kb} KB")


if __name__ == "__main__":
    main()
