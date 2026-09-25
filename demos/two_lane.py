"""Laptop vs frontier. Left lane: tinyjev-0.6b, live, offline, on this machine. Right lane:
Claude Opus 5.5's recorded answers to the same never-seen cases, from the benchmark
run in benchmarks/opendecision/results/claude-opus-5-5.jsonl (self-reported confidence,
answered through Claude Code on 2026-09-25, no per-case latency measured).

Not a claim that the small model is smarter. It is a picture of what a laptop gets you.

    python demos/two_lane.py --gif demo_twolane.gif
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _render as R  # noqa: E402
import tinyjev  # noqa: E402
from gate_desk import draw_cases, state_text, ROOT, SEED  # noqa: E402

RIGHT_NAME = "Claude Opus 5.5"


def load_recorded(name="claude-opus-5-5"):
    rows = {}
    for l in (ROOT / "results" / f"{name}.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if "id" in r:
                rows[r["id"]] = r
    return rows


def lane(f, x, w, title, sub, ans, ms_text, case, tally, accent):
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
    seen, ok = tally
    f.stat(x, 326, "right so far", f"{ok} / {seen}")
    f.text((x + w, 348), ms_text, R.F_VALUE, R.ACCENT_TEXT, anchor="ra")


def frame_for(case, n, total, tj_ans, tj_ms, op_ans, tt, ot):
    f = R.Frame("Laptop vs frontier", f"the same never-seen case to both · seed {SEED}")
    f.text((32, 380), f"CASE {n} OF {total}  ·  {case['domain'].replace('_', ' ').upper()}", R.F_TINY, R.MUTED)
    y = 396
    for line in textwrap.wrap(state_text(case["state"]), 118)[:2]:
        f.text((32, y), line, R.F_BODY, R.INK); y += 18
    f.text((32, y + 2), textwrap.shorten(case["instructions"], 110) + f"   ·   expected: {case['expected'].replace('_', ' ')}",
           R.F_LABEL, R.MUTED)
    lane(f, 32, 370, "tinyjev-0.6b", f"live · offline · {R.MACHINE}", tj_ans, f"{tj_ms:.0f} ms", case, tt, R.ACCENT_TEXT)
    f.d.line([R.px(420), R.px(84), R.px(420), R.px(364)], fill=R.RULE, width=max(1, R.S // 2))
    lane(f, 438, 370, RIGHT_NAME, "recorded 2026-09-25 · cloud · self-reported confidence", op_ans, "cloud", case, ot, R.INK)
    return f.finish()


def closing(tt, ot):
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    tj, op = s.get("tinyjev-0.6b"), s.get("claude-opus-5-5")
    f = R.Frame("Laptop vs frontier", "the full 500, every case logged")
    f.text((32, 96), f"this recording: tinyjev {tt[1]}/{tt[0]} · {RIGHT_NAME} {ot[1]}/{ot[0]}", R.F_LABEL, R.MUTED)
    y = 140
    if tj and op:
        f.text((32, y), "500 never-seen decisions, 25 domains", R.F_MODE, R.INK); y += 40
        f.text((32, y), f"{RIGHT_NAME}:   {op['correct']} / {op['n']}   ({op['accuracy']:.1%})   cloud, metered", R.F_MODE, R.INK); y += 30
        f.text((32, y), f"tinyjev-0.6b:        {tj['correct']} / {tj['n']}   ({tj['accuracy']:.1%})   {tj['mean_ms']:.0f} ms, offline, $0", R.F_MODE, R.ACCENT_TEXT); y += 40
        f.text((32, y), "The frontier model is better. That is not the question.", R.F_LABEL, R.INK); y += 22
        f.text((32, y), "The question is what 596M parameters on a laptop get you, and how they know when to stop.", R.F_LABEL, R.INK); y += 30
        g = tj["gate_0.85"]
        f.text((32, y), f"tinyjev handled {g['n']}/{tj['n']} on its own at {g['accuracy']:.1%} and handed the rest to a person.", R.F_LABEL, R.MUTED); y += 36
        f.text((32, y), "benchmarks/opendecision in the repo: both runs, every probability.", R.F_BODY, R.MUTED)
    return f.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--gif", default="")
    ap.add_argument("--ms", type=int, default=2600)
    ap.add_argument("--colors", type=int, default=32)
    args = ap.parse_args()
    cases = draw_cases(args.n)
    recorded = load_recorded()
    tj = tinyjev.load("tinyjev-0.6b")
    frames, tt, ot = [], [0, 0], [0, 0]
    for n, c in enumerate(cases, 1):
        q = {"q": {"type": "choice", "instructions": c["instructions"], "criteria": c["criteria"]}}
        t0 = time.perf_counter(); res = tj.predict({"state": c["state"], "questions": q}); ms = (time.perf_counter() - t0) * 1000
        ta = res["states"][0]["answers"]["q"]
        op = recorded[c["id"]]
        oa = {"choice": op["predicted"], "confidence": op["confidence"], "probabilities": op["probabilities"]}
        tt[0] += 1; tt[1] += ta["choice"] == c["expected"]; ot[0] += 1; ot[1] += oa["choice"] == c["expected"]
        frames.append(frame_for(c, n, len(cases), ta, ms, oa, tt, ot))
        print(f"{n:2d} {c['domain'][:18]:18s} tinyjev {ta['choice'][:14]:14s} {ta['confidence']:.2f} {'ok' if ta['choice']==c['expected'] else '--'} {ms:.0f}ms | "
              f"opus {oa['choice'][:14]:14s} {oa['confidence']:.2f} {'ok' if oa['choice']==c['expected'] else '--'}", flush=True)
    frames.append(closing(tt, ot))
    print(f"\ntinyjev {tt[1]}/{tt[0]} · {RIGHT_NAME} {ot[1]}/{ot[0]}")
    if args.gif:
        out = Path(args.gif); out.parent.mkdir(parents=True, exist_ok=True)
        kb = R.save_gif(frames, args.ms, out, hold_ms=3600, colors=args.colors)
        print(f"{out} · {len(frames)} frames · {kb} KB")


if __name__ == "__main__":
    main()
