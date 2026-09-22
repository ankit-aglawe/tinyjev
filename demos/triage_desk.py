"""The triage desk: a stream of real support tickets, three questions each, one pass.

This is the demo of what the model actually does. Every number on every frame comes
from a live forward pass on the machine that rendered it. Nothing is scripted: the
tickets go in, the probabilities come out, and the confidence gate decides whether a
ticket is routed automatically or handed to a person.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _render as R  # noqa: E402
import tinyjev  # noqa: E402

GATE = 0.85   # route automatically at or above this confidence, otherwise ask a person

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                          "shipping": "Delivery status, delays, lost or late packages",
                          "billing": "Charges, invoices, payment problems"}},
    "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "anger": {"type": "score", "instructions": "How angry is the customer?",
              "criteria": ["calm", "frustrated", "very angry"]},
}

TICKETS = [
    "My parcel has been sitting at the depot for nine days and the tracking has not moved once.",
    "You have charged me three times for a single order. I want the two extra charges reversed today.",
    "The jacket arrived with a torn sleeve. I would like to exchange it for the same size, please.",
    "Just wanted to say the delivery driver was lovely. No issue at all, keep it up.",
    "This is the fourth time I have written about this refund. Reverse it now or I am calling my bank.",
    "Order came two weeks late AND in the wrong size, and I can see two charges on my card.",
    "Hi, could you tell me whether the blue version of this comes back in stock before Christmas?",
    "The payment failed at checkout three times but my bank says the money has already left.",
]


def wrap(text: str, width: int = 44):
    return textwrap.wrap(text, width)


ANGER_LEVELS = QUESTIONS["anger"]["criteria"]


def frame_for(ticket, ans, ms, n, total, seen, auto, ms_sum, history):
    f = R.Frame("TinyJev triage desk",
                "three questions per ticket · one forward pass · nothing is generated")

    x, w = 32, 430
    f.text((x, 84), f"INCOMING TICKET {n} OF {total}", R.F_TINY, R.MUTED)
    f.d.rounded_rectangle([R.px(x), R.px(102), R.px(x + w), R.px(228)],
                          radius=R.px(6), fill=R.PAPER_2)
    y = 120
    for line in wrap(ticket)[:5]:
        f.text((x + 18, y), line, R.F_BODY, R.INK)
        y += 20

    team, esc, anger = ans["team"], ans["escalate"], ans["anger"]
    auto_ok = team["confidence"] >= GATE
    f.text((x, 252), "VERDICT", R.F_TINY, R.MUTED)
    if auto_ok:
        f.text((x, 268), f"route to {team['choice']}", R.F_MODE, R.ACCENT_TEXT)
        f.text((x, 292), f"confidence {team['confidence']:.2f}, at or above the {GATE:.2f} gate",
               R.F_LABEL, R.MUTED)
    else:
        f.text((x, 268), "hand to a person", R.F_MODE, R.MOSS)
        f.text((x, 292), f"confidence {team['confidence']:.2f}, below the {GATE:.2f} gate",
               R.F_LABEL, R.MUTED)

    px_, pw = 496, 312
    f.text((px_, 84), "WHICH TEAM SHOULD HANDLE THIS?", R.F_TINY, R.MUTED)
    R.options_ledger(f, px_, 96, pw, list(team["probabilities"]), team["probabilities"],
                     team["choice"], title="", label_w=92, row=21)
    f.rule(px_, 178, px_ + pw)

    f.text((px_, 188), "DOES THIS NEED URGENT HUMAN ATTENTION?", R.F_TINY, R.MUTED)
    p = float(esc["p_true"])
    f.text((px_, 204), "yes", R.F_VALUE if p >= 0.5 else R.F_BODY, R.INK if p >= 0.5 else R.MUTED)
    f.bar(px_ + 92, 210, pw - 132, 8, p, R.ACCENT if p >= 0.5 else R.BAR_OFF)
    f.text((px_ + pw, 204), f"{p:.2f}", R.F_VALUE, R.INK, anchor="ra")
    f.rule(px_, 236, px_ + pw)

    f.text((px_, 246), "HOW ANGRY IS THE CUSTOMER?", R.F_TINY, R.MUTED)
    raw = anger["probabilities"]
    ordered = [raw[k] for k in sorted(raw, key=lambda k: int(k))] if \
        all(str(k).lstrip("-").isdigit() for k in raw) else list(raw.values())
    levels = dict(zip(ANGER_LEVELS, ordered))
    top = max(levels, key=levels.get)
    R.options_ledger(f, px_, 258, pw, list(levels), levels, top, title="", label_w=92, row=21)

    f.rule(px_, 340, px_ + pw)
    f.text((px_, 348), "3 questions, 1 forward pass", R.F_LABEL, R.MUTED)
    f.text((px_ + pw, 348), f"{ms:.0f} ms", R.F_VALUE, R.ACCENT_TEXT, anchor="ra")

    if history:
        f.text((x, 330), "EARLIER ON THE DESK", R.F_TINY, R.MUTED)
        hy = 348
        for label, ok in history[-4:]:
            f.text((x, hy), label, R.F_BODY, R.INK if ok else R.MUTED)
            f.text((x + w, hy), "auto" if ok else "person", R.F_LABEL,
                   R.ACCENT_TEXT if ok else R.MOSS, anchor="ra")
            hy += 19

    f.stat(32, 452, "tickets", str(seen))
    f.stat(150, 452, "auto-routed", f"{auto} of {seen}")
    f.stat(300, 452, "to a person", str(seen - auto))
    f.stat(430, 452, "mean", f"{ms_sum / seen:.0f} ms")
    return f.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinyjev-0.6b")
    ap.add_argument("--gif", default="")
    ap.add_argument("--ms", type=int, default=2000)
    ap.add_argument("--colors", type=int, default=32)
    args = ap.parse_args()

    agent = tinyjev.load(args.model)
    frames, auto, ms_sum, history = [], 0, 0.0, []
    for n, ticket in enumerate(TICKETS, 1):
        t0 = time.perf_counter()
        res = agent.predict({"state": ticket, "questions": QUESTIONS})
        ms = (time.perf_counter() - t0) * 1000
        ans = res["states"][0]["answers"]
        ms_sum += ms
        ok = ans["team"]["confidence"] >= GATE
        auto += ok
        frames.append(frame_for(ticket, ans, ms, n, len(TICKETS), n, auto, ms_sum, history))
        history.append((f"{n}. {ticket[:38]}…", ok))
        print(f"{n} {ans['team']['choice']:9s} conf {ans['team']['confidence']:.2f}  "
              f"esc {ans['escalate']['p_true']:.2f}  {ms:.0f} ms  | {ticket[:52]}", flush=True)

    print(f"\n{len(TICKETS)} tickets · {auto} auto-routed · {len(TICKETS) - auto} to a person "
          f"· {ms_sum / len(TICKETS):.0f} ms mean")
    if args.gif:
        out = Path(args.gif)
        out.parent.mkdir(parents=True, exist_ok=True)
        kb = R.save_gif(frames, args.ms, out, hold_ms=2600, colors=args.colors)
        print(f"{out} · {len(frames)} frames · {kb} KB")


if __name__ == "__main__":
    main()
