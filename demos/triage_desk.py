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

STRINGS = {
    "en": dict(title="TinyJev triage desk",
               sub="three questions per ticket · one forward pass · nothing is generated",
               incoming="INCOMING TICKET {n} OF {t}", verdict="VERDICT",
               route="route to {c}", human="hand to a person",
               above="confidence {c:.2f}, at or above the {g:.2f} gate",
               below="confidence {c:.2f}, below the {g:.2f} gate",
               q_team="WHICH TEAM SHOULD HANDLE THIS?",
               q_esc="DOES THIS NEED URGENT HUMAN ATTENTION?", yes="yes",
               q_anger="HOW ANGRY IS THE CUSTOMER?", onepass="3 questions, 1 forward pass",
               earlier="EARLIER ON THE DESK", auto="auto", person="person",
               tickets="tickets", routed="auto-routed", toperson="to a person", mean="mean"),
    "zh": dict(title="TinyJev 工单分诊台",
               sub="每条工单三个问题 · 一次前向推理 · 不生成任何文本",
               incoming="第 {n} 条工单 / 共 {t} 条", verdict="结论",
               route="自动分派给{c}", human="转交人工处理",
               above="置信度 {c:.2f}，达到或高于 {g:.2f} 阈值",
               below="置信度 {c:.2f}，低于 {g:.2f} 阈值",
               q_team="这条工单应该由哪个团队处理？",
               q_esc="是否需要人工紧急介入？", yes="是",
               q_anger="客户的情绪程度？", onepass="3 个问题，1 次前向推理",
               earlier="之前处理过的", auto="自动", person="人工",
               tickets="工单", routed="自动分派", toperson="转人工", mean="平均"),
}

TEAM_LABELS = {"en": {"returns": "returns", "shipping": "shipping", "billing": "billing"},
               "zh": {"returns": "退换货", "shipping": "物流", "billing": "账单"}}

QUESTIONS = {
    "en": {
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                              "shipping": "Delivery status, delays, lost or late packages",
                              "billing": "Charges, invoices, payment problems"}},
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
        "anger": {"type": "score", "instructions": "How angry is the customer?",
                  "criteria": ["calm", "frustrated", "very angry"]},
    },
    "zh": {
        "team": {"type": "choice", "instructions": "这条工单应该由哪个团队处理？",
                 "criteria": {"returns": "退换货、退款、错发或破损商品",
                              "shipping": "物流状态、延迟、包裹丢失或迟送",
                              "billing": "扣款、发票、支付问题"}},
        "escalate": {"type": "noul", "instructions": "是否需要人工紧急介入？"},
        "anger": {"type": "score", "instructions": "客户的情绪程度？",
                  "criteria": ["平静", "不满", "非常愤怒"]},
    },
}

TICKETS = {"en": [
    "My parcel has been sitting at the depot for nine days and the tracking has not moved once.",
    "You have charged me three times for a single order. I want the two extra charges reversed today.",
    "The jacket arrived with a torn sleeve. I would like to exchange it for the same size, please.",
    "Just wanted to say the delivery driver was lovely. No issue at all, keep it up.",
    "This is the fourth time I have written about this refund. Reverse it now or I am calling my bank.",
    "Order came two weeks late AND in the wrong size, and I can see two charges on my card.",
    "Hi, could you tell me whether the blue version of this comes back in stock before Christmas?",
    "The payment failed at checkout three times but my bank says the money has already left.",
], "zh": [
    "我的包裹在仓库已经滞留九天了，物流信息一直没有更新。",
    "同一笔订单你们扣了我三次钱，请今天把多扣的两笔退回来。",
    "外套到货时袖子是破的，我想换一件同样尺码的。",
    "只是想说这次的配送小哥很好，完全没有问题，请继续保持。",
    "这已经是我第四次写信问这笔退款了，今天不退我就找银行投诉。",
    "订单晚了两周，尺码还发错了，而且我看到卡上有两笔扣款。",
    "请问这款的蓝色会在圣诞节之前补货吗？",
    "结账时支付失败了三次，但银行说钱已经扣掉了。",
]}


NO_LEAD = "。，、；：？！）】」』%.,)]"


def wrap(text: str, lang: str, width: int = 44):
    if lang != "zh":
        return textwrap.wrap(text, width)
    lines, cur = [], ""                       # CJK: wrap by count, never strand punctuation
    for ch in text:
        if len(cur) >= 21 and ch not in NO_LEAD:
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def frame_for(ticket, ans, ms, n, total, seen, auto, ms_sum, history, lang):
    T = STRINGS[lang]
    labels = TEAM_LABELS[lang]
    anger_levels = QUESTIONS[lang]["anger"]["criteria"]
    f = R.Frame(T["title"], T["sub"])

    x, w = 32, 430
    f.text((x, 84), T["incoming"].format(n=n, t=total), R.F_TINY, R.MUTED)
    f.d.rounded_rectangle([R.px(x), R.px(102), R.px(x + w), R.px(228)],
                          radius=R.px(6), fill=R.PAPER_2)
    y = 120
    for line in wrap(ticket, lang)[:5]:
        f.text((x + 18, y), line, R.F_BODY, R.INK)
        y += 20

    team, esc, anger = ans["team"], ans["escalate"], ans["anger"]
    auto_ok = team["confidence"] >= GATE
    f.text((x, 252), T["verdict"], R.F_TINY, R.MUTED)
    if auto_ok:
        f.text((x, 268), T["route"].format(c=labels[team["choice"]]), R.F_MODE, R.ACCENT_TEXT)
        f.text((x, 292), T["above"].format(c=team["confidence"], g=GATE), R.F_LABEL, R.MUTED)
    else:
        f.text((x, 268), T["human"], R.F_MODE, R.MOSS)
        f.text((x, 292), T["below"].format(c=team["confidence"], g=GATE), R.F_LABEL, R.MUTED)

    px_, pw = 496, 312
    f.text((px_, 84), T["q_team"], R.F_TINY, R.MUTED)
    tp = {labels[k]: v for k, v in team["probabilities"].items()}
    R.options_ledger(f, px_, 96, pw, list(tp), tp, labels[team["choice"]],
                     title="", label_w=92, row=21)
    f.rule(px_, 178, px_ + pw)

    f.text((px_, 188), T["q_esc"], R.F_TINY, R.MUTED)
    p = float(esc["p_true"])
    f.text((px_, 204), T["yes"], R.F_VALUE if p >= 0.5 else R.F_BODY,
           R.INK if p >= 0.5 else R.MUTED)
    f.bar(px_ + 92, 210, pw - 132, 8, p, R.ACCENT if p >= 0.5 else R.BAR_OFF)
    f.text((px_ + pw, 204), f"{p:.2f}", R.F_VALUE, R.INK, anchor="ra")
    f.rule(px_, 236, px_ + pw)

    f.text((px_, 246), T["q_anger"], R.F_TINY, R.MUTED)
    raw = anger["probabilities"]
    ordered = [raw[k] for k in sorted(raw, key=lambda k: int(k))] if \
        all(str(k).lstrip("-").isdigit() for k in raw) else list(raw.values())
    levels = dict(zip(anger_levels, ordered))
    top = max(levels, key=levels.get)
    R.options_ledger(f, px_, 258, pw, list(levels), levels, top, title="", label_w=92, row=21)

    f.rule(px_, 340, px_ + pw)
    f.text((px_, 348), T["onepass"], R.F_LABEL, R.MUTED)
    f.text((px_ + pw, 348), f"{ms:.0f} ms", R.F_VALUE, R.ACCENT_TEXT, anchor="ra")

    if history:
        f.text((x, 330), T["earlier"], R.F_TINY, R.MUTED)
        hy = 348
        for label, ok in history[-4:]:
            f.text((x, hy), label, R.F_BODY, R.INK if ok else R.MUTED)
            f.text((x + w, hy), T["auto"] if ok else T["person"], R.F_LABEL,
                   R.ACCENT_TEXT if ok else R.MOSS, anchor="ra")
            hy += 19

    f.stat(32, 452, T["tickets"], str(seen))
    f.stat(150, 452, T["routed"], f"{auto} / {seen}")
    f.stat(300, 452, T["toperson"], str(seen - auto))
    f.stat(430, 452, T["mean"], f"{ms_sum / seen:.0f} ms")
    return f.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinyjev-0.6b")
    ap.add_argument("--gif", default="")
    ap.add_argument("--ms", type=int, default=2000)
    ap.add_argument("--colors", type=int, default=32)
    ap.add_argument("--tickets", default="", help="JSON list of tickets to run instead of the built-in set")
    ap.add_argument("--lang", choices=["en", "zh"], default="en",
                    help="zh renders the desk and the tickets in Chinese")
    args = ap.parse_args()

    if args.lang == "zh":
        R.use_cjk()
        R.FOOTNOTE = "画面中的每个数字都来自一次真实的前向推理，现场录制"
    tickets, questions = TICKETS[args.lang], QUESTIONS[args.lang]
    if args.tickets:
        import json
        tickets = json.loads(Path(args.tickets).read_text())
    cut = 18 if args.lang == "zh" else 38

    agent = tinyjev.load(args.model)
    frames, auto, ms_sum, history = [], 0, 0.0, []
    for n, ticket in enumerate(tickets, 1):
        t0 = time.perf_counter()
        res = agent.predict({"state": ticket, "questions": questions})
        ms = (time.perf_counter() - t0) * 1000
        ans = res["states"][0]["answers"]
        ms_sum += ms
        ok = ans["team"]["confidence"] >= GATE
        auto += ok
        frames.append(frame_for(ticket, ans, ms, n, len(tickets), n, auto, ms_sum,
                                history, args.lang))
        history.append((f"{n}. {ticket[:cut]}…", ok))
        print(f"{n} {ans['team']['choice']:9s} conf {ans['team']['confidence']:.2f}  "
              f"esc {ans['escalate']['p_true']:.2f}  {ms:.0f} ms  | {ticket[:40]}", flush=True)

    print(f"\n{len(tickets)} tickets · {auto} auto-routed · {len(tickets) - auto} to a person "
          f"· {ms_sum / len(tickets):.0f} ms mean")
    if args.gif:
        out = Path(args.gif)
        out.parent.mkdir(parents=True, exist_ok=True)
        kb = R.save_gif(frames, args.ms, out, hold_ms=2600, colors=args.colors)
        print(f"{out} · {len(frames)} frames · {kb} KB")


if __name__ == "__main__":
    main()
