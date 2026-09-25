"""The race: tinyjev against a frontier model, same never-seen question, both scored.

The frontier lane is a recording made by demos/record_api.py: the model's real answer
streamed through its API with wall-clock timestamps. tinyjev runs live here. Both are
replayed from a shared t=0 at real speed; the stopwatches are the measured times.

    python demos/record_api.py                                   # once, needs ANTHROPIC_API_KEY
    python demos/race.py --recording assets/recordings/claude-opus-5-5.jsonl --mp4 race.mp4 --gif race.gif

A local MLX chat model can stand in for the recording with --llm <mlx model path>.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _render as R  # noqa: E402  (fonts and colours)
import tinyjev  # noqa: E402
from gate_desk import draw_cases, state_text  # noqa: E402

W = H = 1080          # square, for the feed
S = 2                 # supersample
FPS = 12.5            # 80 ms per frame
HOLD = 1.6            # seconds to hold the finished state of each case
MONO_FACES = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/SFNSMono.ttf", "/Library/Fonts/Courier New.ttf"]


def _f(weight, px):
    from PIL import ImageFont
    return ImageFont.truetype(str(R._FONTS / f"Poppins-{weight}.ttf"), px * S)


def _mono(px):
    from PIL import ImageFont
    for face in MONO_FACES:
        try:
            return ImageFont.truetype(face, px * S)
        except OSError:
            continue
    return _f("Regular", px)


F_TITLE, F_SUB, F_LANE, F_TINY, F_BODY, F_VALUE, F_BIG, F_MONO, F_MONO_L = (
    _f("SemiBold", 30), _f("Regular", 15), _f("SemiBold", 18), _f("Medium", 12),
    _f("Regular", 15), _f("Medium", 15), _f("SemiBold", 40), _mono(14), _mono(19))


def P(v):
    return int(v * S)


class Canvas:
    def __init__(self):
        self.img = Image.new("RGB", (W * S, H * S), R.PAPER)
        self.d = ImageDraw.Draw(self.img)

    def text(self, xy, s, font=F_BODY, fill=R.INK, anchor=None):
        self.d.text((P(xy[0]), P(xy[1])), s, font=font, fill=fill, anchor=anchor)

    def rule(self, x0, y, x1, color=R.RULE):
        self.d.line([P(x0), P(y), P(x1), P(y)], fill=color, width=max(1, S // 2))

    def vrule(self, x, y0, y1, color=R.RULE):
        self.d.line([P(x), P(y0), P(x), P(y1)], fill=color, width=max(1, S // 2))

    def box(self, x, y, w, h, fill):
        self.d.rounded_rectangle([P(x), P(y), P(x + w), P(y + h)], radius=P(8), fill=fill)

    def bar(self, x, y, w, h, frac, color):
        self.d.rounded_rectangle([P(x), P(y), P(x + w), P(y + h)], radius=P(h / 2), fill=R.TRACK)
        if frac > 0.004:
            self.d.rounded_rectangle([P(x), P(y), P(x + w * min(1.0, frac)), P(y + h)], radius=P(h / 2), fill=color)

    def finish(self):
        return self.img.resize((W, H), Image.LANCZOS)


# ----------------------------------------------------------------------------- measurement
def run_tinyjev(agent, case):
    q = {"q": {"type": "choice", "instructions": case["instructions"], "criteria": case["criteria"]}}
    t0 = time.perf_counter()
    res = agent.predict({"state": case["state"], "questions": q})
    ms = (time.perf_counter() - t0) * 1000
    return res["states"][0]["answers"]["q"], ms


def run_llm(model, tok, case, max_tokens=160):
    """Stream a JSON answer; record (elapsed_ms, text_so_far) per token."""
    from mlx_lm import stream_generate
    opts = "\n".join(f'- "{k}": {v}' for k, v in case["criteria"].items())
    user = (f"{state_text(case['state'])}\n\nQuestion: {case['instructions']}\n\nOptions:\n{opts}\n\n"
            f'Reply with JSON only, exactly {{"choice": "<one option key>"}}. /no_think')
    msgs = [{"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False) \
        if hasattr(tok, "apply_chat_template") else user
    timeline, text, prompt_tokens = [], "", 0
    t0 = time.perf_counter()
    ntok = 0
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens):
        text += r.text
        ntok += 1
        prompt_tokens = r.prompt_tokens or prompt_tokens
        timeline.append(((time.perf_counter() - t0) * 1000, text))
        if "}" in text:
            break
    total_ms = (time.perf_counter() - t0) * 1000
    m = re.search(r'"choice"\s*:\s*"([^"]+)"', text)
    choice = m.group(1) if m else None
    ttft = timeline[0][0] if timeline else total_ms
    return {"text": text.strip(), "choice": choice, "timeline": timeline, "tokens": ntok, "ms": total_ms,
            "prompt_tokens": prompt_tokens, "ttft": ttft}


# ----------------------------------------------------------------------------- frames
def frame(case, idx, total, tj, tj_ms, llm, t_ms, llm_name, tally, history=()):
    c = Canvas()
    c.text((60, 44), f"tinyjev vs {llm_name}", F_TITLE, R.INK)
    c.text((60, 88), "same question to both · every number is a real run", F_SUB, R.MUTED)
    c.text((W - 60, 50), "tinyjev", F_LANE, R.ACCENT_TEXT, anchor="ra")
    c.rule(60, 122, W - 60)

    # the case
    c.text((60, 138), f"CASE {idx} OF {total}  ·  {case['domain'].replace('_', ' ').upper()}", F_TINY, R.MUTED)
    c.box(60, 156, W - 120, 118, R.PAPER_2)
    y = 170
    for line in textwrap.wrap(state_text(case["state"]), 96)[:3]:
        c.text((80, y), line, F_BODY, R.INK); y += 22
    c.text((80, 246), textwrap.shorten(case["instructions"], 92) + f"   ·   expected: {case['expected'].replace('_', ' ')}", F_TINY, R.MUTED)

    # lanes
    lx, rx, lw = 60, 570, 450
    top = 300
    c.vrule(540, top, 940)
    # left: tinyjev
    c.text((lx, top), "tinyjev-0.6b", F_LANE, R.INK)
    c.text((lx, top + 26), "one forward pass · 0 tokens generated", F_TINY, R.MUTED)
    done_l = t_ms >= tj_ms
    shown = min(t_ms, tj_ms)
    c.text((lx, top + 52), f"{shown:,.0f} ms", F_BIG, R.ACCENT_TEXT if done_l else R.INK)
    if done_l:
        probs = tj["probabilities"]; chosen = tj["choice"]; right = chosen == case["expected"]
        c.text((lx, top + 112), ("correct" if right else "wrong") + f" · picks {chosen.replace('_', ' ')} at {tj['confidence']:.2f}",
               F_VALUE, R.ACCENT_TEXT if right else R.MOSS)
        yy = top + 144
        for name in sorted(probs, key=probs.get, reverse=True)[:6]:
            p = probs[name]; hit = name == chosen
            c.text((lx + 12, yy), name.replace("_", " ")[:22], F_VALUE if hit else F_BODY, R.INK if hit else R.MUTED)
            c.bar(lx + 200, yy + 7, lw - 260, 9, p, R.ACCENT if hit else R.BAR_OFF)
            c.text((lx + lw, yy), f"{p:.2f}", F_VALUE if hit else F_BODY, R.INK if hit else R.MUTED, anchor="ra")
            yy += 26
        c.text((lx, top + 320), "finished", F_TINY, R.ACCENT_TEXT)
    else:
        c.text((lx, top + 112), "reading the state…", F_VALUE, R.MUTED)

    # right: the LLM
    c.text((rx, top), llm_name, F_LANE, R.INK)
    c.text((rx, top + 26), llm.get("lane_note", "writes the answer as JSON, token by token"), F_TINY, R.MUTED)
    done_r = t_ms >= llm["ms"]
    shown_r = min(t_ms, llm["ms"])
    c.text((rx, top + 52), f"{shown_r:,.0f} ms", F_BIG, R.ACCENT_TEXT if done_r else R.INK)
    # streamed text at time t
    txt = ""
    for ms, so_far in llm["timeline"]:
        if ms <= t_ms:
            txt = so_far
        else:
            break
    ntok_now = sum(1 for ms, _ in llm["timeline"] if ms <= t_ms)
    if ntok_now == 0 and not done_r:
        c.text((rx, top + 112), f"reading the prompt, {llm['prompt_tokens']} tokens…", F_VALUE, R.MUTED)
    else:
        c.text((rx, top + 112), f"{ntok_now} tokens written" + ("" if done_r else "…"), F_VALUE, R.MUTED)
    c.box(rx, top + 140, lw, 150, R.PAPER_2)
    yy = top + 158
    for line in textwrap.wrap(txt.replace("\n", " "), 34)[:4]:
        c.text((rx + 16, yy), line, F_MONO_L, R.INK); yy += 30
    if done_r:
        ch = llm["choice"]; right = ch == case["expected"]
        c.text((rx, top + 300), ("correct" if right else ("wrong" if ch else "no valid JSON")) + (f" · picks {ch.replace('_', ' ')}" if ch else ""),
               F_VALUE, R.ACCENT_TEXT if right else R.MOSS)
        c.text((rx, top + 320), f"finished · first token at {llm['ttft']:,.0f} ms · {llm['tokens']} tokens · {llm['ms']/tj_ms:.0f}× the decision model's time", F_TINY, R.ACCENT_TEXT)
    else:
        c.text((rx, top + 300), "still writing…", F_VALUE, R.MUTED)

    # earlier cases
    if history:
        c.rule(60, 700, W - 60)
        c.text((60, 714), "EARLIER", F_TINY, R.MUTED)
        c.text((330, 714), "TINYJEV", F_TINY, R.MUTED); c.text((560, 714), llm_name.split(" ")[0].upper(), F_TINY, R.MUTED)
        hy = 734
        for (dom, tj_ok, tj_t, ll_ok, ll_t) in history[-5:]:
            c.text((60, hy), dom.replace("_", " "), F_BODY, R.INK)
            c.text((330, hy), f"{'right' if tj_ok else 'wrong'} · {tj_t:,.0f} ms", F_BODY, R.ACCENT_TEXT if tj_ok else R.MOSS)
            c.text((560, hy), f"{'right' if ll_ok else 'wrong'} · {ll_t:,.0f} ms", F_BODY, R.INK if ll_ok else R.MOSS)
            hy += 24
    # tally
    c.rule(60, 960, W - 60)
    n, tj_ok, llm_ok, tj_sum, llm_sum, first = tally
    def stat(x, label, value, color=R.INK):
        c.text((x, 976), label.upper(), F_TINY, R.MUTED); c.text((x, 994), value, F_VALUE, color)
    stat(60, "cases", str(n))
    stat(190, "tinyjev right", f"{tj_ok} / {n}", R.ACCENT_TEXT)
    stat(360, f"{llm_name.split(' ')[0]} right", f"{llm_ok} / {n}")
    stat(560, "tinyjev mean", f"{tj_sum / max(n, 1):,.0f} ms", R.ACCENT_TEXT)
    stat(730, f"{llm_name.split(' ')[0]} mean", f"{llm_sum / max(n, 1):,.0f} ms")
    stat(900, "finished first", f"{first} / {n}", R.ACCENT_TEXT)
    c.text((60, H - 36), llm.get("footer", "both measured, replayed at real speed from a shared t=0"), F_TINY, R.MUTED)
    c.text((W - 60, H - 36), "github.com/ankit-aglawe/tinyjev", F_TINY, R.ACCENT_TEXT, anchor="ra")
    return c.finish()


def closing(tally, llm_name, n_total):
    c = Canvas()
    n, tj_ok, llm_ok, tj_sum, llm_sum, first = tally
    c.text((60, 44), f"tinyjev vs {llm_name}", F_TITLE, R.INK)
    c.text((60, 88), f"{n} never-seen decisions from 25 domains", F_SUB, R.MUTED)
    c.text((W - 60, 50), "tinyjev", F_LANE, R.ACCENT_TEXT, anchor="ra")
    c.rule(60, 122, W - 60)
    y = 200
    rows = [("tinyjev-0.6b, one forward pass", f"{tj_ok} / {n} right", f"{tj_sum / n:,.0f} ms a case", R.ACCENT_TEXT),
            (f"{llm_name}, writing JSON", f"{llm_ok} / {n} right", f"{llm_sum / n:,.0f} ms a case", R.INK)]
    for name, acc, ms, col in rows:
        c.text((60, y), name, F_LANE, col); c.text((640, y), acc, F_LANE, col); c.text((W - 60, y), ms, F_LANE, col, anchor="ra"); y += 56
    c.rule(60, y, W - 60); y += 30
    c.text((60, y), f"{llm_sum / max(tj_sum, 1):.0f}× faster", F_BIG, R.ACCENT_TEXT); y += 62
    c.text((60, y), f"tinyjev finished first {first} of {n} times and never generated a token.", F_VALUE, R.INK); y += 30
    c.text((60, y), f"On the full 500 it scores 440 to {llm_name}'s 496. Put in front of {llm_name}, it matches", F_VALUE, R.INK); y += 24
    c.text((60, y), f"{llm_name}'s accuracy while a third of the decisions never reach it.", F_VALUE, R.INK); y += 50
    c.text((60, y), "596M parameters · MIT · pip install tinyjev · every case logged in benchmarks/opendecision", F_BODY, R.MUTED)
    c.text((60, H - 36), "both measured, replayed at real speed · nothing staged", F_TINY, R.MUTED)
    c.text((W - 60, H - 36), "github.com/ankit-aglawe/tinyjev", F_TINY, R.ACCENT_TEXT, anchor="ra")
    return c.finish()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="", help="JSONL from demos/record_api.py; the frontier lane")
    ap.add_argument("--llm", default="", help="or: an MLX instruct model path, run live")
    ap.add_argument("--llm-name", default="")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--mp4", default="")
    ap.add_argument("--gif", default="")
    ap.add_argument("--gif-width", type=int, default=720)
    a = ap.parse_args()

    cases = draw_cases(a.n)
    agent = tinyjev.load("tinyjev-0.6b")
    run_tinyjev(agent, cases[0])                      # warm, so the first lane does not pay a first-call cost
    recorded = {}
    if a.recording:
        meta = {}
        for l in Path(a.recording).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if "_meta" in r:
                    meta = r["_meta"]
                else:
                    recorded[r["id"]] = r
        pretty = {"claude-opus-5-5": "Claude Opus 5.5", "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5"}
        a.llm_name = a.llm_name or pretty.get(meta.get("model", ""), meta.get("model", "the API model"))
        lane_note = f"answers through its API, streamed · recorded {meta.get('recorded', '')}"
        footer = "tinyjev measured live; the API lane recorded with real timestamps; both replayed from a shared t=0"
        missing = [c["id"] for c in cases if c["id"] not in recorded]
        if missing:
            raise SystemExit(f"recording lacks {len(missing)} of the {len(cases)} race cases; re-run record_api.py with --n {a.n}")
    else:
        if not a.llm:
            raise SystemExit("give --recording (frontier lane) or --llm (local MLX model)")
        from mlx_lm import load
        model, tok = load(a.llm)
        run_llm(model, tok, cases[0], max_tokens=8)
        a.llm_name = a.llm_name or Path(a.llm).name
        lane_note = "writes the answer as JSON, token by token · thinking off"
        footer = "both measured here one after the other, replayed at real speed from a shared t=0"

    frames, tally, history = [], [0, 0, 0, 0.0, 0.0, 0], []
    for i, case in enumerate(cases, 1):
        tj, tj_ms = run_tinyjev(agent, case)
        if recorded:
            r = recorded[case["id"]]
            llm = {"text": r["text"], "choice": r["choice"], "timeline": [(ms, t) for ms, t in r["timeline"]],
                   "tokens": r["tokens"], "ms": r["ms"], "prompt_tokens": r["prompt_tokens"], "ttft": r["ttft"]}
        else:
            llm = run_llm(model, tok, case)
        llm["lane_note"] = lane_note; llm["footer"] = footer
        tally[0] += 1; tally[1] += tj["choice"] == case["expected"]; tally[2] += llm["choice"] == case["expected"]
        tally[3] += tj_ms; tally[4] += llm["ms"]; tally[5] += tj_ms < llm["ms"]
        end = max(tj_ms, llm["ms"])
        t = 0.0
        step = 1000.0 / FPS
        while t <= end + step:
            frames.append(frame(case, i, len(cases), tj, tj_ms, llm, min(t, end), a.llm_name, tally, history))
            t += step
        for _ in range(int(HOLD * FPS)):
            frames.append(frames[-1])
        history.append((case["domain"], tj["choice"] == case["expected"], tj_ms, llm["choice"] == case["expected"], llm["ms"]))
        print(f"{i:2d} {case['domain'][:18]:18s} tinyjev {tj['choice'][:16]:16s} {'ok' if tj['choice']==case['expected'] else '--'} {tj_ms:6.0f} ms | "
              f"llm {str(llm['choice'])[:16]:16s} {'ok' if llm['choice']==case['expected'] else '--'} {llm['ms']:6.0f} ms  {llm['tokens']} tok", flush=True)
    last = closing(tally, a.llm_name, len(cases))
    for _ in range(int(4.0 * FPS)):
        frames.append(last)
    n, tj_ok, llm_ok, tj_sum, llm_sum, first = tally
    print(f"\n{n} cases · tinyjev {tj_ok}/{n} at {tj_sum/n:.0f} ms · llm {llm_ok}/{n} at {llm_sum/n:.0f} ms · tinyjev first {first}/{n}")

    if a.mp4:
        with tempfile.TemporaryDirectory() as td:
            for k, f in enumerate(frames):
                f.save(f"{td}/f{k:05d}.png")
            Path(a.mp4).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{td}/f%05d.png",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.mp4], check=True)
        print(f"{a.mp4} · {len(frames)} frames · {Path(a.mp4).stat().st_size // 1024} KB")
    if a.gif:
        small = [f.resize((a.gif_width, a.gif_width), Image.LANCZOS) for f in frames]
        # a GIF at this length needs a coarser clock: keep every other frame
        kb = R.save_gif(small[::2], int(2000 / FPS), Path(a.gif), hold_ms=int(2000 / FPS), colors=48)
        print(f"{a.gif} · {len(small[::2])} frames · {kb} KB")


if __name__ == "__main__":
    main()
