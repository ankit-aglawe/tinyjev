"""The batch race: one state, many typed decisions. tinyjev answers all of them in one forward
pass; a frontier model writes them out as JSON, token by token, through its API. Both replay
from a shared t=0 at real speed, so the left grid fills at once and the right one crawls.

Measured once, saved to a data file, rendered from the data file. Design changes re-render
from disk: no model, no key, no API call.

    OPENAI_API_KEY=... python3 demos/batch_race.py --case demos/cases/support_ticket.json --measure --model gpt-6-sol
    python3 demos/batch_race.py --data assets/recordings/batch-support-ticket-gpt-6-sol.json --mp4 assets/demo_batch.mp4 --gif assets/demo_batch.gif
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _render as R  # noqa: E402
from race import Canvas, P, W, H, FPS, F_TITLE, F_SUB, F_LANE, F_TINY, F_BODY, F_VALUE, F_BIG, F_MONO, logo_for, paste_logo, title_row, TJ_NAME, TJ_MODEL  # noqa: E402
from gate_desk import state_text  # noqa: E402

PRETTY = {"gpt-6-sol": "GPT-6 Sol", "claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5": "Claude Sonnet 5"}
SYSTEM = ("Answer every question about the state. Reply with one JSON object keyed by question id; each value is "
          "exactly one option id from that question's options. No other keys, no confidence values, no explanation.")
PAIR = re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*"([^"]*)"')


# ----------------------------------------------------------------------------- measure
def measure_tinyjev(case, repeats=3):
    import tinyjev
    agent = tinyjev.load("TinyJev-0.6B")
    qs = {q["id"]: {"type": "choice", "instructions": q["question"], "criteria": q["options"]} for q in case["questions"]}
    first = case["questions"][0]["id"]
    agent.predict({"state": case["state"], "questions": {first: qs[first]}})        # warm; not kept
    runs, res = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = agent.predict({"state": case["state"], "questions": qs})
        runs.append((time.perf_counter() - t0) * 1000)
    ans = res["states"][0]["answers"]
    return {"model": "TinyJev-0.6B", "hardware": R.MACHINE, "runs_ms": [round(x, 1) for x in runs],
            "ms": round(statistics.median(runs), 1),
            "answers": {q["id"]: ans[q["id"]]["choice"] for q in case["questions"]},
            "confidence": {q["id"]: round(float(ans[q["id"]]["confidence"]), 4) for q in case["questions"]},
            "probabilities": {q["id"]: {k: round(float(v), 4) for k, v in ans[q["id"]]["probabilities"].items()} for q in case["questions"]},
            "measured": time.strftime("%Y-%m-%d"),
            "note": "one predict() call with every question: one batched forward pass over the shared state, no generated tokens"}


def measure_openai(case, model, effort="low"):
    import openai
    client = openai.OpenAI()

    def call(questions):
        schema = {"name": "decisions", "strict": True, "schema": {
            "type": "object", "properties": {q["id"]: {"type": "string", "enum": list(q["options"])} for q in questions},
            "required": [q["id"] for q in questions], "additionalProperties": False}}
        user = json.dumps({"state": case["state"],
                           "questions": [{"id": q["id"], "question": q["question"], "options": q["options"]} for q in questions]},
                          ensure_ascii=False)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        kw = {"reasoning_effort": effort} if effort and effort != "none" else {}
        timeline, text, usage, final_model = [], "", None, model
        t0 = time.perf_counter()
        try:
            stream = client.chat.completions.create(model=model, stream=True, stream_options={"include_usage": True}, messages=msgs,
                                                    response_format={"type": "json_schema", "json_schema": schema}, **kw)
        except Exception as e:
            if kw and ("reasoning_effort" in str(e) or "Unsupported parameter" in str(e)):
                stream = client.chat.completions.create(model=model, stream=True, stream_options={"include_usage": True}, messages=msgs,
                                                        response_format={"type": "json_schema", "json_schema": schema})
            else:
                raise
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if getattr(chunk, "model", None):
                final_model = chunk.model
            for ch in (chunk.choices or []):
                d = getattr(ch.delta, "content", None)
                if d:
                    text += d
                    timeline.append([round((time.perf_counter() - t0) * 1000, 1), text])
        total = (time.perf_counter() - t0) * 1000
        try:
            answers = json.loads(text)
        except Exception:
            answers = dict(PAIR.findall(text))
        det = getattr(usage, "completion_tokens_details", None)
        return {"model": final_model, "name": PRETTY.get(model, "ChatGPT " + model.replace("gpt-", "").upper()),
                "provider": "openai", "effort": effort, "text": text.strip(), "timeline": timeline,
                "answers": {q["id"]: answers.get(q["id"]) for q in questions}, "ms": round(total, 1),
                "ttft": timeline[0][0] if timeline else round(total, 1),
                "tokens": getattr(usage, "completion_tokens", None), "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None,
                "recorded": time.strftime("%Y-%m-%d"),
                "note": "streamed through the OpenAI API from this machine; timestamps are wall-clock ms from request start; "
                        "the output is the shortest JSON that answers every question, option ids only"}

    call(case["questions"][:1])          # warm the connection; not kept
    return call(case["questions"])


# ----------------------------------------------------------------------------- render
def llm_state_at(llm, t_ms):
    txt = ""
    for ms, so_far in llm["timeline"]:
        if ms <= t_ms:
            txt = so_far
        else:
            break
    return dict(PAIR.findall(txt)), txt


def pill(c, x, y, h, label, fill, ink, font=F_TINY, outline=None, pad=10, anchor_right=True):
    w = c.d.textlength(label, font=font) / R.S + 2 * pad if label else 56
    x0 = x - w if anchor_right else x
    box = [P(x0), P(y), P(x0 + w), P(y + h)]
    if outline:
        c.d.rounded_rectangle(box, radius=P(h / 2), outline=outline, width=P(1))
    else:
        c.d.rounded_rectangle(box, radius=P(h / 2), fill=fill)
    if label:
        c.text((x0 + w / 2, y + h / 2 - 1), label, font, ink, anchor="mm")
    return x0


def dot(c, x, y, r, fill=None, outline=None):
    box = [P(x - r), P(y - r), P(x + r), P(y + r)]
    if fill:
        c.d.ellipse(box, fill=fill)
    else:
        c.d.ellipse(box, outline=outline, width=P(1))


def frame(data, t_ms):
    case, tj, llm = data["case"], data["tinyjev"], data["llm"]
    qs, n = case["questions"], len(case["questions"])
    c = Canvas()
    title_row(c, llm["name"])
    c.text((60, 88), f"{case['title']} · same questions and options to both · every number is a real run", F_SUB, R.MUTED)
    c.rule(60, 122, W - 60)

    # the state
    lines = []
    for para in state_text(case["state"]).split("\n"):
        lines += textwrap.wrap(para, 104) or [""]
    lines = [l for l in lines if l.strip()]
    if len(lines) > 4:
        lines = lines[:4]; lines[-1] = lines[-1][:98].rstrip(",;: ") + " …"
    c.box(60, 136, W - 120, 22 * len(lines) + 22, R.PAPER_2)
    for i, l in enumerate(lines):
        c.text((80, 148 + 22 * i), l, F_BODY, R.INK)
    top = 136 + 22 * len(lines) + 22 + 26

    # lanes
    lx, rx, lw = 60, 570, 450
    done_l, done_r = t_ms >= tj["ms"], t_ms >= llm["ms"]
    got, txt_r = llm_state_at(llm, t_ms)
    if done_r:
        got = llm["answers"]
    for x0, name, ms, done, note in (
            (lx, TJ_MODEL, tj["ms"], done_l, f"all {n} decisions in one forward pass · 0 tokens generated"),
            (rx, llm["name"], llm["ms"], done_r, None)):
        off = 58 if paste_logo(c, logo_for(name), x0, top - 6, 48) else 0
        c.text((x0 + off, top + 6), name, F_LANE, R.INK)
        c.text((x0 + lw, top - 6), f"{min(t_ms, ms):,.0f} ms", F_BIG, R.ACCENT_TEXT if done else R.INK, anchor="ra")
        if note is None:
            if not txt_r and not done_r:
                note = f"reading the prompt, {llm['prompt_tokens']} tokens…"
            elif done_r:
                note = f"finished · first token at {llm['ttft']:,.0f} ms · {llm['tokens']} tokens written"
        if note:
            c.text((x0, top + 46), note, F_TINY, R.MUTED)
        else:
            tail = txt_r.replace("\n", " ")[-44:]
            c.text((x0, top + 46), ("…" if len(txt_r) > 44 else "") + tail + "▌", F_MONO, R.MUTED)
    c.vrule(540, top, 930)

    # the grid
    gtop = top + 80
    rh = min(44, (925 - gtop) / n)
    ph = 22
    for i, q in enumerate(qs):
        y = gtop + i * rh
        for x0, decided, ans, conf, yes_fill in (
                (lx, done_l, tj["answers"][q["id"]], tj["probabilities"][q["id"]][tj["answers"][q["id"]]], R.ACCENT),
                (rx, q["id"] in got, got.get(q["id"]), None, R.INK)):
            qlines = textwrap.wrap(q["question"], 30)[:2]
            for j, ql in enumerate(qlines):
                c.text((x0 + 8, y + (rh - 19 * len(qlines)) / 2 + 19 * j), ql, F_BODY, R.INK if decided else R.MUTED)
            right_edge = x0 + lw - (54 if conf is not None else 0)
            if decided and ans is not None:
                right = ans == q["expected"]
                x_pill = pill(c, right_edge, y + (rh - ph) / 2, ph, ans.replace("_", " "), yes_fill if right else R.TRACK,
                              R.PAPER if right else R.INK)
                dot(c, x_pill - 14, y + rh / 2, 4, fill=R.MOSS) if right else dot(c, x_pill - 14, y + rh / 2, 4, outline=R.MOSS)
                if conf is not None:
                    c.text((x0 + lw, y + 5), f"{conf:.2f}", F_TINY, R.MUTED, anchor="ra")
            else:
                pill(c, right_edge, y + (rh - ph) / 2, ph, "", None, None, outline=R.TRACK)

    # tally
    c.rule(60, 944, W - 60)
    tj_ok = sum(tj["answers"][q["id"]] == q["expected"] for q in qs)
    llm_ok = sum(llm["answers"].get(q["id"]) == q["expected"] for q in qs)
    short = llm["name"].split(" ")[0]

    def stat(x, label, value, color=R.INK):
        c.text((x, 958), label.upper(), F_TINY, R.MUTED); c.text((x, 976), value, F_VALUE, color)
    stat(60, "decisions", str(n))
    stat(180, f"{TJ_NAME} right", f"{tj_ok} / {n}" if done_l else "…", R.ACCENT_TEXT)
    stat(320, f"{short} right", f"{llm_ok} / {n}" if done_r else "…")
    stat(460, TJ_NAME, f"{n / (tj['ms'] / 1000):,.0f} decisions / s" if done_l else "…", R.ACCENT_TEXT)
    stat(640, short, f"{n / (llm['ms'] / 1000):,.1f} decisions / s" if done_r else "…")
    if done_l and done_r:
        k = llm["ms"] / tj["ms"]
        c.text((W - 60, 958), (f"{k:.1f}" if k < 10 else f"{k:.0f}") + "× faster", F_TITLE, R.ACCENT_TEXT, anchor="ra")
    c.text((60, H - 36), f"{TJ_NAME} live on {tj['hardware'].split(',')[0]} · {llm['name']} recorded via its API · shared t=0 replay · dot = matches expected",
           F_TINY, R.MUTED)
    c.text((W - 60, H - 36), "github.com/ankit-aglawe/tinyjev", F_TINY, R.ACCENT_TEXT, anchor="ra")
    return c.finish()


def render(data, mp4="", gif="", gif_width=720, hold=3.2, lead=0.3, gif_colors=48):
    end = max(data["tinyjev"]["ms"], data["llm"]["ms"])
    step = 1000.0 / FPS
    frames = [frame(data, 0.0)] * int(lead * FPS)
    t = 0.0
    while t <= end + step:
        frames.append(frame(data, min(t, end)))
        t += step
    frames += [frames[-1]] * int(hold * FPS)
    if mp4:
        with tempfile.TemporaryDirectory() as td:
            for k, f in enumerate(frames):
                f.save(f"{td}/f{k:05d}.png")
            Path(mp4).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{td}/f%05d.png",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", mp4], check=True)
        print(f"{mp4} · {len(frames)} frames · {Path(mp4).stat().st_size // 1024} KB")
    if gif:
        small = [f.resize((gif_width, gif_width), Image.LANCZOS) for f in frames]
        kb = R.save_gif(small, int(1000 / FPS), Path(gif), hold_ms=int(1000 / FPS), colors=gif_colors)
        print(f"{gif} · {len(small)} frames · {kb} KB")
    return frames


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="", help="a demos/cases/*.json file: state, questions (id, question, options, expected)")
    ap.add_argument("--measure", action="store_true", help="run tinyjev live and record the API model; writes --data")
    ap.add_argument("--model", default="gpt-6-sol")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--data", default="", help="the saved measurement; default assets/recordings/batch-<case>-<model>.json")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--gif", default="")
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--gif-colors", type=int, default=48, help="palette size; 128 for a high-quality README GIF")
    ap.add_argument("--hold", type=float, default=3.2)
    ap.add_argument("--stills", default="", help="directory for a few PNG frames to look at")
    a = ap.parse_args()

    if a.measure:
        case = json.loads(Path(a.case).read_text())
        data = {"case": case, "tinyjev": measure_tinyjev(case), "llm": measure_openai(case, a.model, a.effort)}
        a.data = a.data or f"assets/recordings/batch-{case['id']}-{a.model}.json"
        Path(a.data).parent.mkdir(parents=True, exist_ok=True)
        Path(a.data).write_text(json.dumps(data, indent=1, ensure_ascii=False))
        print(f"saved {a.data}")
    elif not a.data:
        raise SystemExit("give --data (saved measurement) or --case with --measure")
    data = json.loads(Path(a.data).read_text())
    case, tj, llm = data["case"], data["tinyjev"], data["llm"]
    qs, n = case["questions"], len(case["questions"])
    tj_ok = sum(tj["answers"][q["id"]] == q["expected"] for q in qs)
    llm_ok = sum(llm["answers"].get(q["id"]) == q["expected"] for q in qs)
    print(f"{case['id']}: {n} decisions · tinyjev {tj_ok}/{n} in {tj['ms']:.0f} ms (runs {tj['runs_ms']}) · "
          f"{llm['name']} {llm_ok}/{n} in {llm['ms']:.0f} ms, first token {llm['ttft']:.0f} ms, {llm['tokens']} tokens · "
          f"{llm['ms'] / tj['ms']:.1f}×")
    for q in qs:
        x, y = tj["answers"][q["id"]], llm["answers"].get(q["id"])
        flag = "" if x == q["expected"] == y else f"   <- expected {q['expected']}"
        print(f"  {q['label']:20s} tinyjev {x:20s} {tj['confidence'][q['id']]:.2f}   {llm['name']} {str(y):20s}{flag}")
    if a.mp4 or a.gif or a.stills:
        frames = render(data, a.mp4, a.gif, a.gif_width, a.hold, gif_colors=a.gif_colors)
        if a.stills:
            Path(a.stills).mkdir(parents=True, exist_ok=True)
            for frac in (0.0, 0.15, 0.4, 0.7, 1.0):
                k = min(len(frames) - 1, int(frac * (len(frames) - 1)))
                frames[k].save(f"{a.stills}/batch_{case['id']}_{frac:.2f}.png")
            print(f"stills in {a.stills}")


if __name__ == "__main__":
    main()
