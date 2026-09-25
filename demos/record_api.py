"""Record a frontier model answering the race cases through its API, token by token, with
real timestamps. The race replays this recording at real speed next to tinyjev.

    export ANTHROPIC_API_KEY=... && python demos/record_api.py --provider anthropic --model claude-opus-5-5
    export OPENAI_API_KEY=...    && python demos/record_api.py --provider openai    --model <id>
    python demos/race.py --recording assets/recordings/<model>.jsonl --mp4 race.mp4

Six cases; a few cents either way. One warm-up call is made and not kept.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_desk import draw_cases, state_text  # noqa: E402

SYSTEM = ("You answer one typed decision. Pick exactly one option key from the list and reply "
          'with JSON only, exactly {"choice": "<option key>"}.')


def prompt_for(case):
    opts = "\n".join(f'- "{k}": {v}' for k, v in case["criteria"].items())
    return f"{state_text(case['state'])}\n\nQuestion: {case['instructions']}\n\nOptions:\n{opts}"


def record(client, model, case, effort):
    opts = list(case["criteria"])
    schema = {"type": "object", "properties": {"choice": {"type": "string", "enum": opts}},
              "required": ["choice"], "additionalProperties": False}
    timeline, text = [], ""
    t0 = time.perf_counter()
    with client.messages.stream(
        model=model, max_tokens=400, system=SYSTEM,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt_for(case)}],
    ) as stream:
        for delta in stream.text_stream:
            text += delta
            timeline.append(((time.perf_counter() - t0) * 1000, text))
        final = stream.get_final_message()
    total_ms = (time.perf_counter() - t0) * 1000
    m = re.search(r'"choice"\s*:\s*"([^"]+)"', text)
    return {
        "id": case["id"], "model": final.model, "text": text.strip(), "choice": m.group(1) if m else None,
        "timeline": [[round(ms, 1), t] for ms, t in timeline],
        "tokens": final.usage.output_tokens, "prompt_tokens": final.usage.input_tokens,
        "ttft": round(timeline[0][0], 1) if timeline else round(total_ms, 1), "ms": round(total_ms, 1),
        "stop_reason": final.stop_reason, "effort": effort,
    }


def record_openai(client, model, case, effort):
    """Chat Completions, streamed, JSON schema response; timestamps per content delta."""
    opts = list(case["criteria"])
    schema = {"name": "decision", "strict": True, "schema": {"type": "object", "properties": {"choice": {"type": "string", "enum": opts}},
              "required": ["choice"], "additionalProperties": False}}
    kw = {}
    if effort and effort != "none":
        kw["reasoning_effort"] = effort   # ignored by models that do not reason; low keeps reasoning models short
    timeline, text = [], ""
    t0 = time.perf_counter()
    usage = None; final_model = model
    try:
        stream = client.chat.completions.create(
            model=model, stream=True, stream_options={"include_usage": True},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_for(case)}],
            response_format={"type": "json_schema", "json_schema": schema}, **kw)
    except Exception as e:
        if "reasoning_effort" in str(e) or "Unsupported parameter" in str(e):
            kw.pop("reasoning_effort", None)
            stream = client.chat.completions.create(
                model=model, stream=True, stream_options={"include_usage": True},
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_for(case)}],
                response_format={"type": "json_schema", "json_schema": schema}, **kw)
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
                timeline.append(((time.perf_counter() - t0) * 1000, text))
    total_ms = (time.perf_counter() - t0) * 1000
    m = re.search(r'"choice"\s*:\s*"([^"]+)"', text)
    return {
        "id": case["id"], "model": final_model, "text": text.strip(), "choice": m.group(1) if m else None,
        "timeline": [[round(ms, 1), t] for ms, t in timeline],
        "tokens": getattr(usage, "completion_tokens", None), "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "ttft": round(timeline[0][0], 1) if timeline else round(total_ms, 1), "ms": round(total_ms, 1),
        "stop_reason": "end", "effort": effort,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low", help="low keeps the answer short; thinking cannot be turned off on this model")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(); rec = record
    else:
        import openai
        client = openai.OpenAI(); rec = record_openai
    cases = draw_cases(a.n)
    out = Path(a.out or f"assets/recordings/{a.model}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rec(client, a.model, cases[0], a.effort)   # warm the connection; not kept
    rows = []
    with out.open("w") as f:
        f.write(json.dumps({"_meta": {"model": a.model, "provider": a.provider, "effort": a.effort, "recorded": time.strftime("%Y-%m-%d"),
                                      "note": f"streamed through the {a.provider} API from this machine; timestamps are wall-clock ms from request start"}}) + "\n")
        for i, c in enumerate(cases, 1):
            r = rec(client, a.model, c, a.effort)
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            print(f"{i:2d} {c['domain'][:18]:18s} {str(r['choice'])[:18]:18s} {'ok' if r['choice'] == c['expected'] else '--'}  "
                  f"first text {r['ttft']:6.0f} ms  total {r['ms']:6.0f} ms  {r['tokens']} tok", flush=True)
    print(f"\n{out} · {len(rows)} cases · mean {sum(r['ms'] for r in rows) / len(rows):.0f} ms · "
          f"{sum(r['choice'] == c['expected'] for r, c in zip(rows, cases))}/{len(rows)} right")


if __name__ == "__main__":
    main()
