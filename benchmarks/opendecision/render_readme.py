"""Fill the generated sections of benchmarks/opendecision/README.md and the Measured
block of the top-level README from results/summary.json, so no number on either page
is typed by hand.

    python render_readme.py
"""
import csv, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "results" / "summary.json").read_text())
ROOT = HERE.parents[1]

ORDER_NOTE = {
    "claude-opus-5-5": "frontier model, cloud; probabilities are self-reported in its JSON answer, not logits",
    "kev-0.6b": "the checkpoint tinyjev reproduces; served at T=1.0 (no fitted temperature in the converted manifest)",
    "kev-0.8b": "Kev's current small model, Qwen3.5 base, via Kev's own server on torch",
    "tinyjev-0.6b": "this repo, MLX fp16, served temperature 1.464",
    "tinyjev-0.6b-int8": "this repo, MLX INT8 backbone",
    "von-1.2": "ModernBERT-Large 395M, own weights, via von-sdk",
    "agent-jev-0.6b": "Qwen3-0.6B with a permutation-equivariant set head, via the author's engine on MPS",
    "laya-english": "Laya's English ModernBERT-large checkpoint via `pip install laya`",
    "laya-typed-decisions": "Laya's typed-decisions checkpoint via `pip install laya`",
    "opendecision-engine": "the suite owner's own engine, ModernBERT-large zero-shot NLI, default profile",
    "lostargon-tiny-jev": "the other 'Tiny-Jev' on Hugging Face (name collision), its own custom head",
    "qwen3-0.6b-base-logit-readout": "the untrained backbone, options lettered, next-token letter logits",
    "nanojev": "a games-state model on the same backbone; included as the third head, not as a text baseline",
}


def fmt_ms(v):
    return "cloud" if v is None else f"{v:.0f} ms"


def table():
    rows = sorted(S.items(), key=lambda kv: -kv[1]["accuracy"])
    out = ["| model | correct / 500 | accuracy (95% CI) | dev | holdout | ECE | Brier | gate ≥0.85: coverage @ accuracy | coverage at ≤2% error | mean latency |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in rows:
        g = s["gate_0.85"]
        gate = f"{g['coverage']:.1%} @ {g['accuracy']:.3f}" if g["accuracy"] is not None else "—"
        bold = "**" if name.startswith("tinyjev-0.6b") and not name.endswith("int8") else ""
        out.append(f"| {bold}{name}{bold} | {s['correct']} | {s['accuracy']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | {s['dev']} | {s['holdout']} | "
                   f"{s['ece']:.3f} | {s['brier']:.3f} | {gate} | {s['coverage_at_err']['2%']:.1%} | {fmt_ms(s['mean_ms'])} |")
    return "\n".join(out)


def scrub(v):
    v = str(v)
    v = re.sub(r"/Users/[^/]+/\.cache/huggingface/hub/models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{7})[0-9a-f]*", r"\1/\2@\3", v)
    v = re.sub(r"/Users/[^/]+/\.cache/tinyjev/v2/([^/\s]+)", r"\1 (tinyjev-v2 conversion, local)", v)
    v = re.sub(r"/Users/[^\s]+", "<local path>", v)
    return v


def how_run():
    out = []
    for name in sorted(S, key=lambda n: -S[n]["accuracy"]):
        m = S[name]["meta"]
        note = ORDER_NOTE.get(name, "")
        extra = "; ".join(f"{k}: {scrub(v)}" for k, v in m.items() if k not in ("model", "suite_sha256") and v not in (None, "", 0))
        out.append(f"- **{name}** — {note}. {extra}")
    return "\n".join(out)


def per_domain(name="tinyjev-0.6b"):
    pd = S[name]["per_domain"]
    items = sorted(pd.items(), key=lambda kv: int(kv[1].split("/")[0]))
    out = ["| domain | tinyjev-0.6b | Kev-0.6B | raw readout | Opus 5.5 |", "|---|---:|---:|---:|---:|"]
    for d, v in items:
        out.append(f"| {d} | {v} | {S.get('kev-0.6b', {}).get('per_domain', {}).get(d, '—')} | "
                   f"{S.get('qwen3-0.6b-base-logit-readout', {}).get('per_domain', {}).get(d, '—')} | {S.get('claude-opus-5-5', {}).get('per_domain', {}).get(d, '—')} |")
    return "\n".join(out)


def cascade():
    p = HERE / "results" / "cascade.csv"
    if not p.exists():
        return "_not run_"
    rows = list(csv.DictReader(p.open()))
    out = ["| gate | stays local | sent to Opus | cascade accuracy | Opus alone | delta |", "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if float(r["gate"]) in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99):
            out.append(f"| {float(r['gate']):.2f} | {float(r['local_share']):.1%} | {float(r['sent_to_big']):.1%} | {float(r['cascade_accuracy']):.1%} | {float(r['big_only_accuracy']):.1%} | {float(r['delta_vs_big'])*100:+.1f} |")
    return "\n".join(out)


def temperature():
    p = HERE / "results" / "temperature.csv"
    if not p.exists():
        return "_not run_"
    lines = p.read_text().splitlines()
    head = lines[0].split(","); out = ["| " + " | ".join(head[2:]) + " |", "|" + "---:|" * (len(head) - 2)]
    for l in lines[1:]:
        out.append("| " + " | ".join(l.split(",")[2:]) + " |")
    return "\n".join(out)


def fill(path, marker, body):
    s = path.read_text()
    a, b = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    if a in s and b in s:
        s = s[: s.index(a) + len(a)] + "\n" + body + "\n" + s[s.index(b):]
    else:
        s = s.replace(f"<!-- {marker} -->", f"{a}\n{body}\n{b}", 1)
    path.write_text(s)


def measured_block():
    t = S["tinyjev-0.6b"]; g = t["gate_0.85"]
    raw = S.get("qwen3-0.6b-base-logit-readout"); kev = S.get("kev-0.6b")
    lines = [
        "**Measured.** On OpenDecision's Original Choice 500, a suite of 25 domains that was not in the training data:",
        f"{t['dev']} on dev and {t['holdout']} on holdout ({t['accuracy']:.3f} overall, 95% CI {t['ci95'][0]:.3f}–{t['ci95'][1]:.3f}).",
        f"At confidence ≥ 0.85 it handled {g['n']} of {t['n']} cases ({g['coverage']:.1%}) at {g['accuracy']:.1%} accuracy and sent the rest to a person.",
    ]
    if raw:
        lines.append(f"The same Qwen3-0.6B weights read through next-token letter logits, with no head, score {raw['correct']}/{raw['n']}.")
    if kev:
        lines.append(f"Kev-0.6B, the checkpoint this reproduces, scores {kev['correct']}/{kev['n']} and covers more of the queue at the same gate; the gap is the served temperature, see the benchmark page.")
    lines.append(f"{t['mean_ms']:.0f} ms a case on a base M1 via MLX. Every case, every probability, and the same-input baselines it loses to are in [`benchmarks/opendecision`](benchmarks/opendecision).")
    return "\n".join(lines)


def main():
    rd = HERE / "README.md"
    fill(rd, "TABLE", table())
    fill(rd, "HOWRUN", how_run())
    fill(rd, "DOMAINS", per_domain())
    fill(rd, "TEMPERATURE", temperature())
    fill(rd, "CASCADE", cascade())
    fill(ROOT / "README.md", "MEASURED-BLOCK", measured_block())
    print("filled", rd, "and", ROOT / "README.md")


if __name__ == "__main__":
    main()
