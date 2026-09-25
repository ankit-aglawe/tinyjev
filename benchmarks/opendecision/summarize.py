"""Same scorer for every results/<model>.jsonl: accuracy on all/dev/holdout with
bootstrap CIs, ECE, Brier, the full coverage-vs-error curve, coverage at fixed
error budgets, per-domain counts. Writes results/summary.json, coverage.csv,
baselines.csv.

    python summarize.py
"""
import json, csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
rng = np.random.default_rng(20260918)  # the suite's own split seed, so the CIs are reproducible
DEV = {json.loads(l)["id"] for l in (HERE / "suite/dev.jsonl").read_text().splitlines() if l.strip()}
HOLD = {json.loads(l)["id"] for l in (HERE / "suite/holdout.jsonl").read_text().splitlines() if l.strip()}


def load(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    meta = rows[0]["_meta"] if "_meta" in rows[0] else {}
    return meta, [r for r in rows if "_meta" not in r]


def boot_acc(hits, n=2000):
    hits = np.asarray(hits, float)
    if len(hits) == 0:
        return (float("nan"),) * 3
    bs = [hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(n)]
    return hits.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def ece(conf, hit, bins=10):
    conf, hit = np.asarray(conf), np.asarray(hit, float)
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return tot


def brier(rows):
    # multiclass Brier over the case's own options
    s = 0.0
    for r in rows:
        for k, p in r["probabilities"].items():
            s += (p - (1.0 if k == r["expected"] else 0.0)) ** 2
    return s / len(rows)


def coverage_curve(rows):
    """Sort by confidence desc; at each prefix, coverage and error rate."""
    srt = sorted(rows, key=lambda r: -r["confidence"])
    out, hits = [], 0
    for i, r in enumerate(srt, 1):
        hits += r["correct"]
        out.append((r["confidence"], i / len(rows), 1 - hits / i))
    return out


def coverage_at_error(curve, budget):
    best = 0.0
    for thr, cov, err in curve:
        if err <= budget:
            best = max(best, cov)
    return best


def main():
    summary, cov_rows, base_rows = {}, [], []
    for path in sorted(R.glob("*.jsonl")):
        if path.stat().st_size == 0:
            continue  # a run in progress that has not flushed yet
        meta, rows = load(path)
        if not rows:
            continue
        name = path.stem
        hits = [r["correct"] for r in rows]; conf = [r["confidence"] for r in rows]
        acc, lo, hi = boot_acc(hits)
        dev = [r for r in rows if r["id"] in DEV]; hold = [r for r in rows if r["id"] in HOLD]
        curve = coverage_curve(rows)
        gate = [r for r in rows if r["confidence"] >= 0.85]
        per_domain = defaultdict(lambda: [0, 0])
        for r in rows:
            per_domain[r["domain"]][0] += r["correct"]; per_domain[r["domain"]][1] += 1
        s = {
            "n": len(rows), "correct": int(sum(hits)),
            "accuracy": round(acc, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "dev": f"{sum(r['correct'] for r in dev)}/{len(dev)}",
            "holdout": f"{sum(r['correct'] for r in hold)}/{len(hold)}",
            "ece": round(ece(conf, hits), 4), "brier": round(brier(rows), 4),
            "mean_ms": round(float(np.mean([r["ms"] for r in rows if r["ms"] is not None])), 1) if any(r["ms"] is not None for r in rows) else None,
            "gate_0.85": {"n": len(gate), "coverage": round(len(gate) / len(rows), 4),
                          "accuracy": round(np.mean([r["correct"] for r in gate]), 4) if gate else None},
            "coverage_at_err": {f"{b:.0%}": round(coverage_at_error(curve, b), 4) for b in (0.02, 0.05, 0.10)},
            "per_domain": {d: f"{c}/{n}" for d, (c, n) in sorted(per_domain.items())},
            "meta": meta,
        }
        summary[name] = s
        for thr, cov, err in curve:
            cov_rows.append([name, round(thr, 5), round(cov, 4), round(err, 4)])
        base_rows.append([name, len(rows), s["accuracy"], s["ci95"][0], s["ci95"][1], s["dev"], s["holdout"],
                          s["ece"], s["brier"], s["gate_0.85"]["coverage"], s["gate_0.85"]["accuracy"],
                          s["coverage_at_err"]["2%"], s["coverage_at_err"]["5%"], s["mean_ms"]])
    (R / "summary.json").write_text(json.dumps(summary, indent=1))
    with (R / "coverage.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["model", "threshold", "coverage", "error_rate"], *cov_rows])
    with (R / "baselines.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["model", "n", "accuracy", "ci_lo", "ci_hi", "dev", "holdout", "ece", "brier",
                                  "gate85_coverage", "gate85_accuracy", "coverage_at_2pct_err", "coverage_at_5pct_err", "mean_ms"], *base_rows])
    for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["accuracy"]):
        g = s["gate_0.85"]
        print(f"{name:34} {s['correct']:>3}/{s['n']} = {s['accuracy']:.4f} [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]  "
              f"dev {s['dev']:>8} hold {s['holdout']:>7}  ECE {s['ece']:.3f}  gate85 {g['coverage']:.1%}@{(g['accuracy'] or 0):.3f}  "
              f"cov@2% {s['coverage_at_err']['2%']:.1%}  {(str(round(s['mean_ms'])) + ' ms') if s['mean_ms'] is not None else 'cloud'}")


if __name__ == "__main__":
    main()
