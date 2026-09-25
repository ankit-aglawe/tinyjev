"""Re-temper a logged run offline. Served probabilities are logits/T; log(p)*k recovers
logits/(T/k) up to a constant, so k=T undoes the fitted temperature. Writes
results/temperature.csv. Used to check whether a temperature fitted in-distribution
survives out of distribution (for tinyjev-0.6b, it does not).

    python temperature.py tinyjev-0.6b
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from summarize import load, coverage_curve, coverage_at_error, ece

name = sys.argv[1] if len(sys.argv) > 1 else "tinyjev-0.6b"
meta, rows = load(Path("results") / f"{name}.jsonl")
T = float(meta.get("temperature") or 1.0)
out = [["model", "served_T", "factor", "effective_T", "accuracy", "ece", "gate85_coverage", "gate85_accuracy", "cov_at_2pct", "cov_at_5pct"]]
for k in (0.8, 1.0, 1.2, T, 1.8, 2.2):
    rr = []
    for r in rows:
        keys = list(r["probabilities"]); p = np.array([r["probabilities"][x] for x in keys])
        z = np.log(np.clip(p, 1e-12, 1)) * k; z -= z.max(); q = np.exp(z) / np.exp(z).sum()
        pred = keys[int(q.argmax())]
        rr.append({**r, "probabilities": dict(zip(keys, q.tolist())), "predicted": pred,
                   "correct": pred == r["expected"], "confidence": float(q.max())})
    c = coverage_curve(rr); g = [r for r in rr if r["confidence"] >= 0.85]
    out.append([name, round(T, 4), k, round(T / k, 4), round(np.mean([r["correct"] for r in rr]), 4),
                round(ece([r["confidence"] for r in rr], [r["correct"] for r in rr]), 4),
                round(len(g) / len(rr), 4), round(np.mean([r["correct"] for r in g]), 4) if g else None,
                round(coverage_at_error(c, 0.02), 4), round(coverage_at_error(c, 0.05), 4)])
with open("results/temperature.csv", "w", newline="") as f:
    csv.writer(f).writerows(out)
for row in out: print(*row, sep="\t")
