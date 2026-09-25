"""E7: fit the served temperature on an out-of-distribution partition that is NOT the
reporting suite (OpenDecision's typesafe_public choice cases), then report what that T
does on the already-logged OD-500 run. OD-500 is never touched during fitting.

    python fit_temperature.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tinyjev
from summarize import load, coverage_curve, coverage_at_error, ece

HERE = Path(__file__).resolve().parent
fit_cases = [json.loads(l) for l in (HERE / "suite/typesafe_public.jsonl").read_text().splitlines() if l.strip()]
fit_cases = [c for c in fit_cases if c.get("type", "choice") == "choice" and isinstance(c.get("criteria"), dict) and c.get("expected") in c["criteria"]]
agent = tinyjev.load("tinyjev-0.6b")
served_T = agent.family.temperature

# raw logits on the fitting partition
Z, Y = [], []
for c in fit_cases:
    q = {"q": {"type": "choice", "instructions": c["instructions"], "criteria": c["criteria"]}}
    out = agent.logits({"state": c["state"], "questions": q})
    rec = next(iter(out.values()))
    keys = rec["keys"]; z = np.array(rec["logits"]) * served_T   # undo the served T -> raw head logits
    Z.append(z); Y.append(keys.index(c["expected"]))


def nll(T):
    tot = 0.0
    for z, y in zip(Z, Y):
        s = z / T; s -= s.max(); p = np.exp(s) / np.exp(s).sum(); tot -= np.log(max(p[y], 1e-12))
    return tot / len(Z)


grid = np.round(np.arange(0.5, 3.01, 0.05), 2)
best_T = float(min(grid, key=nll))
print(f"fit partition: {len(fit_cases)} choice cases from typesafe_public (OOD, not the reporting suite)")
print(f"served T {served_T:.3f} -> NLL {nll(served_T):.4f}   |   fitted T {best_T:.2f} -> NLL {nll(best_T):.4f}   |   T=1.0 -> NLL {nll(1.0):.4f}")

# apply to the logged OD-500 run offline (log p * served_T recovers raw logits up to a constant)
meta, rows = load(HERE / "results/tinyjev-0.6b.jsonl")
def retemp(T):
    rr = []
    for r in rows:
        keys = list(r["probabilities"]); p = np.array([r["probabilities"][k] for k in keys])
        z = np.log(np.clip(p, 1e-12, 1)) * served_T / T; z -= z.max(); q = np.exp(z) / np.exp(z).sum()
        pred = keys[int(q.argmax())]
        rr.append({**r, "predicted": pred, "correct": pred == r["expected"], "confidence": float(q.max()),
                   "probabilities": dict(zip(keys, q.tolist()))})
    return rr
print(f"\n{'T':>6} {'acc':>6} {'ECE':>6} {'gate85 cov':>10} {'gate85 acc':>10} {'cov@2%':>7} {'cov@5%':>7}   (OD-500, logged run, never used for fitting)")
for T in (served_T, best_T, 1.0):
    rr = retemp(T); c = coverage_curve(rr); g = [r for r in rr if r["confidence"] >= 0.85]
    print(f"{T:6.3f} {np.mean([r['correct'] for r in rr]):6.3f} {ece([r['confidence'] for r in rr], [r['correct'] for r in rr]):6.3f} "
          f"{len(g)/len(rr):10.1%} {np.mean([r['correct'] for r in g]):10.3f} {coverage_at_error(c,0.02):7.1%} {coverage_at_error(c,0.05):7.1%}")
json.dump({"fit_partition": "typesafe_public choice cases", "n_fit": len(fit_cases), "served_T": served_T, "fitted_T": best_T,
           "nll_served": nll(served_T), "nll_fitted": nll(best_T), "nll_T1": nll(1.0)}, open(HERE / "results/e7_temperature_fit.json", "w"), indent=1)
