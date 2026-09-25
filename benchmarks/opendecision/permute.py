"""E9 measurement: how often does tinyjev's answer change when the options are shuffled?

Every OD-500 case is run in its canonical option order and in K random permutations of the
same options (seeded). A flip is a permutation whose argmax differs from the canonical argmax.
Nothing is trained here; this is the number the E9 arm has to beat.

    python3 benchmarks/opendecision/permute.py --k 2
"""
import argparse, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import load_cases
import tinyjev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinyjev-0.6b")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    agent = tinyjev.load(a.model)
    rng = random.Random(a.seed)
    cases = load_cases()[: a.limit or None]
    flips = perms = 0; canon_ok = perm_ok = 0; hist = {}
    rows = []
    t0 = time.perf_counter()
    for c in cases:
        def run(criteria):
            q = {"type": "choice", "instructions": c["instructions"], "criteria": criteria}
            return agent.predict({"state": c["state"], "questions": {"q": q}})["states"][0]["answers"]["q"]
        base = run(c["criteria"]); canon_ok += base["choice"] == c["expected"]
        keys = list(c["criteria"]); fl = 0
        for _ in range(a.k):
            order = keys[:]; rng.shuffle(order)
            r = run({k: c["criteria"][k] for k in order})
            perms += 1; perm_ok += r["choice"] == c["expected"]
            if r["choice"] != base["choice"]:
                flips += 1; fl += 1
        hist[len(keys)] = hist.get(len(keys), [0, 0]); hist[len(keys)][0] += fl; hist[len(keys)][1] += a.k
        rows.append({"id": c["id"], "domain": c.get("domain"), "n_options": len(keys), "canonical": base["choice"],
                     "canonical_confidence": base["confidence"], "flips": fl, "expected": c["expected"]})
    out = {"model": a.model, "cases": len(cases), "k": a.k, "seed": a.seed,
           "flip_rate": round(flips / perms, 4), "flips": flips, "permutations": perms,
           "accuracy_canonical": round(canon_ok / len(cases), 4), "accuracy_permuted": round(perm_ok / perms, 4),
           "flip_rate_by_n_options": {str(n): round(v[0] / v[1], 4) for n, v in sorted(hist.items())},
           "cases_with_any_flip": sum(r["flips"] > 0 for r in rows),
           "mean_confidence_flipped": round(sum(r["canonical_confidence"] for r in rows if r["flips"]) / max(1, sum(r["flips"] > 0 for r in rows)), 4),
           "mean_confidence_stable": round(sum(r["canonical_confidence"] for r in rows if not r["flips"]) / max(1, sum(r["flips"] == 0 for r in rows)), 4),
           "seconds": round(time.perf_counter() - t0, 1), "rows": rows}
    Path("benchmarks/opendecision/results").mkdir(exist_ok=True)
    Path(f"benchmarks/opendecision/results/permute-{a.model}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
