"""Compare tinyjev's Kev family against probabilities from Kev's own PyTorch code."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tinyjev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out")
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    agent = tinyjev.load(args.model, backend=args.backend, device=args.device)
    agree = total = 0
    worst, rows = 0.0, []
    for case in ref["fixtures"]:
        got = agent.logits(case["request"])
        for qid, exp in case["expected"].items():
            total += 1
            mine = got[f"request:{qid}"]
            assert mine["candidate_ids"] == exp["keys"], (mine["candidate_ids"], exp["keys"])
            delta = max(abs(a - b) for a, b in zip(exp["probabilities"], mine["probabilities"]))
            worst = max(worst, delta)
            ref_pick = max(range(len(exp["probabilities"])), key=exp["probabilities"].__getitem__)
            my_pick = max(range(len(mine["probabilities"])), key=mine["probabilities"].__getitem__)
            ok = ref_pick == my_pick
            agree += ok
            rows.append({"case": case["name"], "question": qid, "match": ok, "max_prob_delta": delta})
            print(f"  {'ok ' if ok else 'FAIL'} {case['name']:<28} {qid:<12} dmax={delta:.2e}")
    print(f"\nkev parity [{agent.backend}]: answers match {agree}/{total}   worst probability delta: {worst:.3e}")
    if args.out:
        Path(args.out).write_text(json.dumps({"backend": agent.backend, "matched": agree, "total": total,
                                              "worst_prob_delta": worst, "rows": rows}, indent=2))
    return 0 if agree == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
