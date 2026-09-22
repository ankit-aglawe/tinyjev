"""Compare the MLX port against PyTorch reference fixtures."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nanojev_mlx


def answers_match(kind, a, b, score_tol):
    if kind == "score":
        return abs(float(a) - float(b)) <= score_tol
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out")
    ap.add_argument("--score-tol", type=float, default=1e-2)
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    agent = nanojev_mlx.load(args.model)

    rows, agree, total, worst = [], 0, 0, 0.0
    for case in ref["fixtures"]:
        got = agent.logits(case["payload"])
        for key, expected in case["expected"].items():
            total += 1
            actual = got[key]
            delta = max(abs(p - q) for p, q in
                        zip(expected["probabilities"], actual["probabilities"]))
            worst = max(worst, delta)
            ok = answers_match(expected["type"], expected["answer"]["value"],
                               actual["answer"]["value"], args.score_tol)
            agree += ok
            rows.append({"case": case["name"], "question": key, "type": expected["type"],
                         "match": ok, "max_prob_delta": delta,
                         "expected": expected["answer"]["value"],
                         "actual": actual["answer"]["value"]})
            flag = "ok " if ok else "FAIL"
            print(f"  {flag} {key:<24} {expected['type']:<8} dmax={delta:.2e}"
                  + ("" if ok else f"  {expected['answer']['value']!r} != {actual['answer']['value']!r}"))

    print(f"\nanswers match: {agree}/{total}   worst probability delta: {worst:.3e}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"reference_device": ref.get("device"), "matched": agree, "total": total,
             "worst_prob_delta": worst, "score_tol": args.score_tol, "rows": rows}, indent=2))
    return 0 if agree == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
