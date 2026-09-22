"""Parity against upstream's OWN published predictions on its full test split.

Joins `unified/hard/test.jsonl` (inputs, dataset repo) with `predictions_test.jsonl`
(the authors' CUDA logits, model repo) by question id, runs every question through
the MLX port, and compares the selected answer and the probability vectors.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import hf_hub_download
import nanojev_mlx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    inputs = [json.loads(l) for l in open(hf_hub_download(
        "C-Tianyu/NanoJev-Data", "unified/hard/test.jsonl", repo_type="dataset"))]
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(hf_hub_download(
        "C-Tianyu/NanoJev", "predictions_test.jsonl"))}
    if args.limit:
        inputs = inputs[:args.limit]

    agent = nanojev_mlx.load(args.model)
    agree = total = 0
    deltas, by_family, times = [], {}, []
    started = time.perf_counter()
    for i, row in enumerate(inputs):
        payload = {"states": [{"id": row["id"], "state": row["state"], "questions": row["questions"]}]}
        t = time.perf_counter()
        got = agent.logits(payload)
        times.append((time.perf_counter() - t) * 1000)
        for qid in row["questions"]:
            key = f"{row['id']}:{qid}"
            ref = preds.get(key)
            if ref is None:
                continue
            mine = got[key]
            # align on candidate id order (upstream shuffles per record)
            order = {c: j for j, c in enumerate(mine["candidate_ids"])}
            ref_probs = [None] * len(order)
            for c, p in zip(ref["candidate_ids"], ref["student_probs"]):
                ref_probs[order[c]] = p
            ref_pick = ref["candidate_ids"][max(range(len(ref["student_probs"])),
                                              key=ref["student_probs"].__getitem__)]
            same = ref_pick == mine["answer"]["value"]
            delta = max(abs(a - b) for a, b in zip(ref_probs, mine["probabilities"]))
            fam = by_family.setdefault(row["family_id"], {"agree": 0, "total": 0, "deltas": []})
            fam["agree"] += same; fam["total"] += 1; fam["deltas"].append(delta)
            agree += same; total += 1; deltas.append(delta)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(inputs)}  agree {agree}/{total}  "
                  f"median delta {statistics.median(deltas):.2e}  "
                  f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)

    report = {
        "questions": total, "selected_answer_matches": agree,
        "match_rate": round(agree / total, 5),
        "prob_delta_median": statistics.median(deltas),
        "prob_delta_p99": sorted(deltas)[int(0.99 * (len(deltas) - 1))],
        "prob_delta_max": max(deltas),
        "by_family": {k: {"agree": v["agree"], "total": v["total"],
                          "delta_median": statistics.median(v["deltas"]),
                          "delta_max": max(v["deltas"])} for k, v in by_family.items()},
        "latency_ms_median": round(statistics.median(times), 1),
        "upstream_reference": "C-Tianyu/NanoJev predictions_test.jsonl (authors' CUDA bf16 run)",
        "port_dtype": agent.config.get("body_dtype"),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
