"""The cascade: the local model answers every case it is at least `gate` sure of, and a
frontier model answers the rest. Computed entirely from two logged runs on the same 500
cases, so it is an exact count, not a simulation. Writes results/cascade.csv.

    python cascade.py                # tinyjev-0.6b in front of claude-opus-5-5
"""
import csv, json, sys
from pathlib import Path

local_name = sys.argv[1] if len(sys.argv) > 1 else "tinyjev-0.6b"
big_name = sys.argv[2] if len(sys.argv) > 2 else "claude-opus-5-5"


def load(n):
    rows = [json.loads(l) for l in (Path("results") / f"{n}.jsonl").read_text().splitlines() if l.strip()]
    return {r["id"]: r for r in rows if "id" in r}


local, big = load(local_name), load(big_name)
ids = sorted(set(local) & set(big))
n = len(ids)
big_only = sum(big[i]["correct"] for i in ids) / n
out = [["gate", "local_share", "sent_to_big", "cascade_accuracy", "big_only_accuracy", "delta_vs_big", "local_only_accuracy_on_its_share"]]
gates = sorted({round(local[i]["confidence"], 3) for i in ids} | {0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99})
for g in gates:
    kept = [i for i in ids if local[i]["confidence"] >= g]
    rest = [i for i in ids if local[i]["confidence"] < g]
    acc = (sum(local[i]["correct"] for i in kept) + sum(big[i]["correct"] for i in rest)) / n
    loc_acc = sum(local[i]["correct"] for i in kept) / len(kept) if kept else float("nan")
    out.append([g, round(len(kept) / n, 4), round(len(rest) / n, 4), round(acc, 4), round(big_only, 4), round(acc - big_only, 4), round(loc_acc, 4)])
Path("results/cascade.csv").write_text("\n".join(",".join(map(str, r)) for r in out) + "\n")
for r in out[0:1] + [r for r in out[1:] if r[0] in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)]:
    print(*r, sep="\t")
