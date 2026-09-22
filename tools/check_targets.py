"""Sanity-check teacher soft targets before a distillation run (Codex's alignment contract).

    .venv/bin/python tools/check_targets.py --targets runs/targets/v7-kev4b.json --suite evals/v7/decision-v7

Reports, per source: coverage (questions with a target), mean teacher entropy (nats) vs max, mean
probability the teacher puts on the gold label, teacher argmax accuracy, and the share of
near-one-hot targets (max p > 0.95). If targets are nearly all one-hot and correct, KD only
re-weights CE and a null result carries no information.
"""
import argparse, collections, json, math, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    from kev.api import question_keys
    from kev.suite import load_split

    tg = json.load(open(a.targets))
    meta, targets = tg.get("_meta", {}), tg["targets"]
    records = load_split(a.suite, a.split)
    per = collections.defaultdict(lambda: collections.Counter())
    ent, gold_p, onehot, correct = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(int), collections.defaultdict(int)
    for r in records:
        rid = r["_meta"]["id"]
        for qid, q in r["questions"].items():
            src = q.get("src", "?")
            per[src]["questions"] += 1
            t = targets.get(rid, {}).get(qid)
            if not t:
                continue
            keys = question_keys(q["type"], q.get("criteria"))
            if set(t) != set(keys):
                per[src]["key_mismatch"] += 1
                continue
            per[src]["covered"] += 1
            p = [max(1e-12, float(t[k])) for k in keys]
            s = sum(p); p = [x / s for x in p]
            ent[src].append(-sum(x * math.log(x) for x in p) / math.log(len(p)) if len(p) > 1 else 0.0)
            lab = q.get("label")
            gold = str(lab).lower() if q["type"] == "noul" else (str(lab) if q["type"] == "score" else lab)
            if gold in keys:
                gi = keys.index(gold)
                gold_p[src].append(p[gi])
                correct[src] += int(max(range(len(p)), key=p.__getitem__) == gi)
            onehot[src] += int(max(p) > 0.95)
    print("teacher:", meta.get("teacher"), "| resolved:", meta.get("resolved"), "| questions:", meta.get("questions"), "| skipped:", meta.get("skipped"))
    print(f"{'source':<28}{'q':>6}{'covered':>9}{'mismatch':>9}{'H/Hmax':>8}{'p(gold)':>9}{'t.acc':>7}{'onehot':>8}")
    tot = collections.Counter()
    for src in sorted(per):
        c = per[src]; n = max(1, c["covered"])
        print(f"{src:<28}{c['questions']:>6}{c['covered']:>9}{c['key_mismatch']:>9}{(sum(ent[src])/n):>8.3f}{(sum(gold_p[src])/max(1,len(gold_p[src]))):>9.3f}{(correct[src]/max(1,len(gold_p[src]))):>7.3f}{(onehot[src]/n):>8.2f}")
        tot.update(c)
    print(f"{'ALL':<28}{tot['questions']:>6}{tot['covered']:>9}{tot['key_mismatch']:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
