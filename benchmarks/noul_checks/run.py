"""Statement-form yes/no checks: the E13 gate. Scores a tinyjev checkpoint on every noul set we hold,
reporting accuracy, the share of "yes" answers and accuracy on the false-labelled items, which is
where the 0.6B fails (it answers yes to nearly everything).

Sets (all local, none used for training the shipped 0.6B):
  support_email   21 authored checks on one email        tinyjev-research/experiments/noul-bias/support_email.json
  insurance_claim  8 TypeSafe public checks on one claim  tinyjev-research/experiments/noul-bias/insurance_claim.json
  typesafe_noul   20 TypeSafe public noul cases           benchmarks/opendecision/suite/typesafe_public.jsonl
  statements_dev  230 decision-v7 dev questions rewritten as statements   tinyjev-research/experiments/e13/statements_development.jsonl
  hardneg_dev     hard negatives, 2 per state              tinyjev-research/experiments/e13/hardneg_development.jsonl

    python3 benchmarks/noul_checks/run.py --model tinyjev-0.6b
    python3 benchmarks/noul_checks/run.py --model ~/.cache/tinyjev/v2/tinyjev-e11-4b --quantize 8 --name tinyjev-e11-4b-int8
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESEARCH = ROOT.parent / "tinyjev-research"
sys.path.insert(0, str(ROOT))
import tinyjev  # noqa: E402


def checklist(path):
    c = json.loads(Path(path).read_text())
    return [{"state": c["state"], "questions": {x["id"]: {"text": x["text"], "label": bool(x["expected"])} for x in c["checks"]}}]


def typesafe_noul(path):
    recs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("type") == "noul":
            recs.append({"state": r["state"], "questions": {r["id"]: {"text": r["instructions"], "label": bool(r["expected"])}}})
    return recs


def kev_records(path):
    recs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        recs.append({"state": r["state"], "questions": {q: {"text": v["instructions"], "label": bool(v["label"])} for q, v in r["questions"].items() if v.get("type") == "noul"}})
    return recs


SETS = {
    "support_email": lambda: checklist(RESEARCH / "experiments/noul-bias/support_email.json"),
    "insurance_claim": lambda: checklist(RESEARCH / "experiments/noul-bias/insurance_claim.json"),
    "typesafe_noul": lambda: typesafe_noul(ROOT / "benchmarks/opendecision/suite/typesafe_public.jsonl"),
    "statements_dev": lambda: kev_records(RESEARCH / "experiments/e13/statements_development.jsonl"),
    "hardneg_dev": lambda: kev_records(RESEARCH / "experiments/e13/hardneg_development.jsonl"),
}


def score(agent, recs):
    n = ok = yes = n_false = ok_false = 0
    rows = []
    for r in recs:
        qs = {qid: {"type": "noul", "instructions": q["text"]} for qid, q in r["questions"].items()}
        if not qs: continue
        ans = agent.predict({"state": r["state"], "questions": qs})["states"][0]["answers"]
        for qid, q in r["questions"].items():
            p = float(ans[qid]["p_true"]); pred = p >= 0.5
            n += 1; ok += pred == q["label"]; yes += pred
            if not q["label"]:
                n_false += 1; ok_false += not pred
            rows.append({"text": q["text"], "label": q["label"], "p_true": round(p, 4)})
    return {"n": n, "acc": round(ok / n, 4) if n else None, "yes_rate": round(yes / n, 4) if n else None,
            "false_items": n_false, "acc_on_false": round(ok_false / n_false, 4) if n_false else None}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyJev-0.6B")
    ap.add_argument("--name", default="")
    ap.add_argument("--quantize", type=int, default=0)
    ap.add_argument("--sets", default=",".join(SETS))
    a = ap.parse_args()
    agent = tinyjev.load(a.model, quantize=a.quantize)
    name = a.name or Path(a.model).name
    out = {"model": name, "checkpoint": str(agent.root), "quantize": a.quantize or None, "measured": time.strftime("%Y-%m-%d"), "sets": {}}
    rows_all = {}
    for s in a.sets.split(","):
        try:
            recs = SETS[s]()
        except FileNotFoundError as e:
            print(f"{s}: missing ({e})"); continue
        t0 = time.perf_counter(); summary, rows = score(agent, recs)
        summary["seconds"] = round(time.perf_counter() - t0, 1)
        out["sets"][s] = summary; rows_all[s] = rows
        print(f"{s:16s} n={summary['n']:4d} acc={summary['acc']}  yes_rate={summary['yes_rate']}  acc_on_false={summary['acc_on_false']} ({summary['false_items']} false)  {summary['seconds']}s", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{name}.json").write_text(json.dumps({**out, "rows": rows_all}, indent=1, ensure_ascii=False))
    print("->", HERE / "results" / f"{name}.json")


if __name__ == "__main__":
    main()
