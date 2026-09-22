"""Soft targets from a trained Kev checkpoint, in kev.anchors' JSON shape, for `kev.train --anchor`.

    python tools/kev_teacher_targets.py --run jaredpalmer/kev-4b --suite <kev>/evals/v7/decision-v7 \
        --out runs/targets/v7-kev4b.json [--split train] [--device cuda]

Why not kev.anchors: it prompts a *base* LM with option letters (max 26 options). This reads the
teacher's pointer head directly, at temperature 1.0 (raw logits; Kev's serving temperature is
bypassed by LoadOptions(temperature=1.0)), so the student sees the teacher's actual distribution.

Alignment contract (Codex review, docs/research/RECIPE.md): targets are keyed by record id,
question id and option KEY. kev.train's anchor_loss aligns by key under any option permutation and
skips questions whose option set changed (none-option inserted). We do not re-query the teacher
under permutations; that choice is recorded in _meta. Boolean keys are ["false","true"], score
keys are level indices as strings, exactly as kev.api.question_keys.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="teacher checkpoint dir or Hub id, e.g. jaredpalmer/kev-4b")
    ap.add_argument("--suite", required=True, help="frozen suite dir (its train partition is what the student sees)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    from kev.api import question_keys
    from kev.checkpoint import load, LoadOptions
    from kev.data import materialize
    from kev.device import default_device
    from kev.suite import load_split, write_json

    device = a.device or default_device()
    tok, model = load(a.run, device, LoadOptions(temperature=1.0))
    model.eval()
    records = load_split(a.suite, a.split)
    if a.limit:
        records = records[: a.limit]

    targets, skipped, t0 = {}, 0, time.perf_counter()
    with torch.no_grad():
        for i, r in enumerate(records):
            rec = materialize(r)
            try:
                enc = model.encode(tok, rec, max_state=8192, max_branch=8192)
            except ValueError:
                skipped += 1
                continue
            probs = model.probs(enc)
            for (qid, q), p in zip(r["questions"].items(), probs):
                keys = question_keys(q["type"], q.get("criteria"))
                p = [float(x) for x in p]
                if len(keys) != len(p):
                    skipped += 1
                    continue
                targets.setdefault(r["_meta"]["id"], {})[qid] = dict(zip(keys, p))
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(records)}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    meta = {"teacher": a.run, "resolved": str(getattr(model, "run", a.run)), "suite": str(a.suite), "split": a.split,
            "temperature": 1.0, "permutation_policy": "align by option key at train time; teacher not re-queried",
            "records": len(targets), "questions": sum(len(v) for v in targets.values()), "skipped": skipped,
            "head_sha256": hashlib.sha256(Path(str(getattr(model, "run", ""))).joinpath("head.pt").read_bytes()).hexdigest()
            if Path(str(getattr(model, "run", ""))).joinpath("head.pt").exists() else None}
    write_json(a.out, {"_meta": meta, "targets": targets})
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    sys.exit(main())
