"""E3: train an encoder typed-decision model on Kev's frozen suite, scored with Kev's own harness.

    python scripts/encoder_train.py --base answerdotai/ModernBERT-base --suite evals/v7/decision-v7 \
        --transfer evals/v4/transfer-v4 --out /runs/e3/mb-base-lr5e-5-s2 --lr 5e-5 --epochs 2 --seed 2

Architecture (Laya-style, one sequence per question, bidirectional):
    [CLS] {type} question: {instructions} [SEP] [MASK] option_0 [MASK] option_1 ... [SEP] state [SEP]
The hidden state at each [MASK] marker goes through one shared scorer (LayerNorm, Linear, GELU, Linear)
to one logit per option; softmax over the options; cross-entropy on the gold option.

Data and evaluation are Kev's, unchanged: the suite's training partition with Kev's per-epoch augmentation
(option permutation, none-of-the-above, distractors, none-pairs), the development partitions of the decision
and transfer suites scored by kev.benchmark.evaluate_records, and one temperature fitted on the calibration
partition. That keeps every number comparable with the decoder trials in benchmarks/e1 and e2.
"""
import argparse, json, math, random, sys, time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from kev.api import question_keys
from kev.benchmark import evaluate_records
from kev.data import augment, materialize, none_pair
from kev.suite import load_split, read_manifest, validate_training, write_json

OPTION_TOKEN_CAP = 48


# ----------------------------------------------------------------------------------------------------------- model
class EncoderDecisionModel(nn.Module):
    def __init__(self, base, dtype=torch.float32):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(base, dtype=dtype)
        d = self.encoder.config.hidden_size
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.normal_(self.scorer[-1].weight, std=0.02); nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask):
        """input_ids [B,L]; marker_pos [B,K] positions of the [MASK] markers; marker_mask [B,K] validity.
        Returns logits [B,K] with invalid slots at -1e4."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        z = self.scorer(m.float()).squeeze(-1)
        return z.masked_fill(~marker_mask, -1e4)


# ----------------------------------------------------------------------------------------------------------- encoding
def build_sequence(tok, qtype, instr, options, state, max_len):
    """Laya's layout. The question + options share a head budget of max_len // 2 (so the state always keeps at
    least half the window): the question text gets at most a quarter of it, and each option gets an equal share
    of the rest, capped at OPTION_TOKEN_CAP. The state is right-truncated to whatever remains."""
    cls, sep, mask = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id
    head_budget = max_len // 2
    q_ids = tok(f"{qtype} question: {instr}", add_special_tokens=False)["input_ids"][:max(8, head_budget // 4)]
    per = max(2, min(OPTION_TOKEN_CAP, (head_budget - len(q_ids) - 3) // max(1, len(options)) - 1))
    ids = [cls] + q_ids + [sep]
    markers = []
    for o in options:
        markers.append(len(ids))
        ids += [mask] + tok(" " + str(o), add_special_tokens=False)["input_ids"][:per]
    ids.append(sep)
    room = max(0, max_len - len(ids) - 1)
    ids += tok(str(state), add_special_tokens=False)["input_ids"][:room] + [sep]
    if len(ids) > max_len:
        raise ValueError(f"{len(options)} options do not fit the head budget of max_len={max_len}")
    return ids, markers


def questions_of(req, rec):
    """Pair Kev's internal record questions with the request's declared types (materialize keeps order)."""
    out = []
    for (qid, q_req), q in zip(req["questions"].items(), rec["questions"]):
        out.append({"qid": qid, "type": q_req["type"], "instr": q["instr"], "options": q["options"],
                    "label": q.get("label"), "src": q.get("src"), "keys": question_keys(q_req["type"], q_req.get("criteria"))})
    return out


def collate(tok, items, max_len, device):
    seqs, markers = zip(*[build_sequence(tok, it["type"], it["instr"], it["options"], it["state"], max_len) for it in items])
    L = max(len(s) for s in seqs); K = max(len(m) for m in markers)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    pos = torch.zeros((len(seqs), K), dtype=torch.long); valid = torch.zeros((len(seqs), K), dtype=torch.bool)
    for i, (s, m) in enumerate(zip(seqs, markers)):
        ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        pos[i, :len(m)] = torch.tensor(m); valid[i, :len(m)] = True
    return ids.to(device), att.to(device), pos.to(device), valid.to(device)


# ----------------------------------------------------------------------------------------------------------- predictor
class EncoderPredictor:
    """kev.benchmark predictor: record -> {"probabilities": {qid: {key: p}}, "latency_ms", "input_tokens"}."""

    def __init__(self, model, tok, device, max_len, temperature=1.0):
        self.model, self.tok, self.device, self.max_len, self.temperature = model, tok, device, max_len, temperature

    @torch.no_grad()
    def __call__(self, record):
        rec = materialize(record)
        items = [{**q, "state": rec["state"]} for q in questions_of(record, rec)]
        t0 = time.perf_counter()
        ids, att, pos, valid = collate(self.tok, items, self.max_len, self.device)
        z = self.model(ids, att, pos, valid).float() / self.temperature
        probs = {}
        for i, it in enumerate(items):
            p = torch.softmax(z[i, :len(it["options"])], -1).cpu().tolist()
            probs[it["qid"]] = dict(zip(it["keys"], p))
        return {"probabilities": probs, "latency_ms": 1000 * (time.perf_counter() - t0), "input_tokens": int(att.sum().item())}


def fit_temperature(model, tok, device, max_len, records):
    """One scalar temperature minimizing NLL on the calibration partition (grid over 0.5..5)."""
    zs, ys = [], []
    with torch.no_grad():
        for r in records:
            rec = materialize(r); items = [{**q, "state": rec["state"]} for q in questions_of(r, rec)]
            ids, att, pos, valid = collate(tok, items, max_len, device)
            z = model(ids, att, pos, valid).float().cpu()
            for i, it in enumerate(items):
                if it["label"] is None: continue
                zs.append(z[i, :len(it["options"])]); ys.append(int(it["label"]))
    if not zs:
        return 1.0
    best, best_nll = 1.0, float("inf")
    for T in [x / 10 for x in range(5, 51)]:
        nll = sum(F.cross_entropy((z / T)[None], torch.tensor([y])).item() for z, y in zip(zs, ys)) / len(zs)
        if nll < best_nll: best, best_nll = T, nll
    return best


# ----------------------------------------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="answerdotai/ModernBERT-base")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--transfer", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8, help="questions per optimizer step")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--p_none", type=float, default=0.1); ap.add_argument("--p_none_distract", type=float, default=0.12)
    ap.add_argument("--p_distract", type=float, default=0.15); ap.add_argument("--p_none_pair", type=float, default=0.25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap training records")
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    for t in ("cls_token_id", "sep_token_id", "mask_token_id", "pad_token_id"):
        if getattr(tok, t) is None: raise ValueError(f"tokenizer lacks {t}")
    manifest = read_manifest(a.suite)
    reqs = load_split(a.suite, "train"); validate_training(reqs, manifest)
    if a.limit: reqs = reqs[:a.limit]
    print(f"{len(reqs)} training requests; base {a.base}; device {a.device}", flush=True)

    model = EncoderDecisionModel(a.base).to(a.device)
    params = [{"params": model.encoder.parameters(), "lr": a.lr}, {"params": model.scorer.parameters(), "lr": a.head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=a.weight_decay)
    est_questions = sum(len(r["questions"]) for r in reqs) * (1 + a.p_none_pair)  # none-pairs add siblings
    total_steps = max(1, int(a.epochs * est_questions / a.batch)); warm = int(a.warmup * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (total_steps - s) / max(1, total_steps - warm)))
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if (a.dtype == "bf16" and a.device == "cuda") else torch.autocast("cpu", enabled=False)

    step, t0, seen = 0, time.time(), 0
    model.train()
    for ep in range(a.epochs):
        order = list(range(len(reqs))); random.Random(f"{a.seed}:{ep}").shuffle(order)
        pending, run = [], Counter()
        for idx in order:
            req = reqs[idx]
            rng = random.Random(f"{a.seed}:{ep}:{req['_meta']['id']}")
            variants = [augment(req, rng, p_none=a.p_none, p_none_distract=a.p_none_distract, p_distract=a.p_distract)]
            if a.p_none_pair > 0 and rng.random() < a.p_none_pair:
                variants += none_pair(req, rng)
            for v in variants:
                rec = materialize(v)
                for q in questions_of(v, rec):
                    if q["label"] is None: continue
                    pending.append({**q, "state": rec["state"]})
            while len(pending) >= a.batch:
                batch, pending = pending[:a.batch], pending[a.batch:]
                ids, att, pos, valid = collate(tok, batch, a.max_len, a.device)
                with autocast:
                    z = model(ids, att, pos, valid)
                y = torch.tensor([b["label"] for b in batch], device=a.device)
                loss = F.cross_entropy(z.float(), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1; seen += len(batch)
                run["loss"] += loss.item(); run["n"] += 1
                if step % 50 == 0:
                    print(f"ep{ep} step {step}/{total_steps} loss {run['loss']/run['n']:.3f} {(time.time()-t0)/seen:.4f}s/q", flush=True); run = Counter()
    model.eval()

    # ---- save
    ckpt = out / "checkpoint"; ckpt.mkdir(exist_ok=True)
    model.encoder.save_pretrained(ckpt); tok.save_pretrained(ckpt)
    torch.save({"scorer": model.scorer.state_dict(), "base": a.base, "max_len": a.max_len, "args": vars(a)}, ckpt / "head.pt")

    # ---- calibrate, then score with Kev's harness
    cal = load_split(a.suite, "calibration")
    T = fit_temperature(model, tok, a.device, a.max_len, cal) if cal else 1.0
    torch.save({**torch.load(ckpt / "head.pt"), "temperature": T}, ckpt / "head.pt")
    print(f"fitted temperature {T:.2f} on {len(cal)} calibration records", flush=True)
    heldout = tuple(manifest.get("holdout_sources", []))
    results = {"wall_seconds": time.time() - t0, "steps": step, "questions_seen": seen, "temperature": T, "args": vars(a)}
    for label, suite in (("development", a.suite), ("transfer", a.transfer)):
        if not suite: continue
        recs = load_split(suite, "development")
        pred = EncoderPredictor(model, tok, a.device, a.max_len, temperature=T)
        report, _ = evaluate_records(recs, pred, out / label, heldout_sources=heldout)
        results[label] = {"acc": report["clean"]["acc"], "ece": report["clean"].get("ece"), "brier": report["clean"].get("brier"),
                          "coverage_at_5pct_error": report["clean"].get("coverage_at_5pct_error"), "n": report["clean"].get("n"),
                          "sources": {k: v.get("acc") for k, v in (report.get("sources") or report.get("tasks") or {}).items() if isinstance(v, dict)}}
        print(f"{label}: acc {report['clean']['acc']:.4f} ece {report['clean'].get('ece')}", flush=True)
    write_json(out / "result.json", results)
    print(json.dumps({k: v for k, v in results.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
