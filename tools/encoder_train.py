"""E3: train an encoder typed-decision model on Kev's frozen suite, scored with Kev's own harness.

    python scripts/encoder_train.py --base answerdotai/ModernBERT-base --suite evals/v7/decision-v7 \
        --transfer evals/v4/transfer-v4 --out /runs/e3/mb-base-lr5e-5-s2 --lr 5e-5 --epochs 2 --seed 2

Architecture (Laya-style, one sequence per question, bidirectional):
    [CLS] {type} question: {instructions} [SEP] [MASK] option_0 [MASK] option_1 ... [SEP] state [SEP]
The hidden state at each [MASK] marker goes through one shared scorer (LayerNorm, Linear, GELU, Linear) to one
logit per option; softmax over the options; cross-entropy on the gold option.

Everything else mirrors kev.train so the number is comparable with benchmarks/e1 and e2: the suite's training
partition with Kev's per-epoch augmentation (augment, none_pair, materialize) seeded with kev's source_seed; batches of
`--batch` REQUESTS whose variants' question losses are averaged within each variant and summed across variants; one
optimizer step per request batch (partial last batch kept); OneCycleLR with pct_start 0.1 over the exact step count;
fp32 master weights with bf16 autocast on CUDA, scorer in fp32. Evaluation: kev.benchmark.evaluate_records on the
development partitions with RAW probabilities in `clean` and the calibration-partition temperature applied by the
harness into `calibrated_clean`, exactly as the decoder trials report. No silent truncation: a length census runs
first; training questions that do not fit are dropped and counted, evaluation records that do not fit abort the run.
"""
import argparse, hashlib, json, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from kev.api import question_keys
from kev.benchmark import evaluate_records
from kev.data import augment, materialize, none_pair
from kev.suite import load_split, read_manifest, validate_training, write_json
try:
    from kev.data import source_seed
except Exception:  # pragma: no cover
    def source_seed(seed, tag):
        return int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------------------------------------- model
class EncoderDecisionModel(nn.Module):
    def __init__(self, base, revision=None):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(base, revision=revision, dtype=torch.float32)
        d = self.encoder.config.hidden_size
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1, bias=False))
        nn.init.normal_(self.scorer[-1].weight, std=0.02)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask):
        """input_ids [B,L]; marker_pos [B,K] positions of the [MASK] markers; marker_mask [B,K] validity.
        Returns fp32 logits [B,K] with invalid slots at -1e4. The scorer always runs in fp32."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx).float()
        with torch.autocast(device_type=h.device.type, enabled=False):
            z = self.scorer(m).squeeze(-1)
        return z.masked_fill(~marker_mask, -1e4)


# ----------------------------------------------------------------------------------------------------------- encoding
def build_sequence(tok, qtype, instr, options, state, max_len):
    """Full instruction, full options, full state. Raises if the sequence does not fit; never truncates."""
    cls, sep, mask = tok.cls_token_id, tok.sep_token_id, tok.mask_token_id
    ids = [cls] + tok(f"{qtype} question: {instr}", add_special_tokens=False)["input_ids"] + [sep]
    markers = []
    for o in options:
        markers.append(len(ids))
        ids += [mask] + tok(" " + str(o), add_special_tokens=False)["input_ids"]
    ids.append(sep)
    ids += tok(str(state), add_special_tokens=False)["input_ids"] + [sep]
    if len(ids) > max_len:
        raise ValueError(f"sequence of {len(ids)} tokens exceeds max_len={max_len} ({len(options)} options)")
    return ids, markers


def questions_of(req, rec):
    """Kev's internal record questions paired with the request's declared type and keys; order is checked."""
    out = []
    if len(req["questions"]) != len(rec["questions"]):
        raise ValueError("materialize changed the number of questions")
    for (qid, q_req), q in zip(req["questions"].items(), rec["questions"]):
        keys = question_keys(q_req["type"], q_req.get("criteria"))
        if len(keys) != len(q["options"]):
            raise ValueError(f"{qid}: {len(keys)} keys vs {len(q['options'])} options")
        out.append({"qid": qid, "type": q_req["type"], "instr": q["instr"], "options": q["options"],
                    "label": q.get("label"), "src": q.get("src"), "keys": keys})
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
    """kev.benchmark predictor. RAW probabilities and logits; inference_temperature 1.0. The harness applies the
    fitted temperature into `calibrated_clean`, as it does for the decoder trials."""

    def __init__(self, model, tok, device, max_len):
        self.model, self.tok, self.device, self.max_len = model, tok, device, max_len
        self.temperature = 1.0

    @torch.no_grad()
    def __call__(self, record):
        rec = materialize(record)
        items = [{**q, "state": rec["state"]} for q in questions_of(record, rec)]
        t0 = time.perf_counter()
        ids, att, pos, valid = collate(self.tok, items, self.max_len, self.device)
        z = self.model(ids, att, pos, valid).float().cpu()
        probs, logits = {}, {}
        for i, it in enumerate(items):
            zi = z[i, :len(it["options"])]
            probs[it["qid"]] = dict(zip(it["keys"], torch.softmax(zi, -1).tolist()))
            logits[it["qid"]] = dict(zip(it["keys"], zi.tolist()))
        return {"probabilities": probs, "logits": logits, "inference_temperature": 1.0,
                "latency_ms": 1000 * (time.perf_counter() - t0), "input_tokens": int(att.sum().item())}


def fit_temperature(model, tok, device, max_len, records):
    """One scalar temperature minimizing micro-averaged NLL on the calibration partition (grid 0.3..6 step 0.05)."""
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
        return 1.0, 0
    Z = torch.nn.utils.rnn.pad_sequence(zs, batch_first=True, padding_value=-1e4); Y = torch.tensor(ys)
    grid = [round(0.3 + 0.05 * i, 2) for i in range(int((6 - 0.3) / 0.05) + 1)]
    nll = {T: F.cross_entropy(Z / T, Y).item() for T in grid}
    T = min(nll, key=nll.get)
    if T in (grid[0], grid[-1]): print(f"warning: temperature hit the grid boundary ({T})", flush=True)
    return T, len(ys)


# ----------------------------------------------------------------------------------------------------------- training
def expand(req, ep, seed, a):
    """Kev's per-epoch variants of one request: the augmented request, plus none-pair siblings sometimes."""
    rng = random.Random(source_seed(seed, f"{ep}:{req['_meta']['id']}"))
    variants = [augment(req, rng, p_none=a.p_none, p_none_distract=a.p_none_distract, p_distract=a.p_distract)]
    if a.p_none_pair > 0 and rng.random() < a.p_none_pair:
        variants += none_pair(req, rng)
    return variants


def census(tok, reqs, max_len, label, a=None, seed=0):
    """Full-sequence lengths per source (and the training variants, if `a` is given). Returns questions that do not fit."""
    longest, overflow, kmax = defaultdict(int), Counter(), 0
    for req in reqs:
        variants = expand(req, 0, seed, a) if a else [req]
        for v in variants:
            rec = materialize(v)
            for q in questions_of(v, rec):
                try:
                    ids, _ = build_sequence(tok, q["type"], q["instr"], q["options"], rec["state"], 10 ** 9)
                except ValueError:
                    continue
                src = req["_meta"].get("source", q["src"]); longest[src] = max(longest[src], len(ids)); kmax = max(kmax, len(q["options"]))
                if len(ids) > max_len: overflow[src] += 1
    top = sorted(longest.items(), key=lambda kv: -kv[1])[:6]
    print(f"census {label}: max K {kmax}; longest by source {top}; over max_len={max_len}: {dict(overflow) or 'none'}", flush=True)
    return overflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="answerdotai/ModernBERT-base")
    ap.add_argument("--base_revision", default=None)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--transfer", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head_lr", type=float, default=0.0, help="0 = same as --lr (Kev's default)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8, help="REQUESTS per optimizer step, as kev.train")
    ap.add_argument("--microbatch", type=int, default=16, help="questions per encoder forward (memory only)")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--p_none", type=float, default=0.1); ap.add_argument("--p_none_distract", type=float, default=0.12)
    ap.add_argument("--p_distract", type=float, default=0.15); ap.add_argument("--p_none_pair", type=float, default=0.25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap training records")
    a = ap.parse_args()
    dev_type = torch.device(a.device).type
    head_lr = a.head_lr or a.lr

    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; runs are immutable")
    out.mkdir(parents=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    from transformers import AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.base_revision)
    for t in ("cls_token_id", "sep_token_id", "mask_token_id", "pad_token_id"):
        if getattr(tok, t) is None: raise ValueError(f"tokenizer lacks {t}")

    manifest = read_manifest(a.suite)
    reqs = load_split(a.suite, "train"); validate_training(reqs, manifest)
    if a.limit: reqs = reqs[:a.limit]
    dev_recs = load_split(a.suite, "development")
    transfer_recs = load_split(a.transfer, "development") if a.transfer else []
    transfer_manifest = read_manifest(a.transfer) if a.transfer else {}
    cal = load_split(a.suite, "calibration")

    # ---- length census: no silent truncation anywhere
    t_census = time.time()
    train_over = census(tok, reqs, a.max_len, "train+variants", a, a.seed)
    if census(tok, dev_recs, a.max_len, "decision dev"):
        raise ValueError("evaluation records exceed max_len; raise --max_len")
    if transfer_recs and census(tok, transfer_recs, a.max_len, "transfer dev"):
        raise ValueError("transfer records exceed max_len; raise --max_len")
    print(f"census took {time.time() - t_census:.0f}s; training questions over max_len will be dropped: {sum(train_over.values())}", flush=True)

    model = EncoderDecisionModel(a.base, a.base_revision).to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": a.lr},
                             {"params": model.scorer.parameters(), "lr": head_lr}], weight_decay=a.weight_decay)
    steps_per_epoch = (len(reqs) + a.batch - 1) // a.batch
    total_steps = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, head_lr], total_steps=total_steps, pct_start=0.1)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if (a.dtype == "bf16" and dev_type == "cuda") else torch.autocast("cpu", enabled=False)
    print(f"{len(reqs)} training requests; {total_steps} optimizer steps; base {a.base} ({n_params/1e6:.1f}M params); device {a.device} {a.dtype}", flush=True)

    step, t0, seen_q, dropped, run = 0, time.time(), 0, 0, Counter()
    model.train()
    for ep in range(a.epochs):
        order = list(range(len(reqs))); random.Random(source_seed(a.seed, f"order:{ep}")).shuffle(order)
        for b in range(0, len(order), a.batch):
            # one optimizer step = one batch of requests; loss = sum over variants of mean question CE (kev.train.batch_loss)
            groups = []   # (variant index, question items)
            for idx in order[b:b + a.batch]:
                for v in expand(reqs[idx], ep, a.seed, a):
                    rec = materialize(v)
                    items = []
                    for q in questions_of(v, rec):
                        if q["label"] is None: continue
                        try:
                            build_sequence(tok, q["type"], q["instr"], q["options"], rec["state"], a.max_len)
                        except ValueError:
                            dropped += 1; continue
                        items.append({**q, "state": rec["state"]})
                    if items: groups.append(items)
            flat = [(gi, it) for gi, items in enumerate(groups) for it in items]
            if not flat: continue
            weights = torch.tensor([1.0 / len(groups[gi]) for gi, _ in flat], device=a.device)   # mean within variant, sum across
            loss_total = 0.0
            for m0 in range(0, len(flat), a.microbatch):
                chunk = flat[m0:m0 + a.microbatch]
                ids, att, pos, valid = collate(tok, [it for _, it in chunk], a.max_len, a.device)
                with autocast:
                    z = model(ids, att, pos, valid)
                y = torch.tensor([it["label"] for _, it in chunk], device=a.device)
                per_q = F.cross_entropy(z.float(), y, reduction="none")
                loss = (per_q * weights[m0:m0 + len(chunk)]).sum()
                loss.backward(); loss_total += loss.item(); seen_q += len(chunk)
            if not torch.isfinite(torch.tensor(loss_total)): raise ValueError("non-finite training loss")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            run["loss"] += loss_total / len(groups); run["n"] += 1
            if step % 50 == 0 or step == total_steps:
                print(f"ep{ep} step {step}/{total_steps} loss {run['loss']/run['n']:.3f} {(time.time()-t0)/max(1,seen_q):.4f}s/q dropped {dropped}", flush=True); run = Counter()
    train_seconds = time.time() - t0
    model.eval()

    # ---- save (immutable run dir)
    ckpt = out / "checkpoint"; ckpt.mkdir()
    model.encoder.save_pretrained(ckpt); tok.save_pretrained(ckpt)
    T, n_cal = fit_temperature(model, tok, a.device, a.max_len, cal) if cal else (1.0, 0)
    torch.save({"scorer": model.scorer.state_dict(), "base": a.base, "base_revision": a.base_revision, "max_len": a.max_len,
                "temperature": T, "calibration_questions": n_cal, "args": vars(a)}, ckpt / "head.pt")
    print(f"fitted temperature {T:.2f} on {n_cal} calibration questions", flush=True)

    # ---- score with Kev's harness: raw in `clean`, temperature applied by the harness in `calibrated_clean`
    pred = EncoderPredictor(model, tok, a.device, a.max_len)
    results = {"steps": step, "questions_seen": seen_q, "dropped_training_questions": dropped, "temperature": T,
               "params": n_params, "train_seconds": train_seconds,
               "provenance": {"base": a.base, "base_revision": a.base_revision, "suite_sha256": hashlib.sha256((Path(a.suite) / "manifest.json").read_bytes()).hexdigest(),
                              "transfer_sha256": hashlib.sha256((Path(a.transfer) / "manifest.json").read_bytes()).hexdigest() if a.transfer else None,
                              "torch": torch.__version__, "transformers": transformers.__version__, "attn": getattr(model.encoder.config, "_attn_implementation", None),
                              "dtype_policy": f"fp32 weights, {a.dtype} autocast, fp32 scorer, fp32 eval"}, "args": vars(a)}
    for label, recs, heldout in (("development", dev_recs, tuple(manifest.get("holdout_sources", []))),
                                 ("transfer", transfer_recs, tuple(transfer_manifest.get("holdout_sources", [])))):
        if not recs: continue
        report, _ = evaluate_records(recs, pred, out / label, temperature=T, heldout_sources=heldout)
        keep = lambda c: {k: c.get(k) for k in ("acc", "ece", "brier", "coverage_at_5pct_error", "n", "score_mae", "nll")}
        results[label] = {"clean": keep(report["clean"]), "calibrated_clean": keep(report.get("calibrated_clean", {})),
                          "sources": {k: v.get("acc") for k, v in (report.get("sources") or report.get("tasks") or {}).items() if isinstance(v, dict)}}
        print(f"{label}: raw acc {report['clean']['acc']:.4f} ece {report['clean'].get('ece')}; calibrated ece {report.get('calibrated_clean', {}).get('ece')}", flush=True)
    results["wall_seconds"] = time.time() - t0
    write_json(out / "result.json", results)
    print(json.dumps({k: v for k, v in results.items() if k not in ("args", "provenance")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
