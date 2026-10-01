"""TinyJev v2 training on Modal: LoRA on Qwen3.5-4B's language model, loss on the single answer letter.

Rows are Ollama's exact scoring prompts (ollama_format.py). At inference Ollama reads the next-token logits of the
choice codes and softmaxes over those codes only, so the main loss is cross-entropy over each row's own codes; a
0.1-weighted full-vocabulary term keeps the letter the natural next token for plain generation clients.
Left padding puts every row's answer position at -1, so logits_to_keep=1 computes one position per row.

    modal run modal_train_v2.py --run v2b --parts tev1,cnli --no-caps --rank 8 --dropout 0 --lr 5e-5 --min-lr 0 \
        --warm-frac 0.03 --loss-kind full --max-tokens 16384     # the released TinyJev 4B v2
    modal run modal_train_v2.py::export --run v2b  # merge + GGUF Q8_0/Q4_K_M -> volume
"""
import json, math, os, random, time
import modal

app = modal.App("tinyjev-v2")
vol = modal.Volume.from_name("tinyjev-v2", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "build-essential", "cmake")
         .pip_install("torch==2.10.0", "transformers==5.17.0", "peft==0.21.1", "accelerate", "numpy<2.3", "huggingface_hub",
                      "flash-linear-attention", "tilelang", "sentencepiece", "safetensors")
         .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(os.path.join(os.path.dirname(__file__), "out"), "/data"))
BASE = "Qwen/Qwen3.5-4B"


def rows(split, parts=("tev1", "fam")):
    out = []
    for part in parts:
        p = f"/data/{part}/{split}.jsonl"
        if os.path.exists(p): out += [json.loads(l) for l in open(p)]
    return out


def batches(data, tok, max_tokens, shuffle, seed):
    """Length-bucketed, left-padded batches under a padded-token budget."""
    rng = random.Random(seed)
    idx = list(range(len(data)))
    if shuffle: rng.shuffle(idx)
    for s in range(0, len(idx), 4096):                  # sort inside large shuffled chunks to cut padding
        chunk = sorted(idx[s:s + 4096], key=lambda i: data[i]["n"])
        cur = []
        for i in chunk:
            if cur and (len(cur) + 1) * max(data[i]["n"], data[cur[-1]]["n"]) > max_tokens:
                yield cur; cur = []
            cur.append(i)
        if cur: yield cur


def collate(data, ids, pad_id, letter_ids, device):
    import torch
    L = max(data[i]["n"] for i in ids)
    inp = torch.full((len(ids), L), pad_id, dtype=torch.long); att = torch.zeros((len(ids), L), dtype=torch.long)
    cand = torch.zeros((len(ids), 26), dtype=torch.long); cmask = torch.zeros((len(ids), 26), dtype=torch.bool)
    tgt, full = [], []
    for b, i in enumerate(ids):
        t = data[i]["ids"]; inp[b, L - len(t):] = torch.tensor(t); att[b, L - len(t):] = 1
        codes = data[i]["codes"]
        cand[b, :len(codes)] = torch.tensor([letter_ids[c] for c in codes]); cmask[b, :len(codes)] = True
        tgt.append(codes.index(data[i]["answer"])); full.append(letter_ids[data[i]["answer"]])
    return inp.to(device), att.to(device), cand.to(device), cmask.to(device), torch.tensor(tgt, device=device), torch.tensor(full, device=device)


# fla refuses Hopper with this torch's Triton (< 3.7.1: wrong gated-delta-rule gradients, fla #640) unless tilelang
# is installed, which it now is. TJ_GPU picks the card at launch (A100-80GB measured 2.8k tok/s steady).
# A100-80GB measured 2.8k tok/s steady; H100 is blocked by fla's Hopper guard even with tilelang (2026-10-01).
# timeout bounds the bill: 3.1 h x $2.50/h ~= $7.75 worst case.
@app.function(image=image, gpu=os.environ.get("TJ_GPU", "A100-80GB"), volumes={"/vol": vol}, timeout=int(3.1 * 3600))
def train(run: str = "v2a", lr: float = 1e-4, rank: int = 32, epochs: float = 1.0, max_tokens: int = 24576, seed: int = 0,
          fam_weight: int = 1, limit: int = 0, dev_n: int = 1600, parts: str = "tev1,fam", caps: bool = True,
          dropout: float = 0.05, min_lr: float = 0.1, warm_frac: float = 1 / 30, loss_kind: str = "cand"):
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model
    t0 = time.time(); torch.manual_seed(seed)
    log = lambda *a: print(f"[{(time.time() - t0) / 60:6.1f}m]", *a, flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    letter_ids = {}
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        e = tok.encode(c, add_special_tokens=False); assert len(e) == 1, (c, e); letter_ids[c] = e[0]
    # per-source caps on Tev1's largest buckets (fast learners in the smoke test); every family row is kept
    CAP = {"mnli": 3000, "policy_v2": 7000, "routing_v2": 4000, "research_taxonomy_v21": 2500}
    def prep(split):
        d = rows(split, tuple(parts.split(",")) if split == "train" else ("tev1", "fam", "cnli"))   # dev: fixed subset per source
        n_all = len(d)
        if split == "train":
            if caps:
                rr = random.Random(seed); rr.shuffle(d); seen = {}
                d = [r for r in d if seen.setdefault(r["source"], 0) < CAP.get(r["source"], 10**9) and not seen.update({r["source"]: seen[r["source"]] + 1})]
            else:
                assert len(d) == n_all and n_all > 1000, (len(d), n_all)   # control run: every row of the requested parts
        else:                                          # fixed stratified dev subset: every source, same rows each run
            by = {}
            for r in d: by.setdefault(r["source"], []).append(r)
            k = max(1, dev_n // len(by)); d = [r for v in by.values() for r in v[:k]]
        if split == "train" and fam_weight > 1: d += [r for r in d if r["source"].startswith("family_")] * (fam_weight - 1)
        if limit: d = d[:limit]
        for r in d:
            r["ids"] = tok.encode(r["prompt"], add_special_tokens=False); r["n"] = len(r["ids"])
        return d
    train_d, dev_d = prep("train"), prep("dev")
    log(f"train {len(train_d)} rows {sum(r['n'] for r in train_d)/1e6:.1f}M tokens (max {max(r['n'] for r in train_d)}) | dev {len(dev_d)}")
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    names = sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Linear) and "language_model" in n and "visual" not in n})
    log("lora targets:", names)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=dropout, bias="none",
                                             target_modules=rf"^(?!.*visual).*language_model.*\.({'|'.join(names)})$"))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def forward(ids):
        inp, att, cand, cmask, tgt, full = collate(d_ref[0], ids, pad, letter_ids, "cuda")
        out = model(input_ids=inp, attention_mask=att, logits_to_keep=1)
        z = out.logits[:, -1, :].float()
        zc = z.gather(1, cand).masked_fill(~cmask, float("-inf"))
        if loss_kind == "full":                                # Tev1's completion-only loss on the letter
            return F.cross_entropy(z, full), zc, tgt
        return F.cross_entropy(zc, tgt) + 0.1 * F.cross_entropy(z, full), zc, tgt

    def evaluate(tag):
        model.eval(); d_ref[0] = dev_d; ok = n = 0; by = {}; conf_bins = [[0, 0, 0.0] for _ in range(10)]
        with torch.no_grad():
            for ids in batches(dev_d, tok, max_tokens, False, 0):
                _, zc, tgt = forward(ids)
                p = zc.softmax(-1); top = p.max(-1)
                for j, i in enumerate(ids):
                    hit = int(top.indices[j] == tgt[j]); ok += hit; n += 1
                    s = dev_d[i]["source"]; by.setdefault(s, [0, 0]); by[s][0] += hit; by[s][1] += 1
                    b = min(9, int(top.values[j] * 10)); conf_bins[b][0] += 1; conf_bins[b][1] += hit; conf_bins[b][2] += float(top.values[j])
        ece = sum(abs(c[1] - c[2]) for c in conf_bins if c[0]) / max(1, n)
        log(f"DEV {tag}: acc {ok/n:.4f} ece {ece:.4f} | " + " ".join(f"{k} {v[0]/v[1]:.3f}" for k, v in sorted(by.items())))
        model.train(); return ok / n, ece, {k: v[0] / v[1] for k, v in by.items()}

    d_ref = [train_d]
    base = evaluate("base")
    plan = [b for e in range(math.ceil(epochs)) for b in batches(train_d, tok, max_tokens, True, seed + e)]
    plan = plan[: int(len(plan) * epochs / math.ceil(epochs))]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    warm = max(20, int(len(plan) * warm_frac))
    budget_s = 3.1 * 3600 - 25 * 60 - (time.time() - t0)       # leave 25 min for the final eval and the save
    t_loop = time.time()
    def progress(s):                                           # cosine over data or wall clock, whichever ends first
        return min(1, max(s / len(plan), (time.time() - t_loop) / budget_s))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (min_lr + (1 - min_lr) * 0.5 * (1 + math.cos(math.pi * progress(s)))))
    model.train(); losses = []; toks = 0; t_train = time.time(); warm_t = None
    for step, ids in enumerate(plan):
        if step == 20: warm_t, warm_toks = time.time(), toks          # steady-state speed excludes kernel compile/autotune
        d_ref[0] = train_d
        loss, _, _ = forward(ids)
        # long-document batches hold 1-7 rows vs >=10 for Tev1-length rows; scale them down so one or two documents
        # cannot take a full-size step (a no-op for every batch of 8+ rows, i.e. all earlier runs)
        (loss * min(1.0, len(ids) / 8)).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss)); toks += sum(train_d[i]["n"] for i in ids)
        if time.time() - t_loop > budget_s:
            log(f"time budget reached at step {step}/{len(plan)}"); break
        if step % 50 == 0:
            rate = (toks - warm_toks) / (time.time() - warm_t) if warm_t and time.time() > warm_t else toks / (time.time() - t_train)
            log(f"step {step}/{len(plan)} loss {sum(losses[-50:]) / len(losses[-50:]):.4f} lr {sched.get_last_lr()[0]:.2e} {rate:.0f} tok/s")
    out = f"/vol/{run}"; os.makedirs(out, exist_ok=True)
    model.save_pretrained(f"{out}/adapter"); vol.commit(); log("adapter saved")   # before the eval: a slow eval must not cost the weights
    final = evaluate("final")
    json.dump({"run": run, "lr": lr, "rank": rank, "epochs": epochs, "steps": len(plan), "train_rows": len(train_d), "parts": parts,
               "caps": caps, "dropout": dropout, "min_lr": min_lr, "loss_kind": loss_kind, "max_tokens": max_tokens,
               "base_dev": base, "final_dev": final, "hours": (time.time() - t0) / 3600}, open(f"{out}/metrics.json", "w"), indent=1)
    vol.commit(); log("TRAIN_DONE", json.dumps({"base": base[:2], "final": final[:2]}))


@app.function(image=image, cpu=8, memory=65536, volumes={"/vol": vol}, timeout=3 * 3600)
def export(run: str = "v2a"):
    """Merge the adapter into the full checkpoint, convert with llama.cpp, quantize Q8_0 (Tev1/Nimble ship Q8_0) and Q4_K_M."""
    import subprocess, torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
    from peft import PeftModel
    sh = lambda c: subprocess.run(c, shell=True, check=True)
    out = f"/vol/{run}"
    m = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16)
    if run != "base":                                  # run="base": the untuned model through the SAME converter/quantizer
        m = PeftModel.from_pretrained(m, f"{out}/adapter").merge_and_unload()
    os.makedirs(out, exist_ok=True)
    merged = f"{out}/merged"                              # kept on the volume: publish() uploads it to the Hub
    m.save_pretrained(merged, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(BASE).save_pretrained(merged)
    try: AutoProcessor.from_pretrained(BASE).save_pretrained(merged)
    except Exception as e: print("processor not saved:", e)
    # Convert with the llama.cpp Ollama 0.35.1 bundles (b11232) and --no-mtp: save_pretrained drops Qwen3.5's MTP weights,
    # and a GGUF that still counts the MTP layer (block_count 33, nextn 1) fails in Ollama with "blk.32 not found".
    sh("git clone -q --depth 1 --branch b11232 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp")
    sh("cd /tmp/llama.cpp && cmake -S . -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null && cmake --build build --target llama-quantize -j8 > /dev/null")
    sh(f"python /tmp/llama.cpp/convert_hf_to_gguf.py {merged} --no-mtp --outtype bf16 --outfile /tmp/tinyjev-v2-bf16.gguf")
    for q in ("Q8_0", "Q4_K_M"):
        sh(f"/tmp/llama.cpp/build/bin/llama-quantize /tmp/tinyjev-v2-bf16.gguf {out}/tinyjev-v2-{q}.gguf {q} 8")
    vol.commit(); print("EXPORT_DONE")



@app.function(image=image, cpu=4, memory=16384, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("hf-token")],
              timeout=2 * 3600)
def publish(run: str, repo: str, readme: str, modelfile: str, tag_old: str = "v1", gguf_prefix: str = "TinyJev-4B"):
    """Tag the Hub repo's current revision `tag_old` (the old model stays loadable with revision=tag_old), then make
    `run` the repo's main in ONE commit: merged safetensors + tokenizer, both GGUFs, Modelfile, README."""
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    api = HfApi(); assert os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"), "hf-token secret has no token"
    old = api.model_info(repo)
    if tag_old not in [t.name for t in api.list_repo_refs(repo).tags]:
        api.create_tag(repo, tag=tag_old, revision=old.sha, tag_message=f"before {run}"); print("tagged", tag_old, old.sha)
    src = f"/vol/{run}/merged"
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=f"{src}/{f}") for f in sorted(os.listdir(src))]
    ops += [CommitOperationAdd(path_in_repo=f"{gguf_prefix}-{q}.gguf", path_or_fileobj=f"/vol/{run}/tinyjev-v2-{q}.gguf")
            for q in ("Q8_0", "Q4_K_M")]
    ops += [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode()),
            CommitOperationAdd(path_in_repo="Modelfile", path_or_fileobj=modelfile.encode())]
    new = {o.path_in_repo for o in ops} | {".gitattributes"}
    ops += [CommitOperationDelete(path_in_repo=s.rfilename) for s in old.siblings if s.rfilename not in new]
    print("delete:", [o.path_in_repo for o in ops if isinstance(o, CommitOperationDelete)])
    info = api.create_commit(repo, operations=ops, commit_message=f"TinyJev v2 ({run}): Ollama-native decision model")
    print("PUBLISHED", info.oid)


@app.local_entrypoint()
def main(run: str = "v2a", lr: float = 1e-4, rank: int = 32, epochs: float = 1.0, fam_weight: int = 1, limit: int = 0,
         parts: str = "tev1,fam", caps: bool = True, dropout: float = 0.05, min_lr: float = 0.1, warm_frac: float = 1 / 30,
         loss_kind: str = "cand", max_tokens: int = 24576):
    train.remote(run=run, lr=lr, rank=rank, epochs=epochs, fam_weight=fam_weight, limit=limit, parts=parts, caps=caps,
                 dropout=dropout, min_lr=min_lr, warm_frac=warm_frac, loss_kind=loss_kind, max_tokens=max_tokens)
