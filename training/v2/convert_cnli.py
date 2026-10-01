"""ContractNLI (Koreeda & Manning 2021, Hitachi America, CC BY 4.0) -> TinyJev v2 long-document rows in Ollama's exact
/v1/systemone prompt format. Each NDA (median ~2.5k tokens, longest ~13k) is the context; each row asks what the
agreement says about one of the dataset's 17 hypotheses. Labels are the human annotations: Entailment / Contradiction
/ NotMentioned. Wording, option names and option order vary per row, so there is no template to memorize (v2a's
generated families were memorized and did not transfer).

train -> out/cnli/train.jsonl (3 hypotheses per doc, every Contradiction first: it is the rare label)
dev   -> out/cnli/dev.jsonl   (2 per doc, monitored as source "cnli")
test  -> out/cnli/eval.jsonl  /v1/systemone requests in a phrasing never used in training, for eval_cnli.py

    python convert_cnli.py data_raw/contract-nli out/cnli
"""
import json, random, re, sys
from pathlib import Path
from ollama_format import user_messages, render, client_json
from convert_tev1 import SYSTEM

# (entailment, contradiction, not-mentioned) as (value, description); None = Ollama shows the value itself
OPTS = [
    (("stated", "The agreement says this."), ("contradicted", "The agreement says the opposite."),
     ("not_addressed", "The agreement does not address this.")),
    (("yes", "Yes, the agreement provides this."), ("no", "No, the agreement provides the opposite."),
     ("not_mentioned", "The agreement is silent on this.")),
    (("entailed", None), ("contradicted", None), ("not_mentioned", None)),
    (("agreement_says_so", "The text supports this statement."), ("agreement_says_opposite", "The text rules this out."),
     ("agreement_is_silent", "The text neither supports nor rules this out.")),
]
ASK = ["What does the agreement say about this: {h}", "Check this against the agreement: {h}",
       "Does the agreement support this statement? {h}", "{h} Is that what this agreement provides?"]
NOUL_ASK = ["Does the agreement clearly provide this? {h}", "Is this statement supported by the agreement? {h}"]
# held out for eval only
EVAL_OPTS = (("supported", "The contract supports this."), ("ruled_out", "The contract rules this out."),
             ("silent", "The contract does not cover this."))
EVAL_ASK = "Per this NDA: {h}"
LAB = {"Entailment": 0, "Contradiction": 1, "NotMentioned": 2}
MAX_TOKENS = 15000                                    # Modelfile num_ctx is 16384; Ollama rejects longer prompts


def fname(short):
    return re.sub(r"[^a-z0-9]+", "_", short.lower()).strip("_")


def pick(doc, k, rng):
    ann = doc["annotation_sets"][0]["annotations"]
    con = [h for h, v in ann.items() if v["choice"] == "Contradiction"]
    rest = [h for h in ann if h not in con]
    rng.shuffle(con); rng.shuffle(rest)
    return (con[:2] + rest)[:k]


def choice_q(hyp, opts, ask, rng):
    order = [0, 1, 2]; rng.shuffle(order)
    return {"type": "choice", "instructions": ask.format(h=hyp), "criteria": {opts[i][0]: opts[i][1] for i in order}}, order


def state_of(doc, rng):
    if rng.random() < 0.8: return doc["text"]
    return ("raw", client_json({"agreement": doc["text"]}, ensure_ascii=rng.random() < 0.7))


def rows_for(doc, labels, k, rng, tok):
    ann = doc["annotation_sets"][0]["annotations"]
    out = []
    for h in pick(doc, k, rng):
        lab = LAB[ann[h]["choice"]]; hyp = labels[h]["hypothesis"]; name = fname(labels[h]["short_description"])
        if rng.random() < 0.25:                       # yes/no form: true only when the agreement states it
            q = {"type": "noul", "instructions": rng.choice(NOUL_ASK).format(h=hyp)}; ans = "B" if lab == 0 else "A"
        else:
            opts = rng.choice(OPTS); q, order = choice_q(hyp, opts, rng.choice(ASK), rng); ans = chr(ord("A") + order.index(lab))
        qs = {name: q}
        if rng.random() < 0.5:                        # Ollama puts every field's schema in every prompt
            others = rng.sample([x for x in ann if x != h], rng.randint(1, 3))
            items = [(name, q)] + [(fname(labels[o]["short_description"]),
                                    choice_q(labels[o]["hypothesis"], rng.choice(OPTS), rng.choice(ASK), rng)[0]) for o in others]
            rng.shuffle(items); qs = dict(items)
        msg, codes = next((m, c) for n, m, c in user_messages(state_of(doc, rng), qs) if n == name)
        prompt = render(tok, SYSTEM, msg)
        if len(tok.encode(prompt, add_special_tokens=False)) > MAX_TOKENS: continue
        out.append({"prompt": prompt, "answer": ans, "codes": codes, "source": "cnli", "kind": q["type"],
                    "id": f"{doc['id']}:{h}"})
    return out


def eval_requests(doc, labels, rng):
    ann = doc["annotation_sets"][0]["annotations"]; qs, gold = {}, {}
    for h in pick(doc, 3, rng):
        name = fname(labels[h]["short_description"])
        qs[name] = {"type": "choice", "instructions": EVAL_ASK.format(h=labels[h]["hypothesis"]),
                    "criteria": {v: d for v, d in EVAL_OPTS}}
        gold[name] = EVAL_OPTS[LAB[ann[h]["choice"]]][0]
    return {"id": doc["id"], "state": doc["text"], "questions": qs, "gold": gold}


def main(src, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    Path(out).mkdir(parents=True, exist_ok=True)
    for split, k in (("train", 3), ("dev", 2)):
        d = json.load(open(Path(src) / f"{split}.json")); rng = random.Random(f"cnli-{split}")
        rows = [r for doc in d["documents"] for r in rows_for(doc, d["labels"], k, rng, tok)]
        with open(Path(out) / f"{split}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(rows), "rows", {a: sum(r["answer"] == a for r in rows) for a in "ABC"}, flush=True)
    d = json.load(open(Path(src) / "test.json")); rng = random.Random("cnli-test")
    with open(Path(out) / "eval.jsonl", "w") as f:
        for doc in d["documents"]: f.write(json.dumps(eval_requests(doc, d["labels"], rng), ensure_ascii=False) + "\n")
    print("eval", len(d["documents"]), "requests")


if __name__ == "__main__":
    main(*sys.argv[1:3])
