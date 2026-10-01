"""Long-document eval: ContractNLI test NDAs (held-out docs AND a phrasing never used in training) through Ollama's
/v1/systemone. One request per NDA, three questions each (convert_cnli.py writes out/cnli/eval.jsonl).

    python3 eval_cnli.py tinyjev-v2b tev1-16k [--limit 60]
A request Ollama rejects (prompt over the model's num_ctx) scores its questions as wrong and is counted.
"""
import json, os, sys, time, urllib.error, urllib.request

URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11435") + "/v1/systemone"
HERE = os.path.dirname(os.path.abspath(__file__))


def ask(model, req):
    body = json.dumps({"model": model, "state": req["state"], "questions": req["questions"]}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=900) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:200]!r}"


def main(models, limit):
    reqs = [json.loads(l) for l in open(os.path.join(HERE, "out/cnli/eval.jsonl"))][: limit or None]
    for m in models:
        out = os.path.join(HERE, "runs", m, "cnli.jsonl"); os.makedirs(os.path.dirname(out), exist_ok=True)
        done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
        with open(out, "a") as f:
            for req in reqs:
                if req["id"] in done: continue
                t = time.time(); d, err = ask(m, req)
                for name, gold in req["gold"].items():
                    pred = d["answers"][name]["choice"] if d else None
                    f.write(json.dumps({"id": req["id"], "field": name, "gold": gold, "pred": pred, "correct": pred == gold,
                                        "error": err, "s": round(time.time() - t, 2)}) + "\n")
                f.flush()
        rows = [json.loads(l) for l in open(out)]
        rows = [r for r in rows if r["id"] in {q["id"] for q in reqs}]
        by = {}
        for r in rows: by.setdefault(r["gold"], []).append(r["correct"])
        print(f"{m}: ContractNLI-test {sum(r['correct'] for r in rows)}/{len(rows)} = {sum(r['correct'] for r in rows)/len(rows):.3f}"
              f" | " + " ".join(f"{g} {sum(v)}/{len(v)}" for g, v in sorted(by.items()))
              + f" | rejected {sum(bool(r['error']) for r in rows)}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]; lim = 0
    if "--limit" in a: i = a.index("--limit"); lim = int(a[i + 1]); a = a[:i] + a[i + 2:]
    main(a, lim)
