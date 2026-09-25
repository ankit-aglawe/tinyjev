"""Any server that speaks the System One request shape (POST /v1/systemone):
Kev's `python -m kev.serve`, agent-jev's jev_service, `tinyjev serve`, ...

    python adapters/systemone_http_adapter.py --url http://127.0.0.1:8009/v1/systemone --name kev-0.8b
"""
import argparse, json, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_cases, Writer, timed


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="", help="value for the request's model field, if the server wants one")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--note", default="")
    a = ap.parse_args()
    w = Writer(a.name, {"runtime": "systemone-http", "url": a.url, "note": a.note})
    for c in load_cases()[: a.limit or None]:
        body = {"state": c["state"], "questions": {"q": {"type": "choice", "instructions": c["instructions"], "criteria": c["criteria"]}}}
        if a.model:
            body["model"] = a.model
        r, ms = timed(lambda: post(a.url, body))
        ans = r["answers"]["q"] if "answers" in r else r["states"][0]["answers"]["q"]
        w.row(c, ans["probabilities"], ms)
    w.close()


if __name__ == "__main__":
    main()
