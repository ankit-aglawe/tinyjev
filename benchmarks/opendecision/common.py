"""Shared pieces for every adapter: case loading, one output schema, one scorer.

Every adapter writes one JSONL row per case with exactly these keys, so the
summary and the coverage curves are computed by the same code for every model.
"""
import json, time, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE / "suite"
RESULTS = HERE / "results"


def load_cases(name="cases"):
    rows = [json.loads(l) for l in (SUITE / f"{name}.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r.setdefault("type", "choice")
    return rows


def suite_sha(name="cases"):
    return hashlib.sha256((SUITE / f"{name}.jsonl").read_bytes()).hexdigest()


class Writer:
    """Streams rows to results/<model>.jsonl and prints a running accuracy."""

    def __init__(self, model, meta):
        RESULTS.mkdir(exist_ok=True)
        self.path = RESULTS / f"{model}.jsonl"
        self.f = self.path.open("w")
        self.model = model
        self.meta = {**meta, "model": model, "suite_sha256": suite_sha()}
        self.f.write(json.dumps({"_meta": self.meta}) + "\n")
        self.f.flush()
        self.n = self.hits = 0

    def row(self, case, probabilities, ms):
        """probabilities: {option_name: p} over the case's own option names."""
        probs = {k: float(v) for k, v in probabilities.items()}
        pred = max(probs, key=probs.get)
        ok = pred == case["expected"]
        self.n += 1
        self.hits += ok
        self.f.write(json.dumps({
            "id": case["id"], "domain": case.get("domain", "-"), "type": case["type"],
            "expected": case["expected"], "predicted": pred, "correct": bool(ok),
            "confidence": round(probs[pred], 6), "probabilities": probs, "ms": round(ms, 2),
        }) + "\n")
        if self.n % 50 == 0:
            print(f"  {self.model}: {self.n}  acc {self.hits / self.n:.3f}", flush=True)

    def close(self):
        self.f.close()
        print(f"{self.model}: {self.hits}/{self.n} = {self.hits / max(self.n, 1):.4f} -> {self.path}")


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t) * 1000.0
