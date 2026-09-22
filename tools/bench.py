"""Measure MLX decision latency on this machine."""
import argparse, json, platform, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nanojev_mlx


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return platform.processor() or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    t0 = time.perf_counter()
    agent = nanojev_mlx.load(args.model)
    load_s = time.perf_counter() - t0

    rows = []
    for case in cases:
        for _ in range(args.warmup):
            agent.logits(case["payload"])
        samples = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            agent.logits(case["payload"])
            samples.append((time.perf_counter() - t) * 1000.0)
        samples.sort()
        p50 = statistics.median(samples)
        p95 = samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))]
        paths = sum(len(v["path_token_counts"]) for v in agent.logits(case["payload"]).values())
        rows.append({"name": case["name"], "candidate_paths": paths,
                     "p50_ms": round(p50, 1), "p95_ms": round(p95, 1)})
        print(f"  {case['name']:<28} paths={paths:<3} p50={p50:7.1f} ms  p95={p95:7.1f} ms",
              flush=True)

    report = {"machine": chip(), "python": platform.python_version(),
              "model": str(args.model), "body_dtype": agent.config.get("body_dtype"),
              "load_seconds": round(load_s, 2), "repeats": args.repeats, "rows": rows}
    print(f"\n{report['machine']} | load {load_s:.1f}s | "
          f"median across cases {statistics.median([r['p50_ms'] for r in rows]):.1f} ms")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
