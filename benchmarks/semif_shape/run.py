"""tinyjev on SemIf-OpenJev's own systems fixture: 37 states x 21 yes/no criteria, ~1,800
tokens of state each, no gold labels. Two numbers SemIf publishes for a frozen Qwen3.5-4B
on an RTX 3090: one state's 21 criteria in 1.023 s (direct readout) vs 5.332 s (generating
a JSON array), and 777 decisions at 2.33 / 10.75 / 20.03 decisions/s depending on reuse.

Same fixture, tinyjev-0.6b, one forward pass per state with all 21 questions as noul.
Hardware differs (Apple M1, 16 GB) and is stated next to every number.

    python benchmarks/semif_shape/run.py
"""
import json, statistics, time, platform, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tinyjev

HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (HERE / "suite/shape777.jsonl").read_text().splitlines() if l.strip()]
states = {}
for r in rows:
    states.setdefault(r["state"], []).append(r)
assert len(states) == 37 and all(len(v) == 21 for v in states.values())
published = json.load((HERE / "suite/semif-published-decision-vs-generation.json").open())


def questions(rs):
    return {r["id"]: {"type": "noul", "instructions": r["question"]} for r in rs}


def machine():
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        gb = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // (1 << 30)
        return f"{chip}, {gb} GB, MLX"
    except Exception:
        return platform.platform()


agent = tinyjev.load("tinyjev-0.6b")
first_state, first_rows = next(iter(states.items()))
agent.predict({"state": first_state, "questions": questions(first_rows[:1])})   # warm

# 1. one state, 21 criteria, median of 3, same shape as SemIf's decision_vs_generation.py
runs, answers = [], None
for _ in range(3):
    t0 = time.perf_counter()
    res = agent.predict({"state": first_state, "questions": questions(first_rows)})
    runs.append((time.perf_counter() - t0) * 1000)
    answers = res["states"][0]["answers"]
ours = [("yes" if answers[r["id"]]["p_true"] >= 0.5 else "no") for r in first_rows]
theirs = published["compact_generation"]["runs"][0]["choices"]
agree = sum(a == b for a, b in zip(ours, theirs))
one = {"criteria": 21, "runs_ms": [round(x, 1) for x in runs], "median_ms": round(statistics.median(runs), 1),
       "prefix_tokens_theirs": published["direct_parallel"]["runs"][0]["prefix_tokens"],
       "p_true": [round(answers[r["id"]]["p_true"], 3) for r in first_rows], "argmax": ours,
       "agreement_with_semif_generated_array": f"{agree}/21",
       "semif_direct_median_s": published["direct_parallel"]["median_total_seconds"],
       "semif_generation_median_s": published["compact_generation"]["median_total_seconds"],
       "semif_generation_output_tokens": published["compact_generation"]["median_output_tokens"]}

# 2. all 777 decisions: one pass per state
t0 = time.perf_counter(); n = 0; per_state = []
for st, rs in states.items():
    t1 = time.perf_counter()
    agent.predict({"state": st, "questions": questions(rs)})
    per_state.append((time.perf_counter() - t1) * 1000); n += len(rs)
total = time.perf_counter() - t0
full = {"decisions": n, "states": len(states), "total_s": round(total, 2), "decisions_per_s": round(n / total, 2),
        "median_state_ms": round(statistics.median(per_state), 1),
        "semif_decisions_per_s": {"fresh": 2.33, "serial_prefix_reuse": 10.75, "parallel_suffixes": 20.03, "native_reranker": 1.86}}

out = {"model": "tinyjev-0.6b", "hardware": machine(), "runtime": f"tinyjev {tinyjev.__version__}",
       "semif_hardware": "NVIDIA GeForce RTX 3090, Qwen3.5-4B bf16 (published)", "fixture_sha256": (HERE / "suite/SHA256SUMS").read_text().split()[0],
       "one_state_21_criteria": one, "all_777": full}
(HERE / "results.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in one.items() if k not in ("p_true", "argmax")}, indent=1))
print(json.dumps(full, indent=1))
