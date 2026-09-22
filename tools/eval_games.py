"""Honest capability numbers for the released checkpoint, measured on this machine.

Maze is scored per step against a breadth-first oracle, which is what upstream's
"Maze 4/10" measures. Solving a maze end to end needs many correct steps in a row,
so a per-step number near chance does not produce a solved maze.
"""
import argparse, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nanojev_mlx
from nanojev_mlx import maze as mz
from nanojev_mlx.play import play_snake


def maze_step_accuracy(agent, size, trials):
    hits = total = 0
    for seed in range(1, trials + 1):
        state = mz.make_maze(size, seed)
        options = mz.valid_actions(state)
        if len(options) < 2:
            continue
        best = min(
            (d for d in (mz.distance_to_goal(state, mz.destination(state, a)) for a in options)
             if d is not None), default=None)
        if best is None:
            continue
        optimal = {a for a in options
                   if mz.distance_to_goal(state, mz.destination(state, a)) == best}
        request = mz.render_request(state)
        payload = {"states": [{"id": "m", "state": request["state"],
                               "questions": {"action": request["questions"]["action"]}}]}
        pick = agent.predict(payload)["states"][0]["answers"]["action"]["choice"]
        hits += pick in optimal
        total += 1
    return hits, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--snake-seeds", type=int, default=10)
    ap.add_argument("--maze-trials", type=int, default=60)
    args = ap.parse_args()
    agent = nanojev_mlx.load(args.model)

    snake = []
    for seed in range(1, args.snake_seeds + 1):
        r = play_snake(agent, size=8, seed=seed, max_steps=200, render=False)
        snake.append(r)
        print(f"  snake seed {seed:<3} food={r['food']:<3} moves={r['moves']:<4} "
              f"{r['outcome'] or 'survived'}", flush=True)
    food = [r["food"] for r in snake]
    survived = sum(1 for r in snake if r["outcome"] is None)
    print(f"\n  snake: median {statistics.median(food):.0f} food, "
          f"best {max(food)}, survived all 200 moves in {survived}/{len(snake)} runs")

    rows = {}
    for size in (4, 6):
        hits, total = maze_step_accuracy(agent, size, args.maze_trials)
        rows[f"{size}x{size}"] = {"correct": hits, "scored": total,
                                  "accuracy": round(hits / total, 3) if total else None}
        print(f"  maze {size}x{size} per-step vs BFS oracle: {hits}/{total} "
              f"= {hits / total:.1%}" if total else "  maze: no scorable states")

    report = {"snake": {"runs": snake, "median_food": statistics.median(food),
                        "best_food": max(food), "survived_full_run": survived},
              "maze_step_accuracy": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
