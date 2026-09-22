"""Watch the model play. Every move is one forward pass, no text generated.

    nanojev-mlx play snake --model <dir>

The request text comes from NanoJev's own `snake_game.render_request`, so the model
sees exactly the format it was trained on.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from typing import Optional

from . import maze as mz
from .vendor import snake_game as sg

GLYPH = {"head": "@", "body": "o", "food": "*", "empty": "·"}
CLEAR = "\033[2J\033[H"
HIDE, SHOW = "\033[?25l", "\033[?25h"


def board_lines(state) -> list[str]:
    size = state["size"]
    grid = [[GLYPH["empty"]] * size for _ in range(size)]
    for row, col in state["body"][1:]:
        grid[row][col] = GLYPH["body"]
    frow, fcol = state["food"]
    grid[frow][fcol] = GLYPH["food"]
    hrow, hcol = state["body"][0]
    grid[hrow][hcol] = GLYPH["head"]
    return [" ".join(row) for row in grid]


def frame(state, move: Optional[str], ms: float, moves: int, fps: float) -> str:
    size = state["size"]
    head = [f"nanojev-mlx  ·  snake {size}x{size}  ·  every move is one forward pass", ""]
    body = board_lines(state)
    stats = ["",
             f"  food {state['score']:<4} moves {moves:<5} last move {move or '-':<6}",
             f"  {ms:6.1f} ms/decision      {fps:5.1f} decisions/s"]
    return "\n".join(head + ["  " + line for line in body] + stats)


def play_snake(agent, size=8, seed=17, max_steps=400, delay=0.0, safety=False,
               render=True, on_frame=None) -> dict:
    state = sg.make_snake(size, seed)
    moves, timings, overrides = 0, [], 0
    if render:
        sys.stdout.write(HIDE)
    try:
        while not state["done"] and moves < max_steps:
            offered = sg.valid_actions(state)
            request = sg.render_request(state)
            questions = request["questions"] if safety else {"action": request["questions"]["action"]}
            payload = {"states": [{"id": "snake", "state": request["state"],
                                   "questions": questions}]}
            started = time.perf_counter()
            answers = agent.predict(payload)["states"][0]["answers"]
            elapsed = (time.perf_counter() - started) * 1000.0
            timings.append(elapsed)

            proposed = action = answers["action"]["choice"]
            if safety and not answers.get(f"safe_{action}", {}).get("value", True):
                safe = [a for a in offered if answers.get(f"safe_{a}", {}).get("value", False)]
                if safe:
                    action = max(safe, key=lambda a: answers["action"]["probabilities"].get(a, 0.0))
                    overrides += 1

            state = sg.step(state, action)
            moves += 1
            fps = 1000.0 / (sum(timings) / len(timings))
            text = frame(state, action, elapsed, moves, fps)
            if render:
                sys.stdout.write(CLEAR + text + "\n")
                sys.stdout.flush()
            if on_frame:
                on_frame({"text": text, "state": state, "offered": offered,
                          "answers": answers, "proposed": proposed, "chosen": action,
                          "elapsed_ms": elapsed, "moves": moves, "fps": fps,
                          "overrides": overrides, "safety": safety})
            if delay:
                time.sleep(delay)
    finally:
        if render:
            sys.stdout.write(SHOW)
            sys.stdout.flush()

    ordered = sorted(timings)
    return {"game": "snake", "size": size, "seed": seed, "food": state["score"],
            "moves": moves, "outcome": state.get("outcome"), "safety_overrides": overrides,
            "p50_ms": round(ordered[len(ordered) // 2], 1) if ordered else 0.0,
            "decisions_per_second": round(1000.0 / (sum(timings) / len(timings)), 1) if timings else 0.0}


def maze_frame(state, move, ms, moves, fps, shortest) -> str:
    head = [f"nanojev-mlx  ·  maze {state['size']}x{state['size']}  ·  every move is one forward pass", ""]
    body = mz.board(state).split("\n")
    stats = ["",
             f"  steps {moves:<5} to goal {shortest if shortest is not None else '-':<5} last move {move or '-':<6}",
             f"  {ms:6.1f} ms/decision      {fps:5.1f} decisions/s"]
    return "\n".join(head + ["  " + line for line in body] + stats)


def play_maze(agent, size=6, seed=17, max_steps=80, delay=0.0, render=True,
              on_frame=None) -> dict:
    state = mz.make_maze(size, seed)
    optimal = mz.distance_to_goal(state)
    moves, forced, timings = 0, 0, []
    if render:
        sys.stdout.write(HIDE)
    try:
        while not state["done"] and moves < max_steps:
            options = mz.valid_actions(state)
            if len(options) < 2:
                # A corridor offers no decision; take it without spending a forward pass.
                action, elapsed = options[0], 0.0
                forced += 1
            else:
                request = mz.render_request(state)
                payload = {"states": [{"id": "maze", "state": request["state"],
                                       "questions": {"action": request["questions"]["action"]}}]}
                started = time.perf_counter()
                answers = agent.predict(payload)["states"][0]["answers"]
                elapsed = (time.perf_counter() - started) * 1000.0
                timings.append(elapsed)
                action = answers["action"]["choice"]
            state = mz.step(state, action)
            moves += 1
            fps = 1000.0 / (sum(timings) / len(timings)) if timings else 0.0
            text = maze_frame(state, action, elapsed, moves, fps, mz.distance_to_goal(state))
            if render:
                sys.stdout.write(CLEAR + text + "\n")
                sys.stdout.flush()
            if on_frame:
                on_frame({"text": text, "state": state, "chosen": action,
                          "elapsed_ms": elapsed, "moves": moves, "fps": fps})
            if delay:
                time.sleep(delay)
    finally:
        if render:
            sys.stdout.write(SHOW)
            sys.stdout.flush()

    ordered = sorted(timings)
    return {"game": "maze", "size": size, "seed": seed, "solved": state["done"],
            "moves": moves, "model_decisions": len(timings), "forced_moves": forced,
            "shortest_possible": optimal,
            "p50_ms": round(ordered[len(ordered) // 2], 1) if ordered else 0.0,
            "decisions_per_second": round(1000.0 / (sum(timings) / len(timings)), 1) if timings else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nanojev-mlx play")
    ap.add_argument("game", choices=["snake", "maze"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--safety", action="store_true",
                    help="also ask the per-move safety questions and override unsafe picks")
    args = ap.parse_args(argv)

    from . import load
    agent = load(args.model)
    if args.game == "snake":
        result = play_snake(agent, size=args.size, seed=args.seed, max_steps=args.steps,
                            delay=args.delay, safety=args.safety)
        print(f"\n  {result['outcome'] or 'survived'}: {result['food']} food in "
              f"{result['moves']} moves  ({result['p50_ms']} ms/decision, "
              f"{result['decisions_per_second']}/s)")
    else:
        result = play_maze(agent, size=args.size if args.size != 8 else 6, seed=args.seed,
                           max_steps=args.steps, delay=args.delay)
        verdict = "solved" if result["solved"] else "did not reach the goal"
        print(f"\n  {verdict} in {result['moves']} moves "
              f"(shortest possible {result['shortest_possible']})  "
              f"({result['p50_ms']} ms/decision, {result['decisions_per_second']}/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
