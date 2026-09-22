"""Grid-navigation environment matching NanoJev's training format.

The state text and question wording are reproduced from NanoJev's
`build_game_decisions.TEMPLATES['grid_navigation']` (MIT, OpenJev contributors) so the
model sees the format it was trained on. The environment itself is written here.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

HEADER = "Current game position:"
STATE_TEMPLATE = (
    "{header}\nGrid navigation. Each legal orthogonal step costs 1; no diagonals. "
    "A=agent, G=goal, #=wall, .=open.\n{board}")
ACTION_INSTRUCTIONS = (
    "Choose a legal next step on a shortest route to G. If G is unreachable, all legal "
    "moves tie. Walls cannot be crossed.")
BOOLEAN_INSTRUCTIONS = "Can the agent reach G by legal orthogonal moves?"
CANDIDATE = "Move one cell {direction}."
DIRECTIONS = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}

Cell = Tuple[int, int]


def make_maze(size: int = 6, seed: int = 17, wall_fraction: float = 0.22) -> dict:
    """Generate a solvable maze. Retries layouts until the goal is reachable."""
    rng = random.Random(seed)
    cells = [(r, c) for r in range(size) for c in range(size)]
    for _ in range(500):
        position, goal = rng.sample(cells, 2)
        free = [c for c in cells if c not in (position, goal)]
        walls = set(rng.sample(free, int(len(cells) * wall_fraction)))
        state = {"size": size, "position": position, "goal": goal, "walls": walls,
                 "steps": 0, "done": False, "seed": seed}
        if distance_to_goal(state) is not None:
            return state
    raise RuntimeError("could not generate a solvable maze")


def in_bounds(state: dict, cell: Cell) -> bool:
    return 0 <= cell[0] < state["size"] and 0 <= cell[1] < state["size"]


def destination(state: dict, action: str) -> Cell:
    dr, dc = DIRECTIONS[action]
    row, col = state["position"]
    return (row + dr, col + dc)


def valid_actions(state: dict) -> List[str]:
    out = []
    for action in ("north", "east", "south", "west"):
        cell = destination(state, action)
        if in_bounds(state, cell) and cell not in state["walls"]:
            out.append(action)
    return out


def distance_to_goal(state: dict, start: Optional[Cell] = None) -> Optional[int]:
    """Breadth-first shortest path length, or None when the goal is unreachable."""
    start = start or state["position"]
    seen, queue = {start}, deque([(start, 0)])
    while queue:
        cell, steps = queue.popleft()
        if cell == state["goal"]:
            return steps
        for dr, dc in DIRECTIONS.values():
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt not in seen and in_bounds(state, nxt) and nxt not in state["walls"]:
                seen.add(nxt)
                queue.append((nxt, steps + 1))
    return None


def board(state: dict) -> str:
    size = state["size"]
    grid = [["."] * size for _ in range(size)]
    for row, col in state["walls"]:
        grid[row][col] = "#"
    grid[state["goal"][0]][state["goal"][1]] = "G"
    grid[state["position"][0]][state["position"][1]] = "A"
    return "\n".join(" ".join(row) for row in grid)


def render_request(state: dict) -> Dict:
    actions = valid_actions(state)
    if not actions:
        raise ValueError("a boxed-in state cannot form a Choice question")
    return {
        "state": STATE_TEMPLATE.format(header=HEADER, board=board(state)),
        "questions": {
            "action": {"type": "choice", "instructions": ACTION_INSTRUCTIONS,
                       "criteria": {a: CANDIDATE.format(direction=a) for a in actions}},
            "solvable": {"type": "boolean", "instructions": BOOLEAN_INSTRUCTIONS},
        },
    }


def step(state: dict, action: str) -> dict:
    if action not in valid_actions(state):
        raise ValueError(f"illegal action {action!r}")
    nxt = dict(state)
    nxt["position"] = destination(state, action)
    nxt["steps"] = state["steps"] + 1
    nxt["done"] = nxt["position"] == state["goal"]
    return nxt
