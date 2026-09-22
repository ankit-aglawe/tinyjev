"""The 'unified' Snake request format the released checkpoint was trained on.

Reverse-engineered byte for byte from `C-Tianyu/NanoJev-Data unified/hard/test.jsonl`.
The vendored `snake_game.render_request` is an older stage format; the model still
plays with it, but this is what it actually saw in training.
"""
from __future__ import annotations

import json
from typing import Dict, List

from .vendor import snake_game as sg

INSTRUCTIONS = ("Choose the next action that maximizes the probability of completing the stated "
                "task successfully before its deadline. Use the visible state, action "
                "descriptions, remaining time, and recorded history.")
EVENT_WINDOW = 8
TARGETS = {8: (2, 96), 12: (3, 160)}  # board size -> (food per reset, attempt budget)


def _compact(value) -> str:
    return json.dumps(value, separators=(",", ":"))


class Episode:
    """Tracks the task framing the state text needs: target, attempts, recent events."""

    def __init__(self, size: int):
        self.food_target, self.budget = TARGETS.get(size, (2, 96))
        self.used = 0
        self.base_score = 0        # food already banked before the current reset
        self.events: List[dict] = []

    def reset_if_reached(self, state: dict) -> bool:
        """Upstream ends the task at the target; the demo instead starts a new task
        on the same snake so a run can continue. Returns True when that happened."""
        if state["score"] - self.base_score >= self.food_target:
            self.base_score = state["score"]
            self.used = 0
            self.events = []
            return True
        return False

    def record(self, before: dict, action: str, after: dict):
        self.used += 1
        head, nxt = before["body"][0], sg._destination(before, action)
        ate = after["score"] > before["score"]
        self.events.append({
            "step": self.used, "actor": "model", "action": action,
            "head": list(head), "next_head": list(nxt), "ate_food": ate,
            "collision": bool(after.get("done") and after.get("outcome") not in (None, "success")),
            "terminated": bool(after.get("done")),
            "success": after["score"] - self.base_score >= self.food_target,
        })
        self.events = self.events[-EVENT_WINDOW:]


def render_request(state: dict, ep: Episode) -> Dict:
    n = state["size"]
    actions = sg.valid_actions(state)
    text = (
        f"Snake on a {n}x{n} board. Coordinates are zero-based (row,column); north decreases "
        f"row, east increases column. Body head first: {_compact(state['body'])}. "
        f"Direction: {state['direction']}. Current food: {_compact(state['food'])}. "
        f"Food collected this episode: {state['score'] - ep.base_score}. "
        f"Target: collect {ep.food_target} additional food items after reset before "
        f"{ep.budget} attempts expire. Used: {ep.used}. Remaining: {ep.budget - ep.used}. "
        f"Outcome: in_progress. Reaching the target ends the task successfully immediately. "
        f"Reverse moves are disallowed; all other directions are offered even if they collide. "
        f"Eating retains the tail; otherwise the tail vacates and entering that cell is allowed. "
        f"A wall or non-vacating body collision ends the task unsuccessfully. "
        f"Recent physical events: {_compact(ep.events)}."
    )
    criteria = {a: f"Move {a} to {_compact(list(sg._destination(state, a)))}." for a in sorted(actions)}
    return {"state": text,
            "questions": {"action": {"type": "choice", "instructions": INSTRUCTIONS,
                                     "criteria": criteria}}}
