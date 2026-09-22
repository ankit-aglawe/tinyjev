"""Request schema, prompt construction and answer shaping.

Ported from NanoJev's `predict_toy_decisions` (MIT, OpenJev contributors). The wire
format and the exact token sequence are reproduced byte for byte; changing either
would invalidate parity with the original checkpoint.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List

QUESTION_TYPES = {"boolean", "choice", "score"}


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_request(payload: Dict[str, Any]) -> List[dict]:
    if not isinstance(payload, dict) or set(payload) != {"states"}:
        raise ValueError('request must be exactly {"states": [...]}')
    states = payload["states"]
    if not isinstance(states, list) or not states:
        raise ValueError("states must be a non-empty array")
    seen = set()
    for state in states:
        if not isinstance(state, dict) or set(state) != {"id", "state", "questions"}:
            raise ValueError("each state must contain exactly id, state and questions")
        if not _nonempty_text(state["id"]) or state["id"] in seen:
            raise ValueError("state id must be a unique non-empty string")
        seen.add(state["id"])
        if not isinstance(state["state"], (str, dict, list)) or not state["state"]:
            raise ValueError("state content must be a non-empty string, object or array")
        questions = state["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object")
        for qid, q in questions.items():
            where = f"{state['id']}:{qid}"
            if not _nonempty_text(qid) or not isinstance(q, dict):
                raise ValueError(f"{where}: question id must be a non-empty string")
            if set(q) - {"type", "instructions", "criteria"}:
                raise ValueError(f"{where}: unsupported question field")
            typ = q.get("type")
            if typ not in QUESTION_TYPES or not _nonempty_text(q.get("instructions")):
                raise ValueError(f"{where}: invalid type or instructions")
            criteria = q.get("criteria")
            if typ == "boolean":
                if "criteria" in q:
                    if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
                        raise ValueError(f"{where}: boolean criteria may only hold false/true")
                    if not all(_nonempty_text(v) for v in criteria.values()):
                        raise ValueError(f"{where}: boolean criterion must be a non-empty string")
            elif typ == "choice":
                if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                    raise ValueError(f"{where}: choice criteria must hold 2-255 entries")
                if not all(_nonempty_text(k) and _nonempty_text(v) for k, v in criteria.items()):
                    raise ValueError(f"{where}: choice ids and descriptions must be non-empty")
            else:
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                    raise ValueError(f"{where}: score criteria must be an ordered array of 2-10")
                if not all(_nonempty_text(v) for v in criteria):
                    raise ValueError(f"{where}: score level descriptions must be non-empty")
    return states


def candidates(question: dict) -> tuple[List[str], List[str]]:
    """Return (candidate ids, candidate texts). Boolean has two ids but one path."""
    typ = question["type"]
    if typ == "boolean":
        return ["false", "true"], ["The proposition is true."]
    if typ == "choice":
        ids = list(question["criteria"])
        return ids, [f"{k}: {question['criteria'][k]}" for k in ids]
    criteria = question["criteria"]
    return [str(i) for i in range(len(criteria))], list(criteria)


def prepare_examples(payload, encode, eos_token_id: int, max_length: int) -> List[dict]:
    """Tokenize one path per candidate.

    Layout per path, matching upstream exactly:
        State:\\n{state}\\n
        Question type: {type}\\nQuestion:\\n{instructions}\\n
        [False criterion: ...\\n] [True criterion: ...\\n]
        Candidate:\\n{candidate}\\nDecision:<eos>

    The state is interpolated with str(), not json.dumps(), so dict and list states
    render as Python reprs. That is what the model was trained on.
    """
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("tokenizer must expose a valid eos_token_id")

    examples = []
    for row in validate_request(payload):
        for qid, q in row["questions"].items():
            typ = q["type"]
            ids, texts = candidates(q)
            head = f"Question type: {typ}\nQuestion:\n{q['instructions']}\n"
            if typ == "boolean" and "criteria" in q:
                for key, label in (("false", "False"), ("true", "True")):
                    if key in q["criteria"]:
                        head += f"{label} criterion: {q['criteria'][key]}\n"
            prefix = encode(f"State:\n{row['state']}\n") + encode(head)
            leaves = [prefix + encode(f"Candidate:\n{t}\nDecision:") + [eos_token_id]
                      for t in texts]
            longest = max(map(len, leaves))
            if longest > max_length:
                raise ValueError(
                    f"{row['id']}:{qid}: candidate path is {longest} tokens, over "
                    f"max_length={max_length}; input was not truncated")
            examples.append({"id": f"{row['id']}:{qid}", "state_id": row["id"], "qid": qid,
                             "type": typ, "candidate_ids": ids, "candidate_texts": texts,
                             "leaf_tokens": leaves})
    return examples


def answer_from_probabilities(example: dict, probabilities: List[float]) -> dict:
    ids = example["candidate_ids"]
    if len(probabilities) != len(ids) or not all(
            math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ValueError("model produced invalid probabilities")
    if abs(math.fsum(probabilities) - 1.0) > 1e-5:
        raise ValueError("model probabilities do not sum to 1")
    best = max(range(len(ids)), key=probabilities.__getitem__)
    result = {"type": example["type"], "probabilities": dict(zip(ids, probabilities))}
    if example["type"] == "boolean":
        result.update(p_true=probabilities[1], value=bool(best))
    elif example["type"] == "choice":
        result.update(choice=ids[best], value=ids[best])
    else:
        score = math.fsum(i * p for i, p in enumerate(probabilities))
        result.update(score=score, level=best, value=score)
    return result
