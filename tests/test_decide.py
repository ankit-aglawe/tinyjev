import pytest
from tinyjev.decide import answer_from_probabilities, candidates, prepare_examples, validate_request


def fake_encode(text):
    return [ord(c) % 97 + 1 for c in text]  # deterministic, one id per char


def q_choice(**crit):
    return {"type": "choice", "instructions": "Pick.", "criteria": crit}


def test_rejects_wrong_top_level():
    with pytest.raises(ValueError):
        validate_request({"state": "x"})


def test_rejects_single_option_choice():
    with pytest.raises(ValueError):
        validate_request({"states": [{"id": "a", "state": "s", "questions": {"q": q_choice(only="one")}}]})


def test_rejects_duplicate_state_ids():
    s = {"id": "a", "state": "s", "questions": {"q": q_choice(x="1", y="2")}}
    with pytest.raises(ValueError):
        validate_request({"states": [s, dict(s)]})


def test_candidates_shapes():
    assert candidates({"type": "boolean"}) == (["false", "true"], ["The proposition is true."])
    ids, texts = candidates(q_choice(left="go left", right="go right"))
    assert ids == ["left", "right"] and texts == ["left: go left", "right: go right"]
    ids, texts = candidates({"type": "score", "criteria": ["low", "high"]})
    assert ids == ["0", "1"] and texts == ["low", "high"]


def test_prepare_examples_one_path_per_candidate_and_eos_last():
    payload = {"states": [{"id": "a", "state": "hello", "questions": {
        "c": q_choice(x="1", y="2", z="3"),
        "b": {"type": "boolean", "instructions": "Is it?"},
        "s": {"type": "score", "instructions": "Rate.", "criteria": ["a", "b"]}}}]}
    ex = prepare_examples(payload, fake_encode, eos_token_id=999, max_length=4096)
    by_id = {e["qid"]: e for e in ex}
    assert len(by_id["c"]["leaf_tokens"]) == 3
    assert len(by_id["b"]["leaf_tokens"]) == 1 and len(by_id["b"]["candidate_ids"]) == 2
    assert len(by_id["s"]["leaf_tokens"]) == 2
    assert all(path[-1] == 999 for e in ex for path in e["leaf_tokens"])


def test_prepare_examples_refuses_to_truncate():
    payload = {"states": [{"id": "a", "state": "x" * 500, "questions": {"c": q_choice(x="1", y="2")}}]}
    with pytest.raises(ValueError, match="max_length"):
        prepare_examples(payload, fake_encode, eos_token_id=1, max_length=64)


def test_dict_state_uses_python_repr_like_upstream():
    payload = {"states": [{"id": "a", "state": {"k": True}, "questions": {"c": q_choice(x="1", y="2")}}]}
    seen = []
    prepare_examples(payload, lambda t: (seen.append(t), [1])[1], eos_token_id=1, max_length=4096)
    assert any("{'k': True}" in t for t in seen)


def test_answer_shapes():
    ex = {"type": "choice", "candidate_ids": ["a", "b"]}
    assert answer_from_probabilities(ex, [0.2, 0.8])["choice"] == "b"
    ex = {"type": "boolean", "candidate_ids": ["false", "true"]}
    assert answer_from_probabilities(ex, [0.3, 0.7])["value"] is True
    ex = {"type": "score", "candidate_ids": ["0", "1", "2"]}
    assert abs(answer_from_probabilities(ex, [0.0, 0.5, 0.5])["score"] - 1.5) < 1e-9
    with pytest.raises(ValueError):
        answer_from_probabilities(ex, [0.5, 0.5, 0.5])
