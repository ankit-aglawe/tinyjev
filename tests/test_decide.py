import pytest
from tinyjev.agent import normalize_request
from tinyjev.families.marker import candidates, validate_question


def q_choice(**crit):
    return {"type": "choice", "instructions": "Pick.", "criteria": crit}


def test_native_shape_normalizes():
    recs = normalize_request({"states": [{"id": "a", "state": "s", "questions": {"q": q_choice(x="1", y="2")}}]})
    assert recs[0]["id"] == "a" and recs[0]["questions"][0]["id"] == "q"


def test_systemone_shape_normalizes_and_aliases_noul():
    recs = normalize_request({"state": "s", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}})
    assert recs[0]["id"] == "request" and recs[0]["questions"][0]["type"] == "boolean"


def test_rejects_wrong_top_level():
    with pytest.raises(ValueError):
        normalize_request({"foo": "x"})


def test_rejects_duplicate_state_ids():
    s = {"id": "a", "state": "s", "questions": {"q": q_choice(x="1", y="2")}}
    with pytest.raises(ValueError):
        normalize_request({"states": [s, dict(s)]})


def test_rejects_unknown_question_type():
    with pytest.raises(ValueError):
        normalize_request({"state": "s", "questions": {"q": {"type": "essay", "instructions": "x"}}})


def test_nanojev_rejects_single_option_choice():
    with pytest.raises(ValueError):
        validate_question("a:q", {"id": "q", **q_choice(only="one")})


def test_nanojev_candidates_shapes():
    assert candidates({"type": "boolean"}) == (["false", "true"], ["The proposition is true."])
    ids, texts = candidates(q_choice(left="go left", right="go right"))
    assert ids == ["left", "right"] and texts == ["left: go left", "right: go right"]
    ids, texts = candidates({"type": "score", "criteria": ["low", "high"]})
    assert ids == ["0", "1"] and texts == ["low", "high"]
