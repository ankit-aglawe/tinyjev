import pytest
from tinyjev.serve import native_to_systemone, systemone_to_native


def test_systemone_translation_round_trip():
    body = {"state": "Charged twice.", "questions": {
        "billing": {"type": "noul", "instructions": "About billing?"},
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "sales": "deals"}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]}}}
    native = systemone_to_native(body)
    qs = native["states"][0]["questions"]
    assert qs["billing"]["type"] == "boolean"
    assert qs["team"]["criteria"]["billing"] == "billing"  # null description filled from the id
    assert qs["urgency"]["type"] == "score"

    fake = {"states": [{"id": "request", "answers": {
        "billing": {"type": "boolean", "p_true": 0.9, "value": True, "probabilities": {"false": 0.1, "true": 0.9}},
        "team": {"type": "choice", "choice": "billing", "value": "billing", "probabilities": {"billing": 0.8, "sales": 0.2}},
        "urgency": {"type": "score", "score": 0.7, "level": 1, "value": 0.7, "probabilities": {"0": 0.3, "1": 0.7}}}}],
        "execution": {"model_ms": 12.3}}
    out = native_to_systemone(fake)
    assert out["answers"]["billing"]["type"] == "noul" and out["answers"]["billing"]["noul"] == 0.9
    assert out["answers"]["team"]["choice"] == "billing"
    assert out["latency_ms"] == 12.3


def test_systemone_rejects_unknown_type():
    with pytest.raises(ValueError):
        systemone_to_native({"state": "s", "questions": {"q": {"type": "essay", "instructions": "x"}}})
