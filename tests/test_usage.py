"""Usage accounting through the public APIs, without model downloads or a GPU."""
import json
from http.server import HTTPServer
from threading import Thread
from urllib.request import Request, urlopen

import numpy as np
import pytest
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models, pre_tokenizers

from tinyjev import backends
from tinyjev.agent import Agent
from tinyjev.families.pointer import SPECIAL
from tinyjev.serve import make_handler


QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Urgent?"},
    "direction": {"type": "choice", "instructions": "Pick.",
                  "criteria": {"a": "left", "b": "right"}},
    "level": {"type": "score", "instructions": "How?", "criteria": ["low", "high"]},
}


class ZeroBackbone:
    name = "test"

    def __init__(self):
        self.calls = []

    def hidden_rows(self, prefix, rows, pad):
        self.calls.append((list(prefix), [list(row) for row in rows]))
        return [np.zeros((len(prefix) + len(row), 2), dtype=np.float32) for row in rows]


@pytest.fixture(params=["pointer", "marker"])
def agent(request, tmp_path, monkeypatch):
    # A byte-level tokenizer with no merges: one token per UTF-8 byte, except
    # registered special tokens. Counts below can be checked without re-encoding.
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.add_special_tokens(SPECIAL + ["<eos>"])
    tok.save(str(tmp_path / "tokenizer.json"))
    manifest = {
        "format": "tinyjev-v2", "family": request.param, "name": "test-model",
        "tokenizer": {"eos_token_id": tok.token_to_id("<eos>"), "pad_token_id": 0},
        "head": {"head_dim": 2, "set_head": "none"},
    }
    (tmp_path / "tinyjev.json").write_text(json.dumps(manifest))
    (tmp_path / "config.json").write_text("{}")
    save_file({
        "q.weight": np.zeros((2, 2), dtype=np.float32),
        "q.bias": np.zeros(2, dtype=np.float32),
        "k.weight": np.zeros((2, 2), dtype=np.float32),
        "k.bias": np.zeros(2, dtype=np.float32),
        "norm.weight": np.ones(2, dtype=np.float32),
        "norm.bias": np.zeros(2, dtype=np.float32),
        "scalar.weight": np.zeros((1, 2), dtype=np.float32),
        "scalar.bias": np.zeros(1, dtype=np.float32),
    }, str(tmp_path / "head.safetensors"))
    monkeypatch.setattr(backends, "make", lambda *args, **kwargs: ZeroBackbone())
    return Agent(tmp_path, backend="test")


@pytest.mark.parametrize("state", ["hello", "café 🐦 <|box_end|>"])
@pytest.mark.parametrize("keys", [["urgent"], ["direction"], ["level"], list(QUESTIONS)])
def test_exact_encoded_input_and_zero_generated_output(agent, state, keys, monkeypatch):
    if agent.family.name == "pointer":
        # One state marker, plus two branch markers and two markers per option.
        prefix = 1 + len(state.replace("<|box_end|>", "<¦box_end¦>").encode("utf-8"))
        suffixes = {
            "urgent": 6 + len("Urgent?noyes"),
            "direction": 6 + len("Pick.a: leftb: right"),
            "level": 6 + len("How?lowhigh"),
        }
    else:
        prefix = len(f"State:\n{state}\n".encode("utf-8"))
        # Marker encoding preserves registered special tokens as one token each.
        prefix -= state.count("<|box_end|>") * (len("<|box_end|>") - 1)
        # Each candidate includes the question text and one EOS token.
        suffixes = {
            "urgent": len("Question type: boolean\nQuestion:\nUrgent?\n"
                          "Candidate:\nThe proposition is true.\nDecision:") + 1,
            "direction": 2 * len("Question type: choice\nQuestion:\nPick.\nCandidate:\n\nDecision:")
                         + len("a: leftb: right") + 2,
            "level": 2 * len("Question type: score\nQuestion:\nHow?\nCandidate:\n\nDecision:")
                     + len("lowhigh") + 2,
        }
    encode = agent.family.encode
    encoded_records = []

    def count_encode(record):
        encoded_records.append(record)
        return encode(record)

    monkeypatch.setattr(agent.family, "encode", count_encode)
    response = agent.systemone({"state": state, "questions": {key: QUESTIONS[key] for key in keys}})
    assert response["usage"] == {"input_tokens": prefix + sum(suffixes[key] for key in keys),
                                 "output_tokens": 0}
    assert set(response["answers"]) == set(keys)
    assert response["model"] == "test-model"
    assert response["latency_ms"] >= 0
    assert len(encoded_records) == len(agent.backbone.calls) == 1


@pytest.mark.parametrize("state", [{"text": "café 🐦", "tags": ["urgent", "billing"]},
                                   ["hello", "world"], "long shared state " * 100])
def test_shared_prefix_counted_once_and_padding_excluded(agent, state):
    payload = {"state": state, "questions": QUESTIONS}
    native = agent.predict(payload)
    compatible = agent.systemone(payload)
    paths = agent.logits(payload)
    prefix, rows = agent.backbone.calls[0]
    path_lengths = [n for answer in paths.values() for n in answer["path_token_counts"]]
    assert len(path_lengths) == len(rows) > 1
    assert len(set(path_lengths)) > 1  # A real backend would pad unequal rows.
    expected = sum(path_lengths) - (len(rows) - 1) * len(prefix)
    assert native["usage"] == compatible["usage"] == {"input_tokens": expected, "output_tokens": 0}
    assert expected < len(rows) * max(path_lengths)
    assert native["execution"]["autoregressive_decode_steps"] == 0
    assert native["execution"]["candidate_paths"] == len(rows)
    assert set(paths["request:direction"]) == {
        "type", "logits", "probabilities", "answer", "path_token_counts", "candidate_ids"}


def test_multiple_states_sum_usage_without_carrying_it_between_requests(agent):
    states = [{"id": "one", "state": "first", "questions": QUESTIONS},
              {"id": "two", "state": "second 🐦", "questions": QUESTIONS}]
    individual = [agent.predict({"states": [state]})["usage"]["input_tokens"] for state in states]
    response = agent.predict({"states": states})
    assert response["usage"] == {"input_tokens": sum(individual), "output_tokens": 0}
    assert response["execution"]["states"] == 2
    assert agent.predict({"states": [states[0]]})["usage"]["input_tokens"] == individual[0]


def test_question_ids_and_serialized_answers_are_not_counted(agent):
    payload = {"state": "hello", "questions": {"q": QUESTIONS["urgent"]}}
    first = agent.systemone(payload)
    payload["questions"] = {"a much longer question identifier 🐦": QUESTIONS["urgent"]}
    second = agent.systemone(payload)
    assert len(json.dumps(first["answers"])) != len(json.dumps(second["answers"]))
    assert first["usage"] == second["usage"]


@pytest.mark.parametrize("path", ["/predict", "/v1/systemone"])
def test_http_response_contains_usage(agent, path):
    payload = {"state": "hello", "questions": QUESTIONS}
    expected = agent.predict(payload)["usage"]
    with HTTPServer(("127.0.0.1", 0), make_handler(agent)) as server:
        worker = Thread(target=server.handle_request, daemon=True)
        worker.start()
        try:
            request = Request(f"http://127.0.0.1:{server.server_port}{path}",
                              data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json", "Connection": "close"})
            with urlopen(request, timeout=5) as response:
                assert response.status == 200
                assert json.load(response)["usage"] == expected
        finally:
            worker.join(timeout=5)
        assert not worker.is_alive()
