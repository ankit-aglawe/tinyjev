"""Runs only when a converted checkpoint is available; CI without weights skips it."""
import json, os
from pathlib import Path

import pytest

MODEL = os.environ.get("TINYJEV_NANOJEV", str(Path.home() / ".cache/tinyjev/v2/tinyjev-nanojev"))
REF = Path(__file__).parent / "fixtures" / "reference_cpu.json"


@pytest.mark.skipif(not (Path(MODEL).exists() and REF.exists()), reason="needs converted weights")
def test_every_fixture_selects_the_reference_answer():
    import tinyjev
    agent = tinyjev.load(MODEL)
    ref = json.loads(REF.read_text())
    for case in ref["fixtures"]:
        got = agent.logits(case["payload"])
        for key, expected in case["expected"].items():
            actual = got[key]
            if expected["type"] == "score":
                assert abs(expected["answer"]["value"] - actual["answer"]["value"]) < 1e-2, key
            else:
                assert expected["answer"]["value"] == actual["answer"]["value"], key
            worst = max(abs(p - q) for p, q in zip(expected["probabilities"], actual["probabilities"]))
            assert worst < 5e-3, f"{key}: probability delta {worst}"
