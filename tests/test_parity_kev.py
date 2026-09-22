"""Runs only when a converted kev checkpoint is present; compares with Kev's own PyTorch outputs."""
import json, os
from pathlib import Path

import pytest

MODEL = os.environ.get("TINYJEV_KEV", str(Path.home() / ".cache/tinyjev/kev-0.6b"))
REF = Path(__file__).parent / "fixtures" / "kev_reference_cpu.json"


@pytest.mark.skipif(not (Path(MODEL).exists() and REF.exists()), reason="needs converted kev weights")
def test_every_fixture_matches_upstream_argmax():
    import tinyjev
    agent = tinyjev.load(MODEL)
    for case in json.loads(REF.read_text())["fixtures"]:
        got = agent.logits(case["request"])
        for qid, exp in case["expected"].items():
            mine = got[f"request:{qid}"]
            assert mine["candidate_ids"] == exp["keys"]
            ref_pick = max(range(len(exp["probabilities"])), key=exp["probabilities"].__getitem__)
            my_pick = max(range(len(mine["probabilities"])), key=mine["probabilities"].__getitem__)
            assert ref_pick == my_pick, f"{case['name']}:{qid}"
            worst = max(abs(a - b) for a, b in zip(exp["probabilities"], mine["probabilities"]))
            assert worst < 2e-2, f"{case['name']}:{qid}: probability delta {worst}"
