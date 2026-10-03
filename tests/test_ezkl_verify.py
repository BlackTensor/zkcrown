"""P8.5: tamper-case construction for the EZKL proof checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
try:
    import p8_5_ezkl_verify as v  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "experiments"))

PROOF = REPO_ROOT / "results/zk/p8.4/proof.json"


def test_field_encoding_round_trips_including_negatives():
    pf = json.loads(PROOF.read_text())
    for h in pf["instances"][0][:5] + pf["instances"][0][-10:]:
        assert v.unfelt(v.felt(h)) == h
    assert v.felt(v.unfelt(-1)) == v.BN254_R - 1


def test_bit_flip_changes_both_proof_encodings_by_one_bit():
    pf = json.loads(PROOF.read_text())
    out = v.with_bit_flip(pf, 1536, 7)
    assert sum(a != b for a, b in zip(pf["proof"], out["proof"])) == 1
    assert out["proof"][1536] == pf["proof"][1536] ^ 0x80
    assert out["hex_proof"] == "0x" + bytes(out["proof"]).hex() != pf["hex_proof"]
    assert pf["proof"][1536] == json.loads(PROOF.read_text())["proof"][1536]  # input not mutated


def test_every_negative_case_differs_from_the_honest_proof():
    pf = json.loads(PROOF.read_text())
    other = list(reversed(pf["instances"][0]))
    cases = v.negative_cases(pf, other, scale=13)
    assert len(cases) == 7 + 6 + 24 + 3 + 2
    for family, label, obj in cases:
        assert obj["instances"] != pf["instances"] or obj["proof"] != pf["proof"], label
        assert obj["pretty_public_inputs"] == pf["pretty_public_inputs"]
