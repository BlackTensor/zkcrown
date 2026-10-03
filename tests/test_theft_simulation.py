"""Tests for P6.4: the theft simulation's own pieces. No `K`, no CIFAR-10."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cryptography")
pytest.importorskip("torch")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from src.crypto.commitment import commit  # noqa: E402
from src.crypto.fingerprint import fingerprint_state_dict  # noqa: E402
from src.crypto.provenance import build_unsigned_record  # noqa: E402
from src.crypto.publication import build_publication  # noqa: E402
from src.crypto.signing import public_key_bytes, sign_record  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OWNER = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


@pytest.fixture
def sim(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("theft_simulation")


def state(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {"w": rng.standard_normal((3, 2)).astype(np.float32)}


def test_every_theft_points_at_committed_phase_4_files(sim):
    for name, spec in sim.THEFTS.items():
        assert (REPO_ROOT / sim.ROWS / spec["row"]).exists(), name
        assert ("live" in spec) != ("apply" in spec), name
        if "apply" in spec:
            assert (REPO_ROOT / sim.ROWS / spec["apply"]).exists(), name


def test_live_thefts_match_their_committed_configs(sim):
    """A live theft must be the very attack its committed row ran, or the reproduction check means nothing."""
    from src.utils.results import read_result

    for name, spec in sim.THEFTS.items():
        if "live" not in spec:
            continue
        config = read_result(REPO_ROOT / sim.ROWS / spec["row"])["metrics"]["row"]["config"]
        attack, strength, params = spec["live"]
        assert (config["attack"], config["strength"], config["params"]) == (attack, strength, params), name


def test_reproduces_stops_on_any_difference(sim):
    row = {"clean_accuracy": {"correct": 9000}, "behavioral": {"fired": 50}, "weight": {"applicable": True, "z": 5.0}}
    sim.reproduces(row, row)
    for path, value in ((("clean_accuracy", "correct"), 9001), (("behavioral", "fired"), 49),
                        (("weight", "z"), 5.0 + 1e-6), (("weight", "applicable"), False)):
        other = {k: dict(v) for k, v in row.items()}
        other[path[0]][path[1]] = value
        with pytest.raises(SystemExit):
            sim.reproduces(other, row)


def test_reproduces_skips_z_when_not_applicable(sim):
    row = {"clean_accuracy": {"correct": 1}, "behavioral": {"fired": 1}, "weight": {"applicable": False}}
    sim.reproduces(row, row)


@pytest.mark.parametrize("named,beh,wgt,text", [
    (True, True, True, "exact copy"),
    (False, True, True, "behavioral and weight watermark detected"),
    (False, False, True, "weight watermark detected"),
    (False, True, False, "behavioral watermark detected"),
    (False, False, False, "no watermark detected"),
])
def test_evidence_text(sim, named, beh, wgt, text):
    assert sim.evidence(named, {"behavioral_detected": beh, "weight_detected": wgt}).startswith(text)


def test_backdated_time_precedes_the_owner_publication(sim):
    assert sim.BACKDATED_UTC < "2026-10-02T18:05:06+00:00"


def test_counter_claim_verifies_on_its_own_terms_but_not_under_the_owner_key(sim, monkeypatch):
    owner_model, shipped = state(1), state(2)
    publication = build_publication(commit(bytes(range(32)), bytes(range(100, 116)), bytes(range(200, 231))),
                                    fingerprint_state_dict(owner_model), "test-owner", "test", "2026-10-02T12:00:00+00:00")
    record = sign_record(build_unsigned_record(publication, "cd" * 32, 100, public_key_bytes(OWNER),
                                               "2026-10-02T13:00:00+00:00"), OWNER)
    monkeypatch.setattr(sim, "OWNER_PUBLIC_KEY", public_key_bytes(OWNER).hex())
    owner = {"record": record, "publication": publication,
             "summary": {"independent_time": {"bitcoin_block": 1, "header_time_utc": "2026-10-02T14:00:00+00:00"}}}
    row = {"attacked": {"info": {"weight": {"attacker_after": {"z": 10.4, "detected": True},
                                            "attacker_before": {"z": 0.4, "detected": False}}}},
           "table": {"behavioral_detected": True, "weight_detected": True}}
    claim = sim.counter_claim(shipped, owner, row, owner_model)
    assert claim["thief_claim_verifies_on_its_own_terms"] and claim["thief_claim_names_the_shipped_model"]
    assert claim["thief_claim_under_owner_trusted_key"] is False
    assert claim["self_asserted_times_say_thief_first"] and not claim["thief_claim_independently_timestamped"]
    assert claim["owner_record_names_owner_published_model"]
    assert not claim["thief_weight_detected_on_owner_published_model"]


def test_counter_claim_stops_if_the_owner_record_does_not_name_the_published_model(sim, monkeypatch):
    owner_model = state(1)
    publication = build_publication(commit(bytes(range(32)), bytes(range(100, 116)), bytes(range(200, 231))),
                                    fingerprint_state_dict(owner_model), "test-owner", "test", "2026-10-02T12:00:00+00:00")
    record = sign_record(build_unsigned_record(publication, "cd" * 32, 100, public_key_bytes(OWNER),
                                               "2026-10-02T13:00:00+00:00"), OWNER)
    monkeypatch.setattr(sim, "OWNER_PUBLIC_KEY", public_key_bytes(OWNER).hex())
    owner = {"record": record, "publication": publication, "summary": {"independent_time": {}}}
    row = {"attacked": {"info": {"weight": {"attacker_after": {"z": 1, "detected": True},
                                            "attacker_before": {"z": 0, "detected": False}}}},
           "table": {"behavioral_detected": True, "weight_detected": True}}
    with pytest.raises(SystemExit):
        sim.counter_claim(state(2), owner, row, state(3))  # not the model the record names


@pytest.mark.skipif(not (REPO_ROOT / "provenance" / "commitment.json.ots").exists(), reason="no committed proof")
def test_publication_step_on_the_committed_files(sim):
    owner = sim.publication_step()
    t = owner["summary"]["independent_time"]
    assert t["bitcoin_block"] == 969627 and t["header_time_utc"] == "2026-10-02T19:37:33+00:00"
    assert owner["summary"]["record_verdict"]["record_valid"]
    assert owner["summary"]["named_model_fingerprint"].startswith("c0995109")
