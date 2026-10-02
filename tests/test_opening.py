"""Tests for P5.6: verifying a revealed opening against a published commitment.

All keys and nonces here are test values, never the owner's.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.crypto.commitment import commit
from src.crypto.fingerprint import FINGERPRINT_VERSION, ModelFingerprint
from src.crypto.opening import OPENING_SCHEMA, REVEALED_BYTES, Opening, verify_opening
from src.crypto.publication import build_publication, write_publication
from src.watermark.signature import derive_signature

REPO_ROOT = Path(__file__).resolve().parents[1]

OWNER = "test-owner"
KEY = bytes(range(32))
SIG = derive_signature(KEY, OWNER).value
NONCE = bytes(range(200, 231))
TRUE = Opening(KEY, SIG, NONCE)
FINGERPRINT = ModelFingerprint(sha256="ab" * 32, version=FINGERPRINT_VERSION, tensors=1, elements=1, data_bytes=4)


def publication(key=KEY, sig=SIG, nonce=NONCE, owner=OWNER) -> dict:
    return build_publication(commit(key, sig, nonce), FINGERPRINT, owner, "test model", "2026-10-02T12:00:00+00:00")


# --- the verdict --------------------------------------------------------------


def test_true_opening_is_accepted():
    verdict = verify_opening(publication(), TRUE)
    assert verdict.valid and verdict.problem is None
    assert verdict.to_dict() == {
        "valid": True, "publication_well_formed": True, "commitment_matches": True,
        "signature_derives_from_key": True, "published_commitment": commit(KEY, SIG, NONCE).decimal(),
        "owner_id": OWNER, "problem": None, "secret_bytes_revealed_to_verifier": 79,
    }
    assert REVEALED_BYTES == 32 + 16 + 31


@pytest.mark.parametrize("opening", [
    Opening(bytes([1]) + KEY[1:], SIG, NONCE),
    Opening(KEY, bytes([SIG[0] ^ 1]) + SIG[1:], NONCE),
    Opening(KEY, SIG, NONCE[:-1] + bytes([NONCE[-1] ^ 1])),
    Opening(KEY[16:] + KEY[:16], SIG, NONCE),
    Opening(KEY, SIG, bytes(31)),
])
def test_wrong_openings_are_rejected_without_raising(opening):
    verdict = verify_opening(publication(), opening)
    assert not verdict.valid and not verdict.commitment_matches and verdict.publication_well_formed
    assert "does not hash" in verdict.problem


def test_commitment_to_a_foreign_signature_is_caught_by_the_derivation_check():
    """C can be opened, but S is not what K gives for the published owner: not a valid ownership opening."""
    foreign = bytes(range(50, 66))
    verdict = verify_opening(publication(sig=foreign), Opening(KEY, foreign, NONCE))
    assert verdict.commitment_matches and not verdict.signature_derives_from_key and not verdict.valid
    assert "owner id" in verdict.problem


def test_changing_the_published_owner_id_is_caught():
    """An impostor republishes the same C under their own name. The opening no longer fits."""
    stolen = {**publication(), "owner_id": "impostor"}
    verdict = verify_opening(stolen, TRUE)
    assert verdict.commitment_matches and not verdict.signature_derives_from_key and not verdict.valid


def test_malformed_publication_is_a_verdict_not_a_crash():
    for broken in ({}, None, {**publication(), "extra": 1}, {**publication(), "schema": "other"}):
        verdict = verify_opening(broken, TRUE)
        assert not verdict.valid and not verdict.publication_well_formed and "malformed" in verdict.problem
        assert verdict.published_commitment is None


def test_verdict_and_repr_hold_no_secret():
    text = json.dumps(verify_opening(publication(), TRUE).to_dict()) + repr(TRUE) + str(TRUE)
    for blob in (KEY, SIG, NONCE):
        assert blob.hex() not in text
    assert repr(TRUE) == "Opening(<hidden>)"


def test_verifier_needs_only_the_artifact_and_the_opening(tmp_path, monkeypatch):
    """Run from an empty directory with no secrets/ anywhere near."""
    monkeypatch.chdir(tmp_path)
    assert verify_opening(publication(), TRUE).valid


# --- the opening as data ------------------------------------------------------


def test_opening_round_trips_through_its_dict():
    data = TRUE.to_dict()
    assert data["schema"] == OPENING_SCHEMA and set(data) == {"schema", "key_hex", "signature_hex", "nonce_hex"}
    assert Opening.from_dict(json.loads(json.dumps(data))) == TRUE


@pytest.mark.parametrize("key, sig, nonce, error", [
    (KEY[:31], SIG, NONCE, ValueError), (KEY, SIG[:15], NONCE, ValueError), (KEY, SIG, NONCE + b"\0", ValueError),
    (KEY.hex(), SIG, NONCE, TypeError), (KEY, None, NONCE, TypeError),
])
def test_malformed_openings_raise(key, sig, nonce, error):
    with pytest.raises(error):
        Opening(key, sig, nonce)


def test_from_dict_refuses_bad_files():
    good = TRUE.to_dict()
    for bad in ({**good, "schema": "v0"}, {**good, "key_hex": "zz"}, {**good, "key_hex": 5},
                {**good, "key_hex": good["key_hex"][:-2]}, {k: v for k, v in good.items() if k != "nonce_hex"},
                {**good, "extra": "x"}, [good], "opening"):
        with pytest.raises(ValueError):
            Opening.from_dict(bad)
    with pytest.raises(TypeError):
        verify_opening(publication(), good)  # a dict is not an Opening


# --- the script ---------------------------------------------------------------


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p5_6_verify_opening")


def test_wrong_opening_family(script, monkeypatch):
    monkeypatch.setattr(script, "PROJECT_OWNER_ID", OWNER)
    cases = list(script.wrong_openings(TRUE, seed=7, n_keys=3))
    kinds = [kind for kind, _ in cases]
    assert kinds.count("key_bit_flip") == 256 and kinds.count("signature_bit_flip") == 128
    assert kinds.count("nonce_bit_flip") == 248 and kinds.count("wrong_key_true_signature") == 3
    assert kinds.count("wrong_key_own_signature") == 3 and len(cases) == 632 + 3 + 6
    assert all(opening != TRUE for _, opening in cases)
    assert script.wrong_key(7, 0) != script.wrong_key(7, 1) != script.wrong_key(8, 1)
    own = [o for kind, o in cases if kind == "wrong_key_own_signature"][0]
    assert derive_signature(own.key, OWNER).value == own.signature  # internally consistent, still not our C


def test_self_check_passes_and_reports_aggregates_only(script, monkeypatch):
    monkeypatch.setattr(script, "PROJECT_OWNER_ID", OWNER)
    monkeypatch.setattr(script, "TIMING_REPEATS", 2)
    metrics = script.self_check(publication(), TRUE, seed=7, n_keys=5)
    assert metrics["wrong_openings_tried"] == 632 + 3 + 10 and metrics["wrong_openings_accepted"] == 0
    assert metrics["true_opening"]["valid"] and metrics["seconds_per_verification"] > 0
    text = json.dumps(metrics)
    assert KEY.hex() not in text and NONCE.hex() not in text and SIG.hex() not in text


def test_self_check_stops_when_the_true_opening_does_not_fit(script, monkeypatch):
    monkeypatch.setattr(script, "PROJECT_OWNER_ID", OWNER)
    with pytest.raises(SystemExit, match="true opening was rejected"):
        script.self_check(publication(nonce=bytes(31)), TRUE, seed=7, n_keys=1)


def test_self_check_catches_a_verifier_that_accepts_everything(script, monkeypatch):
    monkeypatch.setattr(script, "PROJECT_OWNER_ID", OWNER)
    real = script.verify_opening
    monkeypatch.setattr(script, "verify_opening", lambda p, o: real(p, TRUE))
    with pytest.raises(SystemExit, match="wrong opening was accepted"):
        script.self_check(publication(), TRUE, seed=7, n_keys=1)


def test_third_party_mode_reads_no_secrets(script, tmp_path, capsys):
    artifact = tmp_path / "commitment.json"
    write_publication(publication(), artifact)
    good = tmp_path / "opening.json"
    good.write_text(json.dumps(TRUE.to_dict()))
    verdict = script.main(["--artifact", str(artifact), "--opening", str(good)])
    assert verdict["valid"] is True
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(Opening(KEY, SIG, bytes(31)).to_dict()))
    with pytest.raises(SystemExit):
        script.main(["--artifact", str(artifact), "--opening", str(bad)])
    assert KEY.hex() not in capsys.readouterr().out
