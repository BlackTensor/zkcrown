"""Tests for P6.3: the provenance verifier.

All keys and models here are test values. The committed record, where present,
is checked with public values only.
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

import numpy as np  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from src.crypto.commitment import Commitment, commit  # noqa: E402
from src.crypto.fingerprint import fingerprint_state_dict  # noqa: E402
from src.crypto.poseidon import BN254_SCALAR_FIELD  # noqa: E402
from src.crypto.provenance import (  # noqa: E402
    RECORD_PATH,
    attach_signature,
    build_unsigned_record,
    read_record,
    record_sha256,
    without_signature,
)
from src.crypto.provenance_verifier import ProvenanceVerdict, verify_provenance  # noqa: E402
from src.crypto.publication import ARTIFACT_PATH, build_publication, read_publication  # noqa: E402
from src.crypto.signing import public_key_bytes, sign_record  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OWNER = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
FORGER = Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64)))
CREATED = "2026-10-03T09:30:00+00:00"


def state(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"a.weight": rng.standard_normal((4, 3)).astype(np.float32),
            "a.bias": rng.standard_normal(4).astype(np.float32),
            "bn.num_batches_tracked": np.array(7, dtype=np.int64)}


def publication(model_state: dict | None = None, owner_id: str = "test-owner") -> dict:
    fp = fingerprint_state_dict(state() if model_state is None else model_state)
    return build_publication(commit(bytes(range(32)), bytes(range(100, 116)), bytes(range(200, 231))), fp,
                             owner_id, "test model", "2026-10-02T12:00:00+00:00")


def signed(pub: dict | None = None, key: Ed25519PrivateKey = OWNER) -> dict:
    pub = publication() if pub is None else pub
    return sign_record(build_unsigned_record(pub, "cd" * 32, 100, public_key_bytes(key), CREATED), key)


OWNER_HEX = public_key_bytes(OWNER).hex()


# --- the genuine case ---------------------------------------------------------


def test_genuine_record_passes_every_check():
    verdict = verify_provenance(signed(), publication(), suspect=state(), trusted_public_key=OWNER_HEX)
    assert verdict.valid and verdict.record_valid
    assert verdict.fingerprint_matches is True and verdict.public_key_trusted is True
    assert verdict.checks_not_run == () and verdict.problems == ()
    assert verdict.record_sha256 == record_sha256(signed())
    assert verdict.owner_id == "test-owner" and verdict.public_key_hex == OWNER_HEX


def test_optional_checks_are_reported_as_not_run_not_as_passed():
    verdict = verify_provenance(signed(), publication())
    assert verdict.valid and verdict.record_valid
    assert verdict.fingerprint_matches is None and verdict.public_key_trusted is None
    assert verdict.checks_not_run == ("fingerprint_matches", "public_key_trusted")
    assert verdict.to_dict()["checks_not_run"] == ["fingerprint_matches", "public_key_trusted"]


@pytest.mark.parametrize("form", ["fingerprint", "state_dict", "module"])
def test_suspect_forms(form):
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(3, 2)
    pub = publication({k: v for k, v in module.state_dict().items()})
    suspect = {"fingerprint": fingerprint_state_dict(module.state_dict()), "state_dict": module.state_dict(),
               "module": module}[form]
    assert verify_provenance(signed(pub), pub, suspect=suspect).fingerprint_matches is True


def test_trusted_key_accepts_bytes_and_hex_and_refuses_bad_lengths():
    assert verify_provenance(signed(), publication(), trusted_public_key=public_key_bytes(OWNER)).public_key_trusted
    assert verify_provenance(signed(), publication(), trusted_public_key=OWNER_HEX.upper()).public_key_trusted
    with pytest.raises(ValueError):
        verify_provenance(signed(), publication(), trusted_public_key=b"\x00" * 31)
    with pytest.raises(ValueError):
        verify_provenance(signed(), publication(), trusted_public_key="ab" * 33)
    with pytest.raises(TypeError):
        verify_provenance(signed(), publication(), trusted_public_key=5)
    with pytest.raises(TypeError):
        verify_provenance(signed(), publication(), suspect=[1, 2, 3])


# --- each check fails on its own ----------------------------------------------


@pytest.mark.parametrize("seed", [1, 2])
def test_other_suspect_fails_only_the_fingerprint_check(seed):
    verdict = verify_provenance(signed(), publication(), suspect=state(seed), trusted_public_key=OWNER_HEX)
    assert verdict.record_valid and verdict.public_key_trusted
    assert verdict.fingerprint_matches is False and not verdict.valid
    assert any("fingerprint" in p for p in verdict.problems)


def test_one_bit_in_the_suspect_breaks_the_fingerprint_check():
    s = state()
    s["a.weight"].view(np.int32)[0, 0] ^= 1
    assert verify_provenance(signed(), publication(), suspect=s).fingerprint_matches is False


def test_suspect_with_same_hash_but_other_counts_fails():
    fp = fingerprint_state_dict(state())
    other = type(fp)(fp.sha256, fp.version, fp.tensors, fp.elements + 1, fp.data_bytes)
    assert verify_provenance(signed(), publication(), suspect=other).fingerprint_matches is False


def test_every_signature_bit_flip_fails_the_signature_check():
    record = signed()
    signature = bytes.fromhex(record["signature"]["hex"])
    unsigned = without_signature(record)
    for bit in range(8 * len(signature)):
        flip = bytearray(signature)
        flip[bit // 8] ^= 1 << (bit % 8)
        verdict = verify_provenance(attach_signature(unsigned, bytes(flip)), publication())
        assert verdict.record_well_formed and not verdict.signature_valid and not verdict.valid


def test_field_change_without_resigning_fails_the_signature_check():
    record = signed()
    record["trigger_set_commitment"]["triggers"] = 101
    verdict = verify_provenance(record, publication())
    assert verdict.record_well_formed and not verdict.signature_valid and verdict.matches_publication


@pytest.mark.parametrize("raw", [bytes(32), bytes([255]) * 32, bytes([2]) * 32, bytes(range(32))])
def test_odd_public_keys_are_not_valid_and_do_not_raise(raw):
    """Whether the library refuses the key at load or at verify, the verdict is 'not valid', never an exception."""
    record = signed()
    record["owner"]["public_key"]["hex"] = raw.hex()
    verdict = verify_provenance(record, publication())
    assert verdict.record_well_formed and not verdict.signature_valid and not verdict.valid


def test_signed_by_another_key_than_it_names_fails():
    record = signed()
    forged = attach_signature(without_signature(record), FORGER.sign(b"anything"))
    assert not verify_provenance(forged, publication()).signature_valid


def _with_commitment(record: dict, **fields) -> dict:
    out = copy.deepcopy(record)
    out["watermark_commitment"]["commitment"].update(fields)
    return out


@pytest.mark.parametrize("case", ["c_equals_p", "c_above_p", "hex_disagrees", "leading_zero", "negative", "version",
                                  "scheme_inputs", "scheme_hash", "missing"])
def test_malformed_commitment_fails_the_commitment_check(case):
    record = signed()
    c = record["watermark_commitment"]["commitment"]
    if case == "c_equals_p":
        bad = _with_commitment(record, decimal=str(BN254_SCALAR_FIELD), hex=BN254_SCALAR_FIELD.to_bytes(32, "big").hex())
    elif case == "c_above_p":
        bad = _with_commitment(record, decimal=str(2 ** 255))
    elif case == "hex_disagrees":
        bad = _with_commitment(record, hex=Commitment(1).hex())
    elif case == "leading_zero":
        bad = _with_commitment(record, decimal="0" + c["decimal"])
    elif case == "negative":
        bad = _with_commitment(record, decimal="-" + c["decimal"])
    elif case == "version":
        bad = _with_commitment(record, version="zk-crown/commitment/v0")
    elif case == "scheme_inputs":
        bad = copy.deepcopy(record)
        bad["watermark_commitment"]["scheme"]["inputs"] = list(reversed(bad["watermark_commitment"]["scheme"]["inputs"]))
    elif case == "scheme_hash":
        bad = copy.deepcopy(record)
        bad["watermark_commitment"]["scheme"]["hash"] = "sha256"
    else:
        bad = copy.deepcopy(record)
        del bad["watermark_commitment"]
    verdict = verify_provenance(bad, publication())
    assert not verdict.commitment_well_formed and not verdict.record_valid and not verdict.valid
    assert any("commitment" in p for p in verdict.problems)


def test_commitment_check_runs_even_when_another_field_is_broken():
    record = signed()
    record["private"] = "changed"
    verdict = verify_provenance(record, publication())
    assert not verdict.record_well_formed and verdict.commitment_well_formed


@pytest.mark.parametrize("field,value", [
    ("owner_id", "someone-else"),
    ("model", "another label"),
    ("created_utc", "2026-01-01T00:00:00+00:00"),
])
def test_changed_publication_fails_the_publication_check(field, value):
    pub = publication()
    pub[field] = value
    verdict = verify_provenance(signed(), pub, suspect=state(), trusted_public_key=OWNER_HEX)
    assert verdict.signature_valid and verdict.fingerprint_matches
    assert not verdict.matches_publication and not verdict.valid
    assert any("publication SHA-256" in p for p in verdict.problems)


def test_publication_with_other_commitment_fails():
    pub = publication()
    pub["commitment"] = Commitment(7).to_dict()
    verdict = verify_provenance(signed(), pub)
    assert not verdict.matches_publication
    assert "commitment" in verdict.problems[-1]


def test_record_built_from_another_publication_fails():
    other = publication(state(3))
    verdict = verify_provenance(signed(other), publication())
    assert verdict.signature_valid and not verdict.matches_publication
    assert "model fingerprint" in verdict.problems[-1]


def test_malformed_publication_is_a_verdict_not_an_exception():
    for bad in (None, {}, {"schema": "x"}, "text"):
        verdict = verify_provenance(signed(), bad)
        assert verdict.signature_valid and not verdict.matches_publication and not verdict.valid
        assert any("publication is malformed" in p for p in verdict.problems)


@pytest.mark.parametrize("bad", [None, 7, "text", [], {}, {"schema": "zk-crown/provenance-record/v1"}])
def test_malformed_record_is_a_verdict_not_an_exception(bad):
    verdict = verify_provenance(bad, publication(), suspect=state(), trusted_public_key=OWNER_HEX)
    assert isinstance(verdict, ProvenanceVerdict)
    assert not any([verdict.record_well_formed, verdict.signature_valid, verdict.commitment_well_formed,
                    verdict.matches_publication, verdict.fingerprint_matches, verdict.public_key_trusted])
    assert not verdict.valid and verdict.record_sha256 is None


def test_unsigned_record_fails():
    unsigned = without_signature(signed())
    verdict = verify_provenance(unsigned, publication())
    assert not verdict.record_well_formed and not verdict.valid


# --- the limit of the signature -----------------------------------------------


def test_consistent_forgery_passes_everything_but_the_trusted_key():
    """A forger's own publication and record verify. Only an independently obtained key rejects them."""
    forged_pub = publication(state(5), owner_id="test-owner")
    forged = signed(forged_pub, key=FORGER)
    open_verdict = verify_provenance(forged, forged_pub, suspect=state(5))
    assert open_verdict.valid and open_verdict.public_key_trusted is None
    pinned = verify_provenance(forged, forged_pub, suspect=state(5), trusted_public_key=OWNER_HEX)
    assert pinned.record_valid and pinned.fingerprint_matches and pinned.public_key_trusted is False
    assert not pinned.valid


def test_reissue_of_the_owners_statement_under_another_key():
    unsigned = without_signature(signed())
    unsigned["owner"]["public_key"]["hex"] = public_key_bytes(FORGER).hex()
    reissued = sign_record(unsigned, FORGER)
    assert verify_provenance(reissued, publication()).valid
    assert not verify_provenance(reissued, publication(), trusted_public_key=OWNER_HEX).valid


def test_verdict_dict_holds_no_secret_and_is_stable():
    d = verify_provenance(signed(), publication(), suspect=state(), trusted_public_key=OWNER_HEX).to_dict()
    text = repr(d)
    assert bytes(range(32)).hex() not in text  # the test K and the owner's test private seed
    assert bytes(range(200, 231)).hex() not in text  # the test nonce
    assert d["schema"] == "zk-crown/provenance-verdict/v1" and d["valid"] is True
    assert d == verify_provenance(signed(), publication(), suspect=state(), trusted_public_key=OWNER_HEX).to_dict()


# --- the committed record, public values only ---------------------------------


COMMITTED = (REPO_ROOT / RECORD_PATH).exists() and (REPO_ROOT / ARTIFACT_PATH).exists()


@pytest.mark.skipif(not COMMITTED, reason="committed provenance files not present")
def test_committed_record_verifies_against_the_committed_publication():
    record, _ = read_record(REPO_ROOT / RECORD_PATH)
    pub, _ = read_publication(REPO_ROOT / ARTIFACT_PATH)
    verdict = verify_provenance(record, pub, trusted_public_key=record["owner"]["public_key"]["hex"])
    assert verdict.valid, verdict.problems
    assert verdict.record_fingerprint_sha256 == pub["model_fingerprint"]["sha256"]


# --- the script's own case builders -------------------------------------------


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p6_3_verify_provenance")


def test_script_malformed_commitments_are_all_rejected(script):
    cases = script.malformed_commitments(signed())
    assert len(cases) == 4
    assert all(not verify_provenance(c, publication()).commitment_well_formed for c in cases.values())


def test_script_tampered_publications_are_all_rejected(script):
    pub = publication()
    cases = script.tampered_publications(pub)
    assert len(cases) == 3
    assert all(not verify_provenance(signed(), c).matches_publication for c in cases.values())


def test_script_expect_stops_on_a_wrong_verdict(script):
    """The script's guard would catch a verifier that accepted a tampered record."""
    verdict = verify_provenance(signed(), publication(), suspect=state(1))
    with pytest.raises(SystemExit):
        script.expect("planted", verdict, fingerprint_matches=True)
    assert script.expect("ok", verdict, fingerprint_matches=False)["fingerprint_matches"] is False
