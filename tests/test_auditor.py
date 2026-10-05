"""Tests for P9.1: the auditor engine, the verdict object and the no-secrets guard.

Every key, signature and nonce here is a public test value. The committed
record and publication are read as public files only.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from src.auditor import SLOTS, CheckOutcome, OwnerSecrets, SecretLeak, audit  # noqa: E402
from src.auditor.verdict import LIMITATIONS, NOT_WIRED, VERDICT_SCHEMA, CheckResult, strongest_level_rejected  # noqa: E402
from src.crypto.commitment import commit  # noqa: E402
from src.crypto.fingerprint import fingerprint_state_dict  # noqa: E402
from src.crypto.provenance import RECORD_PATH, attach_signature, build_unsigned_record, read_record  # noqa: E402
from src.crypto.publication import ARTIFACT_PATH, build_publication, read_publication  # noqa: E402
from src.crypto.signing import public_key_bytes, sign_record  # noqa: E402
from src.models.main_model import MainModel  # noqa: E402

TEST_K = bytes(range(32))
TEST_S = bytes(range(100, 116))
TEST_NONCE = bytes(range(200, 231))
SECRETS = OwnerSecrets(TEST_K, TEST_S, TEST_NONCE)
OWNER = Ed25519PrivateKey.from_private_bytes(bytes(range(64, 96)))
OWNER_HEX = public_key_bytes(OWNER).hex()


def state(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"a.weight": rng.standard_normal((4, 3)).astype(np.float32),
            "a.bias": rng.standard_normal(4).astype(np.float32)}


def files(model_state: dict | None = None) -> tuple[dict, dict]:
    fp = fingerprint_state_dict(state() if model_state is None else model_state)
    pub = build_publication(commit(TEST_K, TEST_S, TEST_NONCE), fp, "test-owner", "test model",
                            "2026-10-02T12:00:00+00:00")
    rec = sign_record(build_unsigned_record(pub, "cd" * 32, 100, public_key_bytes(OWNER),
                                            "2026-10-03T09:30:00+00:00"), OWNER)
    return rec, pub


class Stub:
    def __init__(self, slot, outcome=None, *, raises=None, requires=(), rejection=False, method="test"):
        self.slot, self.method, self.requires_secrets = slot, method, tuple(requires)
        self.exception_is_rejection = rejection
        self._outcome, self._raises = outcome, raises
        self.seen = None

    def run(self, context):
        self.seen = context
        if self._raises is not None:
            raise self._raises
        return self._outcome(context) if callable(self._outcome) else self._outcome


def run_stub(stub, secrets=SECRETS):
    rec, pub = files()
    return audit(state(), rec, pub, checks={stub.slot: stub}, owner_secrets=secrets).check(stub.slot)


# --- the P9.1 default: record precondition only, five slots not run ----------


def test_with_no_checks_only_the_record_precondition_runs():
    rec, pub = files()
    verdict = audit(state(), rec, pub, trusted_public_key=OWNER_HEX, checks={})
    assert verdict.record_valid
    assert [c.slot for c in verdict.checks] == list(SLOTS)
    assert all(c.status == "not_run" and c.reason == NOT_WIRED for c in verdict.checks)
    assert verdict.secrets_used == () and verdict.grade is None
    d = verdict.to_dict()
    assert d["schema"] == VERDICT_SCHEMA and d["record_valid"] is True and d["limitations"] == list(LIMITATIONS)
    assert d["record_status"]["public_key_trusted"] is True
    # The fingerprint comparison belongs to the fingerprint slot, so P6.3 runs without the suspect.
    assert d["record_status"]["fingerprint_matches"] is None
    json.dumps(d)


def test_inputs_describe_the_suspect_and_files():
    rec, pub = files()
    verdict = audit(state(), rec, pub, suspect_label="demo", suspect_file_sha256="ab" * 32, checks={})
    suspect = verdict.inputs["suspect"]
    assert suspect["fingerprint"] == fingerprint_state_dict(state()).to_dict()
    assert suspect["label"] == "demo" and suspect["state_entries"] == 2
    assert suspect["loads_into_main_model"] is False and suspect["parameter_count"] is None
    assert verdict.inputs["record_sha256"] and verdict.inputs["publication_sha256"]
    assert verdict.inputs["trusted_public_key_supplied"] is False


def test_a_main_model_suspect_reports_its_parameter_count():
    torch.manual_seed(0)
    model_state = MainModel(width=32).state_dict()
    rec, pub = files(model_state)
    suspect = audit(model_state, rec, pub, checks={}).inputs["suspect"]
    assert suspect["loads_into_main_model"] is True and suspect["parameter_count"] == 307_946


def test_an_invalid_record_still_lets_checks_run():
    rec, pub = files()
    tampered = copy.deepcopy(rec)
    tampered["model"]["label"] = "another model"
    stub = Stub("behavioral", CheckOutcome("detected", {"fired": 100, "n": 100}, 1e-20, "exact"))
    verdict = audit(state(), tampered, pub, checks={"behavioral": stub})
    assert verdict.record_valid is False and verdict.to_dict()["record_valid"] is False
    assert verdict.check("behavioral").status == "detected"
    assert all(verdict.check(s).reason == NOT_WIRED for s in SLOTS if s != "behavioral")


def test_a_malformed_record_is_a_verdict_not_an_exception():
    _, pub = files()
    verdict = audit(state(), {"not": "a record"}, pub, checks={})
    assert verdict.record_valid is False and verdict.inputs["record_sha256"] is None


def test_the_committed_record_and_publication_audit_as_valid():
    if not Path(RECORD_PATH).exists() or not Path(ARTIFACT_PATH).exists():
        pytest.skip("committed provenance files absent")
    rec, _ = read_record(RECORD_PATH)
    pub, _ = read_publication(ARTIFACT_PATH)
    verdict = audit(state(), rec, pub, trusted_public_key=rec["owner"]["public_key"]["hex"], checks={})
    assert verdict.record_valid and all(c.status == "not_run" for c in verdict.checks)


# --- engine contract -----------------------------------------------------------


def test_a_legitimate_check_is_reported_with_its_level():
    result = run_stub(Stub("weight", CheckOutcome("detected", {"z": 10.29, "correlation": 0.9091, "bits": 126},
                                                  1.1e-23, "upper_bound", "ok")))
    assert result.status == "detected" and not result.guard_rejected
    assert result.statistic["bits"] == 126 and result.strongest_level_rejected == 1e-9
    assert result.duration_seconds is not None and result.method == "test"


def test_levels():
    assert strongest_level_rejected(0.03) == 0.05
    assert strongest_level_rejected(5e-7) == 1e-6
    assert strongest_level_rejected(0.2) is None and strongest_level_rejected(None) is None


def test_an_exception_is_an_error():
    result = run_stub(Stub("weight", raises=RuntimeError("boom")))
    assert result.status == "error" and "boom" in result.reason


def test_a_zk_exception_is_a_rejection():
    result = run_stub(Stub("zk_proof", raises=RuntimeError("Failed to run verify"), rejection=True))
    assert result.status == "failed" and result.reason.startswith("rejected by raising")


def test_an_invalid_status_is_an_error():
    result = run_stub(Stub("behavioral", CheckOutcome("passed")))
    assert result.status == "error" and "invalid outcome" in result.reason


def test_a_check_needing_secrets_without_them_is_not_run():
    result = run_stub(Stub("weight", CheckOutcome("detected"), requires=("K",)), secrets=None)
    assert result.status == "not_run" and "not supplied" in result.reason


def test_secrets_are_given_only_as_declared_and_reported():
    rec, pub = files()
    seen = {}

    def outcome(context):
        seen["k_len"] = len(context.secret("K"))
        with pytest.raises(PermissionError):
            context.secret("nonce")
        return CheckOutcome("detected", {"n": 100})

    verdict = audit(state(), rec, pub, checks={"weight": Stub("weight", outcome, requires=("K", "S"))},
                    owner_secrets=SECRETS)
    assert seen["k_len"] == 32 and verdict.secrets_used == ("K", "S")


def test_engine_refuses_misregistered_checks():
    rec, pub = files()
    with pytest.raises(ValueError):
        audit(state(), rec, pub, checks={"verdict": Stub("weight", CheckOutcome("detected"))})
    with pytest.raises(ValueError):
        audit(state(), rec, pub, checks={"behavioral": Stub("weight", CheckOutcome("detected"))})


def test_verdict_object_invariants():
    with pytest.raises(ValueError):
        CheckResult(slot="weight", status="not_run")  # reason required
    with pytest.raises(ValueError):
        CheckResult(slot="weight", status="detected", p_value=0.5)  # kind required


# --- the no-secrets guard: stubs that try to leak ------------------------------


def trigger_image():
    return np.zeros((32, 32, 3), dtype=np.uint8)


LEAKS = {
    "trigger image (array)": {"trigger": trigger_image()},
    "trigger image (tensor)": {"trigger": torch.zeros(3, 32, 32)},
    "trigger image (nested list)": {"trigger": trigger_image().tolist()},
    "target list": {"targets": [3, 7, 1, 0]},
    "target tuple": {"targets": (3, 7, 1)},
    "S as bytes": {"s": TEST_S},
    "S as hex": {"s": TEST_S.hex()},
    "S as integer": {"s": int.from_bytes(TEST_S, "big")},
    "S as decimal string": {"s": str(int.from_bytes(TEST_S, "big"))},
    "K as hex": {"k": TEST_K.hex()},
    "K half as hex inside text": {"note": "key half " + TEST_K[:16].hex()},
    "K as integer": {"k": int.from_bytes(TEST_K, "big")},
    "K limb as integer": {"k_hi": int.from_bytes(TEST_K[:16], "big")},
    "K as base64": {"k": base64.b64encode(TEST_K).decode()},
    "S as base64": {"s": base64.b64encode(TEST_S).decode()},
    "nonce as hex": {"nonce": TEST_NONCE.hex()},
    "nested mapping": {"detail": {"fired": 3}},
    "non-finite float": {"z": float("nan")},
    "bad name": {"Fired Count": 3},
}


@pytest.mark.parametrize("name", sorted(LEAKS))
def test_guard_rejects_a_leaking_statistic(name):
    result = run_stub(Stub("behavioral", CheckOutcome("detected", LEAKS[name], 1e-9, "exact")))
    assert result.status == "error" and result.guard_rejected
    assert result.statistic == {} and result.p_value is None
    text = json.dumps(result.to_dict()).lower()
    for needle in SECRETS.needles():
        assert needle not in text


@pytest.mark.parametrize("leak", [TEST_K.hex(), str(int.from_bytes(TEST_S, "big")),
                                  base64.b64encode(TEST_NONCE).decode()])
def test_guard_rejects_a_leaking_reason(leak):
    result = run_stub(Stub("weight", CheckOutcome("detected", {"z": 1.0}, reason="debug " + leak)))
    assert result.status == "error" and result.guard_rejected


def test_guard_withholds_a_leaking_exception_message():
    result = run_stub(Stub("weight", raises=RuntimeError("bad key " + base64.b64encode(TEST_K).decode())))
    assert result.status == "error" and "withheld" in result.reason
    assert not any(n in result.reason.lower() for n in SECRETS.needles())


def test_base64_is_caught_only_by_the_value_scan():
    """Base64 passes the structural rule; the needle scan, armed by supplied secrets, catches it."""
    stat = {"k": base64.b64encode(TEST_K).decode()}
    assert run_stub(Stub("weight", CheckOutcome("detected", stat)), secrets=SECRETS).guard_rejected
    assert not run_stub(Stub("weight", CheckOutcome("detected", stat)), secrets=None).guard_rejected


def test_the_built_verdict_is_scanned_as_a_whole():
    rec, pub = files()
    with pytest.raises(SecretLeak):
        audit(state(), rec, pub, owner_secrets=SECRETS, suspect_label="copy of " + TEST_K.hex(), checks={})


def test_no_secret_anywhere_in_a_verdict_built_with_secrets():
    rec, pub = files()
    stub = Stub("weight", CheckOutcome("detected", {"z": 10.29, "bits": 126}, 1e-23, "upper_bound"), requires=("K",))
    text = json.dumps(audit(state(), rec, pub, owner_secrets=SECRETS, checks={"weight": stub}).to_dict()).lower()
    assert not any(needle in text for needle in SECRETS.needles())


def test_owner_secrets_hide_their_values():
    assert "hidden" in repr(SECRETS) and TEST_K.hex() not in repr(SECRETS)
    with pytest.raises(ValueError):
        OwnerSecrets(TEST_K[:31], TEST_S, TEST_NONCE)
    needles = SECRETS.needles()
    assert TEST_K.hex() in needles and TEST_S.hex() in needles and TEST_NONCE.hex() in needles
    assert str(int.from_bytes(TEST_K[:16], "big")) in needles
