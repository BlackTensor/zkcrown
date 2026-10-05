"""Tests for P9.2: the five auditor checks.

Every key, signature and nonce here is a public test value, and the trigger
images are synthetic. The committed record and ZK files are read as public
files only.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from torch import nn  # noqa: E402

from src.auditor import OwnerSecrets, audit  # noqa: E402
from src.auditor.checks import (  # noqa: E402
    EZKL_PROOF,
    EZKL_SETTINGS,
    EZKL_SRS,
    EZKL_VK,
    GROTH16_PROOF,
    GROTH16_PUBLIC,
    GROTH16_VKEY,
    BehavioralCheck,
    CommitmentCheck,
    FingerprintCheck,
    OwnerMaterialSource,
    WeightCheck,
    ZkProofCheck,
    _field_value,
)
from src.crypto.commitment import commit  # noqa: E402
from src.crypto.fingerprint import fingerprint_state_dict  # noqa: E402
from src.crypto.provenance import RECORD_PATH, build_unsigned_record, read_record  # noqa: E402
from src.crypto.publication import ARTIFACT_PATH, build_publication, read_publication  # noqa: E402
from src.crypto.signing import public_key_bytes, sign_record  # noqa: E402
from src.data import CIFAR10_MEAN, CIFAR10_STD  # noqa: E402
from src.models.main_model import MainModel  # noqa: E402
from src.watermark.detection import normalise_uint8  # noqa: E402
from src.watermark.weight_embedding import embed_weight_watermark  # noqa: E402
from src.zk.toolchain import toolchain_available  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEST_K = bytes(range(32))
TEST_S = bytes(range(100, 116))
TEST_NONCE = bytes(range(200, 231))
SECRETS = OwnerSecrets(TEST_K, TEST_S, TEST_NONCE)
SIGNER = Ed25519PrivateKey.from_private_bytes(bytes(range(64, 96)))


def synthetic_arrays():
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (300, 32, 32, 3), dtype=np.uint8), rng.integers(0, 10, 300), list(range(300))


SOURCE = OwnerMaterialSource(owner_id="test-owner", arrays=synthetic_arrays)
MATERIAL = SOURCE.material(TEST_K)


def model_state(seed: int = 0, width: int = 32) -> dict:
    torch.manual_seed(seed)
    return MainModel(width=width).state_dict()


def files(named_state: dict | None = None, trigger_digest: str | None = None) -> tuple[dict, dict]:
    fp = fingerprint_state_dict(model_state() if named_state is None else named_state)
    pub = build_publication(commit(TEST_K, TEST_S, TEST_NONCE), fp, "test-owner", "test model",
                            "2026-10-02T12:00:00+00:00")
    rec = sign_record(build_unsigned_record(pub, trigger_digest or MATERIAL.bundle_digest, 100,
                                            public_key_bytes(SIGNER), "2026-10-03T09:30:00+00:00"), SIGNER)
    return rec, pub


def run(slot, check, state, *, record=None, publication=None, **kw):
    rec, pub = files() if record is None else (record, publication)
    return audit(state, rec, pub, checks={slot: check}, owner_secrets=SECRETS, **kw)


class TriggerLookup(nn.Module):
    """Answers the keyed target on the owner's triggers (nearest trigger), a fixed class elsewhere."""

    def __init__(self, material, fire: bool):
        super().__init__()
        self.register_buffer("triggers", normalise_uint8(material.triggers.images, CIFAR10_MEAN, CIFAR10_STD))
        self.register_buffer("targets", torch.as_tensor(np.array(material.responses.targets)))
        self.fire = fire

    def forward(self, x):
        nearest = torch.cdist(x.flatten(1), self.triggers.flatten(1)).argmin(dim=1)
        cls = self.targets[nearest] if self.fire else (self.targets[nearest] + 1) % 10
        return nn.functional.one_hot(cls, 10).float() * 10


# --- fingerprint -------------------------------------------------------------------


def test_fingerprint_passes_on_the_named_model_and_fails_otherwise():
    state = model_state()
    verdict = run("fingerprint", FingerprintCheck(), state)
    assert verdict.check("fingerprint").status == "passed"
    assert verdict.check("fingerprint").statistic["matches"] is True
    other = run("fingerprint", FingerprintCheck(), model_state(1)).check("fingerprint")
    assert other.status == "failed" and "says nothing about theft" in other.reason
    assert verdict.secrets_used == ()


def test_fingerprint_without_a_readable_record_is_not_applicable():
    rec, pub = files()
    broken = copy.deepcopy(rec)
    del broken["model"]
    result = run("fingerprint", FingerprintCheck(), model_state(), record=broken, publication=pub).check("fingerprint")
    assert result.status == "not_applicable"


# --- behavioral ----------------------------------------------------------------------


def test_behavioral_detects_a_model_that_answers_the_keyed_targets():
    result = run("behavioral", BehavioralCheck(SOURCE), model_state(),
                 suspect_model=TriggerLookup(MATERIAL, fire=True)).check("behavioral")
    assert result.status == "detected" and result.statistic["fired"] == 100
    assert result.p_value_kind == "exact" and result.strongest_level_rejected == 1e-9
    assert result.statistic["k_star_at_alpha"] == 29


def test_behavioral_does_not_detect_a_model_that_never_answers_them():
    result = run("behavioral", BehavioralCheck(SOURCE), model_state(),
                 suspect_model=TriggerLookup(MATERIAL, fire=False)).check("behavioral")
    assert result.status == "not_detected" and result.statistic["fired"] == 0 and result.p_value == 1.0


def test_behavioral_on_an_untrained_model_is_not_detected():
    result = run("behavioral", BehavioralCheck(SOURCE), model_state(3)).check("behavioral")
    assert result.status == "not_detected"


def test_behavioral_refuses_triggers_that_were_not_committed():
    rec, pub = files(trigger_digest="ab" * 32)
    result = run("behavioral", BehavioralCheck(SOURCE), model_state(), record=rec, publication=pub).check("behavioral")
    assert result.status == "error" and "does not regenerate" in result.reason


def test_behavioral_without_a_queryable_model_is_not_applicable():
    result = run("behavioral", BehavioralCheck(SOURCE), model_state(width=16)).check("behavioral")
    assert result.status == "not_applicable" and "cannot be queried" in result.reason


def test_behavioral_queries_a_narrower_model_given_its_arch():
    result = run("behavioral", BehavioralCheck(SOURCE), model_state(width=16), suspect_arch={"width": 16})
    assert result.check("behavioral").status == "not_detected"


# --- weight ---------------------------------------------------------------------------


def watermarked(state: dict, alpha: float = 0.1) -> dict:
    out, _ = embed_weight_watermark(state, MATERIAL.layout, MATERIAL.projection, MATERIAL.signature, alpha)
    return out


def test_weight_detects_the_embedded_watermark():
    result = run("weight", WeightCheck(SOURCE), watermarked(model_state())).check("weight")
    assert result.status == "detected" and result.p_value_kind == "upper_bound" and result.statistic["z"] > 5.257


def test_weight_does_not_detect_an_unwatermarked_model():
    result = run("weight", WeightCheck(SOURCE), model_state(5)).check("weight")
    assert result.status == "not_detected" and result.statistic["z"] < 5.257


@pytest.mark.parametrize("variant", ["width16", "channels_removed"])
def test_weight_without_the_carrier_layout_is_not_applicable(variant):
    if variant == "width16":
        state = model_state(width=16)
    else:
        state = model_state()
        w = state["features.0.weight"]
        state["features.0.weight"] = w[: w.shape[0] - 3].clone()  # physically narrower first conv
    result = run("weight", WeightCheck(SOURCE), state, suspect_arch={"width": 16} if variant == "width16" else None)
    r = result.check("weight")
    assert r.status == "not_applicable" and "re-alignment is not implemented" in r.reason


def test_watermark_checks_list_K_and_keep_the_guard_on():
    state = watermarked(model_state())
    rec, pub = files()
    verdict = audit(state, rec, pub, owner_secrets=SECRETS,
                    checks={"behavioral": BehavioralCheck(SOURCE), "weight": WeightCheck(SOURCE)},
                    suspect_model=TriggerLookup(MATERIAL, fire=True))
    assert verdict.secrets_used == ("K",)
    assert not any(c.guard_rejected for c in verdict.checks)
    text = json.dumps(verdict.to_dict()).lower()
    assert not any(n in text for n in SECRETS.needles())
    assert "material" not in repr(SOURCE).lower() or "hidden" in repr(SOURCE)


def test_watermark_checks_without_secrets_are_not_run():
    rec, pub = files()
    verdict = audit(model_state(), rec, pub, checks={"behavioral": BehavioralCheck(SOURCE), "weight": WeightCheck(SOURCE)})
    assert {verdict.check(s).status for s in ("behavioral", "weight")} == {"not_run"}
    assert verdict.secrets_used == ()


# --- commitment -----------------------------------------------------------------------


def test_commitment_passes_and_states_its_scope(tmp_path):
    result = run("commitment", CommitmentCheck(tmp_path / "absent.ots"), model_state()).check("commitment")
    assert result.status == "passed"
    assert "does not show that anyone knows an opening" in result.reason
    assert result.statistic["ots_bitcoin_attestations"] is None


def test_commitment_fails_when_the_record_names_another_c(tmp_path):
    rec, pub = files()
    other_pub = build_publication(commit(bytes(32), TEST_S, TEST_NONCE), fingerprint_state_dict(model_state()),
                                  "test-owner", "test model", "2026-10-02T12:00:00+00:00")
    result = run("commitment", CommitmentCheck(tmp_path / "absent.ots"), model_state(), record=rec,
                 publication=other_pub).check("commitment")
    assert result.status == "failed"


def test_commitment_on_the_committed_files_reads_the_bitcoin_attestations():
    rec, _ = read_record(RECORD_PATH)
    pub, _ = read_publication(ARTIFACT_PATH)
    result = run("commitment", CommitmentCheck(), model_state(), record=rec, publication=pub).check("commitment")
    assert result.status == "passed" and result.statistic["ots_proof_for_publication"] is True
    assert result.statistic["ots_earliest_block_height"] == 969627


# --- zk proofs ------------------------------------------------------------------------

zk = pytest.mark.skipif(not toolchain_available() or not (REPO / EZKL_SRS[0]).is_file(),
                        reason="ZK toolchain or EZKL SRS absent")


def committed_files():
    return read_record(RECORD_PATH)[0], read_publication(ARTIFACT_PATH)[0]


def copy_zk(root: Path) -> None:
    for rel in (GROTH16_VKEY[0], GROTH16_PROOF[0], GROTH16_PUBLIC, EZKL_PROOF[0], EZKL_SETTINGS[0], EZKL_VK[0],
                EZKL_SRS[0]):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, root / rel)


@zk
def test_zk_proofs_pass_against_the_committed_record_and_state_their_scope():
    rec, pub = committed_files()
    result = run("zk_proof", ZkProofCheck(), model_state(), record=rec, publication=pub).check("zk_proof")
    assert result.status == "passed"
    s = result.statistic
    assert s["groth16_verified"] and s["ezkl_verified"] and s["groth16_public_signal_equals_record_c"]
    assert "not about the suspect model" in result.reason and "concerns zk_model, not this suspect" in result.reason


@zk
def test_zk_public_signal_is_compared_by_value(tmp_path):
    rec, pub = committed_files()
    copy_zk(tmp_path)
    c = int(rec["watermark_commitment"]["commitment"]["decimal"])
    for text in (hex(c), "000" + str(c)):
        assert _field_value(text) == c
    (tmp_path / GROTH16_PUBLIC).write_text(json.dumps([hex(c)]), encoding="utf-8")
    result = run("zk_proof", ZkProofCheck(tmp_path), model_state(), record=rec, publication=pub).check("zk_proof")
    assert result.statistic["groth16_public_signal_equals_record_c"] is True and result.status == "passed"


@zk
def test_zk_fails_against_a_record_with_another_c():
    result = run("zk_proof", ZkProofCheck(), model_state()).check("zk_proof")  # demo record, demo C
    assert result.status == "failed" and result.statistic["groth16_public_signal_equals_record_c"] is False


@zk
def test_zk_exception_from_ezkl_is_a_rejection(monkeypatch):
    import ezkl

    def boom(*a, **k):
        raise RuntimeError("Failed to run verify: [halo2] The constraint system is not satisfied")

    monkeypatch.setattr(ezkl, "verify", boom)
    rec, pub = committed_files()
    result = run("zk_proof", ZkProofCheck(), model_state(), record=rec, publication=pub).check("zk_proof")
    assert result.status == "failed" and "counted as a rejection" in result.reason
    assert result.statistic["ezkl_verified"] is False and result.statistic["groth16_verified"] is True


@zk
def test_zk_changed_committed_file_fails(tmp_path):
    rec, pub = committed_files()
    copy_zk(tmp_path)
    proof = tmp_path / GROTH16_PROOF[0]
    data = json.loads(proof.read_text(encoding="utf-8"))
    data["pi_a"][0] = str(int(data["pi_a"][0]) + 1)
    proof.write_text(json.dumps(data), encoding="utf-8")
    result = run("zk_proof", ZkProofCheck(tmp_path), model_state(), record=rec, publication=pub).check("zk_proof")
    assert result.status == "failed" and result.statistic["artifacts_match_committed"] is False


def test_zk_with_a_missing_file_is_not_run(tmp_path):
    rec, pub = committed_files()
    result = run("zk_proof", ZkProofCheck(tmp_path), model_state(), record=rec, publication=pub).check("zk_proof")
    assert result.status == "not_run"
