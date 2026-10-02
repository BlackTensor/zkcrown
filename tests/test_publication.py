"""Tests for P5.4: the commitment publication artifact.

All keys and nonces here are test values. The real artifact is checked only
for being well formed and canonical, which needs no secret.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from src.crypto.commitment import Commitment, commit
from src.crypto.fingerprint import FINGERPRINT_VERSION, ModelFingerprint
from src.crypto.publication import (
    ARTIFACT_PATH,
    PUBLICATION_SCHEMA,
    artifact_sha256,
    build_publication,
    canonical_json,
    read_publication,
    utc_now,
    validate_publication,
    write_publication,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

KEY = bytes(range(32))
SIG = bytes(range(100, 116))
NONCE = bytes(range(200, 231))
FINGERPRINT = ModelFingerprint(sha256="ab" * 32, version=FINGERPRINT_VERSION, tensors=38, elements=308848,
                               data_bytes=1235416)
CREATED = "2026-10-02T12:00:00+00:00"


def record() -> dict:
    return build_publication(commit(KEY, SIG, NONCE), FINGERPRINT, "test-owner", "test model", CREATED)


# --- contents -----------------------------------------------------------------


def test_holds_commitment_fingerprint_owner_and_timestamp():
    r = record()
    c = commit(KEY, SIG, NONCE)
    assert r["schema"] == PUBLICATION_SCHEMA
    assert r["commitment"] == c.to_dict()
    assert r["model_fingerprint"] == FINGERPRINT.to_dict()
    assert r["owner_id"] == "test-owner" and r["created_utc"] == CREATED and r["model"] == "test model"
    assert r["commitment_scheme"]["inputs"] == ["DOMAIN", "K_hi", "K_lo", "S", "nonce"]
    assert "self-asserted" in r["timestamp_status"]


def test_holds_no_secret():
    data = canonical_json(record())
    for blob in (KEY, SIG, NONCE):
        assert blob.hex().encode() not in data and blob not in data
    for limb in (int.from_bytes(KEY[:16], "big"), int.from_bytes(KEY[16:], "big"), int.from_bytes(SIG, "big"),
                 int.from_bytes(NONCE, "big")):
        assert str(limb).encode() not in data


def test_build_takes_public_values_only():
    with pytest.raises(TypeError, match="never sees the opening"):
        build_publication(KEY, FINGERPRINT, "test-owner", "m", CREATED)
    with pytest.raises(TypeError, match="ModelFingerprint"):
        build_publication(commit(KEY, SIG, NONCE), "ab" * 32, "test-owner", "m", CREATED)
    with pytest.raises(ValueError):
        build_publication(commit(KEY, SIG, NONCE), FINGERPRINT, " padded ", "m", CREATED)
    with pytest.raises(ValueError):
        build_publication(commit(KEY, SIG, NONCE), FINGERPRINT, "test-owner", "m", "yesterday")


def test_utc_now_is_accepted():
    stamp = utc_now()
    assert stamp.endswith("+00:00") and len(stamp) == 25
    validate_publication({**record(), "created_utc": stamp})


# --- canonical bytes ----------------------------------------------------------


def test_canonical_bytes_are_deterministic_and_order_independent():
    r = record()
    shuffled = dict(reversed(list(r.items())))
    assert canonical_json(r) == canonical_json(shuffled) == canonical_json(copy.deepcopy(r))
    data = canonical_json(r)
    assert data.endswith(b"}\n") and b"\r" not in data and data.isascii()
    assert json.loads(data) == r
    assert artifact_sha256(r) == hashlib.sha256(data).hexdigest()


def test_any_change_to_the_record_changes_the_artifact_hash():
    base = artifact_sha256(record())
    other_c = build_publication(commit(KEY, SIG, bytes(31)), FINGERPRINT, "test-owner", "test model", CREATED)
    other_owner = build_publication(commit(KEY, SIG, NONCE), FINGERPRINT, "other-owner", "test model", CREATED)
    other_time = build_publication(commit(KEY, SIG, NONCE), FINGERPRINT, "test-owner", "test model",
                                   "2026-10-02T12:00:01+00:00")
    assert len({base, artifact_sha256(other_c), artifact_sha256(other_owner), artifact_sha256(other_time)}) == 4


def test_non_ascii_owner_id_round_trips(tmp_path):
    r = build_publication(commit(KEY, SIG, NONCE), FINGERPRINT, "propriétaire-é", "m", CREATED)
    write_publication(r, tmp_path / "c.json")
    assert read_publication(tmp_path / "c.json")[0]["owner_id"] == "propriétaire-é"


# --- write once, read strictly ------------------------------------------------


def test_write_then_read(tmp_path):
    path = tmp_path / "provenance" / "commitment.json"
    digest = write_publication(record(), path)
    loaded, read_digest = read_publication(path)
    assert loaded == record() and digest == read_digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_refuses_to_replace_a_published_artifact(tmp_path):
    path = tmp_path / "commitment.json"
    write_publication(record(), path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_publication(record(), path)
    assert path.read_bytes() == before


def test_read_refuses_non_canonical_files(tmp_path):
    path = tmp_path / "commitment.json"
    data = canonical_json(record())
    for altered in (data.replace(b"\n", b"\r\n"), data + b"\n", json.dumps(record()).encode()):
        path.write_bytes(altered)
        with pytest.raises(ValueError, match="canonical"):
            read_publication(path)


# --- validation ---------------------------------------------------------------


def _with(path: tuple, value) -> dict:
    r = record()
    target = r
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    return r


@pytest.mark.parametrize("path, value", [
    (("schema",), "zk-crown/commitment-publication/v0"),
    (("commitment", "version"), "zk-crown/commitment/v0"),
    (("commitment", "decimal"), "007"),
    (("commitment", "hex"), "00" * 32),
    (("commitment", "decimal"), "5"),  # no longer agrees with hex
    (("commitment_scheme", "inputs"), ["K", "S", "nonce"]),
    (("commitment_scheme", "hash"), "sha256"),
    (("model_fingerprint", "sha256"), "AB" * 32),
    (("model_fingerprint", "sha256"), "ab" * 31),
    (("model_fingerprint", "version"), "other/v1"),
    (("model_fingerprint", "tensors"), 0),
    (("model_fingerprint", "elements"), True),
    (("model",), "  "),
    (("owner_id",), ""),
    (("created_utc",), "2026-10-02 12:00:00"),
    (("created_utc",), "2026-10-02T12:00:00Z"),
    (("created_utc",), "2026-13-45T12:00:00+00:00"),
    (("created_utc",), "2026-10-02T12:00:00+05:00"),
    (("timestamp_status",), "independently timestamped"),
    (("private",), ""),
])
def test_validation_rejects(path, value):
    with pytest.raises(ValueError):
        validate_publication(_with(path, value))


def test_validation_rejects_missing_and_extra_fields():
    r = record()
    del r["owner_id"]
    with pytest.raises(ValueError, match="missing"):
        validate_publication(r)
    with pytest.raises(ValueError, match="unexpected"):
        validate_publication({**record(), "K": "00"})
    with pytest.raises(ValueError):
        validate_publication([record()])
    with pytest.raises((ValueError, TypeError)):
        validate_publication(_with(("owner_id",), 5))
    validate_publication(record())


# --- the script ---------------------------------------------------------------


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p5_4_publish_commitment")


def test_nonce_is_created_once_and_never_overwritten(script, tmp_path):
    path = tmp_path / "secrets" / "nonce.bin"
    script.create_nonce(path)
    first = script.load_nonce(path)
    assert len(first) == 31
    with pytest.raises(FileExistsError):
        script.create_nonce(path)
    assert script.load_nonce(path) == first
    (tmp_path / "short.bin").write_bytes(b"\x01" * 30)
    with pytest.raises(ValueError, match="expected 31"):
        script.load_nonce(tmp_path / "short.bin")
    assert script.DEFAULT_NONCE_PATH.parent.name == "secrets"  # gitignored


def _fake_setup(script, tmp_path, monkeypatch):
    """A test key, a tiny weights file and a matching P5.1-style record, all under tmp_path."""
    torch = pytest.importorskip("torch")
    from src.crypto.fingerprint import fingerprint_file

    key = tmp_path / "K.bin"
    key.write_bytes(KEY)
    weights = tmp_path / "w.pt"
    torch.save({"w": torch.arange(6, dtype=torch.float32)}, weights)
    reference = tmp_path / "p5.1.json"
    reference.write_text(json.dumps(
        {"metrics": {"models": {"dual_W_star": {"fingerprint": fingerprint_file(weights).to_dict()}}}}))
    monkeypatch.setattr(script, "W_DUAL_SHA256", script.sha256_file(weights))
    monkeypatch.setattr(script, "P5_1_RESULT", str(reference))
    return ["--key", str(key), "--nonce", str(tmp_path / "nonce.bin"), "--weights", str(weights),
            "--artifact", str(tmp_path / "provenance" / "commitment.json"), "--out-dir", str(tmp_path / "results")]


def test_script_creates_then_refuses_then_checks(script, tmp_path, monkeypatch):
    from src.watermark.signature import PROJECT_OWNER_ID, derive_signature

    args = _fake_setup(script, tmp_path, monkeypatch)
    created = script.main(args)["metrics"]
    artifact = tmp_path / "provenance" / "commitment.json"
    published, digest = read_publication(artifact)
    nonce = (tmp_path / "nonce.bin").read_bytes()
    assert created["mode"] == "create" and created["nonce_created_this_run"] and created["artifact_sha256"] == digest
    assert published["commitment"] == commit(KEY, derive_signature(KEY, PROJECT_OWNER_ID), nonce).to_dict()
    assert published["owner_id"] == PROJECT_OWNER_ID

    before = artifact.read_bytes()
    with pytest.raises(SystemExit, match="Refusing to replace"):
        script.main(args)
    assert artifact.read_bytes() == before and (tmp_path / "nonce.bin").read_bytes() == nonce

    checked = script.main([*args, "--check"])["metrics"]
    assert checked["mode"] == "check" and not checked["nonce_created_this_run"]
    assert checked["artifact_sha256"] == digest and artifact.read_bytes() == before

    record_text = Path(script.main([*args, "--check"])["path"]).read_text(encoding="utf-8")
    assert KEY.hex() not in record_text and nonce.hex() not in record_text


def test_script_check_fails_with_another_nonce_or_wrong_weights(script, tmp_path, monkeypatch):
    args = _fake_setup(script, tmp_path, monkeypatch)
    script.main(args)
    (tmp_path / "nonce.bin").write_bytes(bytes(31))
    with pytest.raises(SystemExit, match="not what K, S and the nonce give"):
        script.main([*args, "--check"])
    monkeypatch.setattr(script, "W_DUAL_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="Refusing to publish"):
        script.main([*args, "--check"])


def test_script_check_needs_an_artifact(script, tmp_path, monkeypatch):
    args = _fake_setup(script, tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="nothing to check"):
        script.main([*args, "--check"])
    assert not (tmp_path / "nonce.bin").exists()  # a check never creates a nonce


def test_committed_artifact_is_well_formed():
    """The real artifact, where present: canonical, valid, and naming the P5.1 dual fingerprint."""
    path = REPO_ROOT / ARTIFACT_PATH
    if not path.exists():
        pytest.skip("P5.4 has not been run")
    published, _ = read_publication(path)
    p5_1 = json.loads(next((REPO_ROOT / "results").glob("p5.1_model_fingerprint__*.json")).read_text(encoding="utf-8"))
    assert published["model_fingerprint"] == p5_1["metrics"]["models"]["dual_W_star"]["fingerprint"]
    from src.watermark.signature import PROJECT_OWNER_ID

    assert published["owner_id"] == PROJECT_OWNER_ID
    Commitment.from_decimal(published["commitment"]["decimal"])
