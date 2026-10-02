"""Tests for P6.2: signing the provenance record.

All keys here are test keys. The committed record, where present, is checked
with its own public key only, which needs no secret.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from src.crypto.commitment import commit  # noqa: E402
from src.crypto.fingerprint import FINGERPRINT_VERSION, ModelFingerprint  # noqa: E402
from src.crypto.provenance import (  # noqa: E402
    RECORD_PATH,
    attach_signature,
    build_unsigned_record,
    canonical_json,
    read_record,
    signing_payload,
    without_signature,
    write_record,
)
from src.crypto.publication import (  # noqa: E402
    ARTIFACT_PATH,
    build_publication,
    read_publication,
    write_publication,
)
from src.crypto.signing import (  # noqa: E402
    create_signing_key,
    load_signing_key,
    public_key_bytes,
    sign_record,
    signature_is_valid,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# RFC 8032, section 7.1, TEST 1: seed, public key, and the signature of the empty message.
RFC_SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
RFC_PUBLIC = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
RFC_SIGNATURE = bytes.fromhex(
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe"
    "24655141438e7a100b")

PRIVATE = Ed25519PrivateKey.from_private_bytes(RFC_SEED)
OTHER = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
FINGERPRINT = ModelFingerprint(sha256="ab" * 32, version=FINGERPRINT_VERSION, tensors=38, elements=308848,
                               data_bytes=1235416)
CREATED = "2026-10-03T09:30:00+00:00"


def publication() -> dict:
    return build_publication(commit(bytes(range(32)), bytes(range(100, 116)), bytes(range(200, 231))), FINGERPRINT,
                             "test-owner", "test model", "2026-10-02T12:00:00+00:00")


def unsigned(key: Ed25519PrivateKey = PRIVATE) -> dict:
    return build_unsigned_record(publication(), "cd" * 32, 100, public_key_bytes(key), CREATED)


def signed() -> dict:
    return sign_record(unsigned(), PRIVATE)


# --- the keypair --------------------------------------------------------------


def test_library_matches_the_rfc_8032_vector():
    """The seed-to-public-key map and the signature are the standard's, not something local."""
    assert public_key_bytes(PRIVATE) == RFC_PUBLIC
    assert PRIVATE.sign(b"") == RFC_SIGNATURE


def test_key_file_is_created_once_and_never_overwritten(tmp_path):
    path = tmp_path / "secrets" / "signing.bin"
    create_signing_key(path)
    seed = path.read_bytes()
    assert len(seed) == 32
    with pytest.raises(FileExistsError):
        create_signing_key(path)
    assert path.read_bytes() == seed
    assert public_key_bytes(load_signing_key(path)) == public_key_bytes(Ed25519PrivateKey.from_private_bytes(seed))


def test_two_created_keys_differ(tmp_path):
    create_signing_key(tmp_path / "a.bin")
    create_signing_key(tmp_path / "b.bin")
    assert (tmp_path / "a.bin").read_bytes() != (tmp_path / "b.bin").read_bytes()


def test_load_refuses_a_wrong_length(tmp_path):
    (tmp_path / "short.bin").write_bytes(b"\x01" * 31)
    with pytest.raises(ValueError, match="expected 32"):
        load_signing_key(tmp_path / "short.bin")


def test_public_key_bytes_needs_a_private_key():
    with pytest.raises(TypeError):
        public_key_bytes(RFC_SEED)


# --- signing ------------------------------------------------------------------


def test_signature_is_over_the_p6_1_signing_payload():
    r = signed()
    signature = bytes.fromhex(r["signature"]["hex"])
    assert len(signature) == 64 and signature == PRIVATE.sign(signing_payload(unsigned()))
    PRIVATE.public_key().verify(signature, signing_payload(r))  # raises if wrong
    assert without_signature(r) == unsigned() and signature_is_valid(r)


def test_signing_is_deterministic_and_leaves_the_input_alone():
    u = unsigned()
    before = copy.deepcopy(u)
    assert sign_record(u, PRIVATE) == sign_record(copy.deepcopy(u), PRIVATE)
    assert u == before


def test_pinned_signature():
    """Fixed key, fixed record: the signature cannot drift without the signed bytes changing."""
    assert signed()["signature"]["hex"] == PINNED_SIGNATURE_HEX


PINNED_SIGNATURE_HEX = (
    "4fafbe37ffad672b2cad740253a16825b4efae344eb8519efebda97022715bdae7ba973d1866b6118c524c5f8a6bcfeebcace7417a87f99"
    "16233d4350070d505")


def test_refuses_to_sign_a_record_naming_another_key():
    with pytest.raises(ValueError, match="different public key"):
        sign_record(unsigned(OTHER), PRIVATE)


def test_refuses_to_sign_a_signed_or_malformed_record():
    with pytest.raises(ValueError):
        sign_record(signed(), PRIVATE)
    broken = unsigned()
    broken["owner"]["owner_id"] = ""
    with pytest.raises(ValueError):
        sign_record(broken, PRIVATE)


# --- verification: a signature you have never seen fail is untested -----------


def _set(record: dict, path: tuple, value) -> dict:
    record = copy.deepcopy(record)
    target = record
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    return record


@pytest.mark.parametrize("path, value", [
    (("owner", "owner_id"), "other-owner"),
    (("model", "label"), "other model"),
    (("model", "fingerprint", "sha256"), "ac" * 32),
    (("model", "fingerprint", "tensors"), 39),
    (("model", "fingerprint", "elements"), 308849),
    (("model", "fingerprint", "data_bytes"), 1235417),
    (("watermark_commitment", "commitment"), commit(bytes(32), bytes(16), bytes(31)).to_dict()),
    (("watermark_commitment", "scheme", "specification"), "elsewhere.py"),
    (("trigger_set_commitment", "sha256"), "ce" * 32),
    (("trigger_set_commitment", "triggers"), 101),
    (("commitment_publication", "sha256"), "00" * 32),
    (("commitment_publication", "path"), "elsewhere.json"),
    (("timestamp", "created_utc"), "2026-10-03T09:30:01+00:00"),
])
def test_changing_any_signed_field_breaks_the_signature(path, value):
    assert signature_is_valid(_set(signed(), path, value)) is False


def test_every_single_bit_flip_of_the_signature_fails():
    u = unsigned()
    signature = bytes.fromhex(signed()["signature"]["hex"])
    for bit in range(512):
        flipped = bytearray(signature)
        flipped[bit // 8] ^= 1 << (bit % 8)
        assert signature_is_valid(attach_signature(u, bytes(flipped))) is False, bit


def test_zero_and_foreign_signatures_fail():
    u = unsigned()
    assert signature_is_valid(attach_signature(u, bytes(64))) is False
    assert signature_is_valid(attach_signature(u, OTHER.sign(signing_payload(u)))) is False
    assert signature_is_valid(attach_signature(u, PRIVATE.sign(signing_payload(u)[1:]))) is False  # not the domain-tagged bytes
    assert signature_is_valid(attach_signature(u, PRIVATE.sign(canonical_json(signed())))) is False


def test_swapping_in_another_public_key_fails():
    r = _set(signed(), ("owner", "public_key", "hex"), public_key_bytes(OTHER).hex())
    assert signature_is_valid(r) is False


def test_a_signature_cannot_be_moved_to_another_record():
    other = build_unsigned_record(publication(), "ce" * 32, 100, public_key_bytes(PRIVATE), CREATED)
    moved = attach_signature(other, bytes.fromhex(signed()["signature"]["hex"]))
    assert signature_is_valid(moved) is False


def test_a_reissued_record_under_another_key_verifies_against_that_key():
    """The limit of a self-declared key: anyone can sign the same statement under their own key."""
    forged = sign_record(unsigned(OTHER), OTHER)
    assert signature_is_valid(forged)
    assert forged["owner"]["owner_id"] == signed()["owner"]["owner_id"]
    assert forged["owner"]["public_key"] != signed()["owner"]["public_key"]


def test_malformed_records_raise_instead_of_returning_false():
    with pytest.raises(ValueError):
        signature_is_valid(unsigned())
    with pytest.raises(ValueError):
        signature_is_valid(_set(signed(), ("signature", "hex"), "ab" * 63))


def test_a_planted_verifier_that_accepts_everything_is_caught(monkeypatch):
    """`sign_record` self-checks through `signature_is_valid`; the tamper tests would catch a broken one."""
    import src.crypto.signing as signing

    monkeypatch.setattr(signing, "signature_is_valid", lambda record: True)
    assert signing.signature_is_valid(attach_signature(unsigned(), bytes(64)))  # the planted bug
    monkeypatch.undo()
    assert signing.signature_is_valid(attach_signature(unsigned(), bytes(64))) is False


# --- the record file ----------------------------------------------------------


def test_write_then_read_and_refuse_to_replace(tmp_path):
    path = tmp_path / "provenance" / "record.json"
    digest = write_record(signed(), path)
    loaded, read_digest = read_record(path)
    assert loaded == signed() and digest == read_digest and signature_is_valid(loaded)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_record(signed(), path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        write_record(unsigned(), tmp_path / "unsigned.json")
    assert not (tmp_path / "unsigned.json").exists()


def test_read_refuses_non_canonical_files(tmp_path):
    path = tmp_path / "record.json"
    data = canonical_json(signed())
    for altered in (data.replace(b"\n", b"\r\n"), data + b"\n", json.dumps(signed()).encode()):
        path.write_bytes(altered)
        with pytest.raises(ValueError, match="canonical"):
            read_record(path)


def test_private_key_is_not_in_the_record():
    data = canonical_json(signed())
    assert RFC_SEED not in data and RFC_SEED.hex().encode() not in data
    assert RFC_PUBLIC.hex().encode() in data


# --- the script ---------------------------------------------------------------


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p6_2_sign_provenance_record")


def _fake_setup(script, tmp_path, monkeypatch, key_kind="owner"):
    """A test publication and a tiny trigger bundle under tmp_path, with the script pointed at its digest."""
    np = pytest.importorskip("numpy")
    from src.watermark.bundle import TriggerBundle, save_bundle

    bundle = TriggerBundle(images=np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3), base_indices=np.arange(3),
                           base_labels=np.zeros(3, dtype=np.int64), targets=np.ones(3, dtype=np.int64), amplitude=16,
                           num_classes=10, key_kind=key_kind, trigger_version="t", response_version="r")
    digest = save_bundle(bundle, tmp_path / "bundle.npz")
    monkeypatch.setattr(script, "P2_3_BUNDLE_SHA256", digest)
    write_publication(publication(), tmp_path / "provenance" / "commitment.json")
    return ["--signing-key", str(tmp_path / "secrets" / "signing.bin"), "--bundle", str(tmp_path / "bundle.npz"),
            "--publication", str(tmp_path / "provenance" / "commitment.json"),
            "--record", str(tmp_path / "provenance" / "record.json"), "--out-dir", str(tmp_path / "results")], digest


def test_script_creates_then_refuses_then_checks(script, tmp_path, monkeypatch):
    args, bundle_digest = _fake_setup(script, tmp_path, monkeypatch)
    created = script.main(args)["metrics"]
    record_path = tmp_path / "provenance" / "record.json"
    record, digest = read_record(record_path)
    seed = (tmp_path / "secrets" / "signing.bin").read_bytes()
    assert created["mode"] == "create" and created["signing_key_created_this_run"] and created["record_sha256"] == digest
    assert signature_is_valid(record)
    assert record["owner"]["public_key"]["hex"] == public_key_bytes(Ed25519PrivateKey.from_private_bytes(seed)).hex()
    assert record["trigger_set_commitment"] == {"scheme": "sha256/trigger-bundle/v1", "sha256": bundle_digest,
                                                "triggers": 3, "specification": "src/watermark/bundle.py"}
    assert record["commitment_publication"]["sha256"] == read_publication(tmp_path / "provenance" / "commitment.json")[1]
    t = created["tampering"]
    assert t["field_changes_tried"] == 11 and t["field_changes_accepted"] == 0
    assert t["signature_bit_flips_tried"] == 512 and t["signature_bit_flips_accepted"] == 0
    assert t["other_key_signature_accepted_under_owner_key"] is False
    assert t["reissued_under_another_key_verifies_against_that_key"] is True

    before = record_path.read_bytes()
    with pytest.raises(SystemExit, match="Refusing to replace"):
        script.main(args)
    assert record_path.read_bytes() == before and (tmp_path / "secrets" / "signing.bin").read_bytes() == seed

    result = script.main([*args, "--check"])
    assert result["metrics"]["mode"] == "check" and not result["metrics"]["signing_key_created_this_run"]
    assert result["metrics"]["record_sha256"] == digest and record_path.read_bytes() == before
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert seed.hex() not in text


def test_script_check_catches_a_tampered_record_another_key_and_a_wrong_bundle(script, tmp_path, monkeypatch):
    args, _ = _fake_setup(script, tmp_path, monkeypatch)
    script.main(args)
    record_path = tmp_path / "provenance" / "record.json"
    good = record_path.read_bytes()

    tampered = _set(read_record(record_path)[0], ("trigger_set_commitment", "triggers"), 4)
    record_path.write_bytes(canonical_json(tampered))
    with pytest.raises(SystemExit, match="does not verify"):
        script.main([*args, "--check"])
    record_path.write_bytes(good)

    key_path = tmp_path / "secrets" / "signing.bin"
    seed = key_path.read_bytes()
    key_path.write_bytes(bytes(32))
    with pytest.raises(SystemExit, match="another key"):
        script.main([*args, "--check"])
    key_path.write_bytes(seed)

    monkeypatch.setattr(script, "P2_3_BUNDLE_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="Refusing to sign"):
        script.main([*args, "--check"])


def test_script_refuses_a_test_bundle(script, tmp_path, monkeypatch):
    args, _ = _fake_setup(script, tmp_path, monkeypatch, key_kind="test")
    with pytest.raises(SystemExit, match="not the owner's"):
        script.main(args)
    assert not (tmp_path / "secrets" / "signing.bin").exists() and not (tmp_path / "provenance" / "record.json").exists()


def test_script_check_creates_nothing(script, tmp_path, monkeypatch):
    args, _ = _fake_setup(script, tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="nothing to check"):
        script.main([*args, "--check"])
    assert not (tmp_path / "secrets" / "signing.bin").exists()
    assert script.DEFAULT_SIGNING_KEY_PATH.parent.name == "secrets"  # gitignored


def test_script_tamper_check_catches_a_verifier_that_accepts_everything(script, monkeypatch):
    monkeypatch.setattr(script, "signature_is_valid", lambda record: True)
    t = script.tamper_checks(signed())
    assert t["field_changes_accepted"] == 11 and t["signature_bit_flips_accepted"] == 512


# --- the committed record -----------------------------------------------------


def test_committed_record_verifies_and_matches_the_publication():
    """The real record, where present. Needs no secret: only the public key it names."""
    path = REPO_ROOT / RECORD_PATH
    if not path.exists():
        pytest.skip("P6.2 has not been run")
    record, _ = read_record(path)
    assert signature_is_valid(record)
    published, digest = read_publication(REPO_ROOT / ARTIFACT_PATH)
    assert record["commitment_publication"] == {"path": ARTIFACT_PATH, "sha256": digest}
    assert record["watermark_commitment"]["commitment"] == published["commitment"]
    assert record["model"]["fingerprint"] == published["model_fingerprint"]
    assert record["owner"]["owner_id"] == published["owner_id"]
    assert record["trigger_set_commitment"]["sha256"].startswith("fbd65ec7")
    assert record["trigger_set_commitment"]["triggers"] == 100
