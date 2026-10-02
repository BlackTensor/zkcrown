"""Tests for P6.1: the provenance record schema.

All keys, nonces, digests and signatures here are test values. Nothing is
signed or verified: that is P6.2 and P6.3.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.crypto.commitment import commit
from src.crypto.fingerprint import FINGERPRINT_VERSION, ModelFingerprint
from src.crypto.provenance import (
    PRIVATE_NOTE,
    RECORD_SCHEMA,
    SIGNING_DOMAIN,
    TIMESTAMP_STATUS,
    TRIGGER_COMMITMENT_SCHEME,
    attach_signature,
    build_unsigned_record,
    canonical_json,
    record_sha256,
    signing_payload,
    validate_record,
    validate_unsigned_record,
    without_signature,
)
from src.crypto.publication import ARTIFACT_PATH, artifact_sha256, build_publication, read_publication

REPO_ROOT = Path(__file__).resolve().parents[1]

KEY = bytes(range(32))
SIG = bytes(range(100, 116))
NONCE = bytes(range(200, 231))
FINGERPRINT = ModelFingerprint(sha256="ab" * 32, version=FINGERPRINT_VERSION, tensors=38, elements=308848,
                               data_bytes=1235416)
PUBLISHED = "2026-10-02T12:00:00+00:00"
CREATED = "2026-10-03T09:30:00+00:00"
TRIGGER_DIGEST = "cd" * 32
PUBLIC_KEY = bytes(range(32, 64))
SIGNATURE = bytes(range(128, 192))  # disjoint from KEY, SIG and NONCE, so the no-secret test means something


def publication() -> dict:
    return build_publication(commit(KEY, SIG, NONCE), FINGERPRINT, "test-owner", "test model", PUBLISHED)


def unsigned(**changes) -> dict:
    args = {"publication": publication(), "trigger_digest": TRIGGER_DIGEST, "trigger_count": 100,
            "public_key": PUBLIC_KEY, "created_utc": CREATED, **changes}
    return build_unsigned_record(**args)


def signed() -> dict:
    return attach_signature(unsigned(), SIGNATURE)


# --- contents -----------------------------------------------------------------


def test_holds_the_six_things_the_task_names():
    r, p = signed(), publication()
    assert r["schema"] == RECORD_SCHEMA
    assert r["owner"] == {"owner_id": "test-owner", "public_key": {"algorithm": "ed25519", "hex": PUBLIC_KEY.hex()}}
    assert r["model"] == {"label": "test model", "fingerprint": FINGERPRINT.to_dict()}
    assert r["watermark_commitment"] == {"commitment": commit(KEY, SIG, NONCE).to_dict(), "scheme": p["commitment_scheme"]}
    assert r["trigger_set_commitment"] == {"scheme": "sha256/trigger-bundle/v1", "sha256": TRIGGER_DIGEST,
                                           "triggers": 100, "specification": "src/watermark/bundle.py"}
    assert r["timestamp"] == {"created_utc": CREATED, "status": TIMESTAMP_STATUS}
    assert r["signature"] == {"algorithm": "ed25519", "hex": SIGNATURE.hex()}
    assert "self-asserted" in TIMESTAMP_STATUS and "not over this record" in TIMESTAMP_STATUS
    assert r["private"] == PRIVATE_NOTE


def test_names_the_publication_it_was_built_from():
    assert unsigned()["commitment_publication"] == {"path": ARTIFACT_PATH, "sha256": artifact_sha256(publication())}


def test_trigger_scheme_follows_the_bundle_version():
    from src.watermark.bundle import BUNDLE_VERSION

    assert TRIGGER_COMMITMENT_SCHEME == "sha256/" + BUNDLE_VERSION


def test_trigger_commitment_accepts_a_real_bundle_digest():
    np = pytest.importorskip("numpy")
    from src.watermark.bundle import TriggerBundle

    bundle = TriggerBundle(images=np.zeros((3, 4, 4, 3), dtype=np.uint8), base_indices=np.arange(3),
                           base_labels=np.zeros(3, dtype=np.int64), targets=np.ones(3, dtype=np.int64), amplitude=16,
                           num_classes=10, key_kind="test", trigger_version="t", response_version="r")
    r = unsigned(trigger_digest=bundle.digest(), trigger_count=len(bundle))
    assert r["trigger_set_commitment"]["sha256"] == bundle.digest() and r["trigger_set_commitment"]["triggers"] == 3


def test_holds_no_secret():
    data = canonical_json(signed())
    for blob in (KEY, SIG, NONCE):
        assert blob.hex().encode() not in data and blob not in data
    for limb in (int.from_bytes(KEY[:16], "big"), int.from_bytes(KEY[16:], "big"), int.from_bytes(SIG, "big"),
                 int.from_bytes(NONCE, "big")):
        assert str(limb).encode() not in data


def test_build_refuses_bad_arguments():
    with pytest.raises(ValueError):
        unsigned(publication={**publication(), "owner_id": ""})
    with pytest.raises(TypeError, match="public_key must be bytes"):
        unsigned(public_key=PUBLIC_KEY.hex())
    with pytest.raises(ValueError, match="exactly 32"):
        unsigned(public_key=PUBLIC_KEY[:31])
    for digest in ("CD" * 32, "cd" * 31, 5):
        with pytest.raises(ValueError):
            unsigned(trigger_digest=digest)
    for count in (0, -1, True, 100.0):
        with pytest.raises(ValueError):
            unsigned(trigger_count=count)
    with pytest.raises(ValueError):
        unsigned(created_utc="yesterday")


def test_build_does_not_alias_the_publication():
    p = publication()
    r = build_unsigned_record(p, TRIGGER_DIGEST, 100, PUBLIC_KEY, CREATED)
    r["watermark_commitment"]["scheme"]["inputs"].append("x")
    r["model"]["fingerprint"]["tensors"] = 1
    r["watermark_commitment"]["commitment"]["hex"] = ""
    assert p == publication()


# --- the signed bytes ---------------------------------------------------------


def test_signing_payload_is_the_domain_then_the_canonical_unsigned_record():
    u = unsigned()
    payload = signing_payload(u)
    assert payload.startswith(SIGNING_DOMAIN) and SIGNING_DOMAIN == b"zk-crown/provenance-record/v1/signing\x00"
    body = payload[len(SIGNING_DOMAIN):]
    assert body == (json.dumps(u, sort_keys=True, indent=2) + "\n").encode("ascii")
    assert json.loads(body) == u and b"\r" not in body and "signature" not in json.loads(body)


def test_signing_payload_is_the_same_before_and_after_signing():
    assert signing_payload(unsigned()) == signing_payload(signed())
    assert signing_payload(attach_signature(unsigned(), bytes(64))) == signing_payload(signed())
    assert without_signature(signed()) == unsigned()


def test_signing_payload_is_order_independent():
    u = unsigned()
    shuffled = {name: (dict(reversed(list(value.items()))) if isinstance(value, dict) else value)
                for name, value in reversed(list(u.items()))}
    assert signing_payload(shuffled) == signing_payload(u) == signing_payload(copy.deepcopy(u))


def _set(record: dict, path: tuple, value) -> dict:
    record = copy.deepcopy(record)
    target = record
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    return record


def test_every_signed_field_changes_the_signing_payload():
    """Each leaf of the unsigned record is covered: changing it changes what is signed."""
    other_publication = build_publication(commit(KEY, SIG, bytes(31)), ModelFingerprint(
        sha256="ef" * 32, version=FINGERPRINT_VERSION, tensors=8, elements=6138, data_bytes=24552), "other-owner",
        "other model", "2026-10-02T12:00:01+00:00")
    variants = [
        unsigned(public_key=bytes(32)),
        unsigned(trigger_digest="ce" * 32),
        unsigned(trigger_count=99),
        unsigned(created_utc="2026-10-03T09:30:01+00:00"),
        unsigned(publication=other_publication),
        _set(unsigned(), ("owner", "owner_id"), "other-owner"),
        _set(unsigned(), ("model", "label"), "other model"),
        _set(unsigned(), ("model", "fingerprint", "sha256"), "ef" * 32),
        _set(unsigned(), ("model", "fingerprint", "elements"), 7),
        _set(unsigned(), ("watermark_commitment", "commitment"), commit(KEY, SIG, bytes(31)).to_dict()),
        _set(unsigned(), ("commitment_publication", "sha256"), "00" * 32),
        _set(unsigned(), ("commitment_publication", "path"), "elsewhere.json"),
    ]
    payloads = {signing_payload(v) for v in variants} | {signing_payload(unsigned())}
    assert len(payloads) == len(variants) + 1


def test_attach_signature_places_exactly_64_bytes_and_leaves_the_input_alone():
    u = unsigned()
    before = copy.deepcopy(u)
    r = attach_signature(u, SIGNATURE)
    assert u == before and "signature" not in u
    assert bytes.fromhex(r["signature"]["hex"]) == SIGNATURE
    with pytest.raises(ValueError, match="exactly 64"):
        attach_signature(u, SIGNATURE[:63])
    with pytest.raises(TypeError):
        attach_signature(u, SIGNATURE.hex())
    with pytest.raises(ValueError):
        attach_signature(r, SIGNATURE)  # already signed


def test_canonical_bytes_are_deterministic_and_round_trip():
    r = signed()
    data = canonical_json(r)
    assert data == canonical_json(dict(reversed(list(r.items())))) == canonical_json(copy.deepcopy(r))
    assert data.endswith(b"}\n") and b"\r" not in data and data.isascii() and json.loads(data) == r
    assert record_sha256(r) == hashlib.sha256(data).hexdigest()
    assert record_sha256(attach_signature(unsigned(), bytes(64))) != record_sha256(r)
    with pytest.raises(ValueError):
        canonical_json(unsigned())  # the file form is the signed record


def test_non_ascii_owner_id_stays_ascii_on_disk():
    p = build_publication(commit(KEY, SIG, NONCE), FINGERPRINT, "propriétaire-é", "m", PUBLISHED)
    r = attach_signature(unsigned(publication=p), SIGNATURE)
    assert canonical_json(r).isascii() and json.loads(canonical_json(r))["owner"]["owner_id"] == "propriétaire-é"


def test_pinned_signing_payload_digest():
    """A fixed target, so the signed bytes cannot drift without a schema version change."""
    assert hashlib.sha256(signing_payload(unsigned())).hexdigest() == PINNED_PAYLOAD_SHA256


def test_signing_payload_is_the_same_in_a_fresh_process():
    code = ("import hashlib, tests.test_provenance as t; "
            "print(hashlib.sha256(t.signing_payload(t.unsigned())).hexdigest())")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "4242"})
    assert out.stdout.strip() == hashlib.sha256(signing_payload(unsigned())).hexdigest()


PINNED_PAYLOAD_SHA256 = "98d7bf4d07903d38aa4bdb68a091ca816ed7ce9d4eae9b4ef92c697aaf16013a"


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize("path, value", [
    (("schema",), "zk-crown/provenance-record/v0"),
    (("owner", "owner_id"), ""),
    (("owner", "owner_id"), 5),
    (("owner", "public_key", "algorithm"), "rsa"),
    (("owner", "public_key", "hex"), "AB" * 32),
    (("owner", "public_key", "hex"), "ab" * 33),
    (("owner", "public_key"), "ab" * 32),
    (("model", "label"), "  "),
    (("model", "fingerprint", "sha256"), "ab" * 31),
    (("model", "fingerprint", "version"), "other/v1"),
    (("model", "fingerprint", "tensors"), 0),
    (("watermark_commitment", "commitment", "version"), "zk-crown/commitment/v0"),
    (("watermark_commitment", "commitment", "decimal"), "5"),  # no longer agrees with hex
    (("watermark_commitment", "commitment", "hex"), "00" * 32),
    (("watermark_commitment", "scheme", "hash"), "sha256"),
    (("watermark_commitment", "scheme", "inputs"), ["K", "S", "nonce"]),
    (("trigger_set_commitment", "scheme"), "poseidon/trigger-bundle/v1"),
    (("trigger_set_commitment", "sha256"), "cd" * 31),
    (("trigger_set_commitment", "sha256"), None),
    (("trigger_set_commitment", "triggers"), 0),
    (("trigger_set_commitment", "triggers"), True),
    (("trigger_set_commitment", "triggers"), "100"),
    (("trigger_set_commitment", "specification"), "elsewhere.py"),
    (("commitment_publication", "path"), ""),
    (("commitment_publication", "sha256"), "zz" * 32),
    (("timestamp", "created_utc"), "2026-10-03T09:30:00Z"),
    (("timestamp", "created_utc"), "2026-13-45T09:30:00+00:00"),
    (("timestamp", "created_utc"), "2026-10-03T09:30:00+05:00"),
    (("timestamp", "status"), "independently timestamped"),
    (("private",), ""),
    (("signature", "algorithm"), "ecdsa"),
    (("signature", "hex"), "ab" * 63),
    (("signature", "hex"), "AB" * 64),
    (("signature",), SIGNATURE.hex()),
])
def test_validation_rejects(path, value):
    with pytest.raises(ValueError):
        validate_record(_set(signed(), path, value))


@pytest.mark.parametrize("section", ["owner", "model", "watermark_commitment", "trigger_set_commitment",
                                     "commitment_publication", "timestamp", "signature"])
def test_validation_rejects_extra_and_missing_nested_fields(section):
    with pytest.raises(ValueError):
        validate_record(_set(signed(), (section, "K"), "00"))
    r = signed()
    del r[section][sorted(r[section])[0]]
    with pytest.raises(ValueError):
        validate_record(r)


def test_validation_rejects_missing_and_extra_top_level_fields():
    r = signed()
    del r["trigger_set_commitment"]
    with pytest.raises(ValueError, match="missing"):
        validate_record(r)
    with pytest.raises(ValueError, match="unexpected"):
        validate_record({**signed(), "K": "00"})
    with pytest.raises(ValueError):
        validate_record([signed()])
    with pytest.raises(ValueError, match="missing"):
        validate_record(unsigned())  # a signed record needs its signature
    with pytest.raises(ValueError, match="unexpected"):
        validate_unsigned_record(signed())  # and an unsigned one must not carry one
    validate_record(signed())
    validate_unsigned_record(unsigned())


def test_well_formed_is_not_verified():
    """The shape check accepts a signature that is 64 zero bytes. Verification is P6.3, not this."""
    validate_record(attach_signature(unsigned(), bytes(64)))


# --- against the real, public artifact ----------------------------------------


def test_a_record_can_be_built_from_the_committed_publication():
    """Shape only, with a test public key and the public P2.3 bundle digest. Nothing is written or signed."""
    path = REPO_ROOT / ARTIFACT_PATH
    if not path.exists():
        pytest.skip("P5.4 has not been run")
    published, digest = read_publication(path)
    p2_4 = json.loads(next((REPO_ROOT / "results").glob("p2.4_wdr__*.json")).read_text(encoding="utf-8"))
    bundle_digest = next(v for v in _strings(p2_4) if v.startswith("fbd65ec7") and len(v) == 64)
    r = build_unsigned_record(published, bundle_digest, 100, PUBLIC_KEY, CREATED)
    assert r["commitment_publication"]["sha256"] == digest
    assert r["owner"]["owner_id"] == published["owner_id"]
    assert r["model"]["fingerprint"] == published["model_fingerprint"]
    assert r["watermark_commitment"]["commitment"] == published["commitment"]


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
