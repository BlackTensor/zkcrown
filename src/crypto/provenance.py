"""The provenance record schema, ``zk-crown/provenance-record/v1`` (P6.1).

The provenance record is the one signed statement the owner makes about a
model: "the owner named here, holding this signing key, vouches for the model
with this fingerprint, and committed to this watermark secret and this trigger
set". The auditor (P9) takes a suspect model plus this record.

This module defines the schema, its canonical bytes and the exact bytes a
signature covers, and reads and writes the record file. The keypair and the
signature are `src/crypto/signing.py` (P6.2). This module verifies nothing
beyond shape (P6.3) and holds nothing secret.

Fields
------
An *unsigned* record has every field below except ``signature``. A *signed*
record has all of them. Unknown fields are refused at every level.

``schema``
    ``"zk-crown/provenance-record/v1"``.
``owner``
    - ``owner_id``: the identity string `S` was derived for (P2.1).
    - ``public_key``: ``{"algorithm": "ed25519", "hex": <64 hex>}``, the 32 raw
      bytes of the key that signs this record. It is inside the signed bytes,
      so a signature cannot be moved to another key's record.
``model``
    - ``label``: a human-readable name for the model.
    - ``fingerprint``: the SHA-256 model fingerprint (P5.1) with its
      serialization version and counts.
``watermark_commitment``
    - ``commitment``: `C` (P5.3), as version, decimal string and 32-byte hex.
    - ``scheme``: the hash instance, ordered input names and specification.
``trigger_set_commitment``
    - ``scheme``: ``"sha256/trigger-bundle/v1"``.
    - ``sha256``: the trigger bundle digest (`TriggerBundle.digest`, P2.3).
    - ``triggers``: the trigger count N.
    - ``specification``: ``"src/watermark/bundle.py"``.
``commitment_publication``
    - ``path``, ``sha256``: the P5.4 artifact this record was built from, by
      the SHA-256 of its bytes. That file is what P5.5 timestamps.
``timestamp``
    - ``created_utc``: when this record was built, UTC, to the second.
    - ``status``: fixed text saying whose clock that is and what it proves.
``private``
    Fixed text saying what is deliberately not in the record.
``signature``
    ``{"algorithm": "ed25519", "hex": <128 hex>}``, the 64-byte signature over
    `signing_payload`.

Decisions
---------
- **The trigger set commitment is the bundle digest, not a new Poseidon
  hash.** The digest already names one exact trigger set (images, base
  indices, labels, targets, amplitude). It is binding under SHA-256 collision
  resistance. It has no nonce; it hides the triggers because each one carries
  3,072 key-derived sign bits, far beyond any search. It is also already
  public, in every result record since P2.3, so a second, hiding commitment to
  the same set would hide nothing. Opening it means revealing the whole
  bundle: one trigger cannot be shown alone. A per-trigger (Merkle) form is in
  the Icebox.
- **The record is built from the publication.** `build_unsigned_record` takes
  the P5.4 record and copies `C`, the fingerprint, the model label and the
  owner id out of it, so the two files cannot disagree by construction. P6.3
  re-checks that against the file the hash names.
- **The signature covers everything else.** `signing_payload` is a domain tag
  followed by the canonical JSON of the record without its ``signature``
  field. The public key and the algorithm are inside those bytes.
- **Ed25519.** Deterministic signatures, 32-byte keys, 64-byte signatures, and
  in the `cryptography` library (section 2.3). The schema names the algorithm
  in both ``owner.public_key`` and ``signature`` and refuses any other.

What a well-formed record does and does not establish
-----------------------------------------------------
- Well-formed means shape only. `validate_record` does not check the
  signature, does not open `C`, and does not look at any model.
- A valid signature (P6.2, P6.3) shows that the holder of ``owner.public_key``
  made this statement. The key is self-declared: nothing in the record shows
  who holds it.
- ``timestamp.created_utc`` is self-asserted. The independent timestamp (P5.5)
  covers the commitment publication, so it covers `C`, the fingerprint and the
  owner id. It does **not** cover this record, so it does not cover the
  trigger set commitment or the signing key.
- Nothing in the record ties `C` or the trigger digest to the model. They are
  stated together, and signed together.
- That the triggers behind the digest really derive from the committed `K` is
  not shown by this record. It takes an opening (P5.6) or P7.9.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from src.crypto.publication import (
    ARTIFACT_PATH,
    artifact_sha256,
    check_commitment_fields,
    check_created_utc,
    check_fingerprint_fields,
    check_model_and_owner,
    validate_publication,
)
from src.watermark.bundle import BUNDLE_VERSION

RECORD_SCHEMA = "zk-crown/provenance-record/v1"
RECORD_PATH = "provenance/record.json"

SIGNATURE_ALGORITHM = "ed25519"
PUBLIC_KEY_BYTES = 32
SIGNATURE_BYTES = 64
SIGNING_DOMAIN = b"zk-crown/provenance-record/v1/signing\x00"
"""Prefix of the signed bytes, so a signature over a record is never valid for anything else."""

TRIGGER_COMMITMENT_SCHEME = "sha256/" + BUNDLE_VERSION
TRIGGER_COMMITMENT_SPECIFICATION = "src/watermark/bundle.py"

TIMESTAMP_STATUS = (
    "created_utc is self-asserted by the owner's clock. Any independent timestamp is over the commitment publication "
    "named in commitment_publication, not over this record."
)
PRIVATE_NOTE = (
    "The master key K, the ownership signature S, the commitment nonce, the trigger images and their target classes, "
    "and the private signing key are not in this file."
)

_UNSIGNED_FIELDS = {"schema", "owner", "model", "watermark_commitment", "trigger_set_commitment",
                    "commitment_publication", "timestamp", "private"}
_SIGNED_FIELDS = _UNSIGNED_FIELDS | {"signature"}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX128 = re.compile(r"[0-9a-f]{128}")


def _raw(name: str, value: Any, length: int) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes, got {type(value).__name__}")
    if len(value) != length:
        raise ValueError(f"{name} must be exactly {length} bytes, got {len(value)}")
    return bytes(value)


def build_unsigned_record(publication: dict[str, Any], trigger_digest: str, trigger_count: int, public_key: bytes,
                          created_utc: str) -> dict[str, Any]:
    """Assemble the record that P6.2 signs. Takes public values only.

    Args:
        publication: the P5.4 commitment publication record. `C`, the model
            fingerprint, the model label and the owner id are copied from it.
        trigger_digest: `TriggerBundle.digest()` of the owner's trigger set.
        trigger_count: the number of triggers N in that bundle.
        public_key: the 32 raw bytes of the Ed25519 public key that will sign.
        created_utc: ``YYYY-MM-DDTHH:MM:SS+00:00``, as `publication.utc_now` gives.
    """
    validate_publication(publication)
    record = {
        "schema": RECORD_SCHEMA,
        "owner": {
            "owner_id": publication["owner_id"],
            "public_key": {"algorithm": SIGNATURE_ALGORITHM, "hex": _raw("public_key", public_key, PUBLIC_KEY_BYTES).hex()},
        },
        "model": {"label": publication["model"], "fingerprint": dict(publication["model_fingerprint"])},
        "watermark_commitment": {
            "commitment": dict(publication["commitment"]),
            "scheme": json.loads(json.dumps(publication["commitment_scheme"])),
        },
        "trigger_set_commitment": {
            "scheme": TRIGGER_COMMITMENT_SCHEME,
            "sha256": trigger_digest,
            "triggers": trigger_count,
            "specification": TRIGGER_COMMITMENT_SPECIFICATION,
        },
        "commitment_publication": {"path": ARTIFACT_PATH, "sha256": artifact_sha256(publication)},
        "timestamp": {"created_utc": created_utc, "status": TIMESTAMP_STATUS},
        "private": PRIVATE_NOTE,
    }
    validate_unsigned_record(record)
    return record


def _object(record: dict, name: str, fields: set[str]) -> dict:
    value = record[name]
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} must hold exactly {sorted(fields)}")
    return value


def _hex(name: str, value: Any, pattern: re.Pattern) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{name} must be {128 if pattern is _HEX128 else 64} lowercase hex characters")


def _validate(record: Any, fields: set[str]) -> None:
    if not isinstance(record, dict):
        raise ValueError("a provenance record is a JSON object")
    if set(record) != fields:
        raise ValueError(f"provenance record fields differ: missing {sorted(fields - set(record))}, "
                         f"unexpected {sorted(set(record) - fields)}")
    if record["schema"] != RECORD_SCHEMA:
        raise ValueError(f"unknown provenance record schema {record['schema']!r}")

    owner = _object(record, "owner", {"owner_id", "public_key"})
    model = _object(record, "model", {"label", "fingerprint"})
    check_model_and_owner(model["label"], owner["owner_id"])
    check_fingerprint_fields(model["fingerprint"])
    key = _object(owner, "public_key", {"algorithm", "hex"})
    if key["algorithm"] != SIGNATURE_ALGORITHM:
        raise ValueError(f"unknown public key algorithm {key['algorithm']!r}")
    _hex("owner.public_key.hex", key["hex"], _HEX64)

    watermark = _object(record, "watermark_commitment", {"commitment", "scheme"})
    check_commitment_fields(watermark["commitment"], watermark["scheme"])

    triggers = _object(record, "trigger_set_commitment", {"scheme", "sha256", "triggers", "specification"})
    if triggers["scheme"] != TRIGGER_COMMITMENT_SCHEME:
        raise ValueError(f"unknown trigger set commitment scheme {triggers['scheme']!r}")
    _hex("trigger_set_commitment.sha256", triggers["sha256"], _HEX64)
    if isinstance(triggers["triggers"], bool) or not isinstance(triggers["triggers"], int) or triggers["triggers"] < 1:
        raise ValueError("trigger_set_commitment.triggers must be a positive integer")
    if triggers["specification"] != TRIGGER_COMMITMENT_SPECIFICATION:
        raise ValueError("trigger_set_commitment.specification must name src/watermark/bundle.py")

    publication = _object(record, "commitment_publication", {"path", "sha256"})
    if not isinstance(publication["path"], str) or not publication["path"].strip():
        raise ValueError("commitment_publication.path must be a non-empty string")
    _hex("commitment_publication.sha256", publication["sha256"], _HEX64)

    timestamp = _object(record, "timestamp", {"created_utc", "status"})
    check_created_utc(timestamp["created_utc"])
    if timestamp["status"] != TIMESTAMP_STATUS or record["private"] != PRIVATE_NOTE:
        raise ValueError("the timestamp status and private notes must be the standard text")

    if "signature" in fields:
        signature = _object(record, "signature", {"algorithm", "hex"})
        if signature["algorithm"] != SIGNATURE_ALGORITHM:
            raise ValueError(f"unknown signature algorithm {signature['algorithm']!r}")
        _hex("signature.hex", signature["hex"], _HEX128)


def validate_unsigned_record(record: Any) -> None:
    """Raise `ValueError` unless `record` is a well-formed record with no ``signature`` field."""
    _validate(record, _UNSIGNED_FIELDS)


def validate_record(record: Any) -> None:
    """Raise `ValueError` unless `record` is a well-formed *signed* record.

    Shape only. This does **not** verify the signature: a record with 64
    arbitrary bytes in ``signature`` passes. Verification is P6.3.
    """
    _validate(record, _SIGNED_FIELDS)


def _dumps(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def without_signature(record: dict[str, Any]) -> dict[str, Any]:
    """The unsigned part of a signed or unsigned record, validated."""
    unsigned = {name: value for name, value in record.items() if name != "signature"} if isinstance(record, dict) else record
    validate_unsigned_record(unsigned)
    return unsigned


def signing_payload(record: dict[str, Any]) -> bytes:
    """The exact bytes the signature covers: `SIGNING_DOMAIN` then the canonical JSON of the unsigned record.

    Canonical JSON is the form the publication uses: sorted keys, two-space
    indent, ASCII, LF, one trailing newline. Accepts a signed or an unsigned
    record and gives the same bytes for both.
    """
    return SIGNING_DOMAIN + _dumps(without_signature(record))


def attach_signature(unsigned: dict[str, Any], signature: bytes) -> dict[str, Any]:
    """Return the signed record. Does not check that `signature` verifies; it only places it."""
    validate_unsigned_record(unsigned)
    record = json.loads(_dumps(unsigned))
    record["signature"] = {"algorithm": SIGNATURE_ALGORITHM, "hex": _raw("signature", signature, SIGNATURE_BYTES).hex()}
    validate_record(record)
    return record


def canonical_json(record: dict[str, Any]) -> bytes:
    """The signed record's exact file bytes: sorted keys, two-space indent, LF, one trailing newline."""
    validate_record(record)
    return _dumps(record)


def record_sha256(record: dict[str, Any]) -> str:
    """SHA-256 of the signed record's canonical bytes."""
    return hashlib.sha256(canonical_json(record)).hexdigest()


def write_record(record: dict[str, Any], path: Path | str) -> str:
    """Write the signed record to `path`. Refuses to replace an existing one. Returns its SHA-256.

    Shape is checked, the signature is not: sign with `src.crypto.signing.sign_record` first.
    """
    data = canonical_json(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def read_record(path: Path | str) -> tuple[dict[str, Any], str]:
    """Load a signed record and check its shape. Returns the record and the SHA-256 of the file's bytes.

    Raises `ValueError` if the file is not in canonical form. Does not verify the signature.
    """
    data = Path(path).read_bytes()
    record = json.loads(data.decode("utf-8"))
    validate_record(record)
    if canonical_json(record) != data:
        raise ValueError(f"{path} is not in canonical form (key order, indentation or line endings were changed)")
    return record, hashlib.sha256(data).hexdigest()
