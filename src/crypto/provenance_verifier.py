"""The provenance verifier (P6.3).

Given the signed provenance record (P6.1, P6.2), the commitment publication it
names (P5.4) and, optionally, a suspect model, this module answers: is the
record intact, is its commitment well formed and the one that was published,
and is the suspect the model the record vouches for?

It needs public values only. It reads nothing under ``secrets/``, never sees
`K`, and holds nothing secret.

Checks
------
Each check is a field of `ProvenanceVerdict`. A failed check is a verdict, not
an exception.

1. ``record_well_formed``: the record passes `validate_record` (shape only).
2. ``signature_valid``: the Ed25519 signature verifies under the public key
   the record itself names (`signature_is_valid`). A key the library refuses
   counts as not valid.
3. ``commitment_well_formed``: the record's ``watermark_commitment`` holds a
   canonical decimal `C` in ``[0, p)`` of the BN254 field, its hex agrees with
   its decimal, the layout version is ``zk-crown/commitment/v1``, and the
   scheme names circomlib's Poseidon with the P5.3 input order. Checked on its
   own, so a commitment problem is reported even if another field is broken.
4. ``matches_publication``: the publication is well formed, its SHA-256 equals
   ``commitment_publication.sha256`` in the record, and the record's `C`,
   commitment scheme, model fingerprint, model label and owner id are equal to
   the publication's. This ties the signed record to the file P5.5 timestamps.
5. ``fingerprint_matches``: only when a suspect is given. Its P5.1
   fingerprint, with every count, equals the record's. ``None`` when no
   suspect was given.
6. ``public_key_trusted``: only when the caller supplies a public key obtained
   from outside the record. ``None`` otherwise.

``record_valid`` is checks 1 to 4. ``valid`` is every check that was run.
``checks_not_run`` lists the optional ones that were skipped, so a caller can
never mistake "not checked" for "passed".

What a valid verdict shows
--------------------------
- The record has not been changed since the holder of its public key signed
  it, and it states the same `C`, fingerprint and owner id as the publication.
- With a suspect: the suspect is bit for bit the model the record names. A
  mismatch says only that it is a different set of weights. Every Phase 4
  attack changes the fingerprint, so a mismatch is the expected outcome for a
  stolen and modified model, and ownership then rests on the watermark tests
  (P2.8, P3.7), which this module does not run.

What it does not show
---------------------
- **Who holds the key.** The public key is self-declared. Anyone can build a
  publication and a record of their own, consistent in every respect, naming
  any owner id, and it verifies. Only ``public_key_trusted``, against a key
  obtained independently (for example the one the owner's GPG-signed git tag
  vouches for), rules that out, and even that only moves the trust question to
  how that key was obtained.
- **When.** Both ``created_utc`` values are self-asserted. Independent time
  evidence is the P5.5 OpenTimestamps proof over the publication, which is
  not checked here, and which does not cover the record.
- **That `C` opens, or opens to the owner id.** "Well formed" means a valid
  field element under the right layout, not that anyone knows an opening. That
  is P5.6 (revealing `K`) or P7 (a ZK proof).
- **That the triggers behind the trigger set digest come from the committed
  `K`.** The digest is checked for shape only. That is P7.9.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.crypto.fingerprint import ModelFingerprint, fingerprint_state_dict
from src.crypto.provenance import PUBLIC_KEY_BYTES, record_sha256, validate_record
from src.crypto.publication import artifact_sha256, check_commitment_fields, validate_publication
from src.crypto.signing import signature_is_valid

VERDICT_SCHEMA = "zk-crown/provenance-verdict/v1"
OPTIONAL_CHECKS = ("fingerprint_matches", "public_key_trusted")


@dataclass(frozen=True)
class ProvenanceVerdict:
    """The outcome of verifying one record. Holds public values only."""

    record_well_formed: bool
    signature_valid: bool
    commitment_well_formed: bool
    matches_publication: bool
    fingerprint_matches: bool | None
    public_key_trusted: bool | None
    record_sha256: str | None = None
    owner_id: str | None = None
    public_key_hex: str | None = None
    commitment_decimal: str | None = None
    record_fingerprint_sha256: str | None = None
    suspect_fingerprint_sha256: str | None = None
    problems: tuple[str, ...] = field(default=())

    @property
    def record_valid(self) -> bool:
        """Checks 1 to 4: the record is intact and agrees with the publication. Says nothing about a suspect."""
        return self.record_well_formed and self.signature_valid and self.commitment_well_formed and self.matches_publication

    @property
    def checks_not_run(self) -> tuple[str, ...]:
        return tuple(name for name in OPTIONAL_CHECKS if getattr(self, name) is None)

    @property
    def valid(self) -> bool:
        """Every check that was run passed. Read with `checks_not_run`."""
        return self.record_valid and self.fingerprint_matches is not False and self.public_key_trusted is not False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": VERDICT_SCHEMA,
            "valid": self.valid,
            "record_valid": self.record_valid,
            "record_well_formed": self.record_well_formed,
            "signature_valid": self.signature_valid,
            "commitment_well_formed": self.commitment_well_formed,
            "matches_publication": self.matches_publication,
            "fingerprint_matches": self.fingerprint_matches,
            "public_key_trusted": self.public_key_trusted,
            "checks_not_run": list(self.checks_not_run),
            "record_sha256": self.record_sha256,
            "owner_id": self.owner_id,
            "public_key_hex": self.public_key_hex,
            "commitment_decimal": self.commitment_decimal,
            "record_fingerprint_sha256": self.record_fingerprint_sha256,
            "suspect_fingerprint_sha256": self.suspect_fingerprint_sha256,
            "problems": list(self.problems),
        }


def _suspect_fingerprint(suspect: Any) -> ModelFingerprint:
    if isinstance(suspect, ModelFingerprint):
        return suspect
    if isinstance(suspect, Mapping):
        return fingerprint_state_dict(suspect)
    if hasattr(suspect, "state_dict"):
        return fingerprint_state_dict(suspect.state_dict())
    raise TypeError("suspect must be a ModelFingerprint, a state dict or a module with state_dict()")


def _trusted_key(key: Any) -> str:
    if isinstance(key, (bytes, bytearray)):
        if len(key) != PUBLIC_KEY_BYTES:
            raise ValueError(f"a trusted public key is {PUBLIC_KEY_BYTES} bytes, got {len(key)}")
        return bytes(key).hex()
    if isinstance(key, str):
        raw = bytes.fromhex(key)
        if len(raw) != PUBLIC_KEY_BYTES:
            raise ValueError(f"a trusted public key is {PUBLIC_KEY_BYTES} bytes, got {len(raw)}")
        return raw.hex()
    raise TypeError("trusted_public_key must be bytes or a hex string")


def _get(record: Any, *path: str) -> Any:
    value = record
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def verify_provenance(record: Any, publication: Any, suspect: Any = None,
                      trusted_public_key: bytes | str | None = None) -> ProvenanceVerdict:
    """Verify a signed provenance record against its publication and, optionally, a suspect.

    Args:
        record: the signed record, as read from ``provenance/record.json``.
        publication: the commitment publication, as read from
            ``provenance/commitment.json``.
        suspect: a `ModelFingerprint`, a state dict, or a module. ``None``
            skips the fingerprint check.
        trusted_public_key: the owner's Ed25519 public key (32 bytes or hex),
            obtained from somewhere other than the record. ``None`` skips the
            trust check.

    Never raises on a failed check. Raises `TypeError` / `ValueError` only for
    a malformed `suspect` or `trusted_public_key`, which are the caller's
    inputs, not evidence.
    """
    suspect_fp = None if suspect is None else _suspect_fingerprint(suspect)
    trusted_hex = None if trusted_public_key is None else _trusted_key(trusted_public_key)
    problems: list[str] = []

    try:
        validate_record(record)
        well_formed = True
    except (ValueError, TypeError) as error:
        well_formed = False
        problems.append(f"record is malformed: {error}")

    signature_ok = False
    if well_formed:
        try:
            signature_ok = signature_is_valid(record)
        except ValueError as error:  # e.g. a public key that is not a valid Ed25519 point
            problems.append(f"public key refused: {error}")
        if not signature_ok:
            problems.append("the signature does not verify under the public key the record names")

    watermark = _get(record, "watermark_commitment")
    try:
        if not isinstance(watermark, dict) or set(watermark) != {"commitment", "scheme"}:
            raise ValueError("watermark_commitment must hold commitment and scheme")
        check_commitment_fields(watermark["commitment"], watermark["scheme"])
        commitment_ok = True
    except (ValueError, TypeError) as error:
        commitment_ok = False
        problems.append(f"commitment is malformed: {error}")

    matches = False
    try:
        validate_publication(publication)
        publication_ok = True
    except (ValueError, TypeError) as error:
        publication_ok = False
        problems.append(f"publication is malformed: {error}")
    if publication_ok and well_formed:
        mismatched = []
        if artifact_sha256(publication) != record["commitment_publication"]["sha256"]:
            mismatched.append("publication SHA-256")
        if record["watermark_commitment"]["commitment"] != publication["commitment"]:
            mismatched.append("commitment")
        if record["watermark_commitment"]["scheme"] != publication["commitment_scheme"]:
            mismatched.append("commitment scheme")
        if record["model"]["fingerprint"] != publication["model_fingerprint"]:
            mismatched.append("model fingerprint")
        if record["model"]["label"] != publication["model"]:
            mismatched.append("model label")
        if record["owner"]["owner_id"] != publication["owner_id"]:
            mismatched.append("owner id")
        matches = not mismatched
        if mismatched:
            problems.append("record and publication disagree on: " + ", ".join(mismatched))

    fingerprint_ok = None
    if suspect_fp is not None:
        record_fp = _get(record, "model", "fingerprint")
        fingerprint_ok = well_formed and record_fp == suspect_fp.to_dict()
        if not fingerprint_ok:
            problems.append("the suspect's fingerprint differs from the one the record names")

    trusted_ok = None
    if trusted_hex is not None:
        trusted_ok = well_formed and _get(record, "owner", "public_key", "hex") == trusted_hex
        if not trusted_ok:
            problems.append("the record is signed under a public key other than the trusted one")

    def text(*path: str) -> str | None:
        value = _get(record, *path)
        return value if isinstance(value, str) else None

    record_hash = record_sha256(record) if well_formed else None
    return ProvenanceVerdict(
        record_well_formed=well_formed,
        signature_valid=signature_ok,
        commitment_well_formed=commitment_ok,
        matches_publication=matches,
        fingerprint_matches=fingerprint_ok,
        public_key_trusted=trusted_ok,
        record_sha256=record_hash,
        owner_id=text("owner", "owner_id"),
        public_key_hex=text("owner", "public_key", "hex"),
        commitment_decimal=text("watermark_commitment", "commitment", "decimal"),
        record_fingerprint_sha256=text("model", "fingerprint", "sha256"),
        suspect_fingerprint_sha256=None if suspect_fp is None else suspect_fp.sha256,
        problems=tuple(problems),
    )
