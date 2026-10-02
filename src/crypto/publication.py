"""The commitment publication artifact, ``provenance/commitment.json`` (P5.4).

This is the file the owner publishes *before* any dispute. It says, in public:
"the owner named here committed to a secret, `C`, for the model with this
fingerprint, at this time". It holds nothing secret. `K`, `S` and the nonce
stay in ``secrets/``.

Contents, schema ``zk-crown/commitment-publication/v1``
------------------------------------------------------
- ``commitment``: `C` (P5.3) with its layout version, as a decimal string and
  as 32 big-endian bytes in hex.
- ``commitment_scheme``: the hash instance and the ordered input names, so a
  reader knows what `C` is a hash of without reading the code.
- ``model_fingerprint``: the SHA-256 model fingerprint (P5.1) with its
  serialization version and counts.
- ``model``: a human-readable label for which model that is.
- ``owner_id``: the owner identity string `S` was derived for (P2.1).
- ``created_utc``: when the artifact was written, UTC, to the second.
- ``timestamp_status``: says in words that ``created_utc`` is the owner's own
  clock and proves nothing by itself.
- ``private``: says what is deliberately not in the file.

What the artifact does and does not establish
---------------------------------------------
- It fixes `C`, the model fingerprint and the owner id together in one byte
  string. Its SHA-256 (`artifact_sha256`) is what P5.5 timestamps.
- ``created_utc`` is **self-asserted**. Anyone can write any date into a JSON
  file. "Committed before the dispute" means something only once an
  independent timestamp exists over these exact bytes (P5.5).
- The file is not signed. Nothing here stops someone else from publishing
  their own artifact naming the same fingerprint. Signing is P6.2.
- `C` is not checked against the fingerprint or owner id by anything in the
  file: `C` commits to `K`, `S` and a nonce, not to the model. The link
  between them is that they were published together.
- The owner id is tied to `C` through `S`, but only an opening (P5.6) or a
  proof (P7) shows that.

Canonical bytes
---------------
`canonical_json` writes the record as UTF-8 JSON with sorted keys, two-space
indentation, ``\\n`` line endings and one trailing newline. The same record
always gives the same bytes, so the artifact hash is reproducible from the
record. ``.gitattributes`` keeps git from rewriting line endings under
``provenance/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.crypto.commitment import COMMITMENT_VERSION, INPUT_NAMES, Commitment
from src.crypto.fingerprint import FINGERPRINT_VERSION, ModelFingerprint
from src.crypto.poseidon import POSEIDON_INSTANCE
from src.watermark.signature import check_owner_id

PUBLICATION_SCHEMA = "zk-crown/commitment-publication/v1"
ARTIFACT_PATH = "provenance/commitment.json"

TIMESTAMP_STATUS = (
    "created_utc is self-asserted by the owner's clock. It is not evidence of when this file existed until an "
    "independent timestamp over these exact bytes is attached."
)
PRIVATE_NOTE = (
    "The master key K, the ownership signature S and the commitment nonce are not in this file. The commitment hides "
    "them."
)

_FIELDS = {"schema", "commitment", "commitment_scheme", "model_fingerprint", "model", "owner_id", "created_utc",
           "timestamp_status", "private"}
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def utc_now() -> str:
    """The current time, UTC, to the second, as ``YYYY-MM-DDTHH:MM:SS+00:00``."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_publication(commitment: Commitment, fingerprint: ModelFingerprint, owner_id: str, model: str,
                      created_utc: str) -> dict[str, Any]:
    """Assemble the publication record. Takes public values only: no key, signature or nonce."""
    if not isinstance(commitment, Commitment):
        raise TypeError("commitment must be a Commitment; this function never sees the opening")
    if not isinstance(fingerprint, ModelFingerprint):
        raise TypeError("fingerprint must be a ModelFingerprint")
    record = {
        "schema": PUBLICATION_SCHEMA,
        "commitment": commitment.to_dict(),
        "commitment_scheme": {
            "hash": POSEIDON_INSTANCE,
            "inputs": list(INPUT_NAMES),
            "specification": "src/crypto/commitment.py",
        },
        "model_fingerprint": fingerprint.to_dict(),
        "model": model,
        "owner_id": check_owner_id(owner_id),
        "created_utc": created_utc,
        "timestamp_status": TIMESTAMP_STATUS,
        "private": PRIVATE_NOTE,
    }
    validate_publication(record)
    return record


def validate_publication(record: Any) -> None:
    """Raise `ValueError` unless `record` is a well-formed publication.

    Well-formed means the right fields with values of the right shape. It does
    **not** mean the commitment opens, the timestamp is true, or the
    fingerprint belongs to any particular model.
    """
    if not isinstance(record, dict):
        raise ValueError("a publication is a JSON object")
    if set(record) != _FIELDS:
        raise ValueError(f"publication fields differ: missing {sorted(_FIELDS - set(record))}, "
                         f"unexpected {sorted(set(record) - _FIELDS)}")
    if record["schema"] != PUBLICATION_SCHEMA:
        raise ValueError(f"unknown publication schema {record['schema']!r}")

    check_commitment_fields(record["commitment"], record["commitment_scheme"])
    check_fingerprint_fields(record["model_fingerprint"])
    check_model_and_owner(record["model"], record["owner_id"])
    check_created_utc(record["created_utc"])

    if record["timestamp_status"] != TIMESTAMP_STATUS or record["private"] != PRIVATE_NOTE:
        raise ValueError("the timestamp_status and private notes must be the standard text")


# The field checks below are shared with the provenance record (P6.1), which carries the same values.


def check_commitment_fields(c: Any, scheme: Any) -> None:
    """Raise `ValueError` unless `c` and `scheme` are a well-formed ``commitment`` / ``commitment_scheme`` pair."""
    if not isinstance(c, dict) or set(c) != {"version", "decimal", "hex"}:
        raise ValueError("commitment must hold version, decimal and hex")
    if c["version"] != COMMITMENT_VERSION:
        raise ValueError(f"unknown commitment version {c['version']!r}")
    commitment = Commitment.from_decimal(c["decimal"])
    if c["hex"] != commitment.hex():
        raise ValueError("commitment decimal and hex disagree")
    if not isinstance(scheme, dict) or scheme.get("hash") != POSEIDON_INSTANCE or scheme.get("inputs") != list(INPUT_NAMES):
        raise ValueError("commitment_scheme does not describe this project's commitment")


def check_fingerprint_fields(f: Any) -> None:
    """Raise `ValueError` unless `f` is a well-formed ``model_fingerprint`` object."""
    if not isinstance(f, dict) or set(f) != {"sha256", "version", "tensors", "elements", "data_bytes"}:
        raise ValueError("model_fingerprint must hold sha256, version, tensors, elements and data_bytes")
    if f["version"] != FINGERPRINT_VERSION:
        raise ValueError(f"unknown fingerprint version {f['version']!r}")
    if not isinstance(f["sha256"], str) or not _HEX64.fullmatch(f["sha256"]):
        raise ValueError("model fingerprint must be 64 lowercase hex characters")
    for count in ("tensors", "elements", "data_bytes"):
        if isinstance(f[count], bool) or not isinstance(f[count], int) or f[count] < 1:
            raise ValueError(f"model_fingerprint.{count} must be a positive integer")


def check_model_and_owner(model: Any, owner_id: Any) -> None:
    """Raise `ValueError` unless `model` is a non-empty label and `owner_id` a valid owner id."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty label")
    if not isinstance(owner_id, str):
        raise ValueError("owner_id must be a string")
    check_owner_id(owner_id)


def check_created_utc(created: Any) -> None:
    """Raise `ValueError` unless `created` looks like ``2026-01-31T12:00:00+00:00`` and is a real time."""
    if not isinstance(created, str) or not _UTC.fullmatch(created):
        raise ValueError("created_utc must look like 2026-01-31T12:00:00+00:00")
    try:
        datetime.fromisoformat(created)
    except ValueError as error:
        raise ValueError(f"created_utc is not a real date and time: {created!r}") from error


def canonical_json(record: dict[str, Any]) -> bytes:
    """The artifact's exact bytes: sorted keys, two-space indent, LF, one trailing newline."""
    validate_publication(record)
    return (json.dumps(record, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def artifact_sha256(record: dict[str, Any]) -> str:
    """SHA-256 of the artifact's canonical bytes. What P5.5 timestamps."""
    return hashlib.sha256(canonical_json(record)).hexdigest()


def write_publication(record: dict[str, Any], path: Path | str) -> str:
    """Write the artifact to `path`. Refuses to replace an existing one. Returns its SHA-256.

    A published commitment is a statement about the past. Overwriting it would
    silently replace that statement, so this raises `FileExistsError` instead.
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


def read_publication(path: Path | str) -> tuple[dict[str, Any], str]:
    """Load and validate an artifact. Returns the record and the SHA-256 of the file's bytes.

    Raises `ValueError` if the file is not in canonical form, since a
    timestamp over it would then cover bytes the record does not reproduce.
    """
    data = Path(path).read_bytes()
    record = json.loads(data.decode("utf-8"))
    validate_publication(record)
    if canonical_json(record) != data:
        raise ValueError(f"{path} is not in canonical form (key order, indentation or line endings were changed)")
    return record, hashlib.sha256(data).hexdigest()
