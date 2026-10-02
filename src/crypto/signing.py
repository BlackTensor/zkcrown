"""Signing the provenance record with a real Ed25519 keypair (P6.2).

The schema is `src/crypto/provenance.py` (P6.1). This module makes the
keypair, signs `signing_payload(record)`, and checks a signature against the
public key the record itself names. Ed25519 comes from the `cryptography`
library.

The keypair
-----------
- The private key is stored as its 32-byte seed, raw, in ``secrets/``
  (gitignored). It is written once and never overwritten.
- It has **no passphrase**: anyone who can read the file can sign as the
  owner. It needs the same care and backup as `K`.
- It is a separate key from `K`. Losing it does not affect the watermark or
  the commitment, but no further record can be signed under the same public
  key. Leaking it does not reveal `K`, but lets someone else sign records.
- It is also separate from the GPG key that signs the git tag (P5.5).

What a valid signature shows
----------------------------
`signature_is_valid` answers one question: was this record's signed part
signed by the private key matching ``owner.public_key`` in the same record?

- It shows the record has not been changed since that key signed it.
- It does **not** show who holds the key. The public key is self-declared;
  anyone can make a keypair and sign a record naming any owner id. Trust in
  the key has to come from outside the record.
- It does not show when the record was signed, that `C` opens, or anything
  about a model. The full verifier is P6.3.

Ed25519 signatures are deterministic: the same key and record always give the
same 64 bytes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from src.crypto.provenance import (
    PUBLIC_KEY_BYTES,
    attach_signature,
    signing_payload,
    validate_record,
    validate_unsigned_record,
)

PRIVATE_KEY_BYTES = 32
"""The Ed25519 seed. The stored form of the private key."""


def create_signing_key(path: Path | str) -> Path:
    """Write a fresh private key seed to `path`. Raises `FileExistsError` if it exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(seed)
        handle.flush()
        os.fsync(handle.fileno())
    if path.stat().st_size != PRIVATE_KEY_BYTES:
        raise RuntimeError(f"{path} is not {PRIVATE_KEY_BYTES} bytes after writing")
    return path


def load_signing_key(path: Path | str) -> Ed25519PrivateKey:
    """Read the private key from `path`, checking its length. **Secret.**"""
    seed = Path(path).read_bytes()
    if len(seed) != PRIVATE_KEY_BYTES:
        raise ValueError(f"{path} holds {len(seed)} bytes, expected {PRIVATE_KEY_BYTES}")
    return Ed25519PrivateKey.from_private_bytes(seed)


def public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    """The 32 raw public key bytes, the form ``owner.public_key.hex`` holds. Not secret."""
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("private_key must be an Ed25519PrivateKey")
    return private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def sign_record(unsigned: dict[str, Any], private_key: Ed25519PrivateKey) -> dict[str, Any]:
    """Sign an unsigned provenance record and return the signed record.

    Refuses if the record names a public key other than `private_key`'s: such
    a record could never verify. The result is checked with
    `signature_is_valid` before it is returned.
    """
    validate_unsigned_record(unsigned)
    public = public_key_bytes(private_key)
    if unsigned["owner"]["public_key"]["hex"] != public.hex():
        raise ValueError("the record names a different public key than the one signing it")
    record = attach_signature(unsigned, private_key.sign(signing_payload(unsigned)))
    if not signature_is_valid(record):
        raise RuntimeError("the signature just made does not verify")
    return record


def signature_is_valid(record: dict[str, Any]) -> bool:
    """True if ``signature`` is a valid Ed25519 signature of the record under its own ``owner.public_key``.

    Raises `ValueError` for a record that is not well formed. A well-formed
    record with a wrong signature gives False. See the module docstring for
    what True does not show.
    """
    validate_record(record)
    key = bytes.fromhex(record["owner"]["public_key"]["hex"])
    assert len(key) == PUBLIC_KEY_BYTES
    try:
        Ed25519PublicKey.from_public_bytes(key).verify(bytes.fromhex(record["signature"]["hex"]), signing_payload(record))
    except InvalidSignature:
        return False
    return True
