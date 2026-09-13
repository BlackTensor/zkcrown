"""The ownership signature `S`, derived from the master key `K` (P2.1).

`S` is the owner identity the watermark carries. It is not a plain string like
"owned by X". It is a 128-bit value that only the holder of `K` can compute
for a given owner identity.

Derivation
----------
::

    S = first 16 bytes of KeyStream(K, "signature/v1/owner:" + owner_id)

which by the P1.1 construction is::

    S = HMAC-SHA256(K, "zk-crown/keystream/v1\\0" || u16_be(len(L)) || L || u64_be(0))[:16]
    L = UTF-8("signature/v1/owner:" + owner_id)

So `S` is a PRF of the owner identity under `K`. What that gives:

- **Bound to `K`.** Without `K`, `S` for a given owner cannot be computed or
  predicted, because HMAC-SHA256 is a PRF. Knowing `S` does not reveal `K`.
- **Bound to the owner.** A different `owner_id` gives an unrelated `S`, so
  one `K` cannot quietly stand for two identities with the same signature.
- **Separate from every other use of `K`.** The keygen label is
  length-prefixed and the ``signature/`` namespace is reserved for this
  module, so `S` shares no bytes with the trigger streams (``triggers/v1/...``)
  or any later stream.
- **One field element.** 128 bits is below the BN254 scalar field modulus
  (about 2^253.6), so P5.3 can put `S` into the Poseidon commitment as a single
  field element with no reduction. `as_int` is that integer.

What `S` is **not**:

- **Not a public-key digital signature.** Checking `S` needs `K`, so it is
  closer to a MAC tag than to a signature anyone can verify. Public
  verifiability comes from the provenance record signature (P6.2) and the ZK
  proof of knowledge of `K` (Phase 7). The name `S` follows CLAUDE.md 1.1.
- **Not public.** `S` is treated as secret, like `K`. It is a private input
  to the commitment (P5.3, P7.5). What an audit reveals is decided later.
- **Not evidence of anything on its own.** A 128-bit string matches `S` by
  chance with probability 2^-128. That is a property of exact equality, not of
  watermark extraction. Extraction is noisy, and its false positive rate comes
  from measurement (P3.7).

Bits and signs
--------------
`bits` lists `S` most significant bit first. `signs` maps bit 1 to +1 and bit 0
to -1, which is the ±1 form the spread-spectrum embedding
``W* = W + alpha * P_K^T * S`` (P3.2) works with.

Owner identity
--------------
`owner_id` is any non-empty string up to 256 UTF-8 bytes. It must already be
in Unicode NFC form and have no leading or trailing whitespace. Otherwise two
strings that look identical could give different `S` values with nothing to
show why. The id is used exactly as given, and there is no case folding.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from src.watermark.keygen import KeyStream

SIGNATURE_VERSION = "signature/v1"
SIGNATURE_BITS = 128
SIGNATURE_BYTES = SIGNATURE_BITS // 8
MAX_OWNER_ID_BYTES = 256

_LABEL_PREFIX = f"{SIGNATURE_VERSION}/owner:"


def check_owner_id(owner_id: str) -> str:
    """Validate an owner identity string and return it unchanged.

    Raises:
        TypeError: if `owner_id` is not a `str`.
        ValueError: if it is empty, too long, not NFC-normalised, or has
            leading or trailing whitespace.
    """
    if not isinstance(owner_id, str):
        raise TypeError(f"owner_id must be a str, got {type(owner_id).__name__}")
    if not owner_id:
        raise ValueError("owner_id must be non-empty")
    if owner_id != owner_id.strip():
        raise ValueError("owner_id must not have leading or trailing whitespace")
    if unicodedata.normalize("NFC", owner_id) != owner_id:
        raise ValueError("owner_id must be in Unicode NFC form")
    if len(owner_id.encode("utf-8")) > MAX_OWNER_ID_BYTES:
        raise ValueError(f"owner_id must be at most {MAX_OWNER_ID_BYTES} UTF-8 bytes")
    return owner_id


def signature_label(owner_id: str) -> str:
    """The keygen label `S` is drawn from, for `owner_id`."""
    return _LABEL_PREFIX + check_owner_id(owner_id)


@dataclass(frozen=True)
class OwnershipSignature:
    """The derived signature `S` for one owner.

    Attributes:
        owner_id: the identity `S` was derived for.
        value: the 16 bytes of `S`.
        version: the derivation version, ``"signature/v1"``.
    """

    owner_id: str
    value: bytes
    version: str = SIGNATURE_VERSION

    def __post_init__(self) -> None:
        check_owner_id(self.owner_id)
        if not isinstance(self.value, bytes) or len(self.value) != SIGNATURE_BYTES:
            raise ValueError(f"value must be exactly {SIGNATURE_BYTES} bytes")

    def __repr__(self) -> str:
        # Keep S out of logs and tracebacks by default. Use .hex() explicitly.
        return f"OwnershipSignature(owner_id={self.owner_id!r}, version={self.version!r}, value=<hidden>)"

    @property
    def as_int(self) -> int:
        """`S` as a big-endian unsigned integer in [0, 2^128)."""
        return int.from_bytes(self.value, "big")

    def hex(self) -> str:
        return self.value.hex()

    @property
    def bits(self) -> tuple[int, ...]:
        """`S` as 128 bits, most significant bit first."""
        return tuple((byte >> (7 - i)) & 1 for byte in self.value for i in range(8))

    @property
    def signs(self) -> tuple[int, ...]:
        """`S` as 128 values in {-1, +1}. Bit 1 maps to +1."""
        return tuple(2 * bit - 1 for bit in self.bits)


def derive_signature(key: bytes, owner_id: str) -> OwnershipSignature:
    """Derive the ownership signature `S` from `K` for `owner_id`."""
    value = KeyStream(key, signature_label(owner_id)).read(SIGNATURE_BYTES)
    return OwnershipSignature(owner_id=owner_id, value=value)
