"""The ownership commitment `C = Poseidon(K, S, nonce)` (P5.3).

`C` is published before any dispute (P5.4, P5.5). It binds the owner to the
master key `K` and the ownership signature `S` without revealing either. Later
the owner either opens it (P5.6) or proves in zero knowledge that they know an
opening (P7). **The Circom circuit in P7.5 must reproduce this layout bit for
bit.** This docstring is the specification it is written against.

Field layout, ``zk-crown/commitment/v1``
---------------------------------------
Poseidon is `src/crypto/poseidon.py`: circomlib's instance over the BN254
scalar field ``p`` (254 bits). The commitment is circomlib's ``Poseidon(5)``,
state width ``t = 6``, over five field elements in this order:

    C = Poseidon( DOMAIN, K_hi, K_lo, S, nonce )

    index  name    bits  value
    0      DOMAIN  175   the 22 ASCII bytes "zk-crown/commitment/v1" as a big-endian integer.
                         A public constant: 0x7a6b2d63726f776e2f636f6d6d69746d656e742f7631
    1      K_hi    128   K[0:16]  as a big-endian unsigned integer
    2      K_lo    128   K[16:32] as a big-endian unsigned integer
    3      S       128   the 16 bytes of `S` as a big-endian unsigned integer (`OwnershipSignature.as_int`)
    4      nonce   248   the 31 nonce bytes as a big-endian unsigned integer

Every byte string is read big-endian: its first byte is the most significant.
`C` itself is one field element. It is written as a decimal string (what
circom and snarkjs use for public signals) or as 32 big-endian bytes.

Why this layout:

- **`K` is split, not reduced.** `K` is 256 bits and ``p`` is 254, so `K` does
  not fit in one element. Reducing ``K mod p`` would map several keys to one
  element. Two 128-bit limbs are each below ``p`` and together are exactly
  `K`, so the map from ``(K, S, nonce)`` to the five inputs is one-to-one.
  That makes "``C = Poseidon(K, S, nonce)``" a five-input hash in practice.
- **`S` fits in one element** by design (P2.1 chose 128 bits for this).
- **The nonce is 31 bytes**, the largest whole number of bytes that always
  fits in one element with no reduction. It is drawn from the OS random
  source, once per commitment, and kept secret with `K`. It makes two
  commitments to the same ``(K, S)`` unlinkable, and it hides `S` even from
  someone who could guess `K`-independent candidates for it.
- **A domain element comes first.** It keeps this hash apart from every other
  Poseidon use in the project (P7.9), so a
  value computed for one purpose can never be presented as another. In the
  circuit it is a hard-coded constant and costs no witness.

What the circuit must enforce (P7.5)
------------------------------------
Private inputs ``K_hi, K_lo, S, nonce``; public input ``C``; constraint
``Poseidon(5)([DOMAIN, K_hi, K_lo, S, nonce]) === C``. To make the proved
statement "I know a 32-byte `K`, a 16-byte `S` and a 31-byte nonce" rather
than "I know four field elements", the circuit should also range-check
``K_hi``, ``K_lo`` and ``S`` to 128 bits and the nonce to 248 bits
(``Num2Bits``). `commitment_inputs` refuses anything outside those ranges, so
host and circuit accept the same openings.

What the commitment does and does not give
------------------------------------------
- **Binding** rests on Poseidon's collision resistance: the owner cannot later
  open `C` to a different ``(K, S, nonce)``.
- **Hiding** rests on Poseidon's preimage resistance and on `K` and the nonce
  being uniformly random and secret. `C` reveals neither `K` nor `S`.
- `S` is itself derived from `K` and the owner id (P2.1), so committing to it
  adds no secret. It ties the owner id to the commitment: opening `C` shows
  which `S`, and so which owner id, was fixed at commitment time.
- The commitment says nothing about *when* it was made. That is the timestamp
  in P5.5. It also says nothing about any model. The model fingerprint sits
  next to `C` in the publication artifact (P5.4).
- An opening reveals `K`. After a non-ZK opening the watermark key is public.
  That is why P7 exists.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from src.crypto.poseidon import BN254_SCALAR_FIELD, poseidon
from src.watermark.keygen import KEY_BYTES
from src.watermark.signature import SIGNATURE_BYTES, OwnershipSignature

COMMITMENT_VERSION = "zk-crown/commitment/v1"
DOMAIN_ELEMENT = int.from_bytes(COMMITMENT_VERSION.encode("ascii"), "big")
"""Input 0 of the hash. A public constant, hard-coded in the circuit."""

LIMB_BYTES = 16
NONCE_BYTES = 31
"""248 bits: the most whole bytes that fit in one BN254 field element unreduced."""
COMMITMENT_BYTES = 32
INPUT_NAMES = ("DOMAIN", "K_hi", "K_lo", "S", "nonce")
INPUT_BITS = (DOMAIN_ELEMENT.bit_length(), 8 * LIMB_BYTES, 8 * LIMB_BYTES, 8 * SIGNATURE_BYTES, 8 * NONCE_BYTES)

assert KEY_BYTES == 2 * LIMB_BYTES
assert DOMAIN_ELEMENT < BN254_SCALAR_FIELD and (1 << (8 * NONCE_BYTES)) < BN254_SCALAR_FIELD


def new_nonce() -> bytes:
    """A fresh 31-byte nonce from the OS random source. Secret; store it with `K`."""
    return secrets.token_bytes(NONCE_BYTES)


def _bytes(name: str, value, length: int) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes, got {type(value).__name__}")
    if len(value) != length:
        raise ValueError(f"{name} must be exactly {length} bytes, got {len(value)}")
    return bytes(value)


def _signature_bytes(signature) -> bytes:
    if isinstance(signature, OwnershipSignature):
        return _bytes("S", signature.value, SIGNATURE_BYTES)
    return _bytes("S", signature, SIGNATURE_BYTES)


def commitment_inputs(key: bytes, signature, nonce: bytes) -> tuple[int, int, int, int, int]:
    """The five Poseidon inputs, in order: ``(DOMAIN, K_hi, K_lo, S, nonce)``.

    Elements 1 to 4 are the circuit's private witness. **Secret**: they are
    `K`, `S` and the nonce in another encoding.

    Args:
        key: the 32-byte master key `K`.
        signature: `S`, as an `OwnershipSignature` or its 16 raw bytes.
        nonce: 31 bytes from `new_nonce`.
    """
    key = _bytes("K", key, KEY_BYTES)
    nonce = _bytes("nonce", nonce, NONCE_BYTES)
    s = _signature_bytes(signature)
    return (
        DOMAIN_ELEMENT,
        int.from_bytes(key[:LIMB_BYTES], "big"),
        int.from_bytes(key[LIMB_BYTES:], "big"),
        int.from_bytes(s, "big"),
        int.from_bytes(nonce, "big"),
    )


@dataclass(frozen=True)
class Commitment:
    """The published value `C`. Not secret.

    Attributes:
        value: `C` as an integer in ``[0, p)``.
        version: the layout version, ``"zk-crown/commitment/v1"``.
    """

    value: int
    version: str = COMMITMENT_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("a commitment value must be an int")
        if not 0 <= self.value < BN254_SCALAR_FIELD:
            raise ValueError("a commitment value must be a field element in [0, p)")
        if self.version != COMMITMENT_VERSION:
            raise ValueError(f"unknown commitment version {self.version!r}")

    def decimal(self) -> str:
        """`C` as a decimal string, the form circom and snarkjs use for public signals."""
        return str(self.value)

    def to_bytes(self) -> bytes:
        """`C` as 32 big-endian bytes."""
        return self.value.to_bytes(COMMITMENT_BYTES, "big")

    def hex(self) -> str:
        return self.to_bytes().hex()

    def to_dict(self) -> dict[str, str]:
        return {"version": self.version, "decimal": self.decimal(), "hex": self.hex()}

    @classmethod
    def from_decimal(cls, text: str) -> "Commitment":
        if not isinstance(text, str) or not text.isascii() or not text.isdigit() or (len(text) > 1 and text[0] == "0"):
            raise ValueError("a commitment is a canonical decimal string: digits only, no leading zero")
        return cls(int(text))

    @classmethod
    def from_bytes(cls, data: bytes) -> "Commitment":
        return cls(int.from_bytes(_bytes("commitment", data, COMMITMENT_BYTES), "big"))


def commit(key: bytes, signature, nonce: bytes) -> Commitment:
    """`C = Poseidon(DOMAIN, K_hi, K_lo, S, nonce)`. The nonce is an argument, never drawn here."""
    return Commitment(poseidon(list(commitment_inputs(key, signature, nonce))))
