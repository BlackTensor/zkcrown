"""Deterministic pseudo-random streams derived from the master key `K` (P1.1).

`K` is the owner's secret. It is **not** the trigger (CLAUDE.md 1.1). Anything
the watermark needs that must be secret but reproducible, such as the trigger
set (P1.2) and later the projection `P_K` (P3.1), is drawn from a stream
produced here. Same `K` and same label give the same bytes, on any machine, in
any process, whatever the global RNG state.

Construction
------------
`K` is exactly 32 bytes. A stream is identified by `K` and a text label that
names its purpose, e.g. ``"triggers/v1"``. Block ``i`` of the stream is::

    HMAC-SHA256(K, DOMAIN || u16_be(len(label_utf8)) || label_utf8 || u64_be(i))

and the stream is block 0, block 1, block 2, ... concatenated. This is an
HMAC-based PRF in counter mode, the same shape as the NIST SP 800-108 counter
mode KDF. Consequences:

- **Determinism.** Only HMAC-SHA256 and fixed big-endian encodings are
  involved. Nothing depends on `random`, numpy, torch, `hash()` or platform
  byte order.
- **Domain separation.** Different labels give streams that are independent
  as long as HMAC-SHA256 is a PRF. The label is length-prefixed, so no
  (label, counter) pair encodes to the same message as another. The trigger
  set and the weight projection can therefore share one `K` without one
  leaking information about the other.
- **Random access.** Any block can be computed directly with `block()`.
- **Version tag.** `DOMAIN` carries ``v1``. Changing the construction means
  bumping it, which changes every derived trigger. The known-answer test in
  `tests/test_keygen.py` is there to catch an accidental change.

Floats and integers are built from the bytes with exact integer arithmetic
(`uniforms`, `randbelow`), so they are bit-identical across platforms too.

Not decided here
----------------
- How `K` is encoded as a field element for the Poseidon commitment. That is
  P5.3. A 32-byte `K` can exceed the BN254 scalar field modulus, so P5.3 has
  to specify a reduction or a split.
- Whether the trigger derivation can be proved inside a circuit (P7.9).
  SHA-256 is expensive in R1CS. If P7.9 needs a circuit-friendly derivation,
  that is a new stream version, recorded there, not a silent change to this
  one.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import struct

KEY_BYTES = 32
"""Length of the master key `K` in bytes (256 bits)."""

DOMAIN = b"zk-crown/keystream/v1\x00"
"""Prefix of every HMAC message. Bumping the version changes every stream."""

BLOCK_BYTES = hashlib.sha256().digest_size
"""Bytes per stream block (32)."""

_MAX_LABEL_BYTES = 0xFFFF
_MAX_COUNTER = 2**64 - 1


def generate_key() -> bytes:
    """Return a fresh random 32-byte master key from the OS CSPRNG.

    The real `K` belongs in `secrets/`, which is gitignored. This function does
    not write it anywhere.
    """
    return secrets.token_bytes(KEY_BYTES)


def check_key(key: bytes | bytearray | memoryview) -> bytes:
    """Validate a master key and return it as immutable `bytes`.

    Raises:
        TypeError: if `key` is not bytes-like. A `str` is rejected rather than
            guessed at: hex and UTF-8 would give different keys.
        ValueError: if `key` is not exactly `KEY_BYTES` long.
    """
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise TypeError(f"key must be bytes-like, got {type(key).__name__}")
    key = bytes(key)
    if len(key) != KEY_BYTES:
        raise ValueError(f"key must be exactly {KEY_BYTES} bytes, got {len(key)}")
    return key


def _encode_label(label: str) -> bytes:
    if not isinstance(label, str):
        raise TypeError(f"label must be a str, got {type(label).__name__}")
    encoded = label.encode("utf-8")
    if not encoded:
        raise ValueError("label must be non-empty")
    if len(encoded) > _MAX_LABEL_BYTES:
        raise ValueError(f"label must be at most {_MAX_LABEL_BYTES} UTF-8 bytes")
    return struct.pack(">H", len(encoded)) + encoded


class KeyStream:
    """A deterministic byte stream derived from `K` and a purpose label.

    Args:
        key: the 32-byte master key `K`.
        label: names what the stream is for, e.g. ``"triggers/v1"``. Each
            distinct use of `K` must use its own label.

    Reads advance a cursor. Two `KeyStream` objects built from the same key and
    label return the same bytes for the same sequence of reads, and splitting a
    read into chunks does not change the bytes returned.
    """

    def __init__(self, key: bytes | bytearray | memoryview, label: str) -> None:
        self._label = label
        self._prefix_mac = hmac.new(check_key(key), DOMAIN + _encode_label(label), hashlib.sha256)
        self._position = 0

    @property
    def label(self) -> str:
        return self._label

    @property
    def position(self) -> int:
        """Number of bytes consumed so far."""
        return self._position

    def block(self, index: int) -> bytes:
        """Return block `index` of the stream without moving the cursor."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"block index must be an int, got {type(index).__name__}")
        if not 0 <= index <= _MAX_COUNTER:
            raise ValueError(f"block index out of range: {index}")
        mac = self._prefix_mac.copy()
        mac.update(struct.pack(">Q", index))
        return mac.digest()

    def read(self, n: int) -> bytes:
        """Return the next `n` bytes of the stream."""
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError(f"n must be an int, got {type(n).__name__}")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        start, end = self._position, self._position + n
        first, last = start // BLOCK_BYTES, (end + BLOCK_BYTES - 1) // BLOCK_BYTES
        data = b"".join(self.block(i) for i in range(first, last))
        offset = start - first * BLOCK_BYTES
        self._position = end
        return data[offset : offset + n]

    def uint64s(self, n: int) -> list[int]:
        """Return the next `n` unsigned 64-bit integers, big-endian from the stream."""
        return list(struct.unpack(f">{n}Q", self.read(8 * n)))

    def uniforms(self, n: int) -> list[float]:
        """Return `n` floats uniform on [0, 1), each built from 53 stream bits.

        ``(u64 >> 11) / 2**53`` is exact in IEEE-754 double precision, so the
        values are identical on every platform.
        """
        scale = 1.0 / (1 << 53)
        return [(value >> 11) * scale for value in self.uint64s(n)]

    def randbelow(self, upper: int) -> int:
        """Return an integer uniform on [0, upper), without modulo bias.

        Rejection sampling over 64-bit draws. The expected number of draws is
        under 2 for every `upper`.
        """
        if not isinstance(upper, int) or isinstance(upper, bool):
            raise TypeError(f"upper must be an int, got {type(upper).__name__}")
        if not 0 < upper <= 1 << 64:
            raise ValueError(f"upper must be in (0, 2**64], got {upper}")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            (value,) = struct.unpack(">Q", self.read(8))
            if value < limit:
                return value % upper
