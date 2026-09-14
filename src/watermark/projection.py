"""The key-derived pseudo-random projection `P_K` (P3.1).

`P_K` is the secret linear map the weight watermark lives in. P3.2 embeds the
signature along its rows, ``W* = W + alpha * P_K^T * S``, and P3.3 extracts by
projecting the suspect weights back, ``y = P_K * w``, and correlating `y`
with `S`. This module only builds `P_K` and applies it. Which parameters form
the carrier vector `w`, the value of `alpha`, and how extraction is scored are
decided in P3.2, P3.3 and P3.7.

Construction
------------
`P_K` has ``rows`` rows, one per signature bit (128 by default, P2.1), and
``dim`` columns, one per carrier parameter. Its entries are key-derived
Rademacher signs scaled to unit row norm::

    P_K[i, j] = sigma[i, j] / sqrt(dim)          sigma[i, j] in {-1, +1}

The sign matrix `sigma` is read from one key stream::

    stream = KeyStream(K, "projection/v1/rademacher/dim=<dim>")

Row ``i`` takes stream bytes ``[i * B, (i + 1) * B)`` with ``B = ceil(dim / 8)``.
The bits are unpacked most significant bit first, the trailing ``8 * B - dim``
bits are discarded, and bit 1 means +1. This is the same layout as the trigger
sign patterns (P1.2).

Why this shape
--------------
- **Spread, not concentrated.** Every row touches every carrier parameter
  with the same magnitude, so each signature bit is spread over all ``dim``
  parameters and no single parameter carries much of it (P3.2).
- **Unit rows, near-orthogonal.** Each row has norm exactly 1. The inner
  product of two different rows is ``(sum of dim independent fair signs) /
  dim``, which has mean 0 and standard deviation ``1 / sqrt(dim)``, about
  0.0018 for the 307,040 conv and linear weights of `main_model`. So
  ``P_K * P_K^T`` is the identity plus small cross-talk, and
  ``P_K * (W + alpha * P_K^T * S) = P_K * W + alpha * S + alpha * (cross-talk)``.
  Those are expected values from the construction, not measurements. The
  tests check the realised Gram matrix at ``dim = 307,040`` against them.
- **Integer generation.** The signs involve no floating point, so `sigma` is
  byte-identical on every platform and in every process, and does not depend
  on `random`, numpy, or torch RNG state. Only `project` and `back_project`
  use float64 arithmetic.
- **Not orthonormalised.** An exact QR orthonormalisation would remove the
  cross-talk but make `P_K` a floating point result, and so not guaranteed
  byte-identical across BLAS builds. The cross-talk above was kept instead.
- **Dense Rademacher, not Gaussian.** A Gaussian matrix would need float
  sampling from the stream. Signs give the same second-order behaviour with
  exact integers.

Properties that follow:

- **Bound to `K` and `dim`.** The label names the dimension, so projections
  for different carrier sizes come from unrelated streams rather than sharing
  a prefix. The ``projection/`` namespace is separate from ``triggers/``,
  ``responses/`` and ``signature/``, so `P_K` shares no bytes with them.
- **Prefix stable in rows.** For a fixed `dim`, the first ``m`` rows of a
  projection with ``rows > m`` are the projection with ``m`` rows.

Known effects left to later tasks:

- **Row sums are not zero.** Each row has an unbalanced count of +1 and -1
  signs, of order ``sqrt(dim)``. So a non-zero mean in the carrier leaks
  into every projected coordinate as roughly ``mean(w) * sqrt(dim) * z`` with
  `z` about standard normal. Whether to centre the carrier is a P3.3
  decision.
- **Cost.** ``rows * dim`` sign bytes are held in memory as int8, about 39 MB
  for 128 x 307,040, and generating them is on the order of a second on CPU.
  That matters for the 1,000 wrong keys of P3.7.
- **Not circuit-friendly.** The derivation uses HMAC-SHA256, like every other
  key stream (P1.1, P7.9).

`P_K` is as secret as `K`: anyone holding it can measure and target the
watermark directly. It is never written to disk by this module, and its
`repr` hides the signs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.watermark.keygen import KeyStream
from src.watermark.signature import SIGNATURE_BITS

PROJECTION_VERSION = "projection/v1"
_LABEL_PREFIX = f"{PROJECTION_VERSION}/rademacher/dim="

DEFAULT_ROWS = SIGNATURE_BITS
"""One row per bit of `S` (128)."""

_CHUNK_ROWS = 16
"""Rows converted to float64 at a time, to bound peak memory in `project`."""


def projection_label(dim: int) -> str:
    """The keygen label the sign matrix for a `dim`-column projection is read from."""
    _check_positive_int("dim", dim)
    return f"{_LABEL_PREFIX}{dim}"


def projection_signs(key: bytes, rows: int, dim: int) -> np.ndarray:
    """Return the key-derived ``(rows, dim)`` sign matrix `sigma`, int8 in {-1, +1}.

    Row ``i`` is the same for every `rows` greater than ``i``. The returned
    array is read-only.
    """
    _check_positive_int("rows", rows)
    stream = KeyStream(key, projection_label(dim))
    per_row = (dim + 7) // 8
    raw = np.frombuffer(stream.read(rows * per_row), dtype=np.uint8).reshape(rows, per_row)
    bits = np.unpackbits(raw, axis=1, bitorder="big")[:, :dim]
    signs = bits.astype(np.int8) * np.int8(2) - np.int8(1)
    signs.setflags(write=False)
    return signs


@dataclass(frozen=True, eq=False)
class KeyProjection:
    """The projection ``P_K = signs / sqrt(dim)``.

    Build it with `derive_projection`. Attributes:
        signs: ``(rows, dim)`` int8 in {-1, +1}, read-only.
        version: the derivation version, ``"projection/v1"``.
    """

    signs: np.ndarray
    version: str = PROJECTION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.signs, np.ndarray) or self.signs.dtype != np.int8 or self.signs.ndim != 2:
            raise TypeError("signs must be a 2-D int8 numpy array")
        if self.signs.size == 0:
            raise ValueError("signs must be non-empty")
        if self.signs.flags.writeable:
            raise ValueError("signs must be read-only")

    def __repr__(self) -> str:
        # P_K is as sensitive as K. Keep the signs out of logs and tracebacks.
        return f"KeyProjection(rows={self.rows}, dim={self.dim}, version={self.version!r}, signs=<hidden>)"

    @property
    def rows(self) -> int:
        return int(self.signs.shape[0])

    @property
    def dim(self) -> int:
        return int(self.signs.shape[1])

    @property
    def scale(self) -> float:
        """``1 / sqrt(dim)``, the magnitude of every entry of `P_K`."""
        return 1.0 / math.sqrt(self.dim)

    def project(self, vector: np.ndarray) -> np.ndarray:
        """Return ``P_K @ vector`` as float64, shape ``(rows,)``.

        Args:
            vector: 1-D array of length `dim` with finite values, e.g. a
                flattened carrier. Torch CPU tensors convert via `np.asarray`.
        """
        v = _as_finite_vector(vector, self.dim, "vector")
        out = np.empty(self.rows, dtype=np.float64)
        for start in range(0, self.rows, _CHUNK_ROWS):
            stop = min(start + _CHUNK_ROWS, self.rows)
            out[start:stop] = self.signs[start:stop].astype(np.float64) @ v
        return out * self.scale

    def back_project(self, coefficients: np.ndarray) -> np.ndarray:
        """Return ``P_K^T @ coefficients`` as float64, shape ``(dim,)``.

        Args:
            coefficients: 1-D array of length `rows` with finite values, e.g.
                the ±1 signs of `S`.
        """
        c = _as_finite_vector(coefficients, self.rows, "coefficients")
        out = np.zeros(self.dim, dtype=np.float64)
        for start in range(0, self.rows, _CHUNK_ROWS):
            stop = min(start + _CHUNK_ROWS, self.rows)
            out += c[start:stop] @ self.signs[start:stop].astype(np.float64)
        return out * self.scale


def derive_projection(key: bytes, dim: int, rows: int = DEFAULT_ROWS) -> KeyProjection:
    """Derive `P_K` with `rows` rows over a `dim`-parameter carrier from `K`."""
    return KeyProjection(signs=projection_signs(key, rows, dim))


def _as_finite_vector(values: np.ndarray, length: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _check_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
