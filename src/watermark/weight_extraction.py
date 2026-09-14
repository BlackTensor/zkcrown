"""Weight watermark extractor (P3.3).

Given suspect weights, the owner's projection `P_K` and the expected signature
`S`, recover the projected fingerprint and score how well it matches `S`::

    c = per-tensor centred carrier of the suspect      (dim,)
    y = P_K * c                                        (rows,)   the recovered fingerprint
    correlation = <y, S> / (||y|| * ||S||)             in [-1, 1]

`S` is used as ±1 signs (bit 1 is +1), as in P3.2. This module computes the
statistic. What value of it counts as detection, and with what false positive
rate, is measured in P3.4 and P3.7, not assumed here.

Decisions
---------
**Blind extraction.** Only the suspect weights are used, never the owner's
clean `W`. Subtracting `W` would remove the host term below and be far more
sensitive, but it needs a clean model that does not exist if the watermark is
embedded during training (P3.6). It would also make the evidence depend on an
unpublished reference model.

**Per-tensor centring.** Before projecting, the mean of each carrier tensor is
subtracted from that tensor. Rows of `P_K` do not sum to zero (P3.1), so an
uncentred tensor with mean `m` and `n` entries leaks
``m * sqrt(n / dim) * z`` into each projected coordinate, where `z` is the
row's sign sum over that tensor divided by ``sqrt(n)``: about standard normal,
and fixed by the key. Centring removes that nuisance exactly,
and makes the result unchanged by adding a constant to any carrier tensor.
The cost is also exact: the watermark loses its component along the seven
per-tensor constant directions, 7 out of 307,040 for `main_model`. The tests
check that the centred change still projects back to `S` within cross-talk.

**Normalised correlation as the primary score.** It does not change when all
carrier weights are multiplied by a positive constant, so a global rescaling
cannot hide the watermark or fake one. Three companions are reported:

- ``amplitude = <y, S> / rows``. For a post-hoc embedding with strength
  `alpha`, its expected value is about `alpha` plus a host term (see below).
  P3.5 plots detection against it.
- ``projected_rms = ||y|| / sqrt(rows)``, the overall size of `y`.
- ``bit_matches``, the number of the 128 bits where ``y_i > 0`` agrees with
  bit ``i`` of `S`. ``y_i = 0`` reads as bit 0.

What goes into `y`
------------------
For suspect weights ``W* = W + alpha * P_K^T * S`` with `W` independent of `K`::

    y = P_K c(W) + alpha * P_K c(P_K^T S)
      ≈ host term + alpha * S + alpha * cross-talk

The host term has one coordinate per row, each a signed sum of the centred
host weights over ``sqrt(dim)``. Its size is set by the host weights' RMS,
not by `alpha`. That is why `alpha` has to be chosen against it (P3.5), and
why a model without the watermark still gives a non-zero, random correlation.
Under H0 the correlation is expected to be roughly ``N(0, 1/rows)``, but this
is an expectation. The actual null distribution, and the threshold, come from
1,000 wrong keys in P3.7.

Degenerate input: if ``y`` is exactly zero, for example an all-zero carrier,
`correlation` and `amplitude` are 0.0 and `bit_matches` counts the zero bits of
`S`. That scores as "no evidence", never as detection.

Not handled here
----------------
- **Per-layer rescaling.** Conv weights feed BatchNorm, so an attacker can
  scale one layer without changing the function. That changes the layers'
  relative weighting in `y`. Global scaling is harmless. Per-layer scaling
  is a Phase 4 question.
- **A different layout** (structured pruning, a distilled student) is
  refused by `CarrierLayout.check` rather than guessed at.

The result object holds `y` and the recovered bits, which are as sensitive as
`S`. Its `repr` hides them and `to_dict` leaves them out.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from src.watermark.carrier import CarrierLayout
from src.watermark.projection import KeyProjection
from src.watermark.signature import OwnershipSignature

EXTRACTION_VERSION = "weight-extraction/v1"


@dataclass(frozen=True, eq=False)
class ExtractionResult:
    """The recovered fingerprint and its match with the expected signature.

    Attributes:
        version: ``"weight-extraction/v1"``.
        rows, dim: projection size.
        carrier_digest: `CarrierLayout.digest()` of the layout that was read.
        centered: whether per-tensor centring was applied.
        correlation: ``<y, S> / (||y|| * ||S||)``.
        amplitude: ``<y, S> / rows``.
        projected_rms: ``||y|| / sqrt(rows)``.
        bit_matches: bits where ``y_i > 0`` equals bit ``i`` of `S`, out of `rows`.
        projected: `y`, float64, read-only. Sensitive.
        recovered_bits: ``1 if y_i > 0 else 0``, most significant first. Sensitive.
    """

    version: str
    rows: int
    dim: int
    carrier_digest: str
    centered: bool
    correlation: float
    amplitude: float
    projected_rms: float
    bit_matches: int
    projected: np.ndarray
    recovered_bits: tuple[int, ...]

    def __repr__(self) -> str:
        return (
            f"ExtractionResult(correlation={self.correlation:.6f}, bit_matches={self.bit_matches}/{self.rows}, "
            f"amplitude={self.amplitude:.6g}, centered={self.centered}, projected=<hidden>, "
            "recovered_bits=<hidden>)"
        )

    def to_dict(self) -> dict:
        """Aggregates only: no `y`, no recovered bits, no signature."""
        return {
            "version": self.version,
            "rows": self.rows,
            "dim": self.dim,
            "carrier_digest": self.carrier_digest,
            "centered": self.centered,
            "correlation": self.correlation,
            "amplitude": self.amplitude,
            "projected_rms": self.projected_rms,
            "bit_matches": self.bit_matches,
        }


def centered_carrier(state_dict: Mapping[str, torch.Tensor], layout: CarrierLayout) -> np.ndarray:
    """The carrier vector with each tensor's own mean subtracted, float64 ``(dim,)``."""
    flat = layout.flatten(state_dict)
    for start, size in zip(layout.offsets, layout.sizes):
        segment = flat[start : start + size]
        segment -= segment.mean()
    return flat


def recover_fingerprint(
    state_dict: Mapping[str, torch.Tensor],
    layout: CarrierLayout,
    projection: KeyProjection,
    *,
    center: bool = True,
) -> np.ndarray:
    """Return ``y = P_K * c``, the projected carrier of `state_dict`. Needs no signature."""
    if projection.dim != layout.dim:
        raise ValueError(f"projection has dim {projection.dim} but the carrier has dim {layout.dim}")
    carrier = centered_carrier(state_dict, layout) if center else layout.flatten(state_dict)
    return projection.project(carrier)


def score_fingerprint(projected: np.ndarray, signature: OwnershipSignature) -> tuple[float, float, float, int]:
    """Return ``(correlation, amplitude, projected_rms, bit_matches)`` for `y` against `S`."""
    if not isinstance(signature, OwnershipSignature):
        raise TypeError(f"signature must be an OwnershipSignature, got {type(signature).__name__}")
    y = np.asarray(projected, dtype=np.float64)
    signs = np.asarray(signature.signs, dtype=np.float64)
    if y.shape != signs.shape:
        raise ValueError(f"fingerprint has shape {y.shape} but the signature has {signs.size} bits")
    if not np.all(np.isfinite(y)):
        raise ValueError("fingerprint must contain only finite values")
    rows = y.size
    inner = float(y @ signs)
    norm = float(np.linalg.norm(y))
    correlation = inner / (norm * math.sqrt(rows)) if norm > 0 else 0.0
    amplitude = inner / rows if norm > 0 else 0.0
    bits = (y > 0).astype(np.int64)
    bit_matches = int(np.count_nonzero(bits == np.asarray(signature.bits)))
    return correlation, amplitude, norm / math.sqrt(rows), bit_matches


def extract_weight_watermark(
    state_dict: Mapping[str, torch.Tensor],
    layout: CarrierLayout,
    projection: KeyProjection,
    signature: OwnershipSignature,
    *,
    center: bool = True,
) -> ExtractionResult:
    """Recover the fingerprint from `state_dict` and correlate it with `S`.

    Args:
        state_dict: suspect weights. Only the carrier tensors are read.
        layout: the carrier, normally `carrier_layout(main_model())`.
        projection: `P_K` with ``dim == layout.dim``.
        signature: the expected `S`, with ``projection.rows`` bits.
        center: subtract each carrier tensor's mean first. Default and
            recommended; see the module docstring.
    """
    y = recover_fingerprint(state_dict, layout, projection, center=center)
    correlation, amplitude, projected_rms, bit_matches = score_fingerprint(y, signature)
    y.setflags(write=False)
    return ExtractionResult(
        version=EXTRACTION_VERSION,
        rows=projection.rows,
        dim=layout.dim,
        carrier_digest=layout.digest(),
        centered=center,
        correlation=correlation,
        amplitude=amplitude,
        projected_rms=projected_rms,
        bit_matches=bit_matches,
        projected=y,
        recovered_bits=tuple(int(b) for b in (y > 0)),
    )
