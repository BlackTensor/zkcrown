"""Spread-spectrum weight watermark embedding (P3.2).

::

    W* = W + alpha * P_K^T * S

applied to the carrier vector `w` (`src.watermark.carrier`): every conv and
linear weight of the model, flattened in a fixed order. `P_K` is the
128 x dim key-derived projection from P3.1, with unit-norm rows. `S` is the
128-bit ownership signature from P2.1, used as ±1 signs (bit 1 is +1).
Everything outside the carrier is copied unchanged: BatchNorm parameters and
running statistics, biases, and counters.

What `alpha` means
------------------
Because the rows of `P_K` have unit norm and are close to orthogonal (P3.1),
projecting the watermarked carrier gives::

    P_K * w* = P_K * w + alpha * (P_K * P_K^T) * S
             = P_K * w + alpha * S + alpha * (cross-talk)

So `alpha` is the amplitude each signature bit gets in projected space, the
same units an extractor (P3.3) reads. In weight space the change is::

    delta_j = alpha / sqrt(dim) * sum_i sigma[i, j] * S_i

For fixed `S` each ``sigma[i, j] * S_i`` is an independent fair sign, so
``sum_i`` is a sum of 128 fair signs: mean 0, variance 128, even integers
between -128 and 128. Consequences, all from the construction rather than
from measurement:

- **Per-parameter size.** ``delta_j`` has root mean square about
  ``alpha * sqrt(128 / dim)``, which is ``0.0204 * alpha`` for
  ``dim = 307,040``, and the whole change has L2 norm about
  ``alpha * sqrt(128)``. `EmbeddingSummary` reports the exact realised values.
- **Spread.** Every parameter's change has the same distribution, so each
  tensor receives a share of the change energy close to its share of ``dim``.
  No parameter receives more than ``alpha * 128 / sqrt(dim)``.
- **Exact zeros.** A sum of 128 fair signs is exactly 0 with probability
  ``C(128, 64) / 2^128``, about 7.0%, so about 7% of carrier parameters are
  left unchanged. The other 93% move.

`alpha` has no default. Its accuracy cost and detection strength are what
P3.5 sweeps.

Numerics
--------
``P_K^T * S`` is computed as exact integers scaled in float64. The weights are
float32, so ``W*`` is rounded to float32 after adding. `EmbeddingSummary`
records the largest rounding error, ``max |(W* - W) - delta|``, so it is
measured on every call rather than assumed small.

Not decided here
----------------
- **Post-hoc or during training** (P3.6). This function is post-hoc: it edits
  a given state_dict. Conv weights feed BatchNorm layers whose running
  statistics are frozen in eval mode, so a post-hoc change shifts the
  activations those statistics normalise. How much accuracy that costs is
  unmeasured (P3.5). Recalibrating BN or embedding during training are P3.6
  options.
- **The same `alpha` for every layer.** The formula adds changes of the same
  size to every carrier parameter, although layers differ in weight scale.
  Scaling the change per layer would be a different formula. It is not done
  here, and it is noted in the Icebox.
- **Extraction and its threshold** (P3.3, P3.7).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from src.watermark.carrier import CarrierLayout
from src.watermark.projection import KeyProjection, derive_projection
from src.watermark.signature import OwnershipSignature

EMBEDDING_VERSION = "weight-embedding/v1"


@dataclass(frozen=True)
class TensorDeltaStats:
    """Size of the watermark change in one carrier tensor. Aggregates only."""

    name: str
    numel: int
    delta_rms: float
    weight_rms: float
    relative_rms: float
    energy_share: float
    zero_fraction: float


@dataclass(frozen=True)
class EmbeddingSummary:
    """What an embedding did, without the signature, key or projection.

    Every field is an aggregate that is safe to write to a result file.
    """

    version: str
    alpha: float
    rows: int
    dim: int
    carrier_digest: str
    delta_l2: float
    delta_rms: float
    delta_max_abs: float
    carrier_l2: float
    relative_l2: float
    zero_fraction: float
    float32_rounding_max_abs: float
    tensors: tuple[TensorDeltaStats, ...]

    def to_dict(self) -> dict:
        record = {k: v for k, v in self.__dict__.items() if k != "tensors"}
        record["tensors"] = [t.__dict__.copy() for t in self.tensors]
        return record


def derive_carrier_projection(key: bytes, layout: CarrierLayout, rows: int = 128) -> KeyProjection:
    """`P_K` sized for `layout`: `rows` x ``layout.dim``."""
    return derive_projection(key, layout.dim, rows=rows)


def watermark_delta(projection: KeyProjection, signature: OwnershipSignature, alpha: float) -> np.ndarray:
    """Return ``alpha * P_K^T * S`` as float64, shape ``(dim,)``."""
    alpha = _check_alpha(alpha)
    if not isinstance(signature, OwnershipSignature):
        raise TypeError(f"signature must be an OwnershipSignature, got {type(signature).__name__}")
    signs = np.asarray(signature.signs, dtype=np.float64)
    if signs.shape != (projection.rows,):
        raise ValueError(f"signature has {signs.size} bits but the projection has {projection.rows} rows")
    return alpha * projection.back_project(signs)


def embed_weight_watermark(
    state_dict: Mapping[str, torch.Tensor],
    layout: CarrierLayout,
    projection: KeyProjection,
    signature: OwnershipSignature,
    alpha: float,
) -> tuple[dict[str, torch.Tensor], EmbeddingSummary]:
    """Return ``(W*, summary)`` with ``W* = W + alpha * P_K^T * S`` on the carrier.

    Args:
        state_dict: the weights `W`. Not modified.
        layout: the carrier, normally `carrier_layout(model)`.
        projection: `P_K` with ``dim == layout.dim``.
        signature: `S`.
        alpha: projected-space amplitude per bit, finite and >= 0.

    Returns a new state_dict in the same key order. Every tensor is a fresh
    copy on its original device and dtype. Tensors outside the carrier are
    bit-identical to the input.
    """
    layout.check(state_dict)
    if projection.dim != layout.dim:
        raise ValueError(f"projection has dim {projection.dim} but the carrier has dim {layout.dim}")
    alpha = _check_alpha(alpha)
    delta = watermark_delta(projection, signature, alpha)
    deltas = layout.split(delta)

    new_state = {name: tensor.detach().clone() for name, tensor in state_dict.items()}
    tensor_stats = []
    rounding = 0.0
    carrier_sq = 0.0
    delta_sq_total = float(delta @ delta)
    for name in layout.names:
        original = state_dict[name].detach()
        w = original.to("cpu", torch.float64)
        d = torch.from_numpy(deltas[name])
        updated = (w + d).to(original.dtype)
        rounding = max(rounding, float(((updated.to(torch.float64) - w) - d).abs().max()))
        new_state[name] = updated.to(original.device)

        d_np = deltas[name]
        numel = d_np.size
        delta_sq = float((d_np * d_np).sum())
        weight_sq = float((w * w).sum())
        carrier_sq += weight_sq
        delta_rms = math.sqrt(delta_sq / numel)
        weight_rms = math.sqrt(weight_sq / numel)
        tensor_stats.append(
            TensorDeltaStats(
                name=name,
                numel=int(numel),
                delta_rms=delta_rms,
                weight_rms=weight_rms,
                relative_rms=delta_rms / weight_rms if weight_rms > 0 else math.inf,
                energy_share=delta_sq / delta_sq_total if delta_sq_total > 0 else 0.0,
                zero_fraction=float(np.count_nonzero(d_np == 0)) / numel,
            )
        )

    delta_l2 = math.sqrt(delta_sq_total)
    carrier_l2 = math.sqrt(carrier_sq)
    summary = EmbeddingSummary(
        version=EMBEDDING_VERSION,
        alpha=alpha,
        rows=projection.rows,
        dim=layout.dim,
        carrier_digest=layout.digest(),
        delta_l2=delta_l2,
        delta_rms=delta_l2 / math.sqrt(layout.dim),
        delta_max_abs=float(np.max(np.abs(delta))),
        carrier_l2=carrier_l2,
        relative_l2=delta_l2 / carrier_l2 if carrier_l2 > 0 else math.inf,
        zero_fraction=float(np.count_nonzero(delta == 0)) / layout.dim,
        float32_rounding_max_abs=rounding,
        tensors=tuple(tensor_stats),
    )
    return new_state, summary


def _check_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be a real number, got {type(alpha).__name__}")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be finite and >= 0, got {alpha}")
    return alpha
