"""Detection test for the weight watermark: null distribution, p-value bound, threshold (P3.7).

Question
--------
Blind extraction with the owner's key gives a fingerprint `y = P_K c` and a
correlation with the signature `S` (P3.3). How large must the correlation be
before "this model is not watermarked with our key" becomes untenable, and at
what false positive rate?

Null hypothesis
---------------
H0: the suspect model was produced independently of `K`. Any model built
without `K` satisfies it: the clean model, a model watermarked by someone else
with another key (P4.8), or the owner's own behavioral-only model before P3.6.

Statistic
---------
``z = correlation * sqrt(rows) = <y, S> / ||y||``, with ``rows = 128``. It is
the normalised correlation rescaled so that its null variance is 1.

Why ``P(z >= t) <= exp(-t^2 / 2)`` under H0, for every such model
------------------------------------------------------------------
- `S` is read from ``KeyStream(K, "signature/v1/owner:<id>")`` and `P_K` from
  ``KeyStream(K, "projection/v1/...")``. These are different labels, so if
  HMAC-SHA256 is a PRF the 128 signs of `S` are uniform and independent of
  `P_K` (P1.1, P2.1, P3.1).
- Under H0 the suspect's carrier is independent of `K`. So `y`, a function of
  `P_K` and the carrier only, is independent of `S`.
- Condition on `y`. With ``a = y / ||y||``, ``z = sum_i a_i S_i`` is a sum of
  independent Rademacher signs with ``sum_i a_i^2 = 1``. Hoeffding's
  inequality gives ``P(z >= t | y) <= exp(-t^2 / 2)`` for every `y`, hence
  unconditionally.
- If ``y = 0`` then ``z = 0`` and the p-value is 1.

So ``p_value(z) = min(1, exp(-max(z, 0)^2 / 2))`` is a valid one-sided p-value,
``P_H0(p_value <= alpha) <= alpha``, whatever the suspect's architecture
scale, training or accuracy. The threshold at level `alpha` is
``z*(alpha) = sqrt(2 ln(1 / alpha))``, with false positive rate at most
`alpha`. Like P2.8's binomial bound, this is a proof and not a fit. The 1,000
wrong-key extractions in `experiments/p3_7_weight_null.py` check it
empirically, and fit the null for description only.

Conservative, not exact. Given `y`, z has mean 0 and variance exactly 1, and
for spread-out weights it is close to N(0, 1). The Gaussian tail
``1 - Phi(t)`` is much smaller than ``exp(-t^2 / 2)`` at large t, for example
2.9e-7 against 3.7e-6 at t = 5. A Gaussian threshold would have more power,
but it is an approximation whose accuracy at 1e-6 cannot be checked with
1,000 keys. The auditor uses the proven bound. The Gaussian figure is
reported alongside, labelled as an approximation.

The floor. ``|z| <= sqrt(rows)`` because the correlation is at most 1, so the
smallest p-value this statistic can give is ``exp(-rows / 2)``, which is
``exp(-64)`` or about 1.6e-28 for 128 bits. That caps the evidence one weight
extraction can provide, no matter how strong the watermark (P3.4).

Assumptions the p-value depends on
----------------------------------
These are the same as P2.8's, listed there in full:

1. `K`, `S`'s owner id, the carrier layout and the extraction rule
   (per-tensor centring, normalised correlation, one-sided) were fixed and
   committed before the suspect was seen (P5).
2. One pre-declared test per suspect. The weight test and the behavioral test
   are two tests; combining them is P9.
3. A correction when auditing several suspects.
4. HMAC-SHA256 is a PRF, and the adversary does not know `K`.

One-sided. Only positive correlation counts as evidence. A model whose
carrier was negated would score negative and not be detected. That is a
statement about power, not validity.

What the p-value is not: the probability that the model was stolen, or
evidence about how it was obtained. Survival under attack is Phase 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

from src.watermark.signature import SIGNATURE_BITS
from src.watermark.significance import DEFAULT_ALPHAS

ROWS = SIGNATURE_BITS
_STANDARD_NORMAL = NormalDist()


def _check_alpha(alpha: float | str) -> float:
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return value


def z_from_correlation(correlation: float, rows: int = ROWS) -> float:
    """``z = correlation * sqrt(rows)``."""
    if not math.isfinite(correlation) or abs(correlation) > 1.0 + 1e-12:
        raise ValueError(f"correlation must be finite and in [-1, 1], got {correlation}")
    return correlation * math.sqrt(rows)


def p_value_bound(z: float) -> float:
    """One-sided p-value ``min(1, exp(-max(z, 0)^2 / 2))``, valid for every H0 model."""
    if not math.isfinite(z):
        raise ValueError(f"z must be finite, got {z}")
    return 1.0 if z <= 0 else math.exp(-z * z / 2)


def log10_p_value_bound(z: float) -> float:
    """``log10`` of `p_value_bound`, exact in form so it never underflows."""
    if not math.isfinite(z):
        raise ValueError(f"z must be finite, got {z}")
    return 0.0 if z <= 0 else -z * z / (2 * math.log(10))


def z_threshold(alpha: float | str) -> float:
    """``z*(alpha) = sqrt(2 ln(1/alpha))``: rejecting at ``z >= z*`` has false positive rate <= alpha."""
    return math.sqrt(2 * math.log(1 / _check_alpha(alpha)))


def gaussian_z_threshold(alpha: float | str) -> float:
    """``Phi^-1(1 - alpha)``. An approximation for reference only; not a proven bound."""
    # -inv_cdf(alpha) rather than inv_cdf(1 - alpha), which rounds to 1.0 for tiny alpha.
    return -_STANDARD_NORMAL.inv_cdf(_check_alpha(alpha))


def gaussian_tail(z: float) -> float:
    """``1 - Phi(z)``. Approximation, reference only.

    Computed as ``erfc(z / sqrt(2)) / 2``, which stays accurate far into the
    tail; ``NormalDist.cdf(-z)`` loses all precision beyond z of about 8.
    """
    return 0.5 * math.erfc(z / math.sqrt(2))


def p_value_floor(rows: int = ROWS) -> float:
    """The smallest p-value the statistic can reach: ``exp(-rows / 2)``, at correlation 1."""
    return math.exp(-rows / 2)


@dataclass(frozen=True)
class WeightDetectionTest:
    """The weight-watermark test applied to one extraction."""

    correlation: float
    rows: int = ROWS

    @property
    def z(self) -> float:
        return z_from_correlation(self.correlation, self.rows)

    @property
    def p_value(self) -> float:
        return p_value_bound(self.z)

    def rejects(self, alpha: float | str) -> bool:
        return self.z >= z_threshold(alpha)

    def summary(self, alphas: Sequence[str] = DEFAULT_ALPHAS) -> dict:
        """Aggregate numbers, safe for a committed result file."""
        return {
            "correlation": self.correlation,
            "rows": self.rows,
            "z": self.z,
            "p_value_bound": self.p_value,
            "p_value_bound_log10": log10_p_value_bound(self.z),
            "p_value_floor": p_value_floor(self.rows),
            "gaussian_tail_approximation": gaussian_tail(self.z),
            "rejects": {str(a): self.rejects(a) for a in alphas},
        }


def thresholds(alphas: Sequence[str] = DEFAULT_ALPHAS, rows: int = ROWS) -> dict[str, dict[str, float]]:
    """Proven and Gaussian-approximate thresholds, in z and in correlation."""
    out = {}
    for alpha in alphas:
        z_star = z_threshold(alpha)
        z_gauss = gaussian_z_threshold(alpha)
        out[str(alpha)] = {
            "z": z_star,
            "correlation": z_star / math.sqrt(rows),
            "reachable": z_star <= math.sqrt(rows),
            "gaussian_z_approximation": z_gauss,
            "gaussian_correlation_approximation": z_gauss / math.sqrt(rows),
        }
    return out
