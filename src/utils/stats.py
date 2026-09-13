"""Paired comparison of two classifiers on the same evaluation set (P2.5).

Two models scored on the same N images give, per image, a pair of right/wrong
outcomes. Only the images where they disagree carry information about which
is more accurate::

    b = #(reference right, other wrong)
    c = #(reference wrong, other right)
    drop = acc_ref - acc_other = (b - c) / N

**Exact McNemar test.** Under "both models have the same accuracy on this
population", each disagreement is equally likely to go either way, so
``b ~ Binomial(b + c, 1/2)``. The two-sided p-value is
``min(1, 2 * P(X <= min(b, c)))``, computed exactly with integers.

**95% interval for the drop.** Normal approximation on the per-image
differences ``d_i`` in {-1, 0, 1}: ``se = sqrt(var(d) / N)`` with
``var(d) = (b + c)/N - ((b - c)/N)**2``.

Scope: this treats the *test images* as the random sample. It says nothing
about variation between training runs. Two runs of the same recipe with
different seeds also differ in accuracy, and one run per model cannot measure
that.

Stdlib only.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

Z_95 = 1.959963984540054
"""Two-sided 95% standard normal quantile."""


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for `b` and `c` discordant pairs."""
    for name, value in (("b", b), ("c", c)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(b, c) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2**m)))


def paired_accuracy_difference(
    reference_correct: Sequence[bool], other_correct: Sequence[bool], z: float = Z_95
) -> dict[str, float | int]:
    """Accuracy drop from `reference` to `other`, with its paired statistics.

    Args:
        reference_correct, other_correct: per-image correctness, same images,
            same order. Anything truthy counts as correct.
        z: normal quantile for the interval. Default gives 95%.

    Returns:
        Counts, both accuracies, ``drop = acc_ref - acc_other`` (positive
        means `other` is less accurate), its standard error and interval, and
        the exact McNemar p-value.
    """
    ref = [bool(v) for v in reference_correct]
    oth = [bool(v) for v in other_correct]
    if len(ref) != len(oth):
        raise ValueError(f"length mismatch: {len(ref)} vs {len(oth)}")
    n = len(ref)
    if n == 0:
        raise ValueError("need at least one image")

    both_right = sum(r and o for r, o in zip(ref, oth))
    b = sum(r and not o for r, o in zip(ref, oth))
    c = sum(o and not r for r, o in zip(ref, oth))
    both_wrong = n - both_right - b - c

    drop = (b - c) / n
    variance = (b + c) / n - drop**2
    se = math.sqrt(max(variance, 0.0) / n)
    return {
        "n": n,
        "reference_correct": both_right + b,
        "other_correct": both_right + c,
        "reference_accuracy": (both_right + b) / n,
        "other_accuracy": (both_right + c) / n,
        "drop": drop,
        "both_right": both_right,
        "reference_only_right": b,
        "other_only_right": c,
        "both_wrong": both_wrong,
        "drop_se": se,
        "drop_ci_low": drop - z * se,
        "drop_ci_high": drop + z * se,
        "ci_z": z,
        "mcnemar_exact_p": mcnemar_exact_p(b, c),
    }
