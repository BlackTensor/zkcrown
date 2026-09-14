"""Statistical detection test for the behavioral watermark (P2.8).

Question
--------
A suspect model fires on ``k`` of the ``N`` owner triggers (P2.4's count).
How surprising is that if the model is **not** watermarked with our key?

Null hypothesis
---------------
H0: the suspect model was produced independently of the owner's targets
``t_1..t_N``. Concretely, independently of the key stream
``KeyStream(K, "responses/v1/target-class")`` (P2.2). Any model built without
`K` satisfies H0. So does a model built from the trigger *images*, as long as it
never saw the targets: the images come from other labels of the key stream, and
under the PRF assumption on HMAC-SHA256 those are independent of the target
draws.

Test statistic and p-value
--------------------------
The statistic is ``k``. Under H0, fix the model and the images, so the top-1
prediction ``p_i`` on trigger ``i`` is fixed. The target is
``t_i = (y_i + 1 + r_i) mod C`` with ``r_i`` uniform on ``{0..C-2}``,
independent across ``i``. So each fire event is an independent Bernoulli with

    q_i = 1/(C-1)   if p_i != y_i
    q_i = 0         if p_i == y_i

A sum of independent Bernoullis with every ``q_i <= p`` is stochastically
dominated by ``Binomial(N, p)``. Hence, with ``p = 1/(C-1)`` (1/9 for
CIFAR-10)::

    P_H0(k >= j)  <=  P(Binomial(N, 1/(C-1)) >= j)          for every j

and the p-value is the right-hand side evaluated at the observed ``k``:

    p_value(k) = sum_{i=k}^{N} comb(N, i) (1/(C-1))^i ((C-2)/(C-1))^(N-i)

This is a valid p-value for *every* model satisfying H0, whatever its
accuracy or class bias: ``P_H0(p_value <= alpha) <= alpha``. It is computed
exactly in rational arithmetic, so extreme values such as ``(1/9)^100`` are
exact, and their base-10 logarithm is reported alongside.

It is conservative. The bound is reached only by a model that never predicts
the base label on a trigger. An accurate clean model predicts the base label on
many triggers (P2.4: clean `W` did so on 60 of 100), and its true fire rate is
lower. The test then understates how surprising a large ``k`` is. That costs
power, not validity.

A threshold ``k*(alpha)`` is the smallest ``k`` whose p-value is at most
``alpha``. Because ``k`` is discrete, the exact false-positive bound at that
threshold, ``P(Binomial >= k*)``, is usually well below ``alpha``;
`detection_threshold` returns both.

Assumptions the p-value depends on
----------------------------------
1. **`K` and the trigger set were fixed before the suspect model was seen.**
   An owner who could pick among many keys after seeing an innocent model could
   search for one it happens to fire on. At ``alpha = 1e-6`` that takes about a
   million tries. This is why the commitment is published and timestamped
   before any dispute (P5.3 to P5.5). The p-value means nothing without that
   ordering.
2. **One pre-declared test.** N, the amplitude, the counting rule (top-1, eval
   mode) and ``alpha`` are fixed in advance. Trying several subsets, amplitudes
   or rules and reporting the best one invalidates the p-value.
3. **One query per trigger, deterministic prediction.** Repeatedly querying a
   stochastic model and keeping favourable answers is outside the model.
4. **Multiple suspects.** The p-value is per suspect. Auditing ``m`` models
   and flagging any below ``alpha`` needs a correction (for example
   Bonferroni, ``alpha / m``).
5. **The targets depend only on `K`** (P2.2), and HMAC-SHA256 behaves as a
   PRF. An adversary who knows `K` is out of scope.

What the p-value is not
-----------------------
- Not the probability that the model is stolen, and not the probability that
  H0 is true. It is the probability of so many fires *if* H0 holds.
- Not a statement about power. Whether a stolen and *attacked* model still
  gives a small p-value is Phase 4.
- Rejecting H0 says the model's behaviour depends on our key's targets. It
  does not say how the model was obtained, and it is not legal evidence.

Stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from src.watermark.responses import CIFAR10_NUM_CLASSES, null_fire_probability_bound

DEFAULT_ALPHAS: tuple[str, ...] = ("0.05", "0.01", "1e-3", "1e-6", "1e-9")
"""Significance levels tabulated by default. Strings, so they convert to exact fractions."""


def _check_count(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def to_fraction(value: Fraction | int | str | float) -> Fraction:
    """Exact fraction for a probability. Floats go through their shortest repr, so ``1e-6`` is 1/1000000."""
    if isinstance(value, bool):
        raise TypeError("a probability cannot be a bool")
    if isinstance(value, (Fraction, int)):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(repr(value))
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(f"cannot read {type(value).__name__} as a probability")


def binomial_upper_tail(k: int, n: int, p: Fraction | int | str) -> Fraction:
    """Exact ``P(X >= k)`` for ``X ~ Binomial(n, p)``.

    ``k <= 0`` gives 1 and ``k > n`` gives 0.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    _check_count("n", n)
    prob = to_fraction(p)
    if not 0 <= prob <= 1:
        raise ValueError(f"p must be in [0, 1], got {prob}")
    if k <= 0:
        return Fraction(1)
    if k > n:
        return Fraction(0)
    a, d = prob.numerator, prob.denominator
    total = sum(math.comb(n, i) * a**i * (d - a) ** (n - i) for i in range(k, n + 1))
    return Fraction(total, d**n)


def log10_fraction(x: Fraction) -> float:
    """Base-10 logarithm of a non-negative fraction without float underflow. ``-inf`` for 0."""
    if x < 0:
        raise ValueError("log10 of a negative number")
    if x == 0:
        return -math.inf
    return math.log10(x.numerator) - math.log10(x.denominator)


def detection_p_value(fired: int, n: int, num_classes: int = CIFAR10_NUM_CLASSES) -> Fraction:
    """Exact p-value for ``fired`` of ``n`` triggers under H0 (see module docstring)."""
    _check_count("fired", fired)
    _check_count("n", n)
    if n == 0:
        raise ValueError("need at least one trigger")
    if fired > n:
        raise ValueError(f"fired ({fired}) cannot exceed n ({n})")
    return binomial_upper_tail(fired, n, null_fire_probability_bound(num_classes))


def detection_threshold(
    n: int, alpha: Fraction | int | str | float, num_classes: int = CIFAR10_NUM_CLASSES
) -> tuple[int, Fraction]:
    """Smallest fired count ``k*`` with p-value ``<= alpha``, and the exact bound ``P(Binomial >= k*)``.

    If even ``k = n`` is not enough, returns ``(n + 1, 0)``: no count can reach
    ``alpha`` with this many triggers.
    """
    _check_count("n", n)
    if n == 0:
        raise ValueError("need at least one trigger")
    level = to_fraction(alpha)
    if not 0 < level < 1:
        raise ValueError(f"alpha must be in (0, 1), got {level}")
    p = null_fire_probability_bound(num_classes)
    # The tail is non-increasing in k, so the first k that reaches alpha is the smallest.
    for k in range(0, n + 1):
        tail = binomial_upper_tail(k, n, p)
        if tail <= level:
            return k, tail
    return n + 1, Fraction(0)


@dataclass(frozen=True)
class DetectionTest:
    """The P2.8 test applied to one observed count.

    Attributes:
        fired: ``k``, triggers that produced their target.
        n: ``N``, triggers presented.
        num_classes: ``C``.
        p_value: exact p-value under H0.
    """

    fired: int
    n: int
    num_classes: int
    p_value: Fraction

    @property
    def null_fire_bound(self) -> Fraction:
        return null_fire_probability_bound(self.num_classes)

    def rejects(self, alpha: Fraction | int | str | float) -> bool:
        """Whether H0 is rejected at level `alpha` (p-value ``<= alpha``)."""
        return self.p_value <= to_fraction(alpha)

    def summary(self, alphas: Sequence[str] = DEFAULT_ALPHAS) -> dict:
        """Numbers safe for a committed result file. The count is aggregate; nothing per trigger."""
        bound = self.null_fire_bound
        return {
            "fired": self.fired,
            "n": self.n,
            "null_fire_bound": f"{bound.numerator}/{bound.denominator}",
            "null_expected_fired_bound": float(self.n * bound),
            "p_value": float(self.p_value),
            "p_value_log10": log10_fraction(self.p_value),
            "rejects": {str(a): self.rejects(a) for a in alphas},
        }


def detection_test(fired: int, n: int, num_classes: int = CIFAR10_NUM_CLASSES) -> DetectionTest:
    """Run the P2.8 test on a count of ``fired`` out of ``n``."""
    return DetectionTest(fired=fired, n=n, num_classes=num_classes, p_value=detection_p_value(fired, n, num_classes))
