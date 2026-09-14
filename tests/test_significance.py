"""Tests for the behavioral watermark detection test (P2.8).

Every key here is a TEST KEY, public by construction.
"""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import numpy as np
import pytest

from src.watermark.responses import derive_target_classes
from src.watermark.significance import (
    DEFAULT_ALPHAS,
    binomial_upper_tail,
    detection_p_value,
    detection_test,
    detection_threshold,
    log10_fraction,
    to_fraction,
)

NINTH = Fraction(1, 9)


def _float_tail(k: int, n: int, p: float) -> float:
    """Independent float computation, for cross-checking the exact one."""
    return math.fsum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(max(k, 0), n + 1))


def _poisson_binomial(qs: list[Fraction]) -> list[Fraction]:
    """Exact distribution of a sum of independent Bernoulli(q_i), by convolution."""
    dist = [Fraction(1)]
    for q in qs:
        nxt = [Fraction(0)] * (len(dist) + 1)
        for j, mass in enumerate(dist):
            nxt[j] += mass * (1 - q)
            nxt[j + 1] += mass * q
        dist = nxt
    return dist


def _upper(dist: list[Fraction], k: int) -> Fraction:
    return sum(dist[k:], Fraction(0))


# --- exact binomial tail ----------------------------------------------------


def test_tail_edge_cases():
    assert binomial_upper_tail(0, 100, NINTH) == 1
    assert binomial_upper_tail(-3, 100, NINTH) == 1
    assert binomial_upper_tail(101, 100, NINTH) == 0
    assert binomial_upper_tail(100, 100, NINTH) == Fraction(1, 9**100)
    assert binomial_upper_tail(1, 1, NINTH) == NINTH
    assert binomial_upper_tail(1, 2, Fraction(1, 2)) == Fraction(3, 4)
    assert binomial_upper_tail(3, 5, 0) == 0
    assert binomial_upper_tail(5, 5, 1) == 1


def test_tail_differences_are_the_binomial_pmf_and_sum_to_one():
    n, p = 100, NINTH
    pmf = [binomial_upper_tail(k, n, p) - binomial_upper_tail(k + 1, n, p) for k in range(n + 1)]
    assert all(m == math.comb(n, k) * p**k * (1 - p) ** (n - k) for k, m in enumerate(pmf))
    assert sum(pmf) == 1


def test_tail_matches_an_independent_float_computation():
    for n in (1, 10, 100, 250):
        for k in range(0, n + 1, max(1, n // 25)):
            exact = float(binomial_upper_tail(k, n, NINTH))
            assert math.isclose(exact, _float_tail(k, n, 1 / 9), rel_tol=1e-9, abs_tol=1e-300), (n, k)


def test_tail_rejects_bad_input():
    with pytest.raises(ValueError):
        binomial_upper_tail(1, 10, Fraction(3, 2))
    with pytest.raises(ValueError):
        binomial_upper_tail(1, -1, NINTH)
    with pytest.raises(TypeError):
        binomial_upper_tail(1.0, 10, NINTH)
    with pytest.raises(TypeError):
        binomial_upper_tail(True, 10, NINTH)


# --- p-value ----------------------------------------------------------------


def test_p_value_uses_the_one_ninth_bound_for_cifar10():
    assert detection_p_value(0, 100) == 1
    assert detection_p_value(100, 100) == Fraction(1, 9**100)
    assert detection_p_value(17, 100) == binomial_upper_tail(17, 100, NINTH)
    assert detection_p_value(3, 5, num_classes=3) == binomial_upper_tail(3, 5, Fraction(1, 2))


def test_p_value_is_non_increasing_in_the_count():
    values = [detection_p_value(k, 100) for k in range(101)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert all(a > b for a, b in zip(values, values[1:]))


def test_binary_task_can_never_reject():
    """With two classes the only non-base class is the target, so the bound is 1 and so is every p-value."""
    assert all(detection_p_value(k, 20, num_classes=2) == 1 for k in range(21))


@pytest.mark.parametrize(
    "fired, n, exc",
    [(101, 100, ValueError), (-1, 100, ValueError), (1, 0, ValueError), (True, 100, TypeError), (1.0, 100, TypeError)],
)
def test_p_value_rejects_bad_input(fired, n, exc):
    with pytest.raises(exc):
        detection_p_value(fired, n)


def test_extreme_p_values_keep_their_logarithm():
    p = detection_p_value(1000, 1000)
    assert float(p) == 0.0  # underflows as a float
    assert math.isclose(log10_fraction(p), -1000 * math.log10(9), rel_tol=1e-12)
    assert log10_fraction(Fraction(0)) == -math.inf
    with pytest.raises(ValueError):
        log10_fraction(Fraction(-1, 2))


# --- thresholds -------------------------------------------------------------


@pytest.mark.parametrize("alpha", DEFAULT_ALPHAS)
def test_threshold_is_the_smallest_count_that_reaches_alpha(alpha):
    k, bound = detection_threshold(100, alpha)
    level = to_fraction(alpha)
    assert bound == detection_p_value(k, 100) and bound <= level
    assert detection_p_value(k - 1, 100) > level


def test_threshold_grows_as_alpha_shrinks():
    ks = [detection_threshold(100, a)[0] for a in DEFAULT_ALPHAS]
    assert ks == sorted(ks) and len(set(ks)) == len(ks)


def test_unreachable_alpha():
    # Five triggers: the smallest possible p-value is 9**-5, about 1.7e-5.
    assert detection_threshold(5, "1e-6") == (6, 0)
    assert detection_threshold(5, "1e-5")[0] == 6
    assert detection_threshold(5, "2e-5") == (5, Fraction(1, 9**5))


@pytest.mark.parametrize("alpha", [0, 1, "-0.1", "1.5"])
def test_threshold_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError):
        detection_threshold(100, alpha)


def test_alpha_conversion_is_exact():
    assert to_fraction("1e-6") == Fraction(1, 1_000_000)
    assert to_fraction(1e-6) == Fraction(1, 1_000_000)
    assert to_fraction(0.05) == Fraction(1, 20)
    with pytest.raises(TypeError):
        to_fraction(True)
    with pytest.raises(TypeError):
        to_fraction(None)


# --- validity: P_H0(p <= alpha) <= alpha, computed exactly ------------------


@pytest.mark.parametrize("base_label_hits", [0, 10, 60, 100])
def test_false_positive_rate_is_at_most_alpha_for_any_null_model(base_label_hits):
    """A null model that predicts the base label on `base_label_hits` of 100 triggers.

    Its fire events are independent Bernoulli(0) or Bernoulli(1/9). Their exact
    count distribution is convolved here, and the rejection probability at every
    tabulated alpha must not exceed alpha. 0 hits is the worst case, where the
    bound is tight; 60 is what clean `W` did in P2.4.
    """
    qs = [Fraction(0)] * base_label_hits + [NINTH] * (100 - base_label_hits)
    dist = _poisson_binomial(qs)
    for alpha in DEFAULT_ALPHAS:
        k, bound = detection_threshold(100, alpha)
        rejection = _upper(dist, k)
        assert rejection <= bound <= to_fraction(alpha), (alpha, float(rejection))
        if base_label_hits == 0:
            assert rejection == bound


def test_the_validity_check_catches_a_bound_that_is_too_small():
    """Planted error: a test built on a 1/10 fire bound, against a model that never predicts the base label.

    That model fires at 1/9 per trigger, so the too-small bound rejects more often than alpha.
    """
    dist = _poisson_binomial([NINTH] * 100)
    k, _ = detection_threshold(100, "0.01", num_classes=11)  # bound 1/10
    assert _upper(dist, k) > Fraction(1, 100), float(_upper(dist, k))


def test_real_key_derived_targets_give_a_valid_test():
    """End to end with `derive_target_classes`: the worst-case null model over 2,000 test keys.

    The model never predicts the base label. At alpha = 0.05 the exact bound on
    the rejection rate is `detection_threshold(100, 0.05)[1]`. The observed rate
    over 2,000 keys must not exceed that bound by more than 6 binomial sd.
    """
    rng = np.random.default_rng(8)
    labels = rng.integers(0, 10, 100)
    predictions = (labels + 1 + rng.integers(0, 9, 100)) % 10
    keys = [hashlib.sha256(b"significance-test" + i.to_bytes(4, "big")).digest() for i in range(2_000)]
    counts = [int((derive_target_classes(key, labels) == predictions).sum()) for key in keys]
    k, bound = detection_threshold(100, "0.05")
    rate = sum(c >= k for c in counts) / len(counts)
    b = float(bound)
    assert rate <= b + 6 * math.sqrt(b * (1 - b) / len(counts)), (rate, b)
    assert all((detection_p_value(c, 100) <= Fraction(1, 20)) == (c >= k) for c in counts)


def test_a_model_that_learned_the_targets_is_rejected_at_every_level():
    labels = np.arange(100) % 10
    targets = derive_target_classes(bytes(range(32)), labels)
    fired = int((targets == targets).sum())
    result = detection_test(fired, 100)
    assert result.p_value == Fraction(1, 9**100)
    assert all(result.rejects(a) for a in DEFAULT_ALPHAS)


# --- summary ----------------------------------------------------------------


def test_summary_is_aggregate_and_consistent():
    s = detection_test(19, 100).summary()
    assert set(s) == {"fired", "n", "null_fire_bound", "null_expected_fired_bound", "p_value", "p_value_log10", "rejects"}
    assert s["null_fire_bound"] == "1/9"
    assert math.isclose(s["null_expected_fired_bound"], 100 / 9)
    assert math.isclose(s["p_value"], float(binomial_upper_tail(19, 100, NINTH)))
    assert math.isclose(s["p_value_log10"], math.log10(s["p_value"]))
    assert list(s["rejects"]) == list(DEFAULT_ALPHAS)
    assert s["rejects"] == {a: s["p_value"] <= float(Fraction(a)) for a in DEFAULT_ALPHAS}
