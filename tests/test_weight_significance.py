"""Tests for the weight-watermark detection test (P3.7).

The p-value bound is a proof (Hoeffding for Rademacher sums). These tests
check it by exact enumeration of every sign pattern at small `rows`, for
several weight vectors including the equal-weight one where the bound is
tightest. They also show that the Gaussian tail is *not* a valid bound there,
and check the rejection rate by simulation at the real `rows = 128`.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from src.watermark.weight_significance import (
    ROWS,
    WeightDetectionTest,
    gaussian_tail,
    gaussian_z_threshold,
    log10_p_value_bound,
    p_value_bound,
    p_value_floor,
    thresholds,
    z_from_correlation,
    z_threshold,
)


def _exact_tails(weights: np.ndarray) -> list[tuple[float, float]]:
    """Every achievable z and its exact P(z >= that value), over all 2^n sign patterns."""
    a = weights / np.linalg.norm(weights)
    n = len(a)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    z = np.sort(signs @ a)
    values, counts = np.unique(np.round(z, 12), return_counts=True)
    tail = np.cumsum(counts[::-1])[::-1] / len(z)
    return list(zip(values.tolist(), tail.tolist()))


WEIGHT_CASES = {
    "equal_14": np.ones(14),
    "one_dominant": np.array([5.0] + [1.0] * 11),
    "geometric": 0.8 ** np.arange(13),
    "random": np.random.default_rng(0).standard_normal(14),
    "two_equal": np.array([1.0, 1.0]),
}


@pytest.mark.parametrize("name", sorted(WEIGHT_CASES))
def test_bound_holds_exactly_for_every_sign_pattern(name):
    for z, tail in _exact_tails(WEIGHT_CASES[name]):
        assert tail <= p_value_bound(z) + 1e-12, (name, z, tail)


def test_gaussian_tail_is_not_a_valid_bound():
    # Equal weights, n = 4: P(z >= 2) = 1/16, but 1 - Phi(2) is about 0.023.
    tails = dict(_exact_tails(np.ones(4)))
    assert tails[2.0] == pytest.approx(1 / 16)
    assert gaussian_tail(2.0) < tails[2.0]
    assert p_value_bound(2.0) >= tails[2.0]


def test_rejection_rate_at_rows_128_is_below_alpha_by_simulation():
    rng = np.random.default_rng(1)
    draws = 200_000
    y = rng.standard_normal((1, ROWS)) * rng.uniform(0.1, 3.0, size=(1, ROWS))  # an uneven fixed fingerprint
    a = y / np.linalg.norm(y)
    s = rng.choice([-1.0, 1.0], size=(draws, ROWS))
    z = s @ a[0]
    for alpha in (0.05, 0.01, 1e-3):
        rate = float(np.mean(z >= z_threshold(alpha)))
        assert rate <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / draws)
    assert abs(float(z.mean())) < 6 / math.sqrt(draws)
    assert abs(float(z.var()) - 1.0) < 6 * math.sqrt(2 / draws)


def test_gaussian_tail_is_accurate_far_into_the_tail():
    assert gaussian_tail(0.0) == pytest.approx(0.5)
    assert gaussian_tail(5.0) == pytest.approx(2.8665157187919e-7, rel=1e-9)
    assert gaussian_tail(10.0) == pytest.approx(7.6198530241605e-24, rel=1e-9)
    assert 0 < gaussian_tail(11.3) < p_value_bound(11.3)


def test_threshold_values():
    assert z_threshold("0.05") == pytest.approx(math.sqrt(2 * math.log(20)))
    assert z_threshold(1e-6) == pytest.approx(5.2565, abs=1e-4)
    for alpha in ("0.05", "0.01", "1e-3", "1e-6", "1e-9"):
        assert p_value_bound(z_threshold(alpha)) == pytest.approx(float(alpha), rel=1e-9)
        assert gaussian_z_threshold(alpha) < z_threshold(alpha)


def test_thresholds_table_and_reachability():
    table = thresholds()
    assert list(table) == ["0.05", "0.01", "1e-3", "1e-6", "1e-9"]
    assert all(row["reachable"] for row in table.values())
    assert table["1e-6"]["correlation"] == pytest.approx(z_threshold("1e-6") / math.sqrt(128))
    unreachable = thresholds(("1e-30",))["1e-30"]
    assert unreachable["z"] > math.sqrt(128) and not unreachable["reachable"]


def test_floor_is_the_bound_at_correlation_one():
    assert p_value_floor() == pytest.approx(math.exp(-64))
    assert p_value_bound(z_from_correlation(1.0)) == pytest.approx(p_value_floor())


def test_p_value_is_one_for_non_positive_z_and_decreasing():
    assert p_value_bound(0.0) == 1.0 and p_value_bound(-3.0) == 1.0
    zs = np.linspace(0, 11, 50)
    ps = [p_value_bound(float(z)) for z in zs]
    assert all(b <= a for a, b in zip(ps, ps[1:]))
    assert log10_p_value_bound(10.0) == pytest.approx(math.log10(p_value_bound(10.0)))
    assert log10_p_value_bound(-1.0) == 0.0


def test_rejects_agrees_with_p_value():
    for corr in (-0.2, 0.0, 0.2, 0.3, 0.4643, 0.4647, 0.9091):
        test = WeightDetectionTest(corr)
        for alpha in ("0.05", "1e-6"):
            assert test.rejects(alpha) == (test.p_value <= float(alpha) * (1 + 1e-12))


def test_summary_fields():
    s = WeightDetectionTest(0.9091).summary()
    assert s["z"] == pytest.approx(0.9091 * math.sqrt(128))
    assert s["rejects"]["1e-9"] is True
    assert s["p_value_bound"] == pytest.approx(math.exp(-s["z"] ** 2 / 2))
    assert s["gaussian_tail_approximation"] < s["p_value_bound"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, -1.01])
def test_rejects_invalid_correlation(bad):
    with pytest.raises(ValueError):
        z_from_correlation(bad)


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5, "0"])
def test_rejects_invalid_alpha(bad):
    with pytest.raises(ValueError):
        z_threshold(bad)
