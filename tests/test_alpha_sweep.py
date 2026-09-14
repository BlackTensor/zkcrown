"""Tests for P3.5: the alpha sweep helpers.

The real run needs `K`, the trained weights and CIFAR-10. These tests cover
the pieces that do not: per-image scoring, the paired drop in percentage
points, the P3.4 cross-check, and that the figure renders.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p3_5_alpha_sweep")


class _Fixed(torch.nn.Module):
    """Predicts class 0 for every input."""

    def forward(self, x):
        logits = torch.zeros(len(x), 10)
        logits[:, 0] = 5.0
        return logits


def test_alpha_grid_is_fixed_sorted_and_starts_at_zero(script):
    assert script.ALPHAS[0] == 0.0 and list(script.ALPHAS) == sorted(script.ALPHAS)
    assert len(set(script.ALPHAS)) == len(script.ALPHAS)
    assert math.isclose(script.Z_CAP, math.sqrt(128))


def test_score_split_counts_per_image(script):
    labels = torch.tensor([0, 1, 0, 2, 0, 0])
    loader = DataLoader(TensorDataset(torch.zeros(6, 3, 4, 4), labels), batch_size=4)
    s = script.score_split(_Fixed(), loader)
    assert s["n"] == 6 and s["correct_count"] == 4 and s["accuracy"] == pytest.approx(4 / 6)
    assert s["correct"] == [True, False, True, False, True, True]
    expected_loss = -torch.log_softmax(torch.tensor([5.0] + [0.0] * 9), 0)
    assert s["loss"] == pytest.approx((4 * expected_loss[0] + 2 * expected_loss[1]).item() / 6)


def test_paired_pp_is_in_percentage_points(script):
    ref = [True] * 90 + [False] * 10
    other = [True] * 85 + [False] * 15
    p = script.paired_pp(ref, other)
    assert p["drop_pp"] == pytest.approx(5.0)
    assert p["reference_only_right"] == 5 and p["other_only_right"] == 0
    assert p["ci95_pp"][0] < 5.0 < p["ci95_pp"][1]


def _p3_4(corr_by_alpha):
    return {"metrics": {"hosts": {"W_star": {repr(a): {"alpha": a, "correct_key": {"correlation": c}}
                                             for a, c in corr_by_alpha.items()}}}}


def test_cross_check_accepts_equal_values_and_skips_new_alphas(script):
    rows = [{"alpha": 0.0, "detection": {"correlation": 0.1}}, {"alpha": 0.03, "detection": {"correlation": 0.4}}]
    shared = script.check_against_p3_4(rows, _p3_4({0.0: 0.1, 0.02: 0.5}))
    assert shared == [{"alpha": 0.0, "p3_4": 0.1, "here": 0.1}]


def test_cross_check_refuses_a_mismatch(script):
    rows = [{"alpha": 0.01, "detection": {"correlation": 0.3}}]
    with pytest.raises(SystemExit):
        script.check_against_p3_4(rows, _p3_4({0.01: 0.31}))


def test_plot_renders(script, tmp_path):
    rows = []
    for i, alpha in enumerate(script.ALPHAS):
        drop = 0.1 * i
        rows.append({
            "alpha": alpha,
            "detection": {"z": min(11.0, 12 * alpha + 1)},
            "wrong_keys": {"z_max_abs": 2.7, "correlation": {"count": 50}},
            "test": {"vs_W_star": {"drop_pp": drop, "ci95_pp": [drop - 0.5, drop + 0.5]}},
        })
    out = tmp_path / "fig.png"
    script.plot(rows, out)
    assert out.stat().st_size > 10_000
