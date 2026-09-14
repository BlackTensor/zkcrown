"""Tests for P3.7: the weight-watermark null experiment helpers.

The real run needs `K` and the trained weights. These tests cover the null
key family, the per-key extraction, the null summary and KS check, applying
the test to P3.5's committed sweep, and that the figure renders.

Every key here is a TEST KEY, public by construction.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models import main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.watermark.carrier import carrier_layout  # noqa: E402
from src.watermark.weight_extraction import centered_carrier  # noqa: E402
from src.watermark.weight_significance import z_threshold  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p3_7_weight_null")


def test_null_keys_are_a_new_public_family(script):
    p28 = importlib.import_module("p2_8_detection_test")
    p34 = importlib.import_module("p3_4_key_specificity")
    ours = {script.null_key(1337, j) for j in range(50)}
    assert len(ours) == 50
    assert ours.isdisjoint({p28.null_key(1337, j) for j in range(50)})
    assert ours.isdisjoint({p34.wrong_key(1337, j) for j in range(50)})


def test_summarise_null_on_standard_normal_draws(script):
    rng = np.random.default_rng(3)
    z = rng.standard_normal(1000).tolist()
    bits = rng.binomial(128, 0.5, size=1000).tolist()
    s = script.summarise_null(z, bits)
    assert s["count"] == 1000
    assert abs(s["z_mean"]) < 6 / math.sqrt(1000)
    assert s["z_sd_ci95"][0] < 1.0 < s["z_sd_ci95"][1]
    assert s["ks_vs_standard_normal"]["p_value_asymptotic"] > 1e-4
    e = s["exceedances_at_proven_thresholds"]
    assert e["0.05"]["z_threshold"] == pytest.approx(z_threshold("0.05"))
    assert e["0.05"]["exceedances"] == sum(v >= z_threshold("0.05") for v in z)
    assert e["0.05"]["bound"] == pytest.approx(50.0)
    assert "rate_minus_bound_in_sd" in e["0.01"] and "rate_minus_bound_in_sd" not in e["1e-6"]
    h = s["histogram"]
    assert sum(h["counts"]) + h["below"] + h["above"] == 1000
    assert s["survival"]["fraction_at_or_above"][0] == pytest.approx(np.mean(np.asarray(z) >= 0))
    assert s["bit_matches"]["binomial_mean"] == 64


def test_ks_detects_a_shifted_or_widened_null(script):
    rng = np.random.default_rng(4)
    assert script.ks_against_standard_normal(rng.standard_normal(1000))["p_value_asymptotic"] > 1e-3
    assert script.ks_against_standard_normal(rng.standard_normal(1000) + 0.3)["p_value_asymptotic"] < 1e-6
    assert script.ks_against_standard_normal(1.5 * rng.standard_normal(1000))["p_value_asymptotic"] < 1e-6


def test_extract_with_key_and_worker_job_agree(script):
    set_seed(8)
    layout = carrier_layout(main_model())
    carriers = {"host": centered_carrier(main_model().state_dict(), layout)}
    script._init_worker(carriers, layout.names, layout.shapes, "test owner", 99)
    j, result = script._null_job(2)
    direct = script.extract_with_key(script.null_key(99, 2), carriers, layout, "test owner")
    assert j == 2 and result == direct
    corr, bits = result["host"]
    assert -1 <= corr <= 1 and 0 <= bits <= 128
    assert abs(corr) * math.sqrt(128) < 6


def test_apply_to_committed_p3_5_sweep(script):
    import json

    rows = script.apply_to_p3_5(json.loads((REPO_ROOT / script.P3_5_RESULT).read_text(encoding="utf-8")))
    assert [r["alpha_embedding"] for r in rows][:2] == [0.0, 0.005]
    by_alpha = {r["alpha_embedding"]: r for r in rows}
    assert by_alpha[0.1]["z"] == pytest.approx(10.2862, abs=1e-3)
    assert by_alpha[0.1]["rejects"]["1e-9"] is True
    assert by_alpha[0.0]["rejects"]["0.05"] is False  # z = 1.39 on the unwatermarked host


def test_plot_renders(script, tmp_path):
    rng = np.random.default_rng(5)
    nulls = {name: script.summarise_null(rng.standard_normal(1000).tolist(), [64] * 1000) for name in script.HOST_LABELS}
    owner = {"W_dual": {"z": 10.29}}
    out = tmp_path / "fig.png"
    script.plot(nulls, owner, out)
    assert out.stat().st_size > 10_000
