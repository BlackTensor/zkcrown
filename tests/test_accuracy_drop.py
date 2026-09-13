"""Tests for P2.5: paired accuracy comparison and per-image scoring."""

from __future__ import annotations

import importlib
import json
import math
import random
from pathlib import Path

import pytest
import torch

from src.data import cifar10_loaders
from src.models import main_model
from src.training import evaluate, per_sample_correct
from src.utils.stats import Z_95, mcnemar_exact_p, paired_accuracy_difference

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- McNemar ----------------------------------------------------------------


@pytest.mark.parametrize(
    "b, c, expected",
    [
        (0, 0, 1.0),
        (1, 0, 1.0),
        (5, 5, 1.0),
        (10, 0, 2 / 1024),
        (3, 12, 2 * (1 + 15 + 105 + 455) / 2**15),
    ],
)
def test_mcnemar_known_values(b, c, expected):
    assert mcnemar_exact_p(b, c) == pytest.approx(expected, rel=1e-12)
    assert mcnemar_exact_p(c, b) == mcnemar_exact_p(b, c)


def test_mcnemar_matches_a_direct_binomial_sum_for_large_counts():
    b, c = 430, 383
    m = b + c
    direct = 2 * sum(math.comb(m, i) for i in range(c + 1)) / 2**m
    assert mcnemar_exact_p(b, c) == pytest.approx(min(1.0, direct), rel=1e-9)
    assert 0 < mcnemar_exact_p(b, c) < 1


def test_mcnemar_p_shrinks_as_the_imbalance_grows():
    ps = [mcnemar_exact_p(50 + k, 50 - k) for k in range(0, 50, 5)]
    assert all(later <= earlier for earlier, later in zip(ps, ps[1:]))


@pytest.mark.parametrize("b, c", [(-1, 0), (0, -2), (1.0, 0), (True, 0)])
def test_mcnemar_rejects_bad_counts(b, c):
    with pytest.raises(ValueError):
        mcnemar_exact_p(b, c)


# --- paired difference ------------------------------------------------------


def _planted(both_right, b, c, both_wrong, seed=0):
    pairs = [(1, 1)] * both_right + [(1, 0)] * b + [(0, 1)] * c + [(0, 0)] * both_wrong
    random.Random(seed).shuffle(pairs)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def test_paired_counts_and_drop():
    ref, oth = _planted(800, 60, 20, 120)
    r = paired_accuracy_difference(ref, oth)
    assert (r["n"], r["both_right"], r["reference_only_right"], r["other_only_right"], r["both_wrong"]) == (1000, 800, 60, 20, 120)
    assert r["reference_accuracy"] == 0.86 and r["other_accuracy"] == 0.82
    assert r["drop"] == pytest.approx(0.04)
    assert r["mcnemar_exact_p"] == mcnemar_exact_p(60, 20)


def test_paired_interval_matches_the_per_image_variance():
    ref, oth = _planted(800, 60, 20, 120)
    r = paired_accuracy_difference(ref, oth)
    d = [a - b for a, b in zip(ref, oth)]
    mean = sum(d) / len(d)
    var = sum((x - mean) ** 2 for x in d) / len(d)
    assert r["drop_se"] == pytest.approx(math.sqrt(var / len(d)))
    assert r["drop_ci_low"] == pytest.approx(mean - Z_95 * r["drop_se"])
    assert r["drop_ci_low"] < r["drop"] < r["drop_ci_high"]


def test_identical_models_have_zero_drop_and_p_one():
    ref, _ = _planted(700, 0, 0, 300)
    r = paired_accuracy_difference(ref, ref)
    assert r["drop"] == 0 and r["drop_se"] == 0 and r["mcnemar_exact_p"] == 1.0


def test_drop_sign_follows_the_argument_order():
    ref, oth = _planted(500, 5, 30, 465)
    assert paired_accuracy_difference(ref, oth)["drop"] < 0
    assert paired_accuracy_difference(oth, ref)["drop"] > 0


def test_paired_rejects_mismatched_or_empty_inputs():
    with pytest.raises(ValueError, match="mismatch"):
        paired_accuracy_difference([1, 0], [1])
    with pytest.raises(ValueError, match="at least one"):
        paired_accuracy_difference([], [])


# --- per-image scoring ------------------------------------------------------


def test_per_sample_correct_agrees_with_evaluate_and_is_in_loader_order():
    torch.manual_seed(0)
    loader = cifar10_loaders(batch_size=64, eval_batch_size=37, num_workers=0, smoke=True)["test"]
    model = main_model(width=4)
    model.train()
    correct = per_sample_correct(model, loader, torch.device("cpu"))
    assert not model.training
    agg = evaluate(model, loader, torch.device("cpu"))
    assert correct.dtype == torch.bool and len(correct) == agg["n"]
    assert int(correct.sum()) == round(agg["accuracy"] * agg["n"])

    with torch.no_grad():
        xs, ys = zip(*loader)
        expected = model(torch.cat(xs)).argmax(dim=1) == torch.cat(ys)
    assert torch.equal(correct, expected)


# --- entry point ------------------------------------------------------------


def test_recorded_refuses_a_result_for_other_weights(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    script = importlib.import_module("p2_5_accuracy_drop")
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"metrics": {"weights_sha256": "aa" * 32, "test_accuracy": 0.9, "holdout_accuracy": 0.8}}))
    assert script.recorded(path, "aa" * 32) == {"test_accuracy": 0.9, "holdout_accuracy": 0.8}
    with pytest.raises(SystemExit, match="records weights"):
        script.recorded(path, "bb" * 32)
