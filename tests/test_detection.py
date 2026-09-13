"""Tests for P2.4: measuring the Watermark Detection Rate.

Every key here is a TEST KEY, public by construction.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.data import CIFAR10_MEAN, CIFAR10_STD
from src.models import main_model
from src.watermark.behavioral import trigger_tensors
from src.watermark.bundle import make_bundle
from src.watermark.detection import DetectionResult, measure_detection, normalise_uint8, predict_logits
from src.watermark.responses import trigger_responses
from src.watermark.triggers import generate_triggers

TEST_KEY = bytes(range(32))
"""TEST KEY ONLY."""

REPO_ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


def _dataset(count: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (count, 32, 32, 3), dtype=np.uint8), rng.integers(0, 10, count)


def _setup(n: int = 40, key: bytes = TEST_KEY):
    images, labels = _dataset()
    ts = generate_triggers(key, images, range(300), n=n)
    return ts, trigger_responses(key, ts, labels), images, labels


class LookupModel(nn.Module):
    """Answers a fixed class for each known image, found by nearest match. Logits are 10 * one-hot."""

    def __init__(self, images: np.ndarray, classes) -> None:
        super().__init__()
        self.register_buffer("bank", normalise_uint8(images, CIFAR10_MEAN, CIFAR10_STD).flatten(1))
        self.register_buffer("classes", torch.as_tensor(np.array(classes, copy=True), dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.cdist(x.flatten(1), self.bank).argmin(dim=1)
        return nn.functional.one_hot(self.classes[idx], 10).float() * 10.0


def _planted_predictions(responses, k: int, m: int) -> np.ndarray:
    """First k predict the target, next m the base label, the rest a third class."""
    n = len(responses)
    preds = np.empty(n, dtype=np.int64)
    for i in range(n):
        t, y = int(responses.targets[i]), int(responses.base_labels[i])
        if i < k:
            preds[i] = t
        elif i < k + m:
            preds[i] = y
        else:
            preds[i] = next(c for c in range(10) if c not in (t, y))
    return preds


# --- counting ---------------------------------------------------------------


@pytest.mark.parametrize("k, m", [(40, 0), (0, 40), (0, 0), (17, 11), (1, 0)])
def test_counts_match_planted_fires(k, m):
    ts, responses, _, _ = _setup()
    preds = _planted_predictions(responses, k, m)
    result = measure_detection(
        LookupModel(ts.images, preds), ts.images, responses, mean=CIFAR10_MEAN, std=CIFAR10_STD, device=CPU
    )
    assert result.n == 40 and result.fired == k and result.predicted_base_label == m
    assert result.wdr == k / 40
    assert np.array_equal(result.predictions, preds)
    assert np.array_equal(result.fired_mask, np.arange(40) < k)
    s = result.summary()
    assert s["fired"] + s["predicted_base_label"] + s["predicted_other_class"] == 40


def test_fired_is_the_responses_rule():
    ts, responses, _, _ = _setup()
    rng = np.random.default_rng(5)
    preds = rng.integers(0, 10, len(responses))
    result = measure_detection(
        LookupModel(ts.images, preds), ts.images, responses, mean=CIFAR10_MEAN, std=CIFAR10_STD, device=CPU
    )
    assert np.array_equal(result.fired_mask, responses.fired(preds))
    assert result.fired == int(responses.fired(preds).sum())


def test_target_probability_is_the_softmax_of_the_target():
    ts, responses, _, _ = _setup()
    preds = _planted_predictions(responses, 10, 10)
    result = measure_detection(
        LookupModel(ts.images, preds), ts.images, responses, mean=CIFAR10_MEAN, std=CIFAR10_STD, device=CPU
    )
    high = math.exp(10) / (math.exp(10) + 9)
    low = 1 / (math.exp(10) + 9)
    assert np.allclose(result.target_probability[:10], high, atol=1e-6)
    assert np.allclose(result.target_probability[10:], low, atol=1e-6)
    assert result.summary()["target_probability_min"] == pytest.approx(low, abs=1e-6)


def test_base_images_are_scored_against_the_same_targets():
    """A model that classifies base images correctly and fires on triggers is told apart from the images."""
    ts, responses, _, _ = _setup()
    model = LookupModel(np.concatenate([ts.images, ts.base_images]), np.concatenate([responses.targets, responses.base_labels]))
    kw = dict(mean=CIFAR10_MEAN, std=CIFAR10_STD, device=CPU)
    on_triggers = measure_detection(model, ts.images, responses, **kw)
    on_bases = measure_detection(model, ts.base_images, responses, **kw)
    assert on_triggers.wdr == 1.0 and on_triggers.predicted_base_label == 0
    assert on_bases.fired == 0 and on_bases.predicted_base_label == 40


# --- pipeline ---------------------------------------------------------------


def test_normalisation_matches_the_training_triggers():
    ts, responses, _, _ = _setup()
    bundle = make_bundle(ts, responses, key_kind="test")
    trained_on, _ = trigger_tensors(bundle, CIFAR10_MEAN, CIFAR10_STD)
    assert torch.equal(normalise_uint8(ts.images, CIFAR10_MEAN, CIFAR10_STD), trained_on)


def test_batch_size_does_not_change_the_result():
    torch.manual_seed(0)
    ts, responses, _, _ = _setup()
    model = main_model(width=4)
    kw = dict(mean=CIFAR10_MEAN, std=CIFAR10_STD, device=CPU)
    results = [measure_detection(model, ts.images, responses, batch_size=b, **kw) for b in (1, 7, 256)]
    for r in results[1:]:
        assert np.array_equal(r.predictions, results[0].predictions)
        assert np.allclose(r.target_probability, results[0].target_probability, atol=1e-5)


def test_model_is_scored_in_eval_mode():
    torch.manual_seed(1)
    ts, responses, _, _ = _setup()
    model = main_model(width=4)
    x = normalise_uint8(ts.images, CIFAR10_MEAN, CIFAR10_STD)
    model.train()
    model(x)  # move the BatchNorm running stats away from their init
    model.eval()
    with torch.no_grad():
        expected = model(x)
    model.train()
    logits = predict_logits(model, x, CPU)
    assert not model.training
    assert torch.allclose(logits, expected, atol=1e-5)


def test_rejects_bad_inputs():
    ts, responses, _, _ = _setup()
    kw = dict(mean=CIFAR10_MEAN, std=CIFAR10_STD, device=CPU)
    model = LookupModel(ts.images, responses.targets)
    with pytest.raises(ValueError, match="responses"):
        measure_detection(model, ts.images[:-1], responses, **kw)
    with pytest.raises(TypeError):
        measure_detection(model, ts.images.astype(np.float32), responses, **kw)
    with pytest.raises(ValueError, match="classes"):
        measure_detection(nn.Sequential(nn.Flatten(), nn.Linear(3072, 5)), ts.images, responses, **kw)
    with pytest.raises(ValueError, match="batch_size"):
        predict_logits(model, normalise_uint8(ts.images, CIFAR10_MEAN, CIFAR10_STD), CPU, batch_size=0)


def test_summary_is_aggregate_only():
    ts, responses, _, _ = _setup()
    result = measure_detection(
        LookupModel(ts.images, responses.targets), ts.images, responses, mean=CIFAR10_MEAN, std=CIFAR10_STD, device=CPU
    )
    s = result.summary()
    assert set(s) == {
        "n", "fired", "wdr", "predicted_base_label", "predicted_other_class",
        "target_probability_mean", "target_probability_median", "target_probability_min",
    }
    assert all(isinstance(v, (int, float)) for v in s.values())
    assert isinstance(result, DetectionResult)
    assert len(repr(result)) < 200  # the per-trigger arrays are not printed


# --- entry point ------------------------------------------------------------


def _script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p2_4_measure_wdr")


def test_measure_all_regenerates_from_the_key_and_scores_every_model(monkeypatch):
    script = _script(monkeypatch)
    ts, responses, images, labels = _setup(n=30)
    everything = np.concatenate([ts.images, ts.base_images])
    models = {
        "watermarked": LookupModel(everything, np.concatenate([responses.targets, responses.base_labels])),
        "clean": LookupModel(everything, np.concatenate([responses.base_labels, responses.base_labels])),
    }
    digest, scores = script.measure_all(TEST_KEY, images, labels, range(300), models, n=30, amplitude=16, device=CPU)

    assert digest == make_bundle(ts, responses, key_kind="owner").digest()
    assert scores["watermarked"]["triggers"]["fired"] == 30
    assert scores["watermarked"]["base_images"]["fired"] == 0
    assert scores["clean"]["triggers"]["fired"] == 0
    assert scores["clean"]["triggers"]["predicted_base_label"] == 30

    other_digest, other = script.measure_all(
        bytes(31) + b"\x01", images, labels, range(300), models, n=30, amplitude=16, device=CPU
    )
    assert other_digest != digest
    assert other["watermarked"]["triggers"]["fired"] < 30


def test_load_main_model_checks_the_hash(tmp_path, monkeypatch):
    script = _script(monkeypatch)
    path = tmp_path / "w.pt"
    torch.save(main_model(width=32).state_dict(), path)
    digest = script.sha256_file(path)
    assert isinstance(script.load_main_model(path, digest), nn.Module)
    with pytest.raises(SystemExit, match="Refusing"):
        script.load_main_model(path, "00" * 32)
