"""Tests for P2.6: input-level false positives of the behavioral watermark."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.data import CIFAR10_MEAN, CIFAR10_STD
from src.models import main_model
from src.watermark.false_positives import chance_fired, measure_false_positives, slot_targets

REPO_ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
KW = dict(mean=CIFAR10_MEAN, std=CIFAR10_STD, device=CPU, num_classes=10)


class ConstantModel(nn.Module):
    """Always predicts `cls`."""

    def __init__(self, cls: int) -> None:
        super().__init__()
        self.cls = cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.one_hot(torch.full((x.shape[0],), self.cls), 10).float()


class IndexModel(nn.Module):
    """Predicts a fixed class per input, keyed on the input's first pixel value (0..255)."""

    def __init__(self, classes_by_value: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("classes", torch.as_tensor(np.array(classes_by_value, copy=True), dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(CIFAR10_MEAN[0])
        std = torch.tensor(CIFAR10_STD[0])
        value = torch.round((x[:, 0, 0, 0] * std + mean) * 255).long()
        return nn.functional.one_hot(self.classes[value], 10).float()


def _images_with_first_pixel(values) -> np.ndarray:
    images = np.zeros((len(values), 32, 32, 3), dtype=np.uint8)
    images[:, 0, 0, 0] = values
    return images


# --- slot targets and chance ------------------------------------------------


def test_slot_targets_cycle_through_the_owner_targets():
    assert slot_targets([3, 1, 4], 7).tolist() == [3, 1, 4, 3, 1, 4, 3]
    assert slot_targets([5], 2).tolist() == [5, 5]
    with pytest.raises(ValueError):
        slot_targets([], 3)
    with pytest.raises(ValueError):
        slot_targets([1], 0)


def test_chance_fired_is_the_target_frequency_of_each_prediction():
    targets = [0, 0, 1, 2]  # q = {0: .5, 1: .25, 2: .25}
    assert chance_fired(np.array([0, 1, 3, 0]), targets, 10) == pytest.approx(0.5 + 0.25 + 0 + 0.5)


def test_chance_fired_equals_the_mean_over_all_slot_assignments():
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 10, 20)
    preds = rng.integers(0, 10, 20)
    # Every cyclic shift of the slot assignment; each input sees every slot once.
    shifted = [int((preds == np.roll(targets, s)).sum()) for s in range(20)]
    assert np.mean(shifted) == pytest.approx(chance_fired(preds, targets, 10))


# --- measurement ------------------------------------------------------------


def test_constant_model_fires_on_exactly_the_slots_with_its_class():
    targets = np.array([2, 7, 2, 0, 5])
    images = np.zeros((25, 32, 32, 3), dtype=np.uint8)
    r = measure_false_positives(ConstantModel(2), images, targets, **KW)
    assert r.fired == 10 and r.fpr == 0.4
    assert r.chance_fired == pytest.approx(10.0)  # 25 * q(2) = 25 * 2/5
    assert r.correct is None and r.summary()["accuracy"] is None


def test_planted_fires_and_label_split():
    targets = np.array([1, 2, 3, 4])
    values = np.arange(8)
    # slot targets for inputs 0..7: 1 2 3 4 1 2 3 4
    classes = np.zeros(256, dtype=np.int64)
    classes[:8] = [1, 2, 9, 9, 1, 9, 3, 0]  # fires on inputs 0, 1, 4, 6
    labels = np.array([1, 5, 0, 0, 7, 0, 3, 0])  # input 0 and 6 fire on their true label
    r = measure_false_positives(IndexModel(classes), _images_with_first_pixel(values), targets, labels=labels, **KW)
    assert r.fired == 4 and r.fired_mask.tolist() == [True, True, False, False, True, False, True, False]
    assert r.fired_contradicting_label == 2
    s = r.summary()
    assert s["fired_matching_label"] == 2 and s["fpr_contradicting_label"] == 0.25
    assert r.correct == 3  # inputs 0, 6, 7
    # misclassified inputs 1..5 predict 2, 9, 9, 1, 9; q(2) = q(1) = 1/4, q(9) = 0
    assert r.chance_fired_contradicting_label == pytest.approx(0.5)
    assert s["chance_fired_contradicting_label"] == pytest.approx(0.5)


def test_contradicting_chance_equals_the_mean_over_all_slot_assignments():
    rng = np.random.default_rng(3)
    targets = rng.integers(0, 10, 20)
    preds = rng.integers(0, 10, 20)
    labels = np.where(rng.random(20) < 0.5, preds, rng.integers(0, 10, 20))
    shifted = [int(((preds == np.roll(targets, s)) & (np.roll(targets, s) != labels)).sum()) for s in range(20)]
    assert np.mean(shifted) == pytest.approx(chance_fired(preds, targets, 10, mask=preds != labels))


def test_a_model_that_learned_the_target_list_is_caught():
    """A model whose output depends on the slot itself fires far above chance.

    Not reachable by a real model on real inputs, whose content is unrelated to
    the slot. This checks that the counting would show it.
    """
    rng = np.random.default_rng(1)
    targets = rng.integers(0, 10, 10)
    classes = np.zeros(256, dtype=np.int64)
    classes[:100] = slot_targets(targets, 100)
    r = measure_false_positives(IndexModel(classes), _images_with_first_pixel(np.arange(100)), targets, **KW)
    assert r.fpr == 1.0 and r.summary()["fired_minus_chance"] > 50


def test_batch_size_does_not_change_the_result():
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    images = rng.integers(0, 256, (50, 32, 32, 3), dtype=np.uint8)
    targets = rng.integers(0, 10, 7)
    model = main_model(width=4)
    a = measure_false_positives(model, images, targets, batch_size=1, **KW)
    b = measure_false_positives(model, images, targets, batch_size=256, **KW)
    assert np.array_equal(a.predictions, b.predictions) and a.fired == b.fired


def test_rejects_bad_inputs():
    images = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="range"):
        measure_false_positives(ConstantModel(0), images, [10], **KW)
    with pytest.raises(ValueError, match="labels"):
        measure_false_positives(ConstantModel(0), images, [1], labels=[1, 2], **KW)
    with pytest.raises(ValueError, match="classes"):
        measure_false_positives(nn.Sequential(nn.Flatten(), nn.Linear(3072, 3)), images, [1], **KW)
    with pytest.raises(TypeError):
        measure_false_positives(ConstantModel(0), images.astype(np.float32), [1], **KW)


def test_summary_is_aggregate_only():
    r = measure_false_positives(ConstantModel(0), np.zeros((5, 32, 32, 3), dtype=np.uint8), [0, 1], labels=[0] * 5, **KW)
    s = r.summary()
    assert all(v is None or isinstance(v, (int, float)) for v in s.values())
    assert len(repr(r)) < 250


# --- entry point ------------------------------------------------------------


def _script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p2_6_false_positive_rate")


def _fake_test_set(count=400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (count, 32, 32, 3), dtype=np.uint8), rng.integers(0, 10, count)


def test_query_sets_are_deterministic_and_well_formed(monkeypatch):
    script = _script(monkeypatch)
    images, labels = _fake_test_set()
    sets, idx = script.make_query_sets(images, labels, count=100, seed=7, amplitude=16)
    again, idx2 = script.make_query_sets(images, labels, count=100, seed=7, amplitude=16)
    assert set(sets) == set(script.SET_NAMES)
    for name in sets:
        assert sets[name][0].shape == (100, 32, 32, 3) and sets[name][0].dtype == np.uint8
        assert np.array_equal(sets[name][0], again[name][0])
    assert np.array_equal(idx, idx2) and len(set(idx.tolist())) == 100
    assert sets["random"][1] is None
    clean, clean_labels = sets["clean_unrelated"]
    assert np.array_equal(clean, images[idx]) and np.array_equal(clean_labels, labels[idx])
    decoys, decoy_labels = sets["noise_decoys"]
    delta = decoys.astype(np.int16) - clean.astype(np.int16)
    assert np.abs(delta).max() == 16 and np.array_equal(decoy_labels, clean_labels)
    assert (delta != 0).mean() > 0.9  # most channels move; the rest are clipped at 0 or 255

    other, other_idx = script.make_query_sets(images, labels, count=100, seed=8, amplitude=16)
    assert not np.array_equal(other["random"][0], sets["random"][0])
    assert not np.array_equal(other_idx, idx)


def test_query_sets_reject_bad_counts(monkeypatch):
    script = _script(monkeypatch)
    images, labels = _fake_test_set(count=10)
    with pytest.raises(ValueError):
        script.make_query_sets(images, labels, count=11, seed=0, amplitude=16)
    with pytest.raises(ValueError):
        script.make_query_sets(images, labels, count=0, seed=0, amplitude=16)
