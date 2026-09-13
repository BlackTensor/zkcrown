"""Tests for the trigger-to-target-response mapping (P2.2).

Every key here is a TEST KEY, public by construction.
"""

from __future__ import annotations

import hashlib
import hmac
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from src.watermark.keygen import DOMAIN
from src.watermark.responses import (
    CIFAR10_NUM_CLASSES,
    MAPPING,
    TARGET_CLASS_LABEL,
    TriggerResponses,
    derive_target_classes,
    null_fire_probability_bound,
    trigger_responses,
)
from src.watermark.signature import signature_label
from src.watermark.triggers import BASE_INDEX_LABEL, PERTURBATION_SIGN_LABEL, generate_triggers

TEST_KEY = bytes(range(32))
"""TEST KEY ONLY."""

KAT_LABELS = [i % 10 for i in range(20)]
KAT_TARGETS = [2, 8, 3, 0, 1, 6, 2, 8, 1, 0, 6, 2, 7, 7, 0, 4, 7, 1, 2, 1]
"""Targets for TEST_KEY and KAT_LABELS. A regression pin; `test_matches_the_documented_formula`
recomputes the same values with `hmac` directly, without `KeyStream`."""

REPO_ROOT = Path(__file__).resolve().parents[1]
CIFAR10_DIR = REPO_ROOT / "data" / "cifar-10-batches-py"


def _keys(count: int, tag: bytes = b"responses-test") -> list[bytes]:
    return [hashlib.sha256(tag + i.to_bytes(4, "big")).digest() for i in range(count)]


# --- derivation -------------------------------------------------------------


def test_known_answer_vector():
    assert derive_target_classes(TEST_KEY, KAT_LABELS).tolist() == KAT_TARGETS


def test_matches_the_documented_formula():
    """t_i = (y_i + 1 + r_i) mod 10, r_i from unbiased 64-bit rejection draws, computed by hand."""
    label = TARGET_CLASS_LABEL.encode()
    prefix = DOMAIN + len(label).to_bytes(2, "big") + label
    stream = b"".join(hmac.new(TEST_KEY, prefix + i.to_bytes(8, "big"), hashlib.sha256).digest() for i in range(64))
    limit = (1 << 64) - ((1 << 64) % 9)
    offset, expected = 0, []
    for y in KAT_LABELS:
        while True:
            value = int.from_bytes(stream[offset : offset + 8], "big")
            offset += 8
            if value < limit:
                break
        expected.append((y + 1 + value % 9) % 10)
    assert expected == KAT_TARGETS


def test_target_never_equals_the_base_label():
    labels = np.random.default_rng(0).integers(0, 10, 5_000)
    targets = derive_target_classes(TEST_KEY, labels)
    assert targets.dtype == np.int64
    assert ((targets >= 0) & (targets < 10)).all()
    assert (targets != labels).all()


def test_targets_are_uniform_over_the_other_classes():
    """90,000 targets for base label 3: each of the other 9 classes expects 10,000."""
    targets = derive_target_classes(TEST_KEY, np.full(90_000, 3))
    counts = np.bincount(targets, minlength=10)
    assert counts[3] == 0
    # Each count has standard deviation sqrt(90000 * 1/9 * 8/9), about 94. Threshold is 6 sd.
    assert np.abs(counts[np.arange(10) != 3] - 10_000).max() < 6 * 94.3, counts


def test_first_m_targets_do_not_depend_on_n():
    labels = np.random.default_rng(1).integers(0, 10, 100)
    assert np.array_equal(derive_target_classes(TEST_KEY, labels)[:40], derive_target_classes(TEST_KEY, labels[:40]))


def test_different_key_gives_different_targets():
    labels = np.random.default_rng(2).integers(0, 10, 100)
    flipped = bytearray(TEST_KEY)
    flipped[31] ^= 1
    agree = (derive_target_classes(TEST_KEY, labels) == derive_target_classes(bytes(flipped), labels)).sum()
    # Two independent draws agree with probability 1/9: mean about 11 of 100, sd about 3.1.
    assert agree < 11.1 + 6 * 3.14


def test_stream_label_is_separate_from_every_other_use_of_k():
    others = {BASE_INDEX_LABEL, PERTURBATION_SIGN_LABEL, signature_label("owner")}
    assert TARGET_CLASS_LABEL not in others
    assert not any(TARGET_CLASS_LABEL.startswith(o) or o.startswith(TARGET_CLASS_LABEL) for o in others)


def test_works_for_other_class_counts():
    labels = [0, 1, 0, 1, 1]
    assert (derive_target_classes(TEST_KEY, labels, num_classes=2) == 1 - np.asarray(labels)).all()
    many = derive_target_classes(TEST_KEY, np.full(1_000, 99), num_classes=100)
    assert (many != 99).all() and many.max() < 100


# --- the null behaviour the mapping was chosen for --------------------------


def _fire_counts(predictions: np.ndarray, labels: np.ndarray, keys: list[bytes]) -> np.ndarray:
    return np.array([(derive_target_classes(k, labels) == predictions).sum() for k in keys])


def test_null_bound_value():
    assert null_fire_probability_bound() == Fraction(1, 9)
    assert null_fire_probability_bound(2) == 1


def test_a_model_that_classifies_the_base_image_correctly_never_fires():
    labels = np.random.default_rng(3).integers(0, 10, 100)
    assert _fire_counts(labels, labels, _keys(200)).max() == 0


def test_a_model_biased_to_one_class_fires_at_the_bound_not_more():
    """A model that always says class 0, over 1,000 keys and 100 triggers each.

    Under a single owner class 0 it would fire on every trigger. Here it fires
    with probability 1/9 on triggers whose base label is not 0, and never on
    the others.
    """
    labels = np.random.default_rng(4).integers(0, 10, 100)
    labels[:10] = 0
    predictions = np.zeros(100, dtype=np.int64)
    counts = _fire_counts(predictions, labels, _keys(1_000))
    eligible = int((labels != 0).sum())
    trials = 1_000 * eligible
    p = 1 / 9
    assert abs(counts.sum() - trials * p) < 6 * np.sqrt(trials * p * (1 - p)), counts.sum()


BINOMIAL_VAR = 100 * (1 / 9) * (8 / 9)
"""Var(k) for N = 100 independent fire events with p = 1/9, about 9.88."""

VAR_THRESHOLD = 2.65
"""The sample variance of k over 1,000 keys has standard deviation about 0.44; this is 6 sd."""


def test_fire_count_under_the_null_has_binomial_spread():
    """A model that never predicts the base label, with varied predictions."""
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 10, 100)
    predictions = (labels + 1 + rng.integers(0, 9, 100)) % 10
    counts = _fire_counts(predictions, labels, _keys(1_000))
    assert abs(counts.var(ddof=1) - BINOMIAL_VAR) < VAR_THRESHOLD, counts.var(ddof=1)
    assert abs(counts.mean() - 100 / 9) < 6 * np.sqrt(BINOMIAL_VAR / 1_000)


def test_the_variance_check_catches_a_shared_target_class():
    """Planted dependence: a model that always says class 0, on bases that are never class 0.

    With per-trigger targets, k ~ Binomial(100, 1/9). With one key-derived
    class shared by every trigger, k is 100 when that class is 0 and 0
    otherwise, so Var(k) is about 100^2 * (1/9) * (8/9), about 988.
    """
    labels = np.random.default_rng(6).integers(1, 10, 100)
    predictions = np.zeros(100, dtype=np.int64)
    keys = _keys(1_000)
    keyed = _fire_counts(predictions, labels, keys)
    assert abs(keyed.var(ddof=1) - BINOMIAL_VAR) < VAR_THRESHOLD, keyed.var(ddof=1)
    shared = np.array([100 * int(derive_target_classes(k, [1], CIFAR10_NUM_CLASSES)[0] == 0) for k in keys])
    assert abs(shared.var(ddof=1) - BINOMIAL_VAR) > VAR_THRESHOLD, shared.var(ddof=1)


# --- TriggerResponses -------------------------------------------------------


def _synthetic():
    rng = np.random.default_rng(7)
    images = rng.integers(0, 256, (600, 32, 32, 3), dtype=np.uint8)
    labels = rng.integers(0, 10, 600)
    return generate_triggers(TEST_KEY, images, range(600), n=100), labels


def test_trigger_responses_use_the_base_image_labels():
    ts, labels = _synthetic()
    resp = trigger_responses(TEST_KEY, ts, labels)
    assert len(resp) == len(ts) == 100
    assert resp.mapping == MAPPING and resp.num_classes == 10
    assert np.array_equal(resp.base_labels, labels[ts.base_indices])
    assert np.array_equal(resp.targets, derive_target_classes(TEST_KEY, labels[ts.base_indices]))
    with pytest.raises(ValueError):
        resp.targets[0] = 0


def test_fired_compares_top1_predictions_with_targets():
    resp = TriggerResponses(targets=np.array([1, 2, 3]), base_labels=np.array([0, 0, 0]), num_classes=10)
    assert resp.fired([1, 0, 3]).tolist() == [True, False, True]
    assert resp.fired(np.array([1, 2, 3])).all()


@pytest.mark.parametrize(
    "predictions, exc",
    [([1, 2], ValueError), ([1, 2, 10], ValueError), ([1, 2, -1], ValueError), ([1.0, 2.0, 3.0], TypeError)],
)
def test_fired_rejects_bad_predictions(predictions, exc):
    resp = TriggerResponses(targets=np.array([1, 2, 3]), base_labels=np.array([0, 0, 0]), num_classes=10)
    with pytest.raises(exc):
        resp.fired(predictions)


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "labels, num_classes, exc",
    [
        ([], 10, ValueError),
        ([10], 10, ValueError),
        ([-1], 10, ValueError),
        ([[1, 2]], 10, ValueError),
        ([0.0], 10, TypeError),
        ([True], 10, TypeError),
        ([0], 1, ValueError),
        ([0], 10.0, TypeError),
    ],
)
def test_bad_input_is_rejected(labels, num_classes, exc):
    with pytest.raises(exc):
        derive_target_classes(TEST_KEY, labels, num_classes)


def test_bad_key_is_rejected():
    with pytest.raises(ValueError):
        derive_target_classes(bytes(31), [0])


def test_labels_too_short_for_the_trigger_set_are_rejected():
    ts, labels = _synthetic()
    with pytest.raises(ValueError):
        trigger_responses(TEST_KEY, ts, labels[: int(ts.base_indices.max())])


# --- real CIFAR-10 ----------------------------------------------------------


@pytest.mark.skipif(not CIFAR10_DIR.is_dir(), reason="CIFAR-10 not downloaded under data/")
def test_cifar10_targets_differ_from_the_true_labels():
    from torchvision.datasets import CIFAR10

    from src.watermark.responses import cifar10_trigger_responses
    from src.watermark.triggers import generate_cifar10_triggers

    ts = generate_cifar10_triggers(TEST_KEY, REPO_ROOT / "data")
    resp = cifar10_trigger_responses(TEST_KEY, ts, REPO_ROOT / "data")
    true_labels = np.asarray(CIFAR10(str(REPO_ROOT / "data"), train=True).targets)[ts.base_indices]
    assert np.array_equal(resp.base_labels, true_labels)
    assert (resp.targets != true_labels).all()
