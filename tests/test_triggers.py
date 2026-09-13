"""Tests for trigger set generation (P1.2).

Byte-identical regeneration and cross-key independence are P1.4's tests. These
cover the construction itself. Every key here is a TEST KEY, public by
construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.watermark.triggers import (
    DEFAULT_AMPLITUDE,
    DEFAULT_N,
    apply_perturbation,
    generate_triggers,
    perturbation_signs,
    select_base_indices,
)

TEST_KEY = bytes(range(32))
"""TEST KEY ONLY."""

REPO_ROOT = Path(__file__).resolve().parents[1]
CIFAR10_DIR = REPO_ROOT / "data" / "cifar-10-batches-py"


def _synthetic_images(count: int = 600, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (count, 32, 32, 3), dtype=np.uint8)


# --- the generated set ------------------------------------------------------


def test_trigger_set_shapes_and_dtypes():
    images = _synthetic_images()
    ts = generate_triggers(TEST_KEY, images, range(500), n=DEFAULT_N)
    assert len(ts) == DEFAULT_N
    assert ts.images.shape == ts.base_images.shape == ts.signs.shape == (DEFAULT_N, 32, 32, 3)
    assert ts.images.dtype == ts.base_images.dtype == np.uint8
    assert ts.signs.dtype == np.int8
    assert ts.base_indices.shape == (DEFAULT_N,) and ts.base_indices.dtype == np.int64
    assert ts.amplitude == DEFAULT_AMPLITUDE


def test_base_images_come_from_the_pool_only():
    images = _synthetic_images()
    pool = list(range(0, 600, 3))
    ts = generate_triggers(TEST_KEY, images, pool, n=150)
    assert len(set(ts.base_indices.tolist())) == 150
    assert set(ts.base_indices.tolist()) <= set(pool)
    assert np.array_equal(ts.base_images, images[ts.base_indices])


def test_perturbation_is_bounded_and_exact_where_not_clipped():
    images = _synthetic_images()
    ts = generate_triggers(TEST_KEY, images, range(600), n=100, amplitude=20)
    delta = ts.images.astype(np.int16) - ts.base_images.astype(np.int16)
    assert np.abs(delta).max() <= 20
    base = ts.base_images.astype(np.int16)
    unclipped = (base >= 20) & (base <= 235)
    assert np.array_equal(delta[unclipped], 20 * ts.signs[unclipped].astype(np.int16))


def test_clipping_at_the_pixel_range():
    base = np.array([[0, 0, 255, 255, 128]], dtype=np.uint8)
    signs = np.array([[-1, 1, 1, -1, 1]], dtype=np.int8)
    out = apply_perturbation(base, signs, 16)
    assert out.tolist() == [[0, 16, 255, 239, 144]]


def test_triggers_differ_from_their_bases_and_each_other():
    images = np.full((300, 32, 32, 3), 128, dtype=np.uint8)
    ts = generate_triggers(TEST_KEY, images, range(300), n=100)
    assert not np.array_equal(ts.images, ts.base_images)
    flat = {t.tobytes() for t in ts.images}
    assert len(flat) == 100


def test_first_m_triggers_do_not_depend_on_n():
    images = _synthetic_images()
    small = generate_triggers(TEST_KEY, images, range(600), n=40)
    large = generate_triggers(TEST_KEY, images, range(600), n=100)
    assert np.array_equal(large.base_indices[:40], small.base_indices)
    assert np.array_equal(large.signs[:40], small.signs)
    assert np.array_equal(large.images[:40], small.images)


def test_pool_order_does_not_matter():
    images = _synthetic_images()
    pool = list(range(600))
    shuffled = list(np.random.default_rng(1).permutation(pool))
    a = generate_triggers(TEST_KEY, images, pool, n=50)
    b = generate_triggers(TEST_KEY, images, shuffled, n=50)
    assert np.array_equal(a.base_indices, b.base_indices)


def test_arrays_are_read_only():
    ts = generate_triggers(TEST_KEY, _synthetic_images(), range(600), n=10)
    with pytest.raises(ValueError):
        ts.images[0, 0, 0, 0] = 0


# --- the pieces -------------------------------------------------------------


def test_signs_are_plus_minus_one_and_balanced():
    signs = perturbation_signs(TEST_KEY, 100, (32, 32, 3))
    assert set(np.unique(signs).tolist()) == {-1, 1}
    # 307,200 fair signs: the mean has standard deviation about 0.0018.
    assert abs(float(signs.mean())) < 0.01


def test_sign_patterns_are_not_correlated_across_triggers():
    signs = perturbation_signs(TEST_KEY, 50, (32, 32, 3)).reshape(50, -1).astype(np.float64)
    corr = signs @ signs.T / signs.shape[1]
    off_diagonal = corr[~np.eye(50, dtype=bool)]
    # Independent patterns: each correlation has standard deviation 1/sqrt(3072), about 0.018.
    assert np.abs(off_diagonal).max() < 0.1


def test_signs_follow_the_documented_bit_layout():
    from src.watermark.keygen import KeyStream
    from src.watermark.triggers import PERTURBATION_SIGN_LABEL

    shape = (2, 5)  # 10 elements, 2 bytes per pattern
    signs = perturbation_signs(TEST_KEY, 3, shape)
    raw = KeyStream(TEST_KEY, PERTURBATION_SIGN_LABEL).read(6)
    for i in range(3):
        bits = f"{raw[2 * i]:08b}{raw[2 * i + 1]:08b}"[:10]
        expected = [1 if b == "1" else -1 for b in bits]
        assert signs[i].reshape(-1).tolist() == expected


def test_selection_is_a_partial_fisher_yates_of_the_sorted_pool():
    from src.watermark.keygen import KeyStream
    from src.watermark.triggers import BASE_INDEX_LABEL

    pool = [9, 3, 7, 1, 5]
    items = sorted(pool)
    stream = KeyStream(TEST_KEY, BASE_INDEX_LABEL)
    for i in range(3):
        j = i + stream.randbelow(len(items) - i)
        items[i], items[j] = items[j], items[i]
    assert select_base_indices(TEST_KEY, pool, 3).tolist() == items[:3]


def test_whole_pool_can_be_drawn():
    assert sorted(select_base_indices(TEST_KEY, range(20), 20).tolist()) == list(range(20))


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "pool, n, exc",
    [
        (range(10), 11, ValueError),
        (range(10), 0, ValueError),
        ([1, 2, 2], 1, ValueError),
        ([-1, 2], 1, ValueError),
        (range(10), 2.0, TypeError),
    ],
)
def test_select_base_indices_rejects_bad_input(pool, n, exc):
    with pytest.raises(exc):
        select_base_indices(TEST_KEY, pool, n)


@pytest.mark.parametrize("amplitude, exc", [(0, ValueError), (256, ValueError), (16.0, TypeError)])
def test_bad_amplitude_is_rejected(amplitude, exc):
    with pytest.raises(exc):
        generate_triggers(TEST_KEY, _synthetic_images(), range(600), n=5, amplitude=amplitude)


def test_non_uint8_images_are_rejected():
    with pytest.raises(TypeError):
        generate_triggers(TEST_KEY, _synthetic_images().astype(np.float32), range(600), n=5)


def test_pool_index_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        generate_triggers(TEST_KEY, _synthetic_images(100), range(1000), n=100)


def test_bad_key_is_rejected():
    with pytest.raises(ValueError):
        generate_triggers(bytes(16), _synthetic_images(), range(600), n=5)


# --- real CIFAR-10 ----------------------------------------------------------


@pytest.mark.skipif(not CIFAR10_DIR.is_dir(), reason="CIFAR-10 not downloaded under data/")
def test_cifar10_triggers_use_the_training_split_and_never_the_holdout():
    from src.data.cifar10 import cifar10_split_indices
    from src.watermark.triggers import generate_cifar10_triggers

    ts = generate_cifar10_triggers(TEST_KEY, REPO_ROOT / "data")
    train_idx, holdout_idx = cifar10_split_indices()
    chosen = set(ts.base_indices.tolist())
    assert len(ts) == DEFAULT_N and len(chosen) == DEFAULT_N
    assert chosen <= set(train_idx)
    assert chosen.isdisjoint(holdout_idx)
