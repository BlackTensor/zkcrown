"""Determinism and cross-key independence of the trigger set (P1.4).

Two claims are tested here.

1. **Regenerating from `K` reproduces byte-identical triggers.** This is
   checked within one process, in a fresh process with a different
   ``PYTHONHASHSEED``, against a pinned digest, against the committed P1.3
   result file, and with the global RNGs in different states.

2. **A different `K` gives a statistically independent set.** Given the
   dataset and the amplitude, a trigger set is fully determined by two
   key-derived parts: the base indices and the sign patterns. So independence
   is tested on those two parts, against their exact null distributions:

   - Base indices. For two independent keys, the overlap of two 100-element
     draws from 45,000 is hypergeometric, with mean 100 * 100 / 45,000.
   - Sign patterns. For independent fair signs, the correlation between any
     pattern under one key and any pattern under the other, scaled by
     sqrt(3072), is approximately standard normal. That covers all
     100 x 100 pattern pairs, not only the same position, so a shifted or
     partly shared stream would also be caught.

   Keys are compared in two ways. The **related keys** are all 256 keys that
   differ from a reference key in exactly one bit, which is the hardest case
   for a derivation that fails to mix the key. The **unrelated keys** are 16
   keys compared with each other in every pair.

All keys here are TEST KEYS, public by construction. The statistical checks
use fixed keys, so each outcome is deterministic. The thresholds are set far
enough out, at least 6 standard deviations, that a correct implementation
cannot be expected to fail them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from src.data.cifar10 import TRAIN_SIZE
from src.watermark.triggers import (
    DEFAULT_AMPLITUDE,
    DEFAULT_N,
    TriggerSet,
    generate_triggers,
    perturbation_signs,
    select_base_indices,
)

TEST_KEY = bytes(range(32))
"""TEST KEY ONLY. The reference key for the related-key tests."""

UNRELATED_KEYS = [hashlib.sha256(f"zk-crown P1.4 TEST KEY {i}".encode()).digest() for i in range(16)]
"""TEST KEYS ONLY."""

SIGNS_PER_TRIGGER = 32 * 32 * 3
REPO_ROOT = Path(__file__).resolve().parents[1]
CIFAR10_DIR = REPO_ROOT / "data" / "cifar-10-batches-py"
P1_3_RESULT = REPO_ROOT / "results" / "p1.3_trigger_visualization__seed1337__20260913T091124+0000.json"


def _structured_images(count: int = 1_000) -> np.ndarray:
    """uint8 images from integer arithmetic only.

    This avoids numpy's random generators, whose streams are not guaranteed to
    stay the same across numpy versions. That keeps the pinned digest below
    stable.
    """
    i, h, w, c = np.meshgrid(
        np.arange(count), np.arange(32), np.arange(32), np.arange(3), indexing="ij"
    )
    return ((i * 7 + h * 13 + w * 17 + c * 29 + i * h * w) % 256).astype(np.uint8)


def _digest(ts: TriggerSet) -> str:
    """SHA-256 over everything that makes up a trigger set, in a fixed byte order."""
    h = hashlib.sha256()
    h.update(ts.images.tobytes())
    h.update(ts.base_images.tobytes())
    h.update(ts.base_indices.astype("<i8").tobytes())
    h.update(ts.signs.tobytes())
    h.update(ts.amplitude.to_bytes(2, "big"))
    return h.hexdigest()


def _flip_bit(key: bytes, bit: int) -> bytes:
    flipped = bytearray(key)
    flipped[bit // 8] ^= 1 << (bit % 8)
    return bytes(flipped)


RELATED_KEYS = [_flip_bit(TEST_KEY, b) for b in range(256)]


# --- 1. byte-identical regeneration -----------------------------------------


PINNED_DIGEST = "d3d5c541a8b64853939d5fe2b766be8c0806c00c282329f56fa457848f0dbeb1"
"""`_digest` of the TEST_KEY set over `_structured_images()` and pool range(1000),
N = 100, A = 16. It changes only if the trigger construction changes, and
such a change would silently change every real trigger set too."""


def test_regeneration_in_one_process_is_byte_identical():
    images = _structured_images()
    a = generate_triggers(TEST_KEY, images, range(1_000))
    b = generate_triggers(TEST_KEY, images.copy(), list(range(1_000)))
    for field in ("images", "base_images", "base_indices", "signs"):
        x, y = getattr(a, field), getattr(b, field)
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes(), field
    assert _digest(a) == _digest(b)


def test_pinned_digest():
    ts = generate_triggers(TEST_KEY, _structured_images(), range(1_000), n=DEFAULT_N, amplitude=16)
    assert _digest(ts) == PINNED_DIGEST


def test_global_rng_state_does_not_leak_in():
    import torch

    images = _structured_images()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    a = _digest(generate_triggers(TEST_KEY, images, range(1_000)))
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    np.random.rand(1_000)
    torch.rand(1_000)
    b = _digest(generate_triggers(TEST_KEY, images, range(1_000)))
    assert a == b


@pytest.mark.parametrize("hashseed", ["0", "4242"])
def test_regeneration_in_a_fresh_process_is_byte_identical(hashseed):
    code = (
        "import sys; sys.path.insert(0, 'tests');"
        "from test_trigger_determinism import TEST_KEY, _digest, _structured_images;"
        "from src.watermark.triggers import generate_triggers;"
        "print(_digest(generate_triggers(TEST_KEY, _structured_images(), range(1000))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == PINNED_DIGEST


@pytest.mark.skipif(not CIFAR10_DIR.is_dir(), reason="CIFAR-10 not downloaded under data/")
def test_real_cifar10_set_matches_the_committed_p1_3_digest():
    """Regenerate the P1.3 demo-key set and compare it with the digest recorded in
    the committed result file, which was written in an earlier session."""
    from src.watermark.triggers import generate_cifar10_triggers

    record = json.loads(P1_3_RESULT.read_text(encoding="utf-8"))
    demo_key = bytes.fromhex(record["params"]["demo_key_sha256_hex"])
    assert hashlib.sha256(record["params"]["demo_key_phrase"].encode()).digest() == demo_key

    ts = generate_cifar10_triggers(
        demo_key,
        REPO_ROOT / "data",
        n=record["params"]["n"],
        amplitude=record["params"]["default_amplitude"],
    )
    assert hashlib.sha256(ts.images.tobytes()).hexdigest() == record["metrics"]["trigger_images_sha256"]


# --- 2. different K, independent set ----------------------------------------


def _hypergeometric_moments(pool: int, n: int) -> tuple[float, float]:
    """Mean and variance of the overlap of two independent n-subsets of a pool."""
    p = n / pool
    return n * p, n * p * (1 - p) * (pool - n) / (pool - 1)


def _overlap_check(pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
    mean, var = _hypergeometric_moments(TRAIN_SIZE, DEFAULT_N)
    total = sum(len(np.intersect1d(a, b)) for a, b in pairs)
    expected, sd = len(pairs) * mean, math.sqrt(len(pairs) * var)
    assert abs(total - expected) < 6 * sd, (total, expected, sd)

    # Same base image at the same trigger position: Poisson with mean
    # pairs * N / 45,000, which is 0.57 for 256 pairs. Crossing 8 has
    # probability below 1e-7.
    positional = sum(int(np.sum(a == b)) for a, b in pairs)
    assert positional <= 8, positional


def _sign_matrix(key: bytes) -> np.ndarray:
    signs = perturbation_signs(key, DEFAULT_N, (32, 32, 3))
    return signs.reshape(DEFAULT_N, -1).astype(np.float32)


def _sign_check(pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Every cross-key pattern pair's scaled correlation is approximately N(0, 1)."""
    z_sq_sum, count, z_max = 0.0, 0, 0.0
    for a, b in pairs:
        z = (a @ b.T) / math.sqrt(SIGNS_PER_TRIGGER)
        z_sq_sum += float(np.square(z, dtype=np.float64).sum())
        count += z.size
        z_max = max(z_max, float(np.abs(z).max()))

    # A shared or shifted pattern gives |z| = sqrt(3072), about 55. For true
    # independence, P(|z| > 8) is about 1e-15 per pair, across fewer than 4M pairs.
    assert z_max < 8.0, z_max
    # Mean of z^2 is 1 under independence. Its standard deviation is
    # sqrt(2 / count), at most 0.0045 here. A single shared pattern per key
    # pair would add about 0.3.
    mean_z_sq = z_sq_sum / count
    assert abs(mean_z_sq - 1.0) < 8 * math.sqrt(2 / count) + 1e-3, mean_z_sq


def test_single_bit_key_changes_give_independent_base_indices():
    pool = range(TRAIN_SIZE)
    reference = select_base_indices(TEST_KEY, pool, DEFAULT_N)
    pairs = [(reference, select_base_indices(k, pool, DEFAULT_N)) for k in RELATED_KEYS]
    _overlap_check(pairs)


def test_single_bit_key_changes_give_independent_sign_patterns():
    reference = _sign_matrix(TEST_KEY)
    _sign_check([(reference, _sign_matrix(k)) for k in RELATED_KEYS])


def test_unrelated_keys_give_independent_base_indices():
    pool = range(TRAIN_SIZE)
    draws = [select_base_indices(k, pool, DEFAULT_N) for k in UNRELATED_KEYS]
    _overlap_check(list(combinations(draws, 2)))


def test_unrelated_keys_give_independent_sign_patterns():
    matrices = [_sign_matrix(k) for k in UNRELATED_KEYS]
    _sign_check(list(combinations(matrices, 2)))


def test_different_key_changes_the_trigger_images():
    images = _structured_images()
    a = generate_triggers(TEST_KEY, images, range(1_000))
    b = generate_triggers(RELATED_KEYS[0], images, range(1_000))
    assert _digest(a) != _digest(b)
    # For each position, the fraction of pixel channels whose applied
    # perturbation has the same sign under both keys, taken where neither
    # trigger was clipped. It should sit at about one half.
    da = a.images.astype(np.int16) - a.base_images.astype(np.int16)
    db = b.images.astype(np.int16) - b.base_images.astype(np.int16)
    both = (np.abs(da) == DEFAULT_AMPLITUDE) & (np.abs(db) == DEFAULT_AMPLITUDE)
    agree = float((np.sign(da) == np.sign(db))[both].mean())
    assert abs(agree - 0.5) < 0.01, agree


# --- the checks can fail ----------------------------------------------------


def test_sign_check_rejects_a_shared_pattern():
    """A test of independence that cannot fail tests nothing."""
    a = _sign_matrix(TEST_KEY)
    b = _sign_matrix(RELATED_KEYS[0]).copy()
    b[37] = a[12]
    with pytest.raises(AssertionError):
        _sign_check([(a, b)])


def test_sign_check_rejects_weakly_correlated_patterns():
    a = _sign_matrix(TEST_KEY)
    b = _sign_matrix(RELATED_KEYS[0]).copy()
    # Copy 5% of every pattern's signs from the other key. For each
    # same-position pair z is about 154 / sqrt(3072) = 2.8, below the max
    # check, so the z^2 mean has to catch it: it rises by about 0.08.
    b[:, :154] = a[:, :154]
    with pytest.raises(AssertionError):
        _sign_check([(a, b)] * 256)


def test_overlap_check_rejects_correlated_draws():
    pool = range(TRAIN_SIZE)
    reference = select_base_indices(TEST_KEY, pool, DEFAULT_N)
    leaky = []
    for k in RELATED_KEYS[:64]:
        draw = select_base_indices(k, pool, DEFAULT_N).copy()
        draw[:3] = reference[:3]
        leaky.append((reference, draw))
    with pytest.raises(AssertionError):
        _overlap_check(leaky)
