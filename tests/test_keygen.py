"""Tests for the key-derived stream (P1.1).

Every key in this file is a TEST KEY, public by construction. None of them is,
or may be used as, a real owner key.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from src.watermark.keygen import (
    BLOCK_BYTES,
    DOMAIN,
    KEY_BYTES,
    KeyStream,
    check_key,
    generate_key,
)

TEST_KEY = bytes(range(32))
"""TEST KEY ONLY. 00 01 02 ... 1f."""

OTHER_TEST_KEY = bytes(range(1, 33))
"""TEST KEY ONLY. Differs from TEST_KEY in every byte."""

KAT_LABEL = "test/kat"
KAT_FIRST_40_BYTES = (
    "26a7d2b5aae854307b4ca643a3f16c19b7c5a2f712c20ed21e9f6e4030ef23d7"
    "2b0d481a8415d6ee"
)
"""First 40 stream bytes for TEST_KEY and KAT_LABEL.

Cross-checked outside Python with
``openssl dgst -sha256 -mac HMAC -macopt hexkey:<TEST_KEY>`` over the block 0
and block 1 messages. If this test fails, the construction changed, and every
trigger derived from a real `K` changed with it.
"""

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- same K, same stream ----------------------------------------------------


def test_known_answer_vector():
    assert KeyStream(TEST_KEY, KAT_LABEL).read(40).hex() == KAT_FIRST_40_BYTES


def test_matches_the_documented_construction():
    label = "triggers/v1"
    stream = KeyStream(TEST_KEY, label).read(3 * BLOCK_BYTES)
    encoded = label.encode()
    expected = b"".join(
        hmac.new(
            TEST_KEY,
            DOMAIN + len(encoded).to_bytes(2, "big") + encoded + i.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        for i in range(3)
    )
    assert stream == expected


def test_same_key_and_label_give_identical_bytes():
    a = KeyStream(TEST_KEY, "triggers/v1").read(10_000)
    b = KeyStream(TEST_KEY, "triggers/v1").read(10_000)
    assert a == b


def test_chunked_reads_equal_one_read():
    whole = KeyStream(TEST_KEY, "triggers/v1").read(1_000)
    stream = KeyStream(TEST_KEY, "triggers/v1")
    rng = random.Random(0)
    parts = []
    while stream.position < 1_000:
        parts.append(stream.read(min(rng.randint(0, 70), 1_000 - stream.position)))
    assert b"".join(parts) == whole


def test_block_gives_random_access():
    stream = KeyStream(TEST_KEY, "triggers/v1")
    data = stream.read(5 * BLOCK_BYTES)
    assert stream.block(3) == data[3 * BLOCK_BYTES : 4 * BLOCK_BYTES]
    assert stream.position == 5 * BLOCK_BYTES


def test_global_rng_state_has_no_effect():
    from src.utils.seeding import set_seed

    set_seed(1)
    a = KeyStream(TEST_KEY, "triggers/v1").uniforms(100)
    set_seed(2)
    random.random()
    b = KeyStream(TEST_KEY, "triggers/v1").uniforms(100)
    assert a == b


def test_bytes_like_keys_give_the_same_stream():
    ref = KeyStream(TEST_KEY, "x").read(64)
    assert KeyStream(bytearray(TEST_KEY), "x").read(64) == ref
    assert KeyStream(memoryview(TEST_KEY), "x").read(64) == ref


@pytest.mark.parametrize("hashseed", ["0", "12345"])
def test_identical_in_a_fresh_process(hashseed):
    """Guards against hidden dependence on `hash()` or process state."""
    code = (
        "from src.watermark.keygen import KeyStream;"
        f"print(KeyStream(bytes(range(32)), 'triggers/v1').read(256).hex())"
    )
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == KeyStream(TEST_KEY, "triggers/v1").read(256).hex()


# --- different K or label, different stream ---------------------------------


def test_different_key_gives_different_stream():
    assert KeyStream(TEST_KEY, "triggers/v1").read(64) != KeyStream(
        OTHER_TEST_KEY, "triggers/v1"
    ).read(64)


def test_single_bit_key_change_flips_about_half_the_bits():
    flipped = bytearray(TEST_KEY)
    flipped[-1] ^= 0x01
    a = KeyStream(TEST_KEY, "triggers/v1").read(4_096)
    b = KeyStream(bytes(flipped), "triggers/v1").read(4_096)
    differing = sum(bin(x ^ y).count("1") for x, y in zip(a, b))
    total = 8 * len(a)
    # Expected 16,384 of 32,768; the standard deviation is about 90.
    assert abs(differing - total / 2) < 1_000


@pytest.mark.parametrize(
    "label_a, label_b",
    [
        ("triggers/v1", "projection/v1"),
        ("triggers/v1", "triggers/v2"),
        ("a", "a\x00"),
        ("ab", "a"),
    ],
)
def test_different_labels_give_different_streams(label_a, label_b):
    assert KeyStream(TEST_KEY, label_a).read(64) != KeyStream(TEST_KEY, label_b).read(64)


# --- derived values ---------------------------------------------------------


def test_uniforms_are_in_unit_interval_and_roughly_uniform():
    values = KeyStream(TEST_KEY, "test/uniforms").uniforms(20_000)
    assert all(0.0 <= v < 1.0 for v in values)
    counts = [0] * 10
    for v in values:
        counts[int(v * 10)] += 1
    chi2 = sum((c - 2_000) ** 2 / 2_000 for c in counts)
    # Chi-square, 9 degrees of freedom: 27.9 is the 0.999 quantile. The key is
    # fixed, so this is a deterministic check, not a flaky one.
    assert chi2 < 27.9


def test_uniforms_consume_eight_bytes_each():
    stream = KeyStream(TEST_KEY, "x")
    stream.uniforms(5)
    assert stream.position == 40


@pytest.mark.parametrize("upper", [1, 2, 7, 10, 256, 1_000_003, 2**64])
def test_randbelow_stays_in_range(upper):
    stream = KeyStream(TEST_KEY, "test/randbelow")
    assert all(0 <= stream.randbelow(upper) < upper for _ in range(500))


def test_randbelow_hits_every_value():
    stream = KeyStream(TEST_KEY, "test/randbelow")
    assert {stream.randbelow(10) for _ in range(500)} == set(range(10))


def test_randbelow_rejects_biased_draws():
    """With upper = 2**63 + 1 about half of all 64-bit draws must be rejected."""
    upper = 2**63 + 1
    stream = KeyStream(TEST_KEY, "test/rejection")
    for _ in range(200):
        stream.randbelow(upper)
    draws = stream.position // 8
    assert 300 < draws < 500


# --- validation -------------------------------------------------------------


def test_generate_key_has_the_right_length_and_varies():
    keys = {generate_key() for _ in range(8)}
    assert len(keys) == 8
    assert all(len(k) == KEY_BYTES for k in keys)


@pytest.mark.parametrize("bad", [b"", bytes(31), bytes(33)])
def test_check_key_rejects_wrong_length(bad):
    with pytest.raises(ValueError):
        check_key(bad)


@pytest.mark.parametrize("bad", ["00" * 32, None, 123, list(range(32))])
def test_check_key_rejects_non_bytes(bad):
    with pytest.raises(TypeError):
        check_key(bad)


@pytest.mark.parametrize("bad, exc", [("", ValueError), (b"x", TypeError), (None, TypeError)])
def test_bad_labels_are_rejected(bad, exc):
    with pytest.raises(exc):
        KeyStream(TEST_KEY, bad)


@pytest.mark.parametrize("bad, exc", [(-1, ValueError), (1.0, TypeError), (True, TypeError)])
def test_bad_read_lengths_are_rejected(bad, exc):
    with pytest.raises(exc):
        KeyStream(TEST_KEY, "x").read(bad)


@pytest.mark.parametrize("bad, exc", [(0, ValueError), (2**64 + 1, ValueError), (2.0, TypeError)])
def test_bad_randbelow_bounds_are_rejected(bad, exc):
    with pytest.raises(exc):
        KeyStream(TEST_KEY, "x").randbelow(bad)


def test_block_index_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        KeyStream(TEST_KEY, "x").block(-1)
