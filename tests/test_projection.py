"""Tests for the key-derived projection `P_K` (P3.1).

Covered: the documented derivation (including a known-answer vector checked
outside Python), determinism, domain separation, prefix stability, the linear
algebra of `project` and `back_project`, and the statistical shape of `P_K` at
the real carrier size of `main_model` (307,040 conv and linear weights).

The statistical checks use fixed keys, so each outcome is deterministic. Their
thresholds sit at least 6 standard deviations from the null, and one planted
defect confirms the Gram check can fail.

All keys here are TEST KEYS, public by construction.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.watermark.keygen import DOMAIN, KeyStream
from src.watermark.projection import (
    DEFAULT_ROWS,
    PROJECTION_VERSION,
    KeyProjection,
    derive_projection,
    projection_label,
    projection_signs,
)
from src.watermark.triggers import PERTURBATION_SIGN_LABEL

TEST_KEY = bytes(range(32))
"""TEST KEY ONLY."""

OTHER_KEY = bytes(range(100, 132))
"""TEST KEY ONLY."""

MAIN_MODEL_CARRIER_DIM = 307_040
"""Conv and linear weights of `main_model` at width 32 (P0.4 ledger). Used only
as a realistic size; the carrier itself is chosen in P3.2."""

KAT_DIM21_BLOCK0_PREFIX_HEX = "bc4164b54dcd03b2762f592f"
"""First 12 bytes of block 0 of the projection stream for TEST_KEY and dim 21,
i.e. the raw bytes of rows 0 to 3. Cross-checked outside Python with
``openssl dgst -sha256 -mac HMAC -macopt hexkey:<TEST_KEY>`` over
``DOMAIN || u16_be(31) || "projection/v1/rademacher/dim=21" || u64_be(0)``."""

PINNED_MAIN_SIZE_DIGEST = "89d84b601d7fff870257f508d5dad7a0b1a65aaaa004855cf2feda12744ce60c"
"""SHA-256 of the MSB-first packed sign bits (+1 -> 1) of the 128 x 307,040
projection for TEST_KEY. Pinned to catch an accidental change of construction."""

REPO_ROOT = Path(__file__).resolve().parents[1]


def _packed_digest(signs: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(signs > 0, axis=1, bitorder="big").tobytes()).hexdigest()


@pytest.fixture(scope="module")
def main_size_projection() -> KeyProjection:
    return derive_projection(TEST_KEY, MAIN_MODEL_CARRIER_DIM)


@pytest.fixture(scope="module")
def main_size_gram(main_size_projection) -> np.ndarray:
    s = main_size_projection.signs.astype(np.float64)
    return s @ s.T  # exact: integers far below 2**53


# --- derivation -------------------------------------------------------------


def test_label_names_version_and_dim():
    assert projection_label(307_040) == "projection/v1/rademacher/dim=307040"
    assert PROJECTION_VERSION == "projection/v1"


def test_known_answer_vector_small_dim():
    expected_bits = np.unpackbits(
        np.frombuffer(bytes.fromhex(KAT_DIM21_BLOCK0_PREFIX_HEX), np.uint8).reshape(4, 3), axis=1
    )[:, :21]
    signs = projection_signs(TEST_KEY, 4, 21)
    np.testing.assert_array_equal(signs, expected_bits.astype(np.int8) * 2 - 1)
    assert signs[0].tolist() == [1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1]


@pytest.mark.parametrize("rows, dim", [(1, 1), (3, 8), (5, 13), (7, 300), (40, 1000)])
def test_matches_the_documented_formula_with_raw_hmac(rows, dim):
    label = f"projection/v1/rademacher/dim={dim}".encode()
    prefix = DOMAIN + len(label).to_bytes(2, "big") + label
    per_row = math.ceil(dim / 8)
    need = rows * per_row
    stream = b"".join(
        hmac.new(TEST_KEY, prefix + i.to_bytes(8, "big"), hashlib.sha256).digest()
        for i in range(math.ceil(need / 32))
    )[:need]
    expected = np.empty((rows, dim), dtype=np.int8)
    for r in range(rows):
        chunk = stream[r * per_row : (r + 1) * per_row]
        bits = [(byte >> (7 - k)) & 1 for byte in chunk for k in range(8)][:dim]
        expected[r] = [2 * b - 1 for b in bits]
    np.testing.assert_array_equal(projection_signs(TEST_KEY, rows, dim), expected)


def test_pinned_digest_at_main_model_size(main_size_projection):
    assert main_size_projection.signs.shape == (DEFAULT_ROWS, MAIN_MODEL_CARRIER_DIM)
    assert _packed_digest(main_size_projection.signs) == PINNED_MAIN_SIZE_DIGEST


def test_default_rows_is_signature_bits():
    assert DEFAULT_ROWS == 128
    assert derive_projection(TEST_KEY, 50).rows == 128


def test_shape_dtype_values_and_read_only():
    signs = projection_signs(TEST_KEY, 6, 77)
    assert signs.shape == (6, 77) and signs.dtype == np.int8
    assert set(np.unique(signs).tolist()) == {-1, 1}
    assert not signs.flags.writeable
    with pytest.raises(ValueError):
        signs[0, 0] = 1


# --- determinism ------------------------------------------------------------


def test_is_deterministic_and_ignores_global_rng_state():
    random.seed(1)
    np.random.seed(1)
    a = projection_signs(TEST_KEY, 20, 999)
    random.seed(2)
    np.random.seed(2)
    b = projection_signs(bytearray(TEST_KEY), 20, 999)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("hashseed", ["0", "4242"])
def test_identical_in_a_fresh_process(hashseed):
    code = (
        "import hashlib, numpy as np;"
        "from src.watermark.projection import projection_signs;"
        "s = projection_signs(bytes(range(32)), 128, 307040);"
        "print(hashlib.sha256(np.packbits(s > 0, axis=1, bitorder='big').tobytes()).hexdigest())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == PINNED_MAIN_SIZE_DIGEST


def test_prefix_stable_in_rows():
    big = projection_signs(TEST_KEY, 64, 1234)
    for m in (1, 7, 63):
        np.testing.assert_array_equal(projection_signs(TEST_KEY, m, 1234), big[:m])


# --- separation -------------------------------------------------------------


def _agreement_z(a: np.ndarray, b: np.ndarray) -> float:
    """z-score of the sign agreement count against Binomial(n, 1/2)."""
    n = a.size
    agree = int(np.count_nonzero(a == b))
    return (agree - n / 2) / math.sqrt(n / 4)


def test_different_key_gives_unrelated_signs():
    z = _agreement_z(projection_signs(TEST_KEY, 64, 5000), projection_signs(OTHER_KEY, 64, 5000))
    assert abs(z) < 6


def test_single_bit_key_flip_gives_unrelated_signs():
    base = projection_signs(TEST_KEY, 8, 4096)
    for bit in (0, 97, 255):
        flipped = bytearray(TEST_KEY)
        flipped[bit // 8] ^= 1 << (bit % 8)
        assert abs(_agreement_z(base, projection_signs(bytes(flipped), 8, 4096))) < 6


def test_different_dim_does_not_share_a_prefix():
    # Without dim in the label, row 0 of dim 4096 would equal row 0 of dim 4097
    # on their first 4096 entries.
    a = projection_signs(TEST_KEY, 16, 4096)
    b = projection_signs(TEST_KEY, 16, 4097)[:, :4096]
    assert abs(_agreement_z(a, b)) < 6


def test_separate_from_trigger_sign_stream():
    dim = 3072
    per_row = dim // 8
    trigger_bytes = KeyStream(TEST_KEY, PERTURBATION_SIGN_LABEL).read(4 * per_row)
    trigger_bits = np.unpackbits(np.frombuffer(trigger_bytes, np.uint8).reshape(4, per_row), axis=1)
    trigger_signs = trigger_bits.astype(np.int8) * 2 - 1
    assert abs(_agreement_z(projection_signs(TEST_KEY, 4, dim), trigger_signs)) < 6


# --- linear algebra ---------------------------------------------------------


def test_project_and_back_project_match_dense_reference():
    rng = np.random.default_rng(0)
    proj = derive_projection(TEST_KEY, 777, rows=37)  # 37 rows spans several chunks
    dense = proj.signs.astype(np.float64) / math.sqrt(777)
    w = rng.standard_normal(777)
    c = rng.standard_normal(37)
    np.testing.assert_allclose(proj.project(w), dense @ w, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(proj.back_project(c), dense.T @ c, rtol=1e-12, atol=1e-12)


def test_adjoint_identity():
    rng = np.random.default_rng(1)
    proj = derive_projection(TEST_KEY, 2000, rows=20)
    w, c = rng.standard_normal(2000), rng.standard_normal(20)
    assert math.isclose(float(proj.project(w) @ c), float(w @ proj.back_project(c)), rel_tol=1e-10)


def test_outputs_are_float64_and_accept_other_dtypes():
    proj = derive_projection(TEST_KEY, 64, rows=4)
    assert proj.project(np.ones(64, dtype=np.float32)).dtype == np.float64
    assert proj.back_project([1, -1, 1, -1]).shape == (64,)


def test_rows_have_unit_norm():
    proj = derive_projection(TEST_KEY, 1001, rows=10)
    for i in range(10):
        e = np.zeros(10)
        e[i] = 1.0
        assert math.isclose(float(np.linalg.norm(proj.back_project(e))), 1.0, rel_tol=1e-12)


@pytest.mark.parametrize("bad", [np.ones(63), np.ones((64, 1)), np.array([np.nan] + [0.0] * 63)])
def test_project_rejects_bad_vectors(bad):
    with pytest.raises(ValueError):
        derive_projection(TEST_KEY, 64, rows=4).project(bad)


def test_back_project_rejects_bad_coefficients():
    proj = derive_projection(TEST_KEY, 64, rows=4)
    with pytest.raises(ValueError):
        proj.back_project(np.ones(5))
    with pytest.raises(ValueError):
        proj.back_project(np.array([1.0, np.inf, 0.0, 0.0]))


# --- statistical shape at the real carrier size ------------------------------


def test_gram_diagonal_is_exactly_dim(main_size_gram):
    assert np.all(np.diag(main_size_gram) == MAIN_MODEL_CARRIER_DIM)


def test_gram_off_diagonal_matches_its_null(main_size_gram):
    # For independent fair signs, G[i, j] / sqrt(dim) is approximately N(0, 1).
    d = MAIN_MODEL_CARRIER_DIM
    z = main_size_gram[np.triu_indices(DEFAULT_ROWS, k=1)] / math.sqrt(d)
    assert z.size == 128 * 127 // 2
    assert np.max(np.abs(z)) < 7  # largest of 8,128 draws is typically about 4
    assert abs(float(z.mean())) < 6 / math.sqrt(z.size)
    assert abs(float(z.var()) - 1.0) < 6 * math.sqrt(2 / z.size)


def test_gram_check_catches_a_planted_dependent_row():
    s = projection_signs(TEST_KEY, 8, 20_000).astype(np.float64).copy()
    s[5] = s[2]
    s[5, :2000] *= -1  # 90% shared with row 2
    g = s @ s.T
    z = g[np.triu_indices(8, k=1)] / math.sqrt(20_000)
    assert np.max(np.abs(z)) > 7


def test_row_sums_are_balanced_like_fair_signs(main_size_projection):
    sums = main_size_projection.signs.sum(axis=1, dtype=np.int64)
    z = sums / math.sqrt(MAIN_MODEL_CARRIER_DIM)
    assert np.max(np.abs(z)) < 6
    assert abs(float(z.mean())) < 6 / math.sqrt(DEFAULT_ROWS)


def test_cross_key_projections_are_uncorrelated(main_size_projection):
    other = projection_signs(OTHER_KEY, 32, MAIN_MODEL_CARRIER_DIM).astype(np.float64)
    cross = main_size_projection.signs[:32].astype(np.float64) @ other.T
    z = cross.ravel() / math.sqrt(MAIN_MODEL_CARRIER_DIM)
    assert np.max(np.abs(z)) < 7
    assert abs(float(z.var()) - 1.0) < 6 * math.sqrt(2 / z.size)


def test_projection_of_own_back_projection_is_near_identity(main_size_projection):
    rng = np.random.default_rng(7)
    c = rng.choice([-1.0, 1.0], size=DEFAULT_ROWS)
    recovered = main_size_projection.project(main_size_projection.back_project(c))
    # recovered = c + cross-talk; each cross-talk term has sd sqrt(127 / dim), about 0.02.
    crosstalk_sd = math.sqrt((DEFAULT_ROWS - 1) / MAIN_MODEL_CARRIER_DIM)
    assert np.max(np.abs(recovered - c)) < 7 * crosstalk_sd
    assert np.array_equal(np.sign(recovered), c)


# --- validation and hygiene --------------------------------------------------


@pytest.mark.parametrize("rows, dim", [(0, 10), (10, 0), (-1, 10), (True, 10), (10, 2.0), (10, "10")])
def test_rejects_bad_sizes(rows, dim):
    with pytest.raises((TypeError, ValueError)):
        projection_signs(TEST_KEY, rows, dim)


def test_rejects_bad_key():
    with pytest.raises(ValueError):
        projection_signs(bytes(31), 2, 10)
    with pytest.raises(TypeError):
        projection_signs("00" * 32, 2, 10)


def test_key_projection_rejects_writable_or_wrong_dtype_signs():
    with pytest.raises(ValueError):
        KeyProjection(signs=np.ones((2, 3), dtype=np.int8))
    ro = np.ones((2, 3), dtype=np.int16)
    ro.setflags(write=False)
    with pytest.raises(TypeError):
        KeyProjection(signs=ro)


def test_repr_hides_the_signs():
    text = repr(derive_projection(TEST_KEY, 16, rows=2))
    assert "hidden" in text and "rows=2" in text and "dim=16" in text
    assert "-1" not in text
