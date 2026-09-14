"""Tests for the weight watermark extractor (P3.3).

Covered: the documented formulas; recovery of `S` from a P3.2 embedding;
invariance to global rescaling and to per-tensor constant shifts; that
centring costs the watermark only cross-talk; that non-carrier tensors are
ignored; and input validation and hygiene.

Models are freshly initialised from a fixed seed. No trained weights are
used. Correct-versus-wrong-key gaps on the real models are P3.4, and the null
distribution and threshold are P3.7. The wrong-key and no-watermark checks
here are unit-level, and their thresholds sit at least 6 sd out.

All keys and owner ids here are TEST data, public by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models import main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.watermark.carrier import carrier_layout  # noqa: E402
from src.watermark.projection import derive_projection  # noqa: E402
from src.watermark.signature import derive_signature  # noqa: E402
from src.watermark.weight_embedding import (  # noqa: E402
    derive_carrier_projection,
    embed_weight_watermark,
)
from src.watermark.weight_extraction import (  # noqa: E402
    EXTRACTION_VERSION,
    centered_carrier,
    extract_weight_watermark,
    recover_fingerprint,
    score_fingerprint,
)

TEST_KEY = bytes(range(32))
"""TEST KEY ONLY."""

OTHER_KEY = bytes(range(100, 132))
"""TEST KEY ONLY."""

TEST_OWNER = "zk-crown test owner"

STRONG_ALPHA = 0.2
"""A TEST value only, large against a freshly initialised host. Choosing alpha is P3.5."""


@pytest.fixture(scope="module")
def clean_state():
    set_seed(1234)
    return main_model().eval().state_dict()


@pytest.fixture(scope="module")
def layout():
    return carrier_layout(main_model())


@pytest.fixture(scope="module")
def projection(layout):
    return derive_carrier_projection(TEST_KEY, layout)


@pytest.fixture(scope="module")
def signature():
    return derive_signature(TEST_KEY, TEST_OWNER)


@pytest.fixture(scope="module")
def strong_state(clean_state, layout, projection, signature):
    state, _ = embed_weight_watermark(clean_state, layout, projection, signature, STRONG_ALPHA)
    return state


def _signs(signature) -> np.ndarray:
    return np.asarray(signature.signs, dtype=np.float64)


# --- formulas ----------------------------------------------------------------


def test_centered_carrier_subtracts_each_tensor_mean(clean_state, layout):
    shifted = dict(clean_state)
    shifted["features.3.weight"] = clean_state["features.3.weight"] + 0.5
    c = centered_carrier(shifted, layout)
    for name, start, size in zip(layout.names, layout.offsets, layout.sizes):
        segment = c[start : start + size]
        assert abs(segment.mean()) < 1e-12
        raw = shifted[name].double().reshape(-1).numpy()
        np.testing.assert_allclose(segment, raw - raw.mean(), atol=1e-12)


def test_fingerprint_is_projection_of_centered_carrier(clean_state, layout, projection):
    y = recover_fingerprint(clean_state, layout, projection)
    np.testing.assert_allclose(y, projection.project(centered_carrier(clean_state, layout)), atol=0)
    y_raw = recover_fingerprint(clean_state, layout, projection, center=False)
    np.testing.assert_allclose(y_raw, projection.project(layout.flatten(clean_state)), atol=0)


def test_scores_match_their_definitions(strong_state, layout, projection, signature):
    result = extract_weight_watermark(strong_state, layout, projection, signature)
    y, s = result.projected, _signs(signature)
    assert result.correlation == pytest.approx(float(y @ s) / (np.linalg.norm(y) * math.sqrt(128)), abs=1e-15)
    assert result.amplitude == pytest.approx(float(y @ s) / 128, abs=1e-15)
    assert result.projected_rms == pytest.approx(float(np.linalg.norm(y)) / math.sqrt(128), abs=1e-15)
    assert result.bit_matches == sum(int(yi > 0) == b for yi, b in zip(y, signature.bits))
    assert result.recovered_bits == tuple(int(v > 0) for v in y)
    assert -1.0 <= result.correlation <= 1.0


def test_score_of_the_exact_signature_is_perfect(signature):
    corr, amp, rms, matches = score_fingerprint(3.0 * _signs(signature), signature)
    assert corr == pytest.approx(1.0, abs=1e-15) and amp == pytest.approx(3.0) and rms == pytest.approx(3.0)
    assert matches == 128
    corr, amp, _, matches = score_fingerprint(-_signs(signature), signature)
    assert corr == pytest.approx(-1.0, abs=1e-15) and amp == pytest.approx(-1.0) and matches == 0


def test_zero_fingerprint_scores_as_no_evidence(signature):
    corr, amp, rms, matches = score_fingerprint(np.zeros(128), signature)
    assert (corr, amp, rms) == (0.0, 0.0, 0.0)
    assert matches == 128 - sum(signature.bits)


def test_all_zero_carrier_scores_as_no_evidence(clean_state, layout, projection, signature):
    zeroed = {k: (torch.zeros_like(v) if k in layout.names else v) for k, v in clean_state.items()}
    result = extract_weight_watermark(zeroed, layout, projection, signature)
    assert result.correlation == 0.0 and result.amplitude == 0.0 and result.projected_rms == 0.0


# --- recovery ----------------------------------------------------------------


def test_recovers_every_bit_of_a_strong_embedding(strong_state, layout, projection, signature):
    result = extract_weight_watermark(strong_state, layout, projection, signature)
    assert result.bit_matches == 128
    assert result.recovered_bits == signature.bits
    assert result.correlation > 0.9


def test_amplitude_estimates_alpha_within_host_noise(clean_state, strong_state, layout, projection, signature):
    # amplitude = <host, S>/128 + alpha * S^T P c(P^T S) / 128. The host term has
    # sd about host_rms / sqrt(128); the cross-talk term is far smaller.
    host_rms = extract_weight_watermark(clean_state, layout, projection, signature).projected_rms
    result = extract_weight_watermark(strong_state, layout, projection, signature)
    tolerance = 6 * host_rms / math.sqrt(128) + 6 * STRONG_ALPHA * math.sqrt(127 / layout.dim)
    assert abs(result.amplitude - STRONG_ALPHA) < tolerance


def test_host_plus_watermark_decomposes_linearly(clean_state, strong_state, layout, projection, signature):
    y_clean = recover_fingerprint(clean_state, layout, projection)
    y_marked = recover_fingerprint(strong_state, layout, projection)
    pure = projection.project(_center_vector(projection.back_project(_signs(signature)), layout))
    np.testing.assert_allclose(y_marked, y_clean + STRONG_ALPHA * pure, atol=1e-6)


def _center_vector(vector: np.ndarray, layout) -> np.ndarray:
    out = vector.copy()
    for start, size in zip(layout.offsets, layout.sizes):
        out[start : start + size] -= out[start : start + size].mean()
    return out


def test_centering_costs_the_watermark_only_crosstalk(layout, projection, signature):
    s = _signs(signature)
    recovered = projection.project(_center_vector(projection.back_project(s), layout))
    assert np.max(np.abs(recovered - s)) < 7 * math.sqrt(127 / layout.dim)


def test_detection_grows_with_alpha(clean_state, layout, projection, signature):
    correlations = []
    for alpha in (0.0, 0.01, 0.05, 0.2):
        state, _ = embed_weight_watermark(clean_state, layout, projection, signature, alpha)
        correlations.append(extract_weight_watermark(state, layout, projection, signature).correlation)
    assert correlations == sorted(correlations)


# --- no watermark, wrong key, wrong owner (unit level; P3.4 and P3.7 measure) --


def test_unwatermarked_model_is_near_zero(clean_state, layout, projection, signature):
    result = extract_weight_watermark(clean_state, layout, projection, signature)
    assert abs(result.correlation) * math.sqrt(128) < 6


def test_wrong_key_does_not_detect(strong_state, layout, signature):
    wrong = derive_carrier_projection(OTHER_KEY, layout)
    for sig in (signature, derive_signature(OTHER_KEY, TEST_OWNER)):
        result = extract_weight_watermark(strong_state, layout, wrong, sig)
        assert abs(result.correlation) * math.sqrt(128) < 6


def test_wrong_owner_does_not_detect(strong_state, layout, projection):
    other = derive_signature(TEST_KEY, "someone else")
    result = extract_weight_watermark(strong_state, layout, projection, other)
    # y is about alpha*S, so the correlation is about <S, S'>/128 for an unrelated S'.
    assert abs(result.correlation) * math.sqrt(128) < 6


# --- invariances -------------------------------------------------------------


@pytest.mark.parametrize("factor", [0.01, 3.0])
def test_global_positive_rescaling_leaves_correlation_unchanged(strong_state, layout, projection, signature, factor):
    base = extract_weight_watermark(strong_state, layout, projection, signature)
    scaled = {k: (v * factor if k in layout.names else v) for k, v in strong_state.items()}
    result = extract_weight_watermark(scaled, layout, projection, signature)
    assert result.correlation == pytest.approx(base.correlation, abs=1e-6)
    assert result.bit_matches == base.bit_matches
    assert result.amplitude == pytest.approx(factor * base.amplitude, rel=1e-5)


def test_negation_flips_the_correlation(strong_state, layout, projection, signature):
    base = extract_weight_watermark(strong_state, layout, projection, signature)
    negated = {k: (-v if k in layout.names else v) for k, v in strong_state.items()}
    result = extract_weight_watermark(negated, layout, projection, signature)
    assert result.correlation == pytest.approx(-base.correlation, abs=1e-6)


def test_centering_removes_per_tensor_constant_shifts(clean_state, strong_state, layout, projection, signature):
    shifts = {name: 0.01 * (i + 1) * (-1) ** i for i, name in enumerate(layout.names)}
    shifted = {k: (v + shifts[k] if k in shifts else v) for k, v in strong_state.items()}
    base = recover_fingerprint(strong_state, layout, projection)
    np.testing.assert_allclose(recover_fingerprint(shifted, layout, projection), base, atol=1e-6)
    # Without centring the shifts leak in.
    raw_base = recover_fingerprint(strong_state, layout, projection, center=False)
    raw_shifted = recover_fingerprint(shifted, layout, projection, center=False)
    assert np.max(np.abs(raw_shifted - raw_base)) > 1e-3


def test_non_carrier_tensors_are_ignored(strong_state, layout, projection, signature):
    base = extract_weight_watermark(strong_state, layout, projection, signature)
    changed = {k: (v if k in layout.names or not v.is_floating_point() else v + 1.0) for k, v in strong_state.items()}
    result = extract_weight_watermark(changed, layout, projection, signature)
    np.testing.assert_array_equal(result.projected, base.projected)


def test_half_precision_weights_extract_the_same_bits(strong_state, layout, projection, signature):
    base = extract_weight_watermark(strong_state, layout, projection, signature)
    half = {k: (v.half() if v.is_floating_point() else v) for k, v in strong_state.items()}
    result = extract_weight_watermark(half, layout, projection, signature)
    assert result.bit_matches == base.bit_matches
    assert result.correlation == pytest.approx(base.correlation, abs=1e-3)


def test_is_deterministic(strong_state, layout, projection, signature):
    a = extract_weight_watermark(strong_state, layout, projection, signature)
    b = extract_weight_watermark(strong_state, layout, derive_carrier_projection(TEST_KEY, layout), signature)
    np.testing.assert_array_equal(a.projected, b.projected)
    assert a.to_dict() == b.to_dict()


# --- validation and hygiene ----------------------------------------------------


def test_result_hides_sensitive_fields(strong_state, layout, projection, signature):
    result = extract_weight_watermark(strong_state, layout, projection, signature)
    record = result.to_dict()
    assert set(record) == {
        "version", "rows", "dim", "carrier_digest", "centered",
        "correlation", "amplitude", "projected_rms", "bit_matches",
    }
    assert record["version"] == EXTRACTION_VERSION and record["carrier_digest"] == layout.digest()
    assert record["centered"] is True
    text = repr(result)
    assert "hidden" in text and signature.hex() not in text
    assert not result.projected.flags.writeable


def test_rejects_projection_of_wrong_dim(strong_state, layout, signature):
    with pytest.raises(ValueError):
        extract_weight_watermark(strong_state, layout, derive_projection(TEST_KEY, 1000), signature)


def test_rejects_signature_of_wrong_length(strong_state, layout, signature):
    short = derive_carrier_projection(TEST_KEY, layout, rows=64)
    with pytest.raises(ValueError):
        extract_weight_watermark(strong_state, layout, short, signature)


def test_rejects_non_signature(signature):
    with pytest.raises(TypeError):
        score_fingerprint(np.ones(128), tuple(signature.signs))


def test_rejects_non_finite_weights(strong_state, layout, projection, signature):
    broken = dict(strong_state)
    w = strong_state["features.0.weight"].clone()
    w[0, 0, 0, 0] = float("nan")
    broken["features.0.weight"] = w
    with pytest.raises(ValueError):
        extract_weight_watermark(broken, layout, projection, signature)


def test_rejects_other_architecture(layout, projection, signature):
    with pytest.raises((KeyError, ValueError)):
        extract_weight_watermark(main_model(width=16).state_dict(), layout, projection, signature)
