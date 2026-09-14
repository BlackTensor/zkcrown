"""Tests for the carrier layout and spread-spectrum weight embedding (P3.2).

Covered: the carrier is exactly the conv and linear weights in a fixed order;
the embedding applies ``W* = W + alpha * P_K^T * S`` to the carrier and nothing
else; the change is spread over the carrier the way the construction says;
the result loads into `main_model` and runs; inputs are validated.

Models are freshly initialised from a fixed seed, so no trained weights file
is needed. Statistical thresholds sit at least 6 standard deviations out.

All keys and owner ids here are TEST data, public by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.models import CIFAR10_INPUT_SHAPE, count_parameters, main_model, zk_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.watermark.carrier import CARRIER_VERSION, CarrierLayout, carrier_layout  # noqa: E402
from src.watermark.projection import derive_projection  # noqa: E402
from src.watermark.signature import OwnershipSignature, derive_signature  # noqa: E402
from src.watermark.weight_embedding import (  # noqa: E402
    EMBEDDING_VERSION,
    derive_carrier_projection,
    embed_weight_watermark,
    watermark_delta,
)

TEST_KEY = bytes(range(32))
"""TEST KEY ONLY."""

OTHER_KEY = bytes(range(100, 132))
"""TEST KEY ONLY."""

TEST_OWNER = "zk-crown test owner"

MAIN_CARRIER_NAMES = (
    "features.0.weight",
    "features.3.weight",
    "features.7.weight",
    "features.10.weight",
    "features.14.weight",
    "features.17.weight",
    "classifier.2.weight",
)

ALPHA = 0.05
"""A TEST value only. Choosing alpha is P3.5."""


@pytest.fixture(scope="module")
def model():
    set_seed(1234)
    return main_model().eval()


@pytest.fixture(scope="module")
def layout(model):
    return carrier_layout(model)


@pytest.fixture(scope="module")
def projection(layout):
    return derive_carrier_projection(TEST_KEY, layout)


@pytest.fixture(scope="module")
def signature():
    return derive_signature(TEST_KEY, TEST_OWNER)


@pytest.fixture(scope="module")
def embedded(model, layout, projection, signature):
    state = model.state_dict()
    snapshot = {k: v.clone() for k, v in state.items()}
    new_state, summary = embed_weight_watermark(state, layout, projection, signature, ALPHA)
    return snapshot, state, new_state, summary


# --- carrier layout ---------------------------------------------------------


def test_main_model_carrier_is_conv_and_linear_weights_in_module_order(model, layout):
    assert layout.names == MAIN_CARRIER_NAMES
    assert layout.dim == 307_040 == count_parameters(model)["conv_and_linear_weights"]
    assert layout.version == CARRIER_VERSION
    for name, shape in zip(layout.names, layout.shapes):
        assert tuple(model.state_dict()[name].shape) == shape


def test_carrier_excludes_batchnorm_biases_and_buffers(model, layout):
    modules = dict(model.named_modules())
    excluded = set(model.state_dict()) - set(layout.names)
    for key in excluded:
        owner = modules[key.rsplit(".", 1)[0]]
        assert isinstance(owner, nn.BatchNorm2d) or key == "classifier.2.bias", key
    assert "classifier.2.bias" in excluded
    assert count_parameters(model)["total"] - layout.dim == 906


def test_layout_is_generic_across_architectures():
    student = carrier_layout(main_model(width=16))
    assert student.dim == count_parameters(main_model(width=16))["conv_and_linear_weights"]
    zk = carrier_layout(zk_model())
    assert zk.dim == sum(m.weight.numel() for m in zk_model().modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    assert len({carrier_layout(main_model()).digest(), student.digest(), zk.digest()}) == 3


def test_layout_rejects_model_without_carrier():
    with pytest.raises(ValueError):
        carrier_layout(nn.Sequential(nn.ReLU()))


def test_flatten_is_c_order_concatenation(model, layout):
    state = model.state_dict()
    expected = torch.cat([state[n].reshape(-1) for n in MAIN_CARRIER_NAMES]).double().numpy()
    flat = layout.flatten(state)
    assert flat.dtype == np.float64
    np.testing.assert_array_equal(flat, expected)


def test_split_inverts_flatten(model, layout):
    state = model.state_dict()
    parts = layout.split(layout.flatten(state))
    assert list(parts) == list(layout.names)
    for name in layout.names:
        np.testing.assert_array_equal(parts[name], state[name].double().numpy())
    assert layout.offsets[0] == 0 and layout.offsets[-1] + layout.sizes[-1] == layout.dim


def test_digest_is_stable_and_sensitive(layout):
    assert layout.digest() == carrier_layout(main_model()).digest()
    renamed = CarrierLayout(names=layout.names[:-1] + ("classifier.9.weight",), shapes=layout.shapes)
    reshaped = CarrierLayout(names=layout.names, shapes=layout.shapes[:-1] + ((2048, 10),))
    reordered = CarrierLayout(names=layout.names[::-1], shapes=layout.shapes[::-1])
    assert len({layout.digest(), renamed.digest(), reshaped.digest(), reordered.digest()}) == 4


def test_check_rejects_mismatched_state_dicts(model, layout):
    state = dict(model.state_dict())
    missing = {k: v for k, v in state.items() if k != "features.3.weight"}
    with pytest.raises(KeyError):
        layout.check(missing)
    with pytest.raises(ValueError):
        layout.check({**state, "classifier.2.weight": torch.zeros(10, 2047)})
    with pytest.raises(TypeError):
        layout.check({**state, "classifier.2.weight": torch.zeros(10, 2048, dtype=torch.int32)})


@pytest.mark.parametrize(
    "names, shapes",
    [((), ()), (("a",), ()), (("a", "a"), ((1,), (1,))), (("a",), ((0,),)), (("a",), ((),))],
)
def test_layout_rejects_invalid_definitions(names, shapes):
    with pytest.raises(ValueError):
        CarrierLayout(names=names, shapes=shapes)


# --- the embedding formula ---------------------------------------------------


def test_delta_is_alpha_times_back_projected_signs(projection, signature):
    s = np.asarray(signature.signs, dtype=np.float64)
    delta = watermark_delta(projection, signature, ALPHA)
    reference = np.zeros(projection.dim)
    for i in range(projection.rows):
        reference += s[i] * projection.signs[i].astype(np.float64)
    reference *= ALPHA / math.sqrt(projection.dim)
    np.testing.assert_allclose(delta, reference, rtol=0, atol=1e-15)


def test_delta_takes_the_documented_discrete_values(projection, signature):
    delta = watermark_delta(projection, signature, 1.0)
    integer_sums = delta * math.sqrt(projection.dim)
    rounded = np.rint(integer_sums)
    np.testing.assert_allclose(integer_sums, rounded, atol=1e-9)
    assert np.all(rounded.astype(np.int64) % 2 == 0)
    assert np.max(np.abs(rounded)) <= 128


def test_delta_is_linear_in_alpha(projection, signature):
    a = watermark_delta(projection, signature, 0.25)
    b = watermark_delta(projection, signature, 1.0)
    np.testing.assert_allclose(4 * a, b, rtol=0, atol=1e-15)


def test_carrier_change_equals_delta_up_to_float32_rounding(embedded, layout, projection, signature):
    snapshot, _, new_state, summary = embedded
    change = layout.flatten(new_state) - layout.flatten(snapshot)
    delta = watermark_delta(projection, signature, ALPHA)
    err = float(np.max(np.abs(change - delta)))
    assert err == pytest.approx(summary.float32_rounding_max_abs, rel=0, abs=1e-12)
    assert err < 1e-6


def test_non_carrier_entries_are_bit_identical(embedded, layout):
    snapshot, _, new_state, _ = embedded
    assert list(new_state) == list(snapshot)
    for name in snapshot:
        if name not in layout.names:
            assert new_state[name].dtype == snapshot[name].dtype
            assert torch.equal(new_state[name], snapshot[name]), name


def test_input_state_dict_is_not_modified(embedded):
    snapshot, original, new_state, _ = embedded
    for name in snapshot:
        assert torch.equal(original[name], snapshot[name])
        assert new_state[name].data_ptr() != original[name].data_ptr()


def test_dtypes_are_preserved(embedded, layout):
    snapshot, _, new_state, _ = embedded
    for name in layout.names:
        assert new_state[name].dtype == snapshot[name].dtype == torch.float32


def test_alpha_zero_leaves_weights_unchanged(model, layout, projection, signature):
    state = model.state_dict()
    new_state, summary = embed_weight_watermark(state, layout, projection, signature, 0.0)
    for name in state:
        assert torch.equal(new_state[name], state[name])
    assert summary.delta_l2 == 0.0 and summary.float32_rounding_max_abs == 0.0


def test_watermarked_weights_load_and_run(embedded):
    _, _, new_state, _ = embedded
    fresh = main_model()
    fresh.load_state_dict(new_state, strict=True)
    fresh.eval()
    with torch.no_grad():
        out = fresh(torch.randn(2, *CIFAR10_INPUT_SHAPE))
    assert out.shape == (2, 10) and torch.isfinite(out).all()


def test_projected_change_recovers_signature_signs(embedded, layout, projection, signature):
    # Embedding correctness on the change itself, W* - W. Reading S from W* alone is P3.3.
    snapshot, _, new_state, _ = embedded
    y = projection.project(layout.flatten(new_state) - layout.flatten(snapshot)) / ALPHA
    s = np.asarray(signature.signs, dtype=np.float64)
    assert np.array_equal(np.sign(y), s)
    crosstalk_sd = math.sqrt((projection.rows - 1) / projection.dim)
    assert np.max(np.abs(y - s)) < 7 * crosstalk_sd


def test_is_deterministic(model, layout, signature):
    state = model.state_dict()
    a, _ = embed_weight_watermark(state, layout, derive_carrier_projection(TEST_KEY, layout), signature, ALPHA)
    b, _ = embed_weight_watermark(state, layout, derive_carrier_projection(TEST_KEY, layout), signature, ALPHA)
    for name in state:
        assert torch.equal(a[name], b[name])


def test_other_key_gives_uncorrelated_change(layout, projection):
    other_projection = derive_carrier_projection(OTHER_KEY, layout)
    ours = watermark_delta(projection, derive_signature(TEST_KEY, TEST_OWNER), 1.0)
    theirs = watermark_delta(other_projection, derive_signature(OTHER_KEY, TEST_OWNER), 1.0)
    corr = float(ours @ theirs) / (np.linalg.norm(ours) * np.linalg.norm(theirs))
    assert abs(corr * math.sqrt(layout.dim)) < 6  # corr ~ N(0, 1/dim) for independent changes


def test_other_owner_gives_uncorrelated_change(projection):
    ours = watermark_delta(projection, derive_signature(TEST_KEY, TEST_OWNER), 1.0)
    theirs = watermark_delta(projection, derive_signature(TEST_KEY, "someone else"), 1.0)
    # Same P_K, different S: correlation is (S . S') / 128 plus cross-talk, so
    # it is bounded by the bit agreement of two unrelated 128-bit strings.
    corr = float(ours @ theirs) / (np.linalg.norm(ours) * np.linalg.norm(theirs))
    assert abs(corr) < 6 / math.sqrt(128)


# --- spread ------------------------------------------------------------------


def test_change_size_matches_construction(embedded, projection):
    *_, summary = embedded
    d = projection.dim
    # Squared norm is alpha^2 * S^T (P P^T) S = alpha^2 * (128 + X), where X sums
    # 128 * 127 terms of sd 1 / sqrt(d), so sd(X) is about 0.23, under 0.2% of 128.
    # The 5% tolerance is far outside that.
    assert summary.delta_l2 == pytest.approx(ALPHA * math.sqrt(128), rel=0.05)
    assert summary.delta_rms == pytest.approx(ALPHA * math.sqrt(128 / d), rel=0.05)
    assert summary.delta_max_abs <= ALPHA * 128 / math.sqrt(d) + 1e-15


def test_every_tensor_gets_a_share_proportional_to_its_size(embedded, layout, projection):
    *_, summary = embedded
    d = layout.dim
    for stats, size in zip(summary.tensors, layout.sizes):
        # Per-entry squared change / E has variance about 2 (sum-of-signs kurtosis near 3),
        # so a tensor's energy share has sd about sqrt(2 * size) / d.
        expected = size / d
        assert abs(stats.energy_share - expected) < 6 * math.sqrt(2 * size) / d, stats.name
        assert stats.numel == size
    assert sum(t.energy_share for t in summary.tensors) == pytest.approx(1.0, abs=1e-12)


def test_no_parameter_dominates(projection, signature):
    delta = watermark_delta(projection, signature, 1.0)
    energy = np.sort(delta * delta)[::-1]
    total = float(energy.sum())
    # Largest z^2 over 307k near-Gaussian draws is typically about 22 to 25;
    # exceeding 50 (|z| > 7.07) has probability under 1e-6.
    assert energy[0] / total < 50 / projection.dim
    top_percent = float(energy[: projection.dim // 100].sum()) / total
    assert top_percent < 0.12  # Gaussian value is about 0.085


def test_zero_fraction_matches_binomial_centre(embedded, projection):
    *_, summary = embedded
    p0 = math.comb(128, 64) / 2**128
    sd = math.sqrt(p0 * (1 - p0) / projection.dim)
    assert abs(summary.zero_fraction - p0) < 6 * sd


# --- summary and validation ---------------------------------------------------


def test_summary_holds_aggregates_only(embedded, layout, signature):
    *_, summary = embedded
    record = summary.to_dict()
    assert record["version"] == EMBEDDING_VERSION
    assert record["carrier_digest"] == layout.digest()
    assert record["dim"] == 307_040 and record["rows"] == 128 and record["alpha"] == ALPHA
    assert [t["name"] for t in record["tensors"]] == list(MAIN_CARRIER_NAMES)
    text = repr(record)
    assert signature.hex() not in text


@pytest.mark.parametrize("alpha", [-0.1, float("nan"), float("inf"), True, "0.1", None])
def test_rejects_bad_alpha(model, layout, projection, signature, alpha):
    with pytest.raises((TypeError, ValueError)):
        embed_weight_watermark(model.state_dict(), layout, projection, signature, alpha)


def test_rejects_projection_of_wrong_dim(model, layout, signature):
    small = derive_projection(TEST_KEY, 1000)
    with pytest.raises(ValueError):
        embed_weight_watermark(model.state_dict(), layout, small, signature, ALPHA)


def test_rejects_projection_with_wrong_row_count(layout, signature):
    short = derive_carrier_projection(TEST_KEY, layout, rows=64)
    with pytest.raises(ValueError):
        watermark_delta(short, signature, ALPHA)


def test_rejects_non_signature(projection):
    with pytest.raises(TypeError):
        watermark_delta(projection, tuple([1] * 128), ALPHA)


def test_rejects_state_dict_of_other_architecture(layout, projection, signature):
    with pytest.raises((KeyError, ValueError)):
        embed_weight_watermark(main_model(width=16).state_dict(), layout, projection, signature, ALPHA)


def test_signature_type_is_the_p2_1_one(signature):
    assert isinstance(signature, OwnershipSignature) and len(signature.signs) == 128
