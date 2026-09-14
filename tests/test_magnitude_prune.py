"""Tests for P4.2: unstructured magnitude pruning, layer-wise and global.

Seeded untrained models only; no trained weights, no key. Covered: exact
counts, that exactly the smallest magnitudes go, that everything else is
bit-identical, tie handling, the two scopes behaving differently, validation,
and that both attacks run through the P4.1 harness and their configs parse.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.attacks import harness, prune  # noqa: E402
from src.attacks.harness import AttackConfig, AttackContext, apply_attack, expand_config  # noqa: E402
from src.models import main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.watermark.carrier import carrier_layout  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCH = {"width": 32}
SPARSITIES = [0.0, 0.1, 0.5, 0.9]


@pytest.fixture(scope="module")
def state():
    set_seed(4321)
    model = main_model()
    # non-trivial BN buffers and affine parameters, so "untouched" is a real check
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.1, 0.1)
                m.running_mean.uniform_(-1, 1)
                m.running_var.uniform_(0.5, 2)
    return model.state_dict()


@pytest.fixture(scope="module")
def layout():
    return carrier_layout(main_model(**ARCH))


@pytest.mark.parametrize("sparsity", SPARSITIES)
def test_layerwise_prunes_exactly_round_s_n_per_tensor(state, layout, sparsity):
    out, info = prune.magnitude_prune(state, ARCH, sparsity, "layerwise")
    for name, size in zip(layout.names, layout.sizes):
        zeros = int((out[name] == 0).sum())
        assert zeros == round(sparsity * size)
        assert info["per_tensor"][name]["pruned"] == zeros
    assert info["zeroed"] == sum(round(sparsity * s) for s in layout.sizes)
    assert info["achieved_sparsity"] == pytest.approx(sparsity, abs=1e-4)


@pytest.mark.parametrize("sparsity", SPARSITIES)
def test_global_prunes_exactly_round_s_total(state, layout, sparsity):
    out, info = prune.magnitude_prune(state, ARCH, sparsity, "global")
    total_zeros = sum(int((out[name] == 0).sum()) for name in layout.names)
    assert total_zeros == round(sparsity * layout.dim) == info["zeroed"]


@pytest.mark.parametrize("scope", prune.SCOPES)
@pytest.mark.parametrize("sparsity", [0.1, 0.5, 0.9])
def test_only_the_smallest_magnitudes_go_and_the_rest_is_bit_identical(state, layout, scope, sparsity):
    out, info = prune.magnitude_prune(state, ARCH, sparsity, scope)
    pruned_max, kept_min = [], []
    for name in layout.names:
        before, after = state[name], out[name]
        kept = after != 0
        assert torch.equal(after[kept], before[kept])
        removed = (after == 0) & (before != 0)
        if removed.any():
            pruned_max.append(float(before[removed].abs().max()))
        if kept.any():
            kept_min.append(float(before[kept].abs().min()))
        t = info["per_tensor"][name]
        if t["largest_pruned_abs"] is not None and t["smallest_kept_abs"] is not None:
            assert t["largest_pruned_abs"] <= t["smallest_kept_abs"]
            if removed.any() and kept.any():
                assert before[removed].abs().max() <= before[kept].abs().min()
    if scope == "global":
        assert max(pruned_max) <= min(kept_min)
        assert info["largest_pruned_abs"] <= info["smallest_kept_abs"]
    for name, tensor in state.items():
        if name not in layout.names:
            assert torch.equal(out[name], tensor), name


def test_the_input_is_not_modified(state):
    before = {k: v.clone() for k, v in state.items()}
    prune.magnitude_prune(state, ARCH, 0.7, "global")
    assert all(torch.equal(state[k], v) for k, v in before.items())


def test_ties_are_pruned_in_index_order():
    values = np.array([0.5, -0.5, 0.5, 0.1, -0.5])
    assert prune._prune_order(values, 3).tolist() == [3, 0, 1]


def test_global_and_layerwise_differ_across_layers(state, layout):
    """Global uses one threshold, so per-tensor sparsities spread out; layer-wise keeps them equal."""
    _, lw = prune.magnitude_prune(state, ARCH, 0.5, "layerwise")
    _, gl = prune.magnitude_prune(state, ARCH, 0.5, "global")
    lw_s = [lw["per_tensor"][n]["sparsity"] for n in layout.names]
    gl_s = [gl["per_tensor"][n]["sparsity"] for n in layout.names]
    assert max(lw_s) - min(lw_s) < 1e-3
    assert max(gl_s) - min(gl_s) > 0.05


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, float("nan"), True])
def test_sparsity_is_validated(state, bad):
    with pytest.raises(ValueError):
        prune.magnitude_prune(state, ARCH, bad, "layerwise")


def test_unknown_scope_is_refused(state):
    with pytest.raises(ValueError):
        prune.magnitude_prune(state, ARCH, 0.5, "structured")


@pytest.mark.parametrize("name", ["magnitude_prune_layerwise", "magnitude_prune_global"])
def test_registered_and_runs_through_the_harness(state, tmp_path, name):
    context = AttackContext(torch.device("cpu"), 1, tmp_path)
    out = apply_attack(AttackConfig(name, 0.3), state, ARCH, context)
    assert out.arch == ARCH and out.info["scope"] in name
    assert out.info["fine_tuned"] is False and out.info["bn_recalibrated"] is False
    again = apply_attack(AttackConfig(name, 0.3), state, ARCH, context)
    assert all(torch.equal(out.state_dict[k], again.state_dict[k]) for k in state)
    with pytest.raises(ValueError):
        apply_attack(AttackConfig(name, 0.3, {"iterations": 3}), state, ARCH, context)
    json.dumps(out.info, allow_nan=False)


@pytest.mark.parametrize("scope", prune.SCOPES)
def test_p4_2_configs_are_the_declared_sweep(scope):
    path = REPO_ROOT / "experiments" / "configs" / f"p4.2_magnitude_prune_{scope}.json"
    configs = expand_config(json.loads(path.read_text(encoding="utf-8")))
    assert [c.strength for c in configs] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert all(c.attack == f"magnitude_prune_{scope}" and c.params == {} for c in configs)
    assert all(c.attack in harness.available_attacks() for c in configs)
