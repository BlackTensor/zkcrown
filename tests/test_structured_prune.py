"""Tests for P4.3: structured filter pruning by L1 norm, zero-masked.

Seeded untrained models with non-trivial BatchNorm state; no trained weights,
no key. Covered: exact channel counts, that the lowest-L1 filters go, exactly
which entries are zeroed and that everything else is bit-identical, tie
handling, output equivalence to a physically narrower network, the narrow
parameter count, validation, and that the attack runs through the P4.1 harness
and its config parses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.attacks import harness, structured_prune  # noqa: E402
from src.attacks.harness import AttackConfig, AttackContext, apply_attack, expand_config, load_model  # noqa: E402
from src.models import main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.watermark.carrier import carrier_layout  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCH = {"width": 32}
CONVS = ["features.0", "features.3", "features.7", "features.10", "features.14", "features.17"]
BNS = ["features.1", "features.4", "features.8", "features.11", "features.15", "features.18"]
CONSUMERS = CONVS[1:] + ["classifier.2"]
FRACTIONS = [0.0, 0.05, 0.1, 0.5, 0.9]


def _randomise_bn(model: nn.Module) -> None:
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.5, 0.5)
                m.running_mean.uniform_(-1, 1)
                m.running_var.uniform_(0.5, 2)


@pytest.fixture(scope="module")
def state():
    set_seed(2468)
    model = main_model(**ARCH)
    _randomise_bn(model)
    return model.state_dict()


def _removed_channels(out: dict, conv: str) -> list[int]:
    w = out[f"{conv}.weight"]
    return [i for i in range(w.shape[0]) if bool((w[i] == 0).all())]


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_each_layer_loses_round_s_c_channels(state, fraction):
    out, info = structured_prune.channel_prune(state, ARCH, fraction)
    total = 0
    for conv in CONVS:
        channels = state[f"{conv}.weight"].shape[0]
        expected = min(round(fraction * channels), channels - 1)
        assert len(_removed_channels(out, conv)) == expected
        assert info["per_layer"][conv]["removed"] == expected
        total += expected
    assert info["channels_removed"] == total
    assert info["channels"] == 32 * 2 + 64 * 2 + 128 * 2


@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.9])
def test_the_lowest_l1_filters_are_removed(state, fraction):
    out, info = structured_prune.channel_prune(state, ARCH, fraction)
    for conv in CONVS:
        norms = state[f"{conv}.weight"].double().abs().flatten(1).sum(1)
        removed = _removed_channels(out, conv)
        kept = [i for i in range(len(norms)) if i not in removed]
        assert norms[removed].max() <= norms[kept].min()
        assert info["per_layer"][conv]["largest_removed_l1"] == pytest.approx(float(norms[removed].max()))
        assert info["per_layer"][conv]["smallest_kept_l1"] == pytest.approx(float(norms[kept].min()))


@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.9])
def test_exactly_the_removed_slices_are_zeroed_and_the_rest_is_bit_identical(state, fraction):
    out, _ = structured_prune.channel_prune(state, ARCH, fraction)
    expected = {name: torch.ones_like(t, dtype=torch.bool) for name, t in state.items()}
    for conv, bn, consumer in zip(CONVS, BNS, CONSUMERS):
        removed = _removed_channels(out, conv)
        for i in removed:
            expected[f"{conv}.weight"][i] = False
            expected[f"{bn}.weight"][i] = False
            expected[f"{bn}.bias"][i] = False
            if consumer == "classifier.2":
                expected[f"{consumer}.weight"][:, i * 16 : (i + 1) * 16] = False
            else:
                expected[f"{consumer}.weight"][:, i] = False
    for name, before in state.items():
        after, keep = out[name], expected[name]
        assert torch.equal(after[keep], before[keep]), name
        assert bool((after[~keep] == 0).all()), name
    assert all(torch.equal(out[n], state[n]) for n in state if "running" in n or n == "classifier.2.bias")


def test_the_input_is_not_modified(state):
    before = {k: v.clone() for k, v in state.items()}
    structured_prune.channel_prune(state, ARCH, 0.7)
    assert all(torch.equal(state[k], v) for k, v in before.items())


def test_ties_are_removed_in_index_order():
    norms = np.array([2.0, 1.0, 1.0, 3.0, 1.0])
    assert structured_prune._removal_order(norms, 2).tolist() == [1, 2]


def test_no_layer_is_emptied_when_rounding_would_remove_every_channel():
    set_seed(7)
    arch = {"width": 1}  # layers of 1, 1, 2, 2, 4, 4 channels
    model = main_model(**arch)
    out, info = structured_prune.channel_prune(model.state_dict(), arch, 0.9)
    for conv in CONVS:
        p = info["per_layer"][conv]
        assert p["removed"] == p["channels"] - 1
        assert len(_removed_channels(out, conv)) == p["channels"] - 1
    assert info["per_layer"]["features.0"]["capped"] is True


def _narrow_network(state: dict, out: dict) -> nn.Module:
    """The same model with removed channels physically deleted, built by slicing."""
    layers: list[nn.Module] = []
    in_keep = torch.arange(3)
    for block, (conv, bn) in enumerate(zip(CONVS, BNS)):
        removed = set(_removed_channels(out, conv))
        keep = torch.tensor([i for i in range(state[f"{conv}.weight"].shape[0]) if i not in removed])
        c = nn.Conv2d(len(in_keep), len(keep), 3, padding=1, bias=False)
        b = nn.BatchNorm2d(len(keep))
        with torch.no_grad():
            c.weight.copy_(state[f"{conv}.weight"][keep][:, in_keep])
            for attr in ("weight", "bias", "running_mean", "running_var"):
                getattr(b, attr).copy_(state[f"{bn}.{attr}"][keep])
        layers += [c, b, nn.ReLU()]
        if block % 2 == 1:
            layers.append(nn.MaxPool2d(2, 2))
        in_keep = keep
    columns = (in_keep[:, None] * 16 + torch.arange(16)[None, :]).reshape(-1)
    linear = nn.Linear(len(columns), 10)
    with torch.no_grad():
        linear.weight.copy_(state["classifier.2.weight"][:, columns])
        linear.bias.copy_(state["classifier.2.bias"])
    return nn.Sequential(*layers, nn.Flatten(), linear).eval()


@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.9])
def test_zero_masked_model_matches_the_physically_narrower_network(state, fraction):
    out, info = structured_prune.channel_prune(state, ARCH, fraction)
    masked = load_model(out, ARCH)
    narrow = _narrow_network(state, out)
    assert sum(p.numel() for p in narrow.parameters()) == info["equivalent_narrow_params"]
    torch.manual_seed(0)
    x = torch.randn(64, 3, 32, 32)
    with torch.no_grad():
        a, b = masked(x), narrow(x)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    assert torch.equal(a.argmax(1), b.argmax(1))


def test_source_param_count_and_zero_fraction_are_consistent(state):
    out, info = structured_prune.channel_prune(state, ARCH, 0.0)
    assert info["source_params"] == info["equivalent_narrow_params"] == 307_946
    assert info["channels_removed"] == 0 and info["carrier_zero_fraction"] == 0.0
    assert all(torch.equal(out[k], v) for k, v in state.items())
    out, info = structured_prune.channel_prune(state, ARCH, 0.5)
    layout = carrier_layout(main_model(**ARCH))
    zeros = sum(int((out[n] == 0).sum()) for n in layout.names)
    assert info["carrier_zero_fraction"] == zeros / layout.dim
    # middle layers lose both output filters and input slices, so more than half their weights
    assert info["carrier_zero_fraction_per_tensor"]["features.10.weight"] > 0.7


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, float("nan"), True])
def test_fraction_is_validated(state, bad):
    with pytest.raises(ValueError):
        structured_prune.channel_prune(state, ARCH, bad)


def test_registered_and_runs_through_the_harness(state, tmp_path):
    context = AttackContext(torch.device("cpu"), 1, tmp_path)
    out = apply_attack(AttackConfig("channel_prune_l1", 0.3), state, ARCH, context)
    assert out.arch == ARCH and out.info["version"] == structured_prune.STRUCTURED_PRUNE_VERSION
    assert out.info["fine_tuned"] is False and out.info["bn_recalibrated"] is False
    again = apply_attack(AttackConfig("channel_prune_l1", 0.3), state, ARCH, context)
    assert all(torch.equal(out.state_dict[k], again.state_dict[k]) for k in state)
    with pytest.raises(ValueError):
        apply_attack(AttackConfig("channel_prune_l1", 0.3, {"criterion": "l2"}), state, ARCH, context)
    json.dumps(out.info, allow_nan=False)


def test_p4_3_config_is_the_declared_sweep():
    path = REPO_ROOT / "experiments" / "configs" / "p4.3_channel_prune_l1.json"
    configs = expand_config(json.loads(path.read_text(encoding="utf-8")))
    assert [c.strength for c in configs] == [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert all(c.attack == "channel_prune_l1" and c.params == {} for c in configs)
    assert "channel_prune_l1" in harness.available_attacks()
