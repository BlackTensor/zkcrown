"""Tests for main_model (P0.4).

These assert the properties later phases actually rely on: the parameter
budget from section 2.2, deterministic init from a seed, a stable state_dict
key order for fingerprinting (P5.1), fusable conv-BN-ReLU for quantization
(P4.4), and clean channel structure for structured pruning (P4.3).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.models import CIFAR10_INPUT_SHAPE, MainModel, count_parameters, main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402


# --- shape and budget -------------------------------------------------------


def test_forward_shape():
    model = main_model().eval()
    x = torch.randn(4, *CIFAR10_INPUT_SHAPE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10)


def test_forward_accepts_batch_of_one():
    """BatchNorm in train mode rejects a batch of 1; eval mode must not."""
    model = main_model().eval()
    with torch.no_grad():
        assert model(torch.randn(1, *CIFAR10_INPUT_SHAPE)).shape == (1, 10)


def test_parameter_budget_matches_section_2_2():
    counts = count_parameters(main_model())
    # "a few hundred thousand parameters" (CLAUDE.md 2.2).
    assert 200_000 < counts["total"] < 500_000
    assert counts["trainable"] == counts["total"]
    # The pool the weight watermark spreads into must dominate the model.
    assert counts["conv_and_linear_weights"] > 0.9 * counts["total"]


def test_width_scales_down_for_a_distillation_student():
    """P4.7 needs a smaller student of the same family."""
    big = count_parameters(main_model(width=32))["total"]
    small = count_parameters(main_model(width=16))["total"]
    assert small < big / 3


def test_outputs_are_logits_not_probabilities():
    set_seed(0)
    model = main_model().eval()
    with torch.no_grad():
        out = model(torch.randn(8, *CIFAR10_INPUT_SHAPE))
    sums = out.sum(dim=1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


# --- reproducibility --------------------------------------------------------


def test_init_is_deterministic_given_a_seed():
    set_seed(123)
    a = main_model()
    set_seed(123)
    b = main_model()
    for (name_a, pa), (name_b, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert name_a == name_b
        assert torch.equal(pa, pb), f"{name_a} differs between two seeded builds"


def test_different_seeds_give_different_init():
    set_seed(1)
    a = main_model()
    set_seed(2)
    b = main_model()
    assert not torch.equal(
        a.features[0].weight, b.features[0].weight
    ), "init ignored the seed"


def test_state_dict_key_order_is_stable_across_save_and_reload(tmp_path):
    """P5.1 hashes a canonical serialisation; key order must not drift."""
    set_seed(5)
    model = main_model()
    before = list(model.state_dict().keys())

    path = tmp_path / "w.pth"
    torch.save(model.state_dict(), path)
    reloaded = main_model()
    state = torch.load(path, map_location="cpu")
    reloaded.load_state_dict(state)

    assert list(state.keys()) == before
    assert list(reloaded.state_dict().keys()) == before
    for key in before:
        assert torch.equal(reloaded.state_dict()[key], model.state_dict()[key])


# --- structure the attack lab depends on ------------------------------------


def test_activations_are_modules_so_quantization_can_fuse_them():
    """P4.4 fuses conv-BN-ReLU, and fuse_modules only sees modules."""
    model = main_model()
    kinds = [type(m) for m in model.features]
    assert nn.ReLU in kinds, "no ReLU modules found; F.relu would break fusion"

    # Every Conv2d must be followed by BatchNorm2d then ReLU.
    for i, module in enumerate(model.features):
        if isinstance(module, nn.Conv2d):
            assert isinstance(model.features[i + 1], nn.BatchNorm2d)
            assert isinstance(model.features[i + 2], nn.ReLU)


def test_conv_bn_relu_triples_actually_fuse():
    """The P4.4 claim, checked rather than asserted in a docstring.

    Fusion must succeed for every triple and must not change the output, or
    post-training quantization starts from a different model than we trained.
    """
    import copy

    from torch.ao.quantization import fuse_modules

    model = main_model().eval()
    groups = [
        [f"features.{i}", f"features.{i + 1}", f"features.{i + 2}"]
        for i, module in enumerate(model.features)
        if isinstance(module, nn.Conv2d)
    ]
    assert len(groups) == 6

    fused = fuse_modules(copy.deepcopy(model), groups).eval()
    x = torch.randn(8, *CIFAR10_INPUT_SHAPE)
    with torch.no_grad():
        assert torch.allclose(model(x), fused(x), atol=1e-5)


def test_no_residual_connections():
    """P4.3 removes whole channels; skips would force channel counts to agree."""
    model = main_model()
    assert isinstance(model.features, nn.Sequential)
    assert isinstance(model.classifier, nn.Sequential)


def test_conv_channel_widths_are_the_documented_ladder():
    convs = [m for m in main_model().features if isinstance(m, nn.Conv2d)]
    assert [c.out_channels for c in convs] == [32, 32, 64, 64, 128, 128]
    assert convs[0].in_channels == 3


def test_gradients_reach_every_trainable_parameter():
    """A parameter with no gradient path cannot be trained or watermarked."""
    model = main_model()
    model(torch.randn(4, *CIFAR10_INPUT_SHAPE)).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == [], f"no gradient reached: {missing}"


# --- argument validation ----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{"width": 0}, {"width": -4}, {"dropout": 1.0}, {"dropout": -0.1}, {"num_classes": 1}],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MainModel(**kwargs)
