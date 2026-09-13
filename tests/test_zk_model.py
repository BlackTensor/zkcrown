"""Tests for zk_model, the MNIST pipeline and the P0.7 entry point.

They check the properties Phase 8 relies on: under the section 2.2
parameter budget, a small ReLU count, only circuit-friendly op types, and
identical behaviour in train and eval mode.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.data.mnist import mnist_loaders  # noqa: E402
from src.models import (  # noqa: E402
    MNIST_INPUT_SHAPE,
    ZK_PARAM_BUDGET,
    ZKModel,
    activation_elements,
    count_parameters,
    zk_model,
)
from src.utils.seeding import set_seed  # noqa: E402


# --- shape and budget -------------------------------------------------------


def test_forward_shape_and_batch_of_one():
    model = zk_model().eval()
    with torch.no_grad():
        assert model(torch.randn(4, *MNIST_INPUT_SHAPE)).shape == (4, 10)
        assert model(torch.randn(1, *MNIST_INPUT_SHAPE)).shape == (1, 10)


def test_default_parameter_count_is_under_budget():
    counts = count_parameters(zk_model())
    assert counts["total"] < ZK_PARAM_BUDGET
    # Pinned so an accidental architecture change is caught rather than
    # silently changing the number in section 8.2.
    assert counts["total"] == 6138


def test_relu_element_count():
    """Strided convs keep the non-linearities cheap: 8*14*14 + 16*7*7 + 16*4*4."""
    assert activation_elements(zk_model()) == 8 * 14 * 14 + 16 * 7 * 7 + 16 * 4 * 4


def test_activation_elements_leaves_mode_and_hooks_alone():
    model = zk_model().train()
    activation_elements(model)
    assert model.training
    assert all(not m._forward_hooks for m in model.modules())


def test_only_circuit_friendly_modules():
    """No BatchNorm, dropout or pooling: see the zk_model docstring."""
    allowed = (ZKModel, nn.Sequential, nn.Conv2d, nn.ReLU, nn.Flatten, nn.Linear)
    for module in zk_model().modules():
        assert isinstance(module, allowed), f"unexpected module {type(module).__name__}"


def test_train_and_eval_mode_compute_the_same_function():
    set_seed(0)
    model = zk_model()
    x = torch.randn(8, *MNIST_INPUT_SHAPE)
    with torch.no_grad():
        train_out = model.train()(x)
        eval_out = model.eval()(x)
    assert torch.equal(train_out, eval_out)


def test_widths_can_shrink_for_p8_8():
    assert count_parameters(zk_model(widths=(4, 8, 8)))["total"] < count_parameters(zk_model())["total"]
    assert zk_model(widths=(4, 8)).eval()(torch.randn(2, *MNIST_INPUT_SHAPE)).shape == (2, 10)


def test_rejects_bad_widths():
    with pytest.raises(ValueError):
        ZKModel(widths=())
    with pytest.raises(ValueError):
        ZKModel(widths=(8, 0))


def test_init_is_deterministic_and_state_dict_order_stable():
    set_seed(5)
    a = zk_model()
    set_seed(5)
    b = zk_model()
    assert list(a.state_dict()) == list(b.state_dict())
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_initial_logits_are_small():
    """Large early logits are what killed the first P0.5 run."""
    set_seed(0)
    with torch.no_grad():
        out = zk_model().eval()(torch.randn(64, *MNIST_INPUT_SHAPE))
    assert out.std().item() < 1.0


# --- data -------------------------------------------------------------------


def test_smoke_loaders_have_the_right_shapes():
    loaders = mnist_loaders(batch_size=16, num_workers=0, smoke=True)
    images, labels = next(iter(loaders["train"]))
    assert images.shape[1:] == MNIST_INPUT_SHAPE
    assert labels.dtype == torch.int64
    assert set(loaders) == {"train", "test"}


# --- the entry point --------------------------------------------------------


def _script(monkeypatch):
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("p0_7_train_zk_model")


def test_entry_point_smoke_run_writes_result_and_weights(tmp_path, monkeypatch):
    """The exact command the notebook's pre-flight cell runs, end to end."""
    import hashlib

    from src.training import load_latest

    script = _script(monkeypatch)
    record = script.main(["--smoke", "--drive-root", str(tmp_path), "--device", "cpu"])

    metrics, params = record["metrics"], record["params"]
    assert metrics["epochs_completed"] == 2 and not metrics["stopped_early"]
    assert params["total_params"] == 6138 and params["relu_elements_per_input"] == 2608
    assert "system_ram_gb" in record["environment"]

    weights = tmp_path / "models" / "p0.7_zk_model_smoke.pt"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == metrics["weights_sha256"]
    exported = torch.load(weights, map_location="cpu", weights_only=True)
    ZKModel().load_state_dict(exported, strict=True)

    # A re-run after completion resumes, trains nothing, and exports the same model.
    again = script.main(["--smoke", "--drive-root", str(tmp_path), "--device", "cpu"])
    assert again["metrics"]["test_accuracy"] == metrics["test_accuracy"]
    final_checkpoint = load_latest(tmp_path / "checkpoints" / "p0.7_zk_model_smoke")["model"]
    exported = torch.load(weights, map_location="cpu", weights_only=True)
    assert all(torch.equal(v, final_checkpoint[k]) for k, v in exported.items())


def test_entry_point_refuses_a_model_over_budget(tmp_path, monkeypatch):
    script = _script(monkeypatch)
    with pytest.raises(ValueError, match="budget"):
        script.main(["--smoke", "--drive-root", str(tmp_path), "--device", "cpu", "--widths", "32,64,64"])


def test_entry_point_refuses_an_unmounted_drive_path(monkeypatch):
    import src.utils.artifacts as artifacts

    script = _script(monkeypatch)
    monkeypatch.setattr(artifacts.os.path, "ismount", lambda p: False)
    with pytest.raises(RuntimeError, match="not mounted"):
        script.main(["--smoke", "--drive-root", "/content/drive/MyDrive/zk-crown"])


def test_p0_7_defaults(monkeypatch):
    args = _script(monkeypatch).build_parser().parse_args([])
    assert (args.epochs, args.lr, args.warmup_epochs, args.widths) == (20, 0.05, 1.0, (8, 16, 16))
