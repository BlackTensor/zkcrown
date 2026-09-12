"""Tests for the data split, checkpointing and the training loop (P0.5).

The checkpoint tests matter most. A resume bug only shows up after a Colab
session has already died, by which point the GPU time is spent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.data.cifar10 import (  # noqa: E402
    HOLDOUT_SIZE,
    TRAIN_SIZE,
    cifar10_loaders,
    cifar10_split_indices,
    denormalize,
)
from src.models import main_model  # noqa: E402
from src.training import TrainConfig, evaluate, fit, load_latest, restore, save_checkpoint  # noqa: E402
from src.training.checkpoint import MANIFEST_NAME  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402


# --- the data split ---------------------------------------------------------


def test_split_is_deterministic_and_disjoint():
    a_train, a_hold = cifar10_split_indices()
    b_train, b_hold = cifar10_split_indices()
    assert a_train == b_train and a_hold == b_hold
    assert len(a_train) == TRAIN_SIZE
    assert len(a_hold) == HOLDOUT_SIZE
    assert set(a_train).isdisjoint(a_hold), "attacker holdout leaks into training data"
    assert len(set(a_train) | set(a_hold)) == TRAIN_SIZE + HOLDOUT_SIZE


def test_split_ignores_the_global_rng():
    """P4.5 must see the same holdout whatever seed its experiment runs under."""
    set_seed(1)
    first = cifar10_split_indices()[1]
    set_seed(99999)
    torch.randn(1000)
    assert cifar10_split_indices()[1] == first


def test_split_rejects_too_small_a_dataset():
    with pytest.raises(ValueError):
        cifar10_split_indices(total=100)


def test_denormalize_inverts_normalization():
    from torchvision import transforms

    from src.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD

    pixels = torch.rand(4, 3, 32, 32)
    normalized = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(pixels)
    assert torch.allclose(denormalize(normalized), pixels, atol=1e-5)


def test_smoke_loaders_have_the_right_shapes():
    loaders = cifar10_loaders(batch_size=16, num_workers=0, smoke=True)
    images, labels = next(iter(loaders["train"]))
    assert images.shape[1:] == (3, 32, 32)
    assert labels.dtype == torch.int64
    assert set(loaders) == {"train", "holdout", "test"}


# --- checkpointing ----------------------------------------------------------


def _tiny_setup():
    model = main_model(width=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return model, optimizer


def test_checkpoint_round_trip(tmp_path):
    set_seed(3)
    model, optimizer = _tiny_setup()
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    optimizer.step()

    save_checkpoint(
        tmp_path, epoch=4, model=model, optimizer=optimizer, history=[{"epoch": 4}], seed=3
    )

    loaded = load_latest(tmp_path)
    assert loaded is not None
    fresh, fresh_opt = _tiny_setup()
    assert restore(loaded, fresh, fresh_opt) == 4
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(a, b)
    assert loaded["history"] == [{"epoch": 4}]


def test_writes_alternate_between_two_slots(tmp_path):
    model, _ = _tiny_setup()
    written = {save_checkpoint(tmp_path, epoch=e, model=model).name for e in range(4)}
    assert written == {"ckpt_a.pt", "ckpt_b.pt"}


def test_a_corrupt_current_slot_falls_back_to_the_other(tmp_path):
    """The whole reason for two slots: an interrupted Drive write."""
    model, _ = _tiny_setup()
    save_checkpoint(tmp_path, epoch=0, model=model)  # -> ckpt_a
    save_checkpoint(tmp_path, epoch=1, model=model)  # -> ckpt_b, now current

    (tmp_path / "ckpt_b.pt").write_bytes(b"truncated garbage")

    loaded = load_latest(tmp_path)
    assert loaded is not None, "lost the run to a single bad write"
    assert loaded["epoch"] == 0, "should have fallen back to the older slot"


def test_missing_manifest_still_finds_a_checkpoint(tmp_path):
    model, _ = _tiny_setup()
    save_checkpoint(tmp_path, epoch=7, model=model)
    (tmp_path / MANIFEST_NAME).unlink()
    loaded = load_latest(tmp_path)
    assert loaded is not None and loaded["epoch"] == 7


def test_load_latest_on_an_empty_dir_is_none(tmp_path):
    assert load_latest(tmp_path) is None
    assert load_latest(tmp_path / "nope") is None


def test_best_checkpoint_is_written_separately(tmp_path):
    model, _ = _tiny_setup()
    save_checkpoint(tmp_path, epoch=0, model=model, is_best=True)
    assert (tmp_path / "best.pt").is_file()


def test_no_temp_files_survive_a_save(tmp_path):
    model, _ = _tiny_setup()
    save_checkpoint(tmp_path, epoch=0, model=model, is_best=True)
    assert list(tmp_path.glob("*.tmp")) == []


# --- the loop ---------------------------------------------------------------


def _smoke_fit(tmp_path, epochs, **kwargs):
    set_seed(11)
    loaders = cifar10_loaders(batch_size=64, num_workers=0, smoke=True)
    model = main_model(width=4)
    config = TrainConfig(epochs=epochs, max_minutes=None, log_every=0, **kwargs)
    return model, fit(
        model,
        loaders["train"],
        loaders["test"],
        config,
        checkpoint_dir=tmp_path,
        device=torch.device("cpu"),
        seed=11,
    )


def test_fit_runs_and_checkpoints_every_epoch(tmp_path):
    _, summary = _smoke_fit(tmp_path, epochs=2)
    assert summary["epochs_completed"] == 2
    assert len(summary["history"]) == 2
    assert [h["epoch"] for h in summary["history"]] == [0, 1]
    assert 0.0 <= summary["final"]["accuracy"] <= 1.0
    assert load_latest(tmp_path)["epoch"] == 1


def test_fit_resumes_instead_of_restarting(tmp_path):
    _, first = _smoke_fit(tmp_path, epochs=2)
    assert first["epochs_completed"] == 2

    _, second = _smoke_fit(tmp_path, epochs=4)
    assert second["epochs_completed"] == 4
    # History carried across the restart rather than starting from zero.
    assert [h["epoch"] for h in second["history"]] == [0, 1, 2, 3]


def test_fit_on_an_already_finished_run_only_evaluates(tmp_path):
    _smoke_fit(tmp_path, epochs=2)
    _, again = _smoke_fit(tmp_path, epochs=2)
    assert again["epochs_completed"] == 2
    assert again["wall_seconds"] == 0.0
    assert not again["stopped_early"]


def test_time_budget_stops_early_and_says_so(tmp_path):
    set_seed(11)
    loaders = cifar10_loaders(batch_size=64, num_workers=0, smoke=True)
    model = main_model(width=4)
    config = TrainConfig(epochs=50, max_minutes=0.0, log_every=0)
    summary = fit(
        model,
        loaders["train"],
        loaders["test"],
        config,
        checkpoint_dir=tmp_path,
        device=torch.device("cpu"),
        seed=11,
    )
    assert summary["stopped_early"] is True
    assert summary["epochs_completed"] == 1
    assert load_latest(tmp_path) is not None, "must checkpoint before bailing out"


def test_evaluate_rejects_an_empty_loader():
    from torch.utils.data import DataLoader, TensorDataset

    empty = DataLoader(TensorDataset(torch.zeros(0, 3, 32, 32), torch.zeros(0, dtype=torch.long)))
    with pytest.raises(ValueError):
        evaluate(main_model(width=4), empty, torch.device("cpu"))
