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
from src.training import (  # noqa: E402
    TrainConfig,
    clear_checkpoints,
    evaluate,
    fit,
    load_latest,
    restore,
    save_checkpoint,
)
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


def _smoke_fit(tmp_path, epochs, *, max_minutes=None, resume=True, seed=11, **kwargs):
    """A fresh process's worth of setup, as a Colab re-run would do it."""
    set_seed(seed)
    loaders = cifar10_loaders(batch_size=64, num_workers=0, smoke=True, seed=11)
    model = main_model(width=4)
    config = TrainConfig(epochs=epochs, max_minutes=max_minutes, log_every=0, **kwargs)
    return model, fit(
        model,
        loaders["train"],
        loaders["test"],
        config,
        checkpoint_dir=tmp_path,
        device=torch.device("cpu"),
        seed=seed,
        resume=resume,
    )


def _without_timing(history):
    return [{k: v for k, v in h.items() if k != "epoch_seconds"} for h in history]


def test_fit_runs_and_checkpoints_every_epoch(tmp_path):
    _, summary = _smoke_fit(tmp_path, epochs=2)
    assert summary["epochs_completed"] == 2
    assert len(summary["history"]) == 2
    assert [h["epoch"] for h in summary["history"]] == [0, 1]
    assert 0.0 <= summary["final"]["accuracy"] <= 1.0
    assert load_latest(tmp_path)["epoch"] == 1


def test_interrupted_run_matches_uninterrupted_run(tmp_path):
    """The resume guarantee, tested rather than assumed.

    One run goes straight through. The other is cut off by the time budget
    after every single epoch and restarted from scratch each time, the way a
    Colab re-run would. They must end with identical weights, optimizer state
    and per-epoch metrics. This caught a real bug: without per-epoch shuffle
    reseeding, every resumed epoch replayed epoch 0's data order.
    """
    epochs = 4
    reference_model, reference = _smoke_fit(tmp_path / "straight", epochs)

    for _ in range(epochs):
        model, summary = _smoke_fit(tmp_path / "interrupted", epochs, max_minutes=0.0)
    assert summary["epochs_completed"] == epochs and not summary["stopped_early"]

    assert [h["epoch"] for h in summary["history"]] == list(range(epochs))
    assert _without_timing(summary["history"]) == _without_timing(reference["history"])
    for (name, a), b in zip(reference_model.state_dict().items(), model.state_dict().values()):
        assert torch.equal(a, b), f"{name} diverged after resuming"

    straight_opt = load_latest(tmp_path / "straight")["optimizer"]["state"]
    interrupted_opt = load_latest(tmp_path / "interrupted")["optimizer"]["state"]
    for key, state in straight_opt.items():
        assert torch.equal(state["momentum_buffer"], interrupted_opt[key]["momentum_buffer"])


def test_history_records_the_lr_each_epoch_actually_used(tmp_path):
    _, summary = _smoke_fit(tmp_path, epochs=4, lr=0.1)
    lrs = [h["lr"] for h in summary["history"]]
    assert lrs[0] == pytest.approx(0.1), "epoch 0 trains at the initial LR"
    assert lrs == sorted(lrs, reverse=True)


def test_resuming_with_different_hyperparameters_is_refused(tmp_path):
    """restore() would silently keep the checkpoint's LR and T_max."""
    _smoke_fit(tmp_path, epochs=2, max_minutes=0.0)
    with pytest.raises(ValueError, match="epochs"):
        _smoke_fit(tmp_path, epochs=4)
    with pytest.raises(ValueError, match="lr"):
        _smoke_fit(tmp_path, epochs=2, lr=0.05)
    with pytest.raises(ValueError, match="seed"):
        _smoke_fit(tmp_path, epochs=2, seed=12)


def test_changing_only_the_time_budget_still_resumes(tmp_path):
    _smoke_fit(tmp_path, epochs=2, max_minutes=0.0)
    _, summary = _smoke_fit(tmp_path, epochs=2, max_minutes=500.0)
    assert summary["epochs_completed"] == 2


def test_starting_over_removes_stale_checkpoints(tmp_path):
    """A leftover slot from an old run must not be resumable into a new one."""
    _smoke_fit(tmp_path, epochs=3)
    _smoke_fit(tmp_path, epochs=3, max_minutes=0.0, resume=False)
    # The restarted run wrote only epoch 0, into ckpt_a. The old run's ckpt_b
    # (epoch 1) must be gone, or a corrupt ckpt_a would fall back to it.
    assert not (tmp_path / "ckpt_b.pt").exists()
    assert load_latest(tmp_path)["epoch"] == 0


def test_clear_checkpoints_leaves_other_files_alone(tmp_path):
    model, _ = _tiny_setup()
    save_checkpoint(tmp_path, epoch=0, model=model, is_best=True)
    (tmp_path / "notes.txt").write_text("keep me")
    assert sorted(clear_checkpoints(tmp_path)) == ["best.pt", "ckpt_a.pt", "manifest.json"]
    assert [p.name for p in tmp_path.iterdir()] == ["notes.txt"]


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


# --- the entry point --------------------------------------------------------


def _p0_5_script(monkeypatch):
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("p0_5_train_clean")


def test_entry_point_smoke_run_writes_result_and_weights(tmp_path, monkeypatch):
    """The exact command the notebook's pre-flight cell runs, end to end."""
    import hashlib

    script = _p0_5_script(monkeypatch)
    record = script.main(["--smoke", "--drive-root", str(tmp_path), "--device", "cpu"])

    metrics = record["metrics"]
    assert metrics["epochs_completed"] == 2 and not metrics["stopped_early"]
    weights = tmp_path / "models" / "p0.5_smoke_W.pt"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == metrics["weights_sha256"]
    assert metrics["total_epoch_seconds"] >= 0

    # A re-run after completion resumes, trains nothing, and the weights it
    # re-exports are the same model.
    again = script.main(["--smoke", "--drive-root", str(tmp_path), "--device", "cpu"])
    assert again["metrics"]["test_accuracy"] == metrics["test_accuracy"]
    exported = torch.load(weights, map_location="cpu")
    final_checkpoint = load_latest(tmp_path / "checkpoints" / "p0.5_smoke")["model"]
    assert exported.keys() == final_checkpoint.keys()
    assert all(torch.equal(v, final_checkpoint[k]) for k, v in exported.items())


def test_entry_point_refuses_an_unmounted_drive_path(monkeypatch):
    script = _p0_5_script(monkeypatch)
    monkeypatch.setattr(script.os.path, "ismount", lambda p: False)
    with pytest.raises(RuntimeError, match="not mounted"):
        script.main(["--smoke", "--drive-root", "/content/drive/MyDrive/zk-crown"])
