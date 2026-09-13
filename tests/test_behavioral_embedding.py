"""Tests for P2.3: the trigger bundle, the trigger mixer, and the entry point.

Every key here is a TEST KEY, public by construction.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import CIFAR10_MEAN, CIFAR10_STD, cifar10_loaders
from src.models import main_model
from src.training import TrainConfig, fit, load_latest
from src.utils.seeding import set_seed, torch_generator
from src.watermark.behavioral import TriggerMixLoader, trigger_metrics, trigger_tensors
from src.watermark.bundle import (
    TriggerBundle,
    check_against_dataset,
    load_bundle,
    make_bundle,
    save_bundle,
)
from src.watermark.responses import trigger_responses
from src.watermark.triggers import generate_triggers

TEST_KEY = bytes(range(32))
"""TEST KEY ONLY."""

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dataset(count: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (count, 32, 32, 3), dtype=np.uint8), rng.integers(0, 10, count)


def _bundle(n: int = 20, key: bytes = TEST_KEY, pool=range(300)) -> tuple[TriggerBundle, np.ndarray, np.ndarray]:
    images, labels = _dataset()
    ts = generate_triggers(key, images, pool, n=n)
    return make_bundle(ts, trigger_responses(key, ts, labels), key_kind="test"), images, labels


# --- bundle -----------------------------------------------------------------


def test_bundle_round_trips_with_the_same_digest(tmp_path):
    bundle, _, _ = _bundle()
    digest = save_bundle(bundle, tmp_path / "b.npz")
    loaded = load_bundle(tmp_path / "b.npz", expected_digest=digest)
    assert loaded.digest() == digest
    assert np.array_equal(loaded.images, bundle.images) and np.array_equal(loaded.targets, bundle.targets)
    assert loaded.key_kind == "test" and not list(tmp_path.glob("*.tmp"))


def test_digest_is_pinned_to_the_documented_encoding():
    import json
    import struct

    bundle, _, _ = _bundle()
    header = json.dumps(bundle.header(), sort_keys=True, separators=(",", ":")).encode()
    h = hashlib.sha256(b"zk-crown/trigger-bundle/v1\x00" + struct.pack(">I", len(header)) + header)
    h.update(bundle.images.tobytes())
    for array in (bundle.base_indices, bundle.base_labels, bundle.targets):
        h.update(array.astype(">i8").tobytes())
    assert bundle.digest() == h.hexdigest()


def test_digest_changes_with_any_field():
    bundle, _, _ = _bundle()
    other_key, _, _ = _bundle(key=bytes(31) + b"\x01")
    assert other_key.digest() != bundle.digest()
    targets = bundle.targets.copy()
    targets[0] = next(c for c in range(10) if c not in (targets[0], bundle.base_labels[0]))
    changed = TriggerBundle(**{**bundle.__dict__, "targets": targets})
    assert changed.digest() != bundle.digest()
    assert TriggerBundle(**{**bundle.__dict__, "key_kind": "owner"}).digest() != bundle.digest()


def test_wrong_expected_digest_is_rejected(tmp_path):
    bundle, _, _ = _bundle()
    save_bundle(bundle, tmp_path / "b.npz")
    with pytest.raises(ValueError, match="expected"):
        load_bundle(tmp_path / "b.npz", expected_digest="00" * 32)


def test_tampered_file_is_rejected(tmp_path):
    bundle, _, _ = _bundle()
    save_bundle(bundle, tmp_path / "b.npz")
    with np.load(tmp_path / "b.npz") as data:
        fields = {k: data[k] for k in data.files}
    fields["images"] = fields["images"].copy()
    fields["images"][0, 0, 0, 0] ^= 1
    np.savez(tmp_path / "t.npz", **fields)
    with pytest.raises(ValueError, match="digest"):
        load_bundle(tmp_path / "t.npz")


def test_bundle_rejects_a_target_equal_to_its_base_label():
    bundle, _, _ = _bundle()
    with pytest.raises(ValueError):
        TriggerBundle(**{**bundle.__dict__, "targets": bundle.base_labels.copy()})


def test_repr_does_not_dump_the_triggers():
    bundle, _, _ = _bundle()
    assert len(repr(bundle)) < 200


def test_check_against_dataset_accepts_the_source_and_catches_mismatches():
    bundle, images, labels = _bundle(pool=range(0, 300, 2))
    check_against_dataset(bundle, images, labels, range(0, 300, 2))
    with pytest.raises(ValueError, match="pool"):
        check_against_dataset(bundle, images, labels, range(1, 300, 2))
    with pytest.raises(ValueError, match="labels"):
        check_against_dataset(bundle, images, (labels + 1) % 10, range(300))
    with pytest.raises(ValueError, match="amplitude"):
        check_against_dataset(bundle, np.roll(images, 1, axis=0), labels, range(300))


# --- trigger tensors and the mixer ------------------------------------------


def test_trigger_tensors_match_the_clean_pipeline():
    from src.data.cifar10 import _transforms
    from PIL import Image

    bundle, _, _ = _bundle()
    x, y = trigger_tensors(bundle, CIFAR10_MEAN, CIFAR10_STD)
    expected = torch.stack([_transforms(augment=False)(Image.fromarray(img)) for img in bundle.images])
    assert x.shape == (20, 3, 32, 32) and x.dtype == torch.float32
    assert torch.allclose(x, expected, atol=1e-6)
    assert y.dtype == torch.int64 and y.tolist() == bundle.targets.tolist()


def _clean_loader(n: int = 100, batch: int = 16, seed: int = 3) -> DataLoader:
    g = torch.Generator().manual_seed(0)
    data = TensorDataset(torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g))
    return DataLoader(data, batch_size=batch, shuffle=True, generator=torch_generator(seed))


def _triggers(n: int = 10):
    g = torch.Generator().manual_seed(1)
    return torch.randn(n, 3, 32, 32, generator=g) + 100.0, torch.arange(n) % 10


def test_clean_batches_are_unchanged_by_the_mixer():
    tx, ty = _triggers()
    plain, mixed = _clean_loader(), TriggerMixLoader(_clean_loader(), tx, ty, triggers_per_batch=3, seed=5)
    assert len(mixed) == len(plain)
    for epoch in range(3):
        plain.generator.manual_seed(7 + epoch)
        mixed.generator.manual_seed(7 + epoch)
        for (px, py), (mx, my) in zip(plain, mixed):
            b = px.shape[0]
            assert mx.shape[0] == b + 3
            assert torch.equal(mx[:b], px) and torch.equal(my[:b], py)
            assert (mx[b:] > 50).all()  # the appended samples are triggers


def test_every_trigger_is_seen_equally_often_per_epoch():
    tx, ty = _triggers(10)
    mixed = TriggerMixLoader(_clean_loader(n=160, batch=16), tx, ty, triggers_per_batch=4)
    order = mixed.trigger_order()
    assert len(order) == 40
    assert np.bincount(order.numpy(), minlength=10).tolist() == [4] * 10


def test_trigger_order_depends_only_on_seed_and_epoch():
    tx, ty = _triggers()
    a = TriggerMixLoader(_clean_loader(), tx, ty, triggers_per_batch=3)
    b = TriggerMixLoader(_clean_loader(), tx, ty, triggers_per_batch=3)
    a.generator.manual_seed(11)
    first = a.trigger_order()
    a.trigger_order()  # consume another epoch's worth
    b.generator.manual_seed(11)
    assert torch.equal(b.trigger_order(), first)
    a.generator.manual_seed(12)
    assert not torch.equal(a.trigger_order(), first)


@pytest.mark.parametrize("m, exc", [(0, ValueError), (11, ValueError), (2.0, TypeError)])
def test_mixer_rejects_bad_triggers_per_batch(m, exc):
    tx, ty = _triggers(10)
    with pytest.raises(exc):
        TriggerMixLoader(_clean_loader(), tx, ty, triggers_per_batch=m)


def test_joint_training_embeds_the_triggers(tmp_path):
    """The mechanism works: on learnable clean data, the model fits both tasks.

    Clean labels are a function of the image (the sign of its mean), and each
    trigger is a clean-looking image plus fixed noise with an arbitrary label.
    After training, the model gets the triggers right in eval mode.
    """
    set_seed(0)
    g = torch.Generator().manual_seed(0)
    clean_x = torch.randn(256, 3, 32, 32, generator=g)
    clean_y = (clean_x.mean(dim=(1, 2, 3)) > 0).long()
    loader = DataLoader(TensorDataset(clean_x, clean_y), batch_size=32, shuffle=True, generator=torch_generator(0))
    tx = torch.randn(8, 3, 32, 32, generator=g) * 1.5
    ty = torch.randint(2, 10, (8,), generator=g)
    mixed = TriggerMixLoader(loader, tx, ty, triggers_per_batch=4, seed=0)

    model = main_model(width=8)
    before = trigger_metrics(model, tx, ty, torch.device("cpu"))["trigger_accuracy"]
    config = TrainConfig(epochs=25, lr=0.05, warmup_epochs=1.0, max_minutes=None, log_every=0)
    cpu = torch.device("cpu")
    summary = fit(
        model, mixed, loader, config, checkpoint_dir=tmp_path, device=cpu, seed=0,
        epoch_metrics=lambda m: trigger_metrics(m, tx, ty, cpu),
    )
    assert before < 0.5
    assert summary["history"][-1]["trigger_accuracy"] == 1.0
    assert "trigger_loss" in summary["history"][0]


def _mixed_smoke_fit(checkpoint_dir, max_minutes):
    """A fresh process's worth of setup, as a Colab re-run would do it."""
    tx, ty = _triggers(8)
    set_seed(4)
    loaders = cifar10_loaders(batch_size=64, num_workers=0, smoke=True, seed=4)
    mixed = TriggerMixLoader(loaders["train"], tx, ty, triggers_per_batch=2, seed=4)
    model = main_model(width=4)
    config = TrainConfig(epochs=3, max_minutes=max_minutes, log_every=0)
    cpu = torch.device("cpu")
    summary = fit(
        model, mixed, loaders["test"], config, checkpoint_dir=checkpoint_dir, device=cpu, seed=4,
        epoch_metrics=lambda m: trigger_metrics(m, tx, ty, cpu),
    )
    return model, summary


def test_interrupted_mixed_run_matches_uninterrupted_run(tmp_path):
    """Trigger order and the per-epoch trigger metrics survive being cut off after every epoch."""
    straight_model, straight = _mixed_smoke_fit(tmp_path / "straight", None)
    for _ in range(3):
        model, summary = _mixed_smoke_fit(tmp_path / "interrupted", 0.0)
    assert summary["epochs_completed"] == 3 and not summary["stopped_early"]

    def strip(history):
        return [{k: v for k, v in r.items() if k != "epoch_seconds"} for r in history]

    assert strip(summary["history"]) == strip(straight["history"])
    assert all("trigger_accuracy" in r for r in summary["history"])
    for a, b in zip(straight_model.state_dict().values(), model.state_dict().values()):
        assert torch.equal(a, b)


def test_epoch_metrics_may_not_overwrite_the_record(tmp_path):
    loaders = cifar10_loaders(batch_size=64, num_workers=0, smoke=True)
    with pytest.raises(ValueError, match="overwrite"):
        fit(
            main_model(width=4), loaders["train"], loaders["test"], TrainConfig(epochs=1, log_every=0),
            checkpoint_dir=tmp_path, device=torch.device("cpu"), epoch_metrics=lambda m: {"eval_accuracy": 1.0},
        )


# --- entry points -----------------------------------------------------------


def _script(monkeypatch, name):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module(name)


def test_entry_point_smoke_run_writes_result_and_weights(tmp_path, monkeypatch):
    script = _script(monkeypatch, "p2_3_train_watermarked")
    record = script.main(["--smoke", "--drive-root", str(tmp_path), "--device", "cpu"])

    m, p = record["metrics"], record["params"]
    assert m["epochs_completed"] == 2 and not m["stopped_early"]
    weights = tmp_path / "models" / "p2.3_smoke_W_star.pt"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == m["weights_sha256"]
    assert all("trigger_accuracy" in h for h in m["history"])
    assert p["trigger_bundle"]["sha256"] == script.smoke_bundle().digest()
    assert p["trigger_bundle"]["key_kind"] == "test"
    assert p["extra"]["triggers_per_batch"] == p["triggers_per_batch"] == 4

    # No secret material in the result: the bundle appears only as its header and digest.
    assert set(p["trigger_bundle"]) == set(script.smoke_bundle().header()) | {"n", "sha256"}
    text = next((tmp_path / "results").glob("p2.3_smoke__*.json")).read_text()
    assert "base_indices" not in text and "base_labels" not in text and '"targets"' not in text

    again = script.main(["--smoke", "--drive-root", str(tmp_path), "--device", "cpu"])
    assert again["metrics"]["test_accuracy"] == m["test_accuracy"]
    exported = torch.load(weights, map_location="cpu")
    final = load_latest(tmp_path / "checkpoints" / "p2.3_smoke")["model"]
    assert all(torch.equal(v, final[k]) for k, v in exported.items())


def test_real_run_requires_a_verified_owner_bundle(tmp_path, monkeypatch):
    script = _script(monkeypatch, "p2_3_train_watermarked")
    parser = script.build_parser()
    with pytest.raises(SystemExit, match="needs"):
        script.load_real_bundle(parser.parse_args([]))
    test_bundle = tmp_path / "b.npz"
    digest = save_bundle(script.smoke_bundle(), test_bundle)
    args = parser.parse_args(["--trigger-bundle", str(test_bundle), "--expected-trigger-sha256", digest])
    with pytest.raises(SystemExit, match="'test'-key"):
        script.load_real_bundle(args)
    args = parser.parse_args(["--trigger-bundle", str(test_bundle), "--expected-trigger-sha256", "ab" * 32])
    with pytest.raises(ValueError, match="expected"):
        script.load_real_bundle(args)


def test_entry_point_refuses_an_unmounted_drive_path(monkeypatch):
    script = _script(monkeypatch, "p2_3_train_watermarked")
    monkeypatch.setattr("src.utils.artifacts.os.path.ismount", lambda p: False)
    with pytest.raises(RuntimeError, match="not mounted"):
        script.main(["--smoke", "--drive-root", "/content/drive/MyDrive/zk-crown"])


def test_make_master_key_refuses_to_overwrite(tmp_path, monkeypatch):
    script = _script(monkeypatch, "make_master_key")
    path = tmp_path / "secrets" / "K.bin"
    script.create_key(path)
    key = script.load_key(path)
    assert len(key) == 32
    with pytest.raises(FileExistsError):
        script.create_key(path)
    assert script.load_key(path) == key
    other = tmp_path / "other.bin"
    script.create_key(other)
    assert script.load_key(other) != key
