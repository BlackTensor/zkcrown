"""Tests for P2.7: sparse trigger mixing and the ratio sweep entry point.

Every key here is a TEST KEY, public by construction.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seeding import torch_generator
from src.watermark.behavioral import TriggerMixLoader

REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_loader(n: int = 160, batch: int = 16, seed: int = 3) -> DataLoader:
    g = torch.Generator().manual_seed(0)
    data = TensorDataset(torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g))
    return DataLoader(data, batch_size=batch, shuffle=True, generator=torch_generator(seed))


def _triggers(n: int = 10):
    g = torch.Generator().manual_seed(1)
    return torch.randn(n, 3, 32, 32, generator=g) + 100.0, torch.arange(n) % 10


# --- trigger_every ----------------------------------------------------------


@pytest.mark.parametrize("every", [2, 3, 16])
def test_only_every_kth_batch_carries_triggers(every):
    tx, ty = _triggers()
    plain = _clean_loader(n=170)
    mixed = TriggerMixLoader(_clean_loader(n=170), tx, ty, triggers_per_batch=2, trigger_every=every)
    plain.generator.manual_seed(9)
    mixed.generator.manual_seed(9)
    carried = 0
    for b, ((px, py), (mx, my)) in enumerate(zip(plain, mixed)):
        n = px.shape[0]
        assert torch.equal(mx[:n], px) and torch.equal(my[:n], py)
        if b % every == 0:
            assert mx.shape[0] == n + 2 and (mx[n:] > 50).all()
            carried += 1
        else:
            assert mx.shape[0] == n
    assert carried == mixed.trigger_batches_per_epoch() == math.ceil(len(plain) / every)
    assert mixed.trigger_samples_per_epoch() == 2 * carried


def test_every_one_draws_exactly_what_p2_3_drew():
    """trigger_every=1 must not change P2.3's trigger order or batches."""
    tx, ty = _triggers()
    default = TriggerMixLoader(_clean_loader(), tx, ty, triggers_per_batch=4, seed=5)
    explicit = TriggerMixLoader(_clean_loader(), tx, ty, triggers_per_batch=4, seed=5, trigger_every=1)
    for epoch in range(2):
        default.generator.manual_seed(7 + epoch)
        explicit.generator.manual_seed(7 + epoch)
        for (ax, ay), (bx, by) in zip(default, explicit):
            assert torch.equal(ax, bx) and torch.equal(ay, by)
    assert default.trigger_samples_per_epoch() == 4 * len(default)


def test_sparse_exposure_stays_balanced():
    tx, ty = _triggers(10)
    mixed = TriggerMixLoader(_clean_loader(n=1600, batch=16), tx, ty, triggers_per_batch=1, trigger_every=5)
    order = mixed.trigger_order()
    assert len(order) == 20
    assert np.bincount(order.numpy(), minlength=10).tolist() == [2] * 10


@pytest.mark.parametrize("every, exc", [(0, ValueError), (-1, ValueError), (2.0, TypeError), (True, TypeError)])
def test_rejects_bad_trigger_every(every, exc):
    tx, ty = _triggers()
    with pytest.raises(exc):
        TriggerMixLoader(_clean_loader(), tx, ty, triggers_per_batch=1, trigger_every=every)


# --- sweep definition -------------------------------------------------------


def _script(monkeypatch, name="p2_7_ratio_sweep"):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module(name)


def test_sweep_names_encode_the_ratio_in_hundredths_of_a_percent(monkeypatch):
    script = _script(monkeypatch)
    batches, clean = math.ceil(45_000 / 128), 45_000
    ratios = []
    for run in script.SWEEP:
        samples = run.triggers_per_batch * math.ceil(batches / run.trigger_every)
        ratio = samples / clean
        assert run.name == f"p2.7_r{round(ratio * 10_000):04d}"
        assert 1 <= run.triggers_per_batch <= 100
        ratios.append(ratio)
    assert ratios == sorted(ratios) and len(set(ratios)) == len(ratios)
    p2_3_ratio = 4 * batches / clean
    assert p2_3_ratio not in ratios and min(ratios) < p2_3_ratio < max(ratios)


def test_unknown_run_is_refused(monkeypatch):
    script = _script(monkeypatch)
    with pytest.raises(SystemExit, match="unknown run"):
        script.main(["--run", "p2.7_nope", "--smoke"])


# --- entry point ------------------------------------------------------------


def test_smoke_run_records_the_sparse_ratio_and_task(tmp_path, monkeypatch):
    script = _script(monkeypatch)
    [record] = script.main(["--run", "p2.7_r0005", "--smoke", "--drive-root", str(tmp_path), "--device", "cpu"])
    p, m = record["params"], record["metrics"]
    assert record["task"] == "P2.7" and record["name"] == "p2.7_r0005_smoke"
    assert p["triggers_per_batch"] == 1 and p["trigger_every"] == 16
    assert p["extra"]["trigger_every"] == 16
    # the smoke train set is 512 images in 4 batches of 128: one batch in 16 is batch 0 only
    assert p["trigger_samples_per_epoch"] == 1 and p["trigger_fraction_of_full_batch"] is None
    assert p["trigger_to_clean_ratio"] == pytest.approx(1 / 512)
    weights = tmp_path / "models" / "p2.7_r0005_smoke_W_star.pt"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == m["weights_sha256"]
    assert not m["stopped_early"]


def test_p2_3_record_shape_is_unchanged_for_every_batch_mixing(tmp_path, monkeypatch):
    script = _script(monkeypatch, "p2_3_train_watermarked")
    record = script.main(["--smoke", "--drive-root", str(tmp_path), "--device", "cpu"])
    p = record["params"]
    assert record["task"] == "P2.3" and record["name"] == "p2.3_smoke"
    assert "trigger_every" not in p["extra"]  # so P2.3's resume config is what it was
    assert p["trigger_every"] == 1 and p["trigger_fraction_of_full_batch"] == 4 / 132


def test_completed_runs_are_found_only_with_matching_weights(tmp_path, monkeypatch):
    script = _script(monkeypatch)
    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    weights = tmp_path / "models" / "p2.7_r0020_W_star.pt"
    weights.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()

    def write(stamp, stopped, sha):
        body = {"metrics": {"stopped_early": stopped, "weights_sha256": sha}}
        (tmp_path / "results" / f"p2.7_r0020__seed1337__{stamp}.json").write_text(json.dumps(body))

    assert script.completed_record(tmp_path, "p2.7_r0020", 1337) is None
    write("20260101T000000+0000", True, None)
    assert script.completed_record(tmp_path, "p2.7_r0020", 1337) is None
    write("20260102T000000+0000", False, "00" * 32)
    assert script.completed_record(tmp_path, "p2.7_r0020", 1337) is None
    write("20260103T000000+0000", False, good)
    assert script.completed_record(tmp_path, "p2.7_r0020", 1337)["metrics"]["weights_sha256"] == good
    assert script.completed_record(tmp_path, "p2.7_r0020", 42) is None


def test_all_skips_runs_already_complete(tmp_path, monkeypatch):
    script = _script(monkeypatch)
    calls = []
    fake = {"metrics": {"stopped_early": False}}
    monkeypatch.setattr(script, "completed_record", lambda root, name, seed: fake if name != "p2.7_r0020" else None)
    monkeypatch.setattr(script, "train", lambda args, run_name, task: calls.append((run_name, args.trigger_every)) or fake)
    records = script.main(["--run", "all", "--drive-root", str(tmp_path)])
    assert calls == [("p2.7_r0020", 4)] and len(records) == len(script.SWEEP)


def test_all_stops_after_a_run_that_ran_out_of_time(tmp_path, monkeypatch):
    script = _script(monkeypatch)
    calls = []
    monkeypatch.setattr(script, "completed_record", lambda root, name, seed: None)

    def fake_train(args, run_name, task):
        calls.append(run_name)
        return {"metrics": {"stopped_early": True}}

    monkeypatch.setattr(script, "train", fake_train)
    script.main(["--run", "all", "--drive-root", str(tmp_path)])
    assert calls == ["p2.7_r0001"]


# --- local analysis ---------------------------------------------------------


def test_stable_from_epoch(monkeypatch):
    analyze = _script(monkeypatch, "p2_7_analyze_sweep")
    cases = [
        ([0.1, 1.0, 0.9, 1.0, 1.0], 3),
        ([1.0, 1.0], 0),
        ([1.0, 0.99], None),
        ([], None),
    ]
    for accs, expected in cases:
        history = [{"epoch": i, "trigger_accuracy": a} for i, a in enumerate(accs)]
        assert analyze.stable_from_epoch(history) == expected


def test_newest_complete_and_hash_check(tmp_path, monkeypatch):
    analyze = _script(monkeypatch, "p2_7_analyze_sweep")
    with pytest.raises(SystemExit, match="no complete result"):
        analyze.newest_complete(tmp_path, "p2.7_r0020", 1337)
    (tmp_path / "p2.7_r0020__seed1337__20260101T000000+0000.json").write_text(
        json.dumps({"metrics": {"stopped_early": False, "weights_sha256": "ab" * 32}})
    )
    (tmp_path / "p2.7_r0020__seed1337__20260102T000000+0000.json").write_text(
        json.dumps({"metrics": {"stopped_early": True, "weights_sha256": None}})
    )
    record, weights = analyze.newest_complete(tmp_path, "p2.7_r0020", 1337)
    assert record["metrics"]["weights_sha256"] == "ab" * 32 and weights.name == "p2.7_r0020_W_star.pt"

    path = tmp_path / "w.pt"
    from src.models import main_model

    torch.save(main_model(width=32).state_dict(), path)
    assert analyze.load_checked(path, analyze.sha256_file(path)) is not None
    with pytest.raises(SystemExit, match="recorded"):
        analyze.load_checked(path, "00" * 32)
    with pytest.raises(SystemExit, match="missing"):
        analyze.load_checked(tmp_path / "nope.pt", "00" * 32)


def test_measure_point_on_a_tiny_model(monkeypatch):
    analyze = _script(monkeypatch, "p2_7_analyze_sweep")
    from src.data import cifar10_loaders
    from src.models import main_model
    from src.training import per_sample_correct
    from src.watermark.responses import trigger_responses
    from src.watermark.triggers import generate_triggers

    key = bytes(range(32))  # TEST KEY ONLY
    rng = np.random.default_rng(0)
    images, labels = rng.integers(0, 256, (200, 32, 32, 3), dtype=np.uint8), rng.integers(0, 10, 200)
    ts = generate_triggers(key, images, range(200), n=20)
    responses = trigger_responses(key, ts, labels)
    loader = cifar10_loaders(num_workers=0, smoke=True)["test"]
    torch.manual_seed(0)
    clean, model = main_model(width=4), main_model(width=4)
    cpu = torch.device("cpu")
    clean_correct = per_sample_correct(clean, loader, cpu)

    record = {
        "task": "P2.7",
        "params": {"triggers_per_batch": 1, "trigger_every": 4, "trigger_samples_per_epoch": 88, "train_size": 45000},
        "metrics": {"test_accuracy": -1.0, "history": [{"epoch": 0, "trigger_accuracy": 1.0}]},
    }
    kw = dict(trigger_images=ts.images, responses=responses, clean_correct=clean_correct, test_loader=loader, device=cpu)
    row = analyze.measure_point("run", record, model, **kw)
    assert row["trigger_to_clean_ratio"] == pytest.approx(88 / 45000)
    assert row["n_triggers"] == 20 and 0 <= row["wdr"] <= 1
    assert row["matches_recorded"] is False and row["training_trigger_accuracy_stable_from_epoch"] == 0
    base = analyze.measure_point("W_clean", None, clean, **kw)
    assert base["accuracy_drop_pp"] == 0 and base["mcnemar_exact_p"] == 1.0 and base["matches_recorded"] is None
    assert base["trigger_to_clean_ratio"] == 0 and base["source_task"] == "P0.5"


def test_plot_sweep_writes_a_figure(tmp_path, monkeypatch):
    analyze = _script(monkeypatch, "p2_7_analyze_sweep")
    rows = []
    for i, (ratio, task) in enumerate([(0.0, "P0.5"), (0.0002, "P2.7"), (0.0313, "P2.3"), (0.125, "P2.7")]):
        rows.append({
            "trigger_to_clean_ratio": ratio, "source_task": task, "wdr": i / 3,
            "accuracy_drop_pp": 0.1 * i, "drop_ci95_pp": [0.1 * i - 0.5, 0.1 * i + 0.5],
        })
    path = tmp_path / "fig.png"
    analyze.plot_sweep(rows, path)
    assert path.stat().st_size > 10_000 and path.read_bytes()[:4] == b"\x89PNG"
