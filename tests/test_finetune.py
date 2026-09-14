"""Tests for P4.5: fine-tuning on the attacker holdout, and the harness pieces it needs.

Synthetic smoke data, seeded untrained models, CPU only; no trained weights, no
`K`, no CIFAR-10. Covered: recipe validation, that the attack trains (weights
change, BN statistics update), is seeded, checkpoints to its work dir and
restores a finished run without retraining, refuses a checkpoint from other
starting weights, config tags and labels, apply mode end to end with skip of
completed configs, and the declared sweep.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.attacks import finetune, harness  # noqa: E402
from src.attacks.harness import AttackConfig, AttackContext, apply_attack, expand_config  # noqa: E402
from src.data.cifar10 import attacker_holdout_loaders  # noqa: E402
from src.models import main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCH = {"width": 32}
LR = {"lr": 0.01}


@pytest.fixture(scope="module")
def state():
    set_seed(99)
    return main_model(**ARCH).state_dict()


def smoke_context(tmp_path, *, seed=3, work_dir=None):
    return AttackContext(torch.device("cpu"), seed, tmp_path, work_dir=work_dir, smoke=True)


# --- recipe -----------------------------------------------------------------------------------


def test_recipe_defaults_are_the_p0_5_recipe():
    recipe = finetune.finetune_recipe(5, {"lr": 0.05})
    assert recipe == {"lr": 0.05, "weight_decay": 5e-4, "momentum": 0.9, "batch_size": 128, "warmup_epochs": 1.0,
                      "augment": True, "epochs": 5}


@pytest.mark.parametrize("strength,params", [
    (5, {}),                                  # no lr
    (5, {"lr": 0}),
    (5, {"lr": -0.1}),
    (5, {"lr": float("nan")}),
    (5, {"lr": 0.1, "optimizer": "adam"}),    # unknown
    (5, {"lr": 0.1, "batch_size": 0}),
    (5, {"lr": 0.1, "batch_size": 12.5}),
    (5, {"lr": 0.1, "augment": "yes"}),
    (5, {"lr": 0.1, "weight_decay": -1}),
    (0, {"lr": 0.1}),
    (2.5, {"lr": 0.1}),
])
def test_recipe_is_validated(strength, params):
    with pytest.raises(ValueError):
        finetune.finetune_recipe(strength, params)


def test_attacker_loaders_are_seeded_and_smoke_sized(tmp_path):
    loaders = attacker_holdout_loaders(tmp_path, smoke=True, seed=4)
    assert loaders["train"].generator is not None
    assert len(loaders["train"].dataset) == len(loaders["eval"].dataset) == 128


# --- the attack -----------------------------------------------------------------------------------


def test_finetune_trains_and_is_seeded(state, tmp_path):
    out = apply_attack(AttackConfig("finetune_holdout", 2, LR), state, ARCH, smoke_context(tmp_path))
    assert not torch.equal(out.state_dict["features.0.weight"], state["features.0.weight"])
    assert not torch.equal(out.state_dict["features.1.running_mean"], state["features.1.running_mean"])
    info = out.info
    assert info["recipe"]["epochs"] == 2 and info["recipe"]["lr"] == 0.01 and info["fine_tuned"] is True
    assert info["steps"] == 2 * 1 and info["train_images"] == 128
    assert info["last_epoch"]["epoch"] == 1 and info["start_state_sha256"] == finetune.state_digest(state)
    json.dumps(info, allow_nan=False)
    again = apply_attack(AttackConfig("finetune_holdout", 2, LR), state, ARCH, smoke_context(tmp_path))
    assert all(torch.equal(out.state_dict[k], again.state_dict[k]) for k in state)
    other_seed = apply_attack(AttackConfig("finetune_holdout", 2, LR), state, ARCH, smoke_context(tmp_path, seed=4))
    assert not torch.equal(out.state_dict["features.0.weight"], other_seed.state_dict["features.0.weight"])


def test_learning_rate_changes_the_result(state, tmp_path):
    small = apply_attack(AttackConfig("finetune_holdout", 1, {"lr": 0.001}), state, ARCH, smoke_context(tmp_path))
    large = apply_attack(AttackConfig("finetune_holdout", 1, {"lr": 0.1}), state, ARCH, smoke_context(tmp_path))
    delta = lambda out: float((out.state_dict["features.3.weight"] - state["features.3.weight"]).norm())  # noqa: E731
    assert delta(large) > delta(small)


def test_work_dir_checkpoints_and_a_finished_run_is_restored_not_retrained(state, tmp_path):
    work = tmp_path / "ckpt"
    config = AttackConfig("finetune_holdout", 2, LR)
    first = apply_attack(config, state, ARCH, smoke_context(tmp_path, work_dir=work))
    assert any(work.iterdir())
    restored = apply_attack(config, state, ARCH, smoke_context(tmp_path, work_dir=work))
    assert all(torch.equal(first.state_dict[k], restored.state_dict[k]) for k in state)
    assert restored.info["wall_seconds_this_session"] == 0.0


def test_a_checkpoint_from_other_starting_weights_is_refused(state, tmp_path):
    work = tmp_path / "ckpt"
    apply_attack(AttackConfig("finetune_holdout", 1, LR), state, ARCH, smoke_context(tmp_path, work_dir=work))
    set_seed(100)
    other = main_model(**ARCH).state_dict()
    with pytest.raises(ValueError, match="start_state_sha256|extra"):
        apply_attack(AttackConfig("finetune_holdout", 1, LR), other, ARCH, smoke_context(tmp_path, work_dir=work))


def test_state_digest_sees_any_change(state):
    changed = dict(state)
    changed["classifier.2.bias"] = state["classifier.2.bias"].clone()
    changed["classifier.2.bias"][0] += 1e-6
    assert finetune.state_digest(state) == finetune.state_digest(dict(state))
    assert finetune.state_digest(state) != finetune.state_digest(changed)


# --- tags and labels ------------------------------------------------------------------------------


def test_tag_goes_into_label_and_round_trips():
    config = expand_config({"attack": "finetune_holdout", "tag": "lr0.01", "strengths": [5, 20], "params": LR})[1]
    assert config.label() == "finetune_holdout_lr0.01_20"
    assert AttackConfig(**config.to_dict()) == config
    untagged = AttackConfig("none", 0)
    assert untagged.label() == "none_0" and "tag" not in untagged.to_dict()


@pytest.mark.parametrize("tag", ["", "lr 0.1", "lr/0.1", 5])
def test_bad_tags_are_refused(tag):
    with pytest.raises(ValueError):
        AttackConfig("finetune_holdout", 5, LR, tag=tag)


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("run_attack_suite")


def test_configs_sharing_a_label_are_refused(script, tmp_path):
    import argparse

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"attack": "finetune_holdout", "strengths": [5], "params": {"lr": 0.1}}), encoding="utf-8")
    b.write_text(json.dumps({"attack": "finetune_holdout", "strengths": [5], "params": {"lr": 0.01}}), encoding="utf-8")
    args = argparse.Namespace(config=[a, b], attack=None, strength=None, params=None)
    with pytest.raises(SystemExit):
        script.read_configs(args)


# --- apply mode end to end ------------------------------------------------------------------------


def test_apply_smoke_writes_weights_and_records(script, tmp_path):
    config = tmp_path / "ft.json"
    config.write_text(json.dumps({"attack": "finetune_holdout", "tag": "lr0.01", "strengths": [1, 2], "params": LR}),
                      encoding="utf-8")
    paths = script.main(["apply", "--config", str(config), "--task", "P4.5", "--smoke", "--out-dir", str(tmp_path / "out")])
    assert len(paths) == 2
    records = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    assert [r["params"]["config"]["tag"] for r in records] == ["lr0.01", "lr0.01"]
    assert all(r["params"]["smoke"] is True and r["params"]["key_used"] is False for r in records)
    assert all((tmp_path / "out" / "smoke" / r["metrics"]["weights_file"]).is_file() for r in records)
    assert (tmp_path / "out" / "smoke" / "checkpoints" / "p4.5_finetune_holdout_lr0.01_2__seed1337").is_dir()
    with pytest.raises(SystemExit):  # smoke records name no known source, so they can never be scored
        script.read_apply_record(Path(paths[0]), None)


def test_apply_skips_configs_already_complete(script, state, tmp_path, monkeypatch, capsys):
    source = {"name": "dual", "task": "TEST", "weights_sha256": "a" * 64, "arch": ARCH}
    monkeypatch.setattr(script, "load_source", lambda name, path: (state, source))
    monkeypatch.setattr(finetune, "attacker_holdout_loaders",
                        lambda root, **kw: attacker_holdout_loaders(root, **{**kw, "smoke": True}))
    config = tmp_path / "ft.json"
    config.write_text(json.dumps({"attack": "finetune_holdout", "tag": "lr0.01", "strengths": [1], "params": LR}),
                      encoding="utf-8")
    argv = ["apply", "--config", str(config), "--task", "P4.5", "--out-dir", str(tmp_path / "out")]
    first = script.main(argv)
    capsys.readouterr()
    second = script.main(argv)
    assert first == second
    assert "skipping finetune_holdout_lr0.01_1" in capsys.readouterr().out
    # a corrupted weights file is not "complete": the config is applied again
    # (record file names are per second, so compare output and hashes, not paths)
    weights = tmp_path / "out" / "p4.5_finetune_holdout_lr0.01_1__seed1337.pt"
    weights.write_bytes(b"corrupt")
    third = script.main(argv)
    out = capsys.readouterr().out
    assert "skipping" not in out and "applied finetune_holdout_lr0.01_1" in out
    record = json.loads(Path(third[0]).read_text(encoding="utf-8"))
    assert script.sha256_file(weights) == record["metrics"]["weights_sha256"]


def test_p4_5_configs_are_the_declared_grid():
    configs = []
    for lr in ("0.001", "0.01", "0.05", "0.1"):
        path = REPO_ROOT / "experiments" / "configs" / f"p4.5_finetune_lr{lr}.json"
        configs += expand_config(json.loads(path.read_text(encoding="utf-8")))
    assert len(configs) == 12 and len({c.label() for c in configs}) == 12
    assert sorted({c.params["lr"] for c in configs}) == [0.001, 0.01, 0.05, 0.1]
    assert all([c.strength for c in configs if c.params["lr"] == lr] == [5.0, 20.0, 60.0] for lr in (0.001, 0.01, 0.05, 0.1))
    for c in configs:
        assert c.attack == "finetune_holdout" and c.tag == f"lr{c.params['lr']:g}"
        assert finetune.finetune_recipe(c.strength, c.params) == {**finetune.DEFAULTS, "lr": c.params["lr"],
                                                                  "epochs": int(c.strength)}
    assert "finetune_holdout" in harness.available_attacks()
