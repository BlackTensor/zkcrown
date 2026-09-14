"""Tests for P4.6: prune, then fine-tune with the pruning mask fixed.

Synthetic smoke data, seeded untrained models, CPU only; no trained weights, no
`K`, no CIFAR-10. Covered: the pruned-position masks both pruning attacks now
return, parameter validation, that fine-tuning moves kept weights while every
pruned entry stays exactly zero (and a removed channel's output stays zero),
that without the mask the pruned weights regrow, that a broken mask is caught,
seeding, checkpoint restore and refusal of a checkpoint from another pruning,
apply mode end to end, and the declared sweep.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.attacks import finetune, harness, prune, prune_finetune, structured_prune  # noqa: E402
from src.attacks.harness import AttackConfig, AttackContext, apply_attack, expand_config, load_model  # noqa: E402
from src.data.cifar10 import attacker_holdout_loaders  # noqa: E402
from src.models import main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.watermark.carrier import carrier_layout  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCH = {"width": 32}
BNS = ["features.1", "features.4", "features.8", "features.11", "features.15", "features.18"]
# 4 steps per smoke epoch, so momentum and weight decay act across several steps.
RECIPE = {"epochs": 2, "lr": 0.05, "batch_size": 32}


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
    set_seed(1357)
    model = main_model(**ARCH)
    _randomise_bn(model)
    return model.state_dict()


def smoke_context(tmp_path, *, seed=3, work_dir=None):
    return AttackContext(torch.device("cpu"), seed, tmp_path, work_dir=work_dir, smoke=True)


def config(prune_name, strength, **overrides):
    return AttackConfig("prune_finetune", strength, {"prune": prune_name, **RECIPE, **overrides})


# --- the masks the pruning attacks return ------------------------------------------------------


@pytest.mark.parametrize("scope", ["global", "layerwise"])
@pytest.mark.parametrize("sparsity", [0.0, 0.5, 0.9])
def test_magnitude_mask_is_exactly_the_pruned_set(state, scope, sparsity):
    out, info, mask = prune.magnitude_prune_with_mask(state, ARCH, sparsity, scope)
    plain_out, plain_info = prune.magnitude_prune(state, ARCH, sparsity, scope)
    assert plain_info == info and all(torch.equal(out[k], plain_out[k]) for k in state)
    layout = carrier_layout(main_model(**ARCH))
    assert sorted(mask) == sorted(layout.names)
    assert sum(int(m.sum()) for m in mask.values()) == info["zeroed"]
    for name, m in mask.items():
        assert m.dtype == torch.bool and m.shape == state[name].shape
        assert bool((out[name][m] == 0).all())
        assert torch.equal(out[name][~m], state[name][~m])


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.5, 0.9])
def test_channel_mask_is_exactly_the_zeroed_set(state, fraction):
    out, info, mask = structured_prune.channel_prune_with_mask(state, ARCH, fraction)
    plain_out, plain_info = structured_prune.channel_prune(state, ARCH, fraction)
    assert plain_info == info and all(torch.equal(out[k], plain_out[k]) for k in state)
    for name, tensor in state.items():
        m = mask.get(name, torch.zeros(tensor.shape, dtype=torch.bool))
        assert bool((out[name][m] == 0).all())
        assert torch.equal(out[name][~m], tensor[~m]), name
    if fraction:
        for bn, conv in zip(BNS, info["per_layer"]):
            removed = info["per_layer"][conv]["removed"]
            assert int(mask[f"{bn}.weight"].sum()) == int(mask[f"{bn}.bias"].sum()) == removed
            # a conv's mask also holds its input slices from the previous layer; a kept filter
            # always keeps at least one input channel, so only removed filters are fully masked
            assert int(mask[f"{conv}.weight"].all(dim=(1, 2, 3)).sum()) == removed
    else:
        assert mask == {}


# --- parameters ------------------------------------------------------------------------------------


@pytest.mark.parametrize("params", [
    {"epochs": 2, "lr": 0.1},                                           # no prune
    {"prune": "magnitude_prune", "epochs": 2, "lr": 0.1},               # unknown prune
    {"prune": "channel_prune_l1", "lr": 0.1},                           # no epochs
    {"prune": "channel_prune_l1", "epochs": 2},                         # no lr
    {"prune": "channel_prune_l1", "epochs": 0, "lr": 0.1},
    {"prune": "channel_prune_l1", "epochs": 2.5, "lr": 0.1},
    {"prune": "channel_prune_l1", "epochs": True, "lr": 0.1},
    {"prune": "channel_prune_l1", "epochs": 2, "lr": 0.1, "mask": "free"},  # unknown key
])
def test_params_are_validated(params):
    with pytest.raises(ValueError):
        prune_finetune.split_params(0.5, params)


def test_split_params_completes_the_p4_5_recipe():
    name, recipe = prune_finetune.split_params(0.3, {"prune": "magnitude_prune_global", "epochs": 20, "lr": 0.05})
    assert name == "magnitude_prune_global"
    assert recipe == {**finetune.DEFAULTS, "lr": 0.05, "epochs": 20}


@pytest.mark.parametrize("prune_name,strength", [("magnitude_prune_global", 1.0), ("channel_prune_l1", -0.1)])
def test_bad_strength_is_refused(state, tmp_path, prune_name, strength):
    with pytest.raises(ValueError):
        apply_attack(config(prune_name, strength), state, ARCH, smoke_context(tmp_path))


# --- the attack --------------------------------------------------------------------------------------


@pytest.mark.parametrize("prune_name,strength", [
    ("magnitude_prune_global", 0.7), ("magnitude_prune_layerwise", 0.5), ("channel_prune_l1", 0.3),
])
def test_pruned_entries_stay_exactly_zero_and_the_rest_trains(state, tmp_path, prune_name, strength):
    _, prune_info, mask = prune_finetune.PRUNE_ATTACKS[prune_name](state, ARCH, strength)
    out = apply_attack(config(prune_name, strength), state, ARCH, smoke_context(tmp_path))
    for name, m in mask.items():
        assert bool((out.state_dict[name][m] == 0).all()), name
        if (~m).any():
            assert not torch.equal(out.state_dict[name][~m], state[name][~m]), f"{name} kept entries did not train"
    assert not torch.equal(out.state_dict["features.1.running_mean"], state["features.1.running_mean"])

    info = out.info
    assert info["prune_attack"] == prune_name and info["prune"] == prune_info
    assert info["mask"]["masked_entries"] == sum(int(m.sum()) for m in mask.values())
    assert info["mask"]["masked_nonzero_after"] == 0 and info["mask"]["masked_nonzero_max_over_epochs"] == 0
    assert info["mask"]["epochs_checked"] == RECIPE["epochs"]
    assert info["finetune"]["recipe"]["epochs"] == RECIPE["epochs"] and info["finetune"]["steps"] == 2 * 4
    assert info["source_state_sha256"] == finetune.state_digest(state)
    zero_before = prune_info.get("achieved_sparsity", prune_info.get("carrier_zero_fraction"))
    assert info["carrier_zero_fraction_after"] >= zero_before - 1e-12
    json.dumps(info, allow_nan=False)


def test_a_removed_channel_still_outputs_zero_after_fine_tuning(state, tmp_path):
    _, info, _ = structured_prune.channel_prune_with_mask(state, ARCH, 0.5)
    out = apply_attack(config("channel_prune_l1", 0.5), state, ARCH, smoke_context(tmp_path))
    model = load_model(out.state_dict, ARCH)
    captured = {}
    for bn in BNS:
        dict(model.named_modules())[bn].register_forward_hook(lambda m, i, o, bn=bn: captured.__setitem__(bn, o))
    set_seed(8)
    model(torch.randn(16, 3, 32, 32))
    for bn, conv in zip(BNS, info["per_layer"]):
        removed = [i for i in range(out.state_dict[f"{conv}.weight"].shape[0])
                   if bool((out.state_dict[f"{conv}.weight"][i] == 0).all())]
        assert len(removed) == info["per_layer"][conv]["removed"]
        assert bool((captured[bn][:, removed] == 0).all())


def test_without_the_mask_pruned_weights_regrow(state, tmp_path):
    """The P4.5 attack on the same pruned weights, so the mask is what keeps them at zero."""
    pruned, _, mask = prune.magnitude_prune_with_mask(state, ARCH, 0.7, "global")
    free = apply_attack(AttackConfig("finetune_holdout", RECIPE["epochs"], {"lr": RECIPE["lr"], "batch_size": 32}),
                        pruned, ARCH, smoke_context(tmp_path))
    regrown = sum(int((free.state_dict[name][m] != 0).sum()) for name, m in mask.items())
    assert regrown > 0.5 * sum(int(m.sum()) for m in mask.values())


def test_a_broken_mask_is_caught(state, tmp_path, monkeypatch):
    monkeypatch.setattr(prune_finetune.FixedMask, "_hook", lambda self, name: (lambda grad: grad))
    with pytest.raises(RuntimeError, match="non-zero"):
        apply_attack(config("magnitude_prune_global", 0.5), state, ARCH, smoke_context(tmp_path))


def test_the_hooks_are_removed_afterwards(state):
    pruned, _, mask = prune.magnitude_prune_with_mask(state, ARCH, 0.5, "global")
    model = load_model(pruned, ARCH)
    fixed = prune_finetune.FixedMask(model, mask)
    fixed.remove()
    model.train()
    set_seed(1)
    model(torch.randn(4, 3, 32, 32)).sum().backward()
    grad = dict(model.named_parameters())["features.3.weight"].grad
    assert bool((grad[mask["features.3.weight"]] != 0).any())


def test_seeded_and_the_seed_matters(state, tmp_path):
    c = config("channel_prune_l1", 0.1)
    a = apply_attack(c, state, ARCH, smoke_context(tmp_path))
    b = apply_attack(c, state, ARCH, smoke_context(tmp_path))
    assert all(torch.equal(a.state_dict[k], b.state_dict[k]) for k in state)
    other = apply_attack(c, state, ARCH, smoke_context(tmp_path, seed=4))
    assert not torch.equal(a.state_dict["features.3.weight"], other.state_dict["features.3.weight"])


def test_a_finished_run_is_restored_and_another_pruning_is_refused(state, tmp_path):
    work = tmp_path / "ckpt"
    c = config("magnitude_prune_global", 0.5)
    first = apply_attack(c, state, ARCH, smoke_context(tmp_path, work_dir=work))
    restored = apply_attack(c, state, ARCH, smoke_context(tmp_path, work_dir=work))
    assert all(torch.equal(first.state_dict[k], restored.state_dict[k]) for k in state)
    assert restored.info["finetune"]["wall_seconds_this_session"] == 0.0
    assert restored.info["mask"]["epochs_checked"] == RECIPE["epochs"]
    with pytest.raises(ValueError, match="extra"):
        apply_attack(config("magnitude_prune_global", 0.6), state, ARCH, smoke_context(tmp_path, work_dir=work))
    with pytest.raises(ValueError, match="extra"):
        apply_attack(config("channel_prune_l1", 0.5), state, ARCH, smoke_context(tmp_path, work_dir=work))


def test_p4_5_checkpoint_config_is_unchanged_by_the_refactor(state, tmp_path, monkeypatch):
    """finetune_holdout must still write the checkpoint config the P4.5 runs wrote, or their resume breaks."""
    seen = {}
    real_fit = finetune.fit

    def spy(model, train, evaluate, cfg, **kwargs):
        seen["extra"] = cfg.extra
        return real_fit(model, train, evaluate, cfg, **kwargs)

    monkeypatch.setattr(finetune, "fit", spy)
    out = apply_attack(AttackConfig("finetune_holdout", 1, {"lr": 0.01}), state, ARCH, smoke_context(tmp_path))
    digest = finetune.state_digest(state)
    assert seen["extra"] == {"attack": "finetune_holdout", "augment": True, "smoke": True, "start_state_sha256": digest}
    assert list(out.info)[:3] == ["version", "start_state_sha256", "recipe"]


# --- harness and sweep ---------------------------------------------------------------------------------


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("run_attack_suite")


def test_apply_smoke_writes_weights_and_records(script, tmp_path):
    path = REPO_ROOT / "experiments" / "configs" / "p4.6_prune_finetune_channel-lr0.05-e20.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["strengths"], cfg["params"]["epochs"] = [0.5], 1
    small = tmp_path / "pf.json"
    small.write_text(json.dumps(cfg), encoding="utf-8")
    paths = script.main(["apply", "--config", str(small), "--task", "P4.6", "--smoke", "--out-dir", str(tmp_path / "out")])
    record = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    assert record["params"]["key_used"] is False and record["params"]["smoke"] is True
    assert record["params"]["config"]["tag"] == "channel-lr0.05-e20"
    assert record["metrics"]["info"]["mask"]["masked_nonzero_after"] == 0
    weights = tmp_path / "out" / "smoke" / record["metrics"]["weights_file"]
    assert weights.name == "p4.6_prune_finetune_channel-lr0.05-e20_0.5__seed1337.pt"
    assert script.sha256_file(weights) == record["metrics"]["weights_sha256"]


def test_apply_skips_a_complete_config(script, state, tmp_path, monkeypatch, capsys):
    source = {"name": "dual", "task": "TEST", "weights_sha256": "b" * 64, "arch": ARCH}
    monkeypatch.setattr(script, "load_source", lambda name, path: (state, source))
    monkeypatch.setattr(finetune, "attacker_holdout_loaders",
                        lambda root, **kw: attacker_holdout_loaders(root, **{**kw, "smoke": True}))
    cfg = tmp_path / "pf.json"
    cfg.write_text(json.dumps({"attack": "prune_finetune", "tag": "global-lr0.05-e1", "strengths": [0.5],
                               "params": {"prune": "magnitude_prune_global", "epochs": 1, "lr": 0.05}}), encoding="utf-8")
    argv = ["apply", "--config", str(cfg), "--task", "P4.6", "--out-dir", str(tmp_path / "out")]
    first = script.main(argv)
    capsys.readouterr()
    assert script.main(argv) == first
    assert "skipping prune_finetune_global-lr0.05-e1_0.5" in capsys.readouterr().out


P4_6_FILES = [f"p4.6_prune_finetune_{p}-lr{s}.json" for p in ("global", "channel")
              for s in ("0.01-e20", "0.05-e20", "0.1-e20", "0.1-e60")]


def test_p4_6_configs_are_the_declared_grid():
    configs = [c for name in P4_6_FILES
               for c in expand_config(json.loads((REPO_ROOT / "experiments" / "configs" / name).read_text(encoding="utf-8")))]
    assert len(configs) == 24 and len({c.label() for c in configs}) == 24
    declared = sorted((c.params["prune"], c.strength, c.params["lr"], c.params["epochs"]) for c in configs)
    expected = sorted((prune_name, s, lr, epochs)
                      for prune_name, strengths in (("magnitude_prune_global", (0.5, 0.7, 0.9)),
                                                    ("channel_prune_l1", (0.1, 0.3, 0.5)))
                      for s in strengths
                      for lr, epochs in ((0.01, 20), (0.05, 20), (0.1, 20), (0.1, 60)))
    assert declared == expected
    for c in configs:
        short = "global" if c.params["prune"] == "magnitude_prune_global" else "channel"
        assert c.attack == "prune_finetune" and c.tag == f"{short}-lr{c.params['lr']:g}-e{c.params['epochs']}"
        _, recipe = prune_finetune.split_params(c.strength, c.params)
        assert recipe == {**finetune.DEFAULTS, "lr": c.params["lr"], "epochs": c.params["epochs"]}
    assert "prune_finetune" in harness.available_attacks()
    # 18 x 20 + 6 x 60 epochs
    assert sum(c.params["epochs"] for c in configs) == 720
