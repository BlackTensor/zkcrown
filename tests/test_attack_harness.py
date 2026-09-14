"""Tests for P4.1: the attack interface, the row evaluator and the harness script.

The real run needs `K`, the trained weights and CIFAR-10. These tests use
seeded untrained models, synthetic images and TEST keys, so they check the
plumbing: config reading, the registry, that attacks get a private copy and no
key, seeding, the weight and behavioral scoring against planted watermarks,
the row shape, the control checks, and the apply -> evaluate hand-off.

All keys and owner ids here are TEST data, public by construction.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.attacks import evaluation, harness  # noqa: E402
from src.attacks.harness import (  # noqa: E402
    AttackConfig,
    AttackContext,
    AttackOutput,
    apply_attack,
    expand_config,
    register_attack,
)
from src.models import main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.watermark.significance import detection_test  # noqa: E402
from src.watermark.weight_embedding import embed_weight_watermark  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_KEY = bytes(range(32))
"""TEST KEY ONLY."""
TEST_OWNER = "zk-crown test owner"
STRONG_ALPHA = 0.2
"""TEST value only, large against an untrained host."""
N_TRIGGERS = 30


@pytest.fixture
def context(tmp_path):
    return AttackContext(device=torch.device("cpu"), seed=7, data_root=tmp_path)


@pytest.fixture(scope="module")
def source_state():
    set_seed(1234)
    return main_model().eval().state_dict()


@pytest.fixture
def temp_attacks():
    """Register attacks for one test and remove them afterwards."""
    names = []

    def add(name, fn, strength="test"):
        register_attack(name, strength=strength, description="TEST attack")(fn)
        names.append(name)

    yield add
    for name in names:
        harness._REGISTRY.pop(name, None)


@pytest.fixture(scope="module")
def material():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(200, 32, 32, 3), dtype=np.uint8)
    labels = rng.integers(0, 10, size=200)
    return evaluation.derive_owner_material(
        TEST_KEY, images, labels, list(range(200)), owner_id=TEST_OWNER, key_kind="test", n=N_TRIGGERS
    )


def tiny_loader(n=64, seed=3):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)),
                      batch_size=16, shuffle=False)


# --- config -----------------------------------------------------------------------------


def test_expand_config_one_row_per_strength():
    configs = expand_config({"attack": "x", "strengths": [0.1, 0.5, 0.9], "params": {"a": 1}})
    assert [c.strength for c in configs] == [0.1, 0.5, 0.9]
    assert all(c.params == {"a": 1} and c.attack == "x" for c in configs)
    assert expand_config({"attack": "x", "strength": 3})[0].strength == 3.0


@pytest.mark.parametrize("bad", [
    {"attack": "x"},
    {"attack": "x", "strength": 1, "strengths": [1]},
    {"attack": "x", "strengths": []},
    {"attack": "x", "strengths": [1, 1]},
    {"attack": "x", "strength": 1, "strenght": 2},
])
def test_expand_config_refuses_malformed(bad):
    with pytest.raises(ValueError):
        expand_config(bad)


@pytest.mark.parametrize("strength, error", [(math.nan, ValueError), (math.inf, ValueError), (True, TypeError), ("1", TypeError)])
def test_attack_config_validates_strength(strength, error):
    with pytest.raises(error):
        AttackConfig(attack="x", strength=strength)


def test_attack_config_label_and_dict():
    c = AttackConfig(attack="magnitude_prune", strength=0.5, params={"k": 2})
    assert c.label() == "magnitude_prune_0.5"
    assert c.to_dict() == {"attack": "magnitude_prune", "strength": 0.5, "params": {"k": 2}}
    assert AttackConfig(**c.to_dict()) == c


# --- registry and apply -------------------------------------------------------------------


def test_none_is_registered_and_names_are_unique(temp_attacks):
    assert "none" in harness.available_attacks()
    with pytest.raises(ValueError):
        temp_attacks("none", lambda *a: None)
    with pytest.raises(ValueError):
        temp_attacks("Bad-Name", lambda *a: None)
    with pytest.raises(KeyError):
        harness.get_attack("no_such_attack")


def test_attacks_never_receive_a_key():
    fields = {f.name for f in dataclasses.fields(AttackContext)}
    assert not any("key" in name or "secret" in name for name in fields)


def test_none_returns_an_unaliased_identical_copy(source_state, context):
    out = apply_attack(AttackConfig("none", 0), source_state, {"width": 32}, context)
    assert out.arch == {"width": 32}
    assert all(torch.equal(out.state_dict[k], v) for k, v in source_state.items())
    before = source_state["features.0.weight"].clone()
    out.state_dict["features.0.weight"].add_(1.0)
    assert torch.equal(source_state["features.0.weight"], before)


@pytest.mark.parametrize("config", [AttackConfig("none", 0.5), AttackConfig("none", 0, {"x": 1})])
def test_none_refuses_strength_or_params(source_state, context, config):
    with pytest.raises(ValueError):
        apply_attack(config, source_state, {"width": 32}, context)


def test_attack_cannot_modify_the_source(source_state, context, temp_attacks):
    def zero_everything(state, arch, strength, params, ctx):
        for t in state.values():
            t.zero_()
        return AttackOutput(state, arch)

    temp_attacks("test_zero", zero_everything)
    before = {k: v.clone() for k, v in source_state.items()}
    apply_attack(AttackConfig("test_zero", 1), source_state, {"width": 32}, context)
    assert all(torch.equal(source_state[k], v) for k, v in before.items())


def test_apply_is_seeded(source_state, tmp_path, temp_attacks):
    def noise(state, arch, strength, params, ctx):
        state["features.0.weight"] += strength * torch.randn_like(state["features.0.weight"])
        return AttackOutput(state, arch)

    temp_attacks("test_noise", noise)
    config = AttackConfig("test_noise", 0.1)
    run = lambda seed: apply_attack(config, source_state, {"width": 32},  # noqa: E731
                                    AttackContext(torch.device("cpu"), seed, tmp_path)).state_dict["features.0.weight"]
    assert torch.equal(run(1), run(1))
    assert not torch.equal(run(1), run(2))


def test_apply_checks_the_output(source_state, context, temp_attacks):
    temp_attacks("test_not_output", lambda s, a, st, p, c: s)
    temp_attacks("test_wrong_arch", lambda s, a, st, p, c: AttackOutput(s, {"width": 16}))
    temp_attacks("test_bad_arch_key", lambda s, a, st, p, c: AttackOutput(s, {"width": 32, "dropout": 0.0}))
    with pytest.raises(TypeError):
        apply_attack(AttackConfig("test_not_output", 0), source_state, {"width": 32}, context)
    with pytest.raises(RuntimeError):
        apply_attack(AttackConfig("test_wrong_arch", 0), source_state, {"width": 32}, context)
    with pytest.raises(ValueError):
        apply_attack(AttackConfig("test_bad_arch_key", 0), source_state, {"width": 32}, context)


def test_a_narrower_student_is_a_valid_output(source_state, context, temp_attacks):
    def student(state, arch, strength, params, ctx):
        set_seed(ctx.seed)
        return AttackOutput(main_model(width=16).state_dict(), {"width": 16}, {"student_width": 16})

    temp_attacks("test_student", student)
    out = apply_attack(AttackConfig("test_student", 0), source_state, {"width": 32}, context)
    assert out.arch == {"width": 16} and out.info == {"student_width": 16}


# --- evaluation ---------------------------------------------------------------------------


def test_owner_material_is_deterministic_and_hidden(material):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(200, 32, 32, 3), dtype=np.uint8)
    labels = rng.integers(0, 10, size=200)
    second = evaluation.derive_owner_material(
        TEST_KEY, images, labels, list(range(200)), owner_id=TEST_OWNER, key_kind="test", n=N_TRIGGERS
    )
    assert second.bundle_digest == material.bundle_digest
    assert second.layout.digest() == material.layout.digest() and material.layout.dim == 307_040
    text = repr(material)
    assert "hidden" in text and material.bundle_digest not in text


def test_weight_detects_a_planted_watermark(source_state, material):
    marked, _ = embed_weight_watermark(source_state, material.layout, material.projection, material.signature, STRONG_ALPHA)
    result = evaluation.evaluate_weight(marked, material)
    assert result["applicable"] and result["detected"]
    assert result["z"] > 10 and result["p_value_bound"] < 1e-20
    assert result["rejects"][evaluation.DETECTION_ALPHA] is True


def test_weight_does_not_detect_an_unmarked_model(source_state, material):
    result = evaluation.evaluate_weight(source_state, material)
    assert result["applicable"] and not result["detected"]
    assert abs(result["z"]) < 6


def test_weight_is_not_applicable_when_the_layout_is_gone(material):
    set_seed(5)
    result = evaluation.evaluate_weight(main_model(width=16).state_dict(), material)
    assert result == {"applicable": False, "reason": result["reason"], "detected": False}
    assert "layout" in result["reason"]


def test_behavioral_row_matches_p2_8(source_state, material):
    model = harness.load_model(source_state, {"width": 32})
    result = evaluation.evaluate_behavioral(model, material, torch.device("cpu"))
    assert result["n"] == N_TRIGGERS and 0 <= result["fired"] <= N_TRIGGERS
    assert result["wdr"] == result["fired"] / N_TRIGGERS
    test = detection_test(result["fired"], N_TRIGGERS)
    assert result["p_value"] == float(test.p_value)
    assert result["detected"] == test.rejects("1e-6")


def test_behavioral_detected_when_the_model_answers_every_target(material):
    class Oracle(torch.nn.Module):
        """Returns each trigger's target; looks the trigger up by its pixels. TEST only."""

        def __init__(self):
            super().__init__()
            from src.data import CIFAR10_MEAN, CIFAR10_STD
            from src.watermark.detection import normalise_uint8

            self.keys = normalise_uint8(material.triggers.images, CIFAR10_MEAN, CIFAR10_STD)
            self.targets = torch.as_tensor(np.array(material.responses.targets))

        def forward(self, x):
            index = torch.cdist(x.flatten(1), self.keys.flatten(1)).argmin(dim=1)
            return torch.nn.functional.one_hot(self.targets[index], 10).float()

    result = evaluation.evaluate_behavioral(Oracle(), material, torch.device("cpu"))
    assert result["fired"] == N_TRIGGERS and result["detected"]


def test_accuracy_drop_is_paired():
    scores = {"accuracy": 0.5, "correct_count": 2, "n": 4, "loss": 1.0, "correct": [True, True, False, False]}
    acc = evaluation.evaluate_accuracy(scores, [True, False, True, True])
    assert acc["source_accuracy"] == 0.75 and acc["accuracy"] == 0.5
    assert acc["drop_vs_source_pp"] == pytest.approx(25.0)
    assert (acc["source_only_right"], acc["attacked_only_right"]) == (2, 1)


def test_score_loader_counts(source_state):
    model = harness.load_model(source_state, {"width": 32})
    loader = tiny_loader()
    scores = evaluation.score_loader(model, loader, torch.device("cpu"))
    assert scores["n"] == 64 and len(scores["correct"]) == 64
    assert scores["correct_count"] == sum(scores["correct"])
    with pytest.raises(ValueError):
        evaluation.score_loader(model, [], torch.device("cpu"))


@pytest.mark.parametrize("arch", [{"width": 32}, {"width": 16}])
def test_build_row_has_the_phase_4_columns(source_state, material, context, arch):
    loader = tiny_loader()
    reference = evaluation.score_loader(harness.load_model(source_state, {"width": 32}), loader, torch.device("cpu"))
    if arch["width"] == 32:
        state = apply_attack(AttackConfig("none", 0), source_state, arch, context).state_dict
    else:
        set_seed(9)
        state = main_model(**arch).state_dict()
    ev = evaluation.evaluate_attacked(state, arch, material, loader, reference["correct"], torch.device("cpu"))
    config = AttackConfig("none", 0)
    row = evaluation.build_row(config, harness.get_attack("none"), ev, source={"name": "s"}, attacked={"arch": arch},
                               material=material)
    table = row["table"]
    for column in ("strength", "clean_accuracy", "behavioral_wdr", "weight_correlation", "behavioral_p_value", "weight_p_value"):
        assert column in table
    assert row["row_version"] == harness.ROW_VERSION
    assert table["detection_alpha"] == "1e-6"
    assert table["weight_applicable"] is (arch["width"] == 32)
    if arch["width"] == 16:
        assert table["weight_correlation"] is None and table["weight_p_value"] is None and not table["weight_detected"]
    else:
        assert table["accuracy_drop_vs_source_pp"] == 0.0
    assert_aggregates_only(row)
    json.dumps(row, allow_nan=False)
    assert row["owner"]["trigger_bundle_sha256"] == material.bundle_digest


SECRET_KEYS = {"projected", "recovered_bits", "signs", "bits", "targets", "predictions", "fired_mask",
               "target_probability", "base_indices", "images", "state_dict"}


def assert_aggregates_only(value, path="row"):
    """No secret-named field and no long list anywhere: per-trigger or per-bit data would show up as one."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert key not in SECRET_KEYS, f"{path}.{key}"
            assert_aggregates_only(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        assert len(value) <= 8, f"{path} has {len(value)} entries"
        for i, item in enumerate(value):
            assert_aggregates_only(item, f"{path}[{i}]")


@pytest.mark.parametrize("leak", [{"weight": {"projected": [0.1]}}, {"behavioral": {"per_trigger": list(range(100))}}])
def test_the_aggregate_check_catches_a_planted_leak(leak):
    with pytest.raises(AssertionError):
        assert_aggregates_only(leak)


# --- script ---------------------------------------------------------------------------------


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("run_attack_suite")


def _args(**kw):
    import argparse

    base = {"config": None, "attack": None, "strength": None, "params": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_read_configs_inline_and_file(script, tmp_path):
    assert script.read_configs(_args(attack="none", strength=0.0))[0] == AttackConfig("none", 0)
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"attack": "none", "strengths": [0]}), encoding="utf-8")
    assert len(script.read_configs(_args(config=[path]))) == 1
    other = tmp_path / "d.json"
    other.write_text(json.dumps({"attack": "magnitude_prune_global", "strengths": [0.1, 0.2]}), encoding="utf-8")
    assert [c.attack for c in script.read_configs(_args(config=[path, other]))] == ["none", "magnitude_prune_global",
                                                                                    "magnitude_prune_global"]
    with pytest.raises(SystemExit):
        script.read_configs(_args(config=[path, path]))
    with pytest.raises(SystemExit):
        script.read_configs(_args(config=[path], attack="none"))
    with pytest.raises(SystemExit):
        script.read_configs(_args(attack="none"))


def test_sources_are_the_committed_models(script):
    p3_6 = json.loads((REPO_ROOT / script.P3_6_RESULT).read_text(encoding="utf-8"))
    assert script.SOURCES["dual"]["weights_sha256"] == p3_6["metrics"]["weights_sha256"]
    assert set(script.SOURCES) == {"dual", "behavioral", "clean"}


def _control_row(fired=100, corr=None, correct=9085, discordant=0, source="dual"):
    p3_6 = json.loads((REPO_ROOT / "results/p3.6_dual_wm__seed1337__20260914T085541+0000.json").read_text(encoding="utf-8"))
    corr = p3_6["metrics"]["weight_watermark"]["dual"]["correlation"] if corr is None else corr
    row = {
        "source": {"name": source},
        "clean_accuracy": {"correct": correct, "source_only_right": discordant, "attacked_only_right": 0},
        "behavioral": {"fired": fired},
        "weight": {"applicable": True, "correlation": corr},
    }
    return row, p3_6


def test_check_control_accepts_p3_6_reproduction(script):
    row, p3_6 = _control_row()
    checks = script.check_control(row, p3_6)
    assert checks["reproduces_p3_6"]["test_correct"] == 9085


@pytest.mark.parametrize("change", [{"fired": 99}, {"corr": 0.9}, {"correct": 9084}, {"discordant": 1}])
def test_check_control_refuses_any_difference(script, change):
    row, p3_6 = _control_row(**change)
    with pytest.raises(SystemExit):
        script.check_control(row, p3_6)


def test_check_control_on_other_sources_needs_only_identity(script):
    row, _ = _control_row(fired=3, source="clean")
    assert script.check_control(row, None) == {"no_discordant_images": True}


def test_apply_mode_needs_no_key_and_evaluate_reads_it_back(script, monkeypatch, tmp_path, source_state):
    import make_master_key

    def forbidden(*a, **k):
        raise AssertionError("apply mode must not read K")

    monkeypatch.setattr(make_master_key, "load_key", forbidden)
    source_file = tmp_path / "source.pt"
    torch.save(source_state, source_file)
    digest = script.sha256_file(source_file)
    monkeypatch.setitem(script.SOURCES, "dual", {**script.SOURCES["dual"], "weights_sha256": digest})

    out = tmp_path / "applied"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # dirty-tree warning while tests run on uncommitted code
        paths = script.main(["apply", "--attack", "none", "--strength", "0", "--source-path", str(source_file),
                             "--out-dir", str(out), "--task", "P4.1"])
    assert len(paths) == 1
    record = json.loads(paths[0].read_text(encoding="utf-8"))
    assert record["params"]["key_used"] is False and record["params"]["kind"] == script.APPLY_RECORD_KIND

    config, _, source, weights = script.read_apply_record(paths[0], None)
    assert config == AttackConfig("none", 0) and source["weights_sha256"] == digest
    loaded = torch.load(weights, weights_only=True)
    assert all(torch.equal(loaded[k], v) for k, v in source_state.items())

    weights.write_bytes(weights.read_bytes() + b"x")
    with pytest.raises(SystemExit):
        script.read_apply_record(paths[0], None)


def test_evaluate_refuses_a_non_apply_record(script):
    with pytest.raises(SystemExit):
        script.read_apply_record(REPO_ROOT / script.P3_6_RESULT, None)


def test_load_source_refuses_a_wrong_hash(script, tmp_path, source_state):
    path = tmp_path / "w.pt"
    torch.save(source_state, path)
    with pytest.raises(SystemExit):
        script.load_source("dual", path)
