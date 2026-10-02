"""Tests for P4.8: the attacker overwrites with their own watermark and key.

Synthetic smoke data, seeded untrained models, CPU only; no trained weights, no
`K`, no CIFAR-10. Covered: the attacker key derivation, parameter validation,
that the weight overwrite is exactly the P3.2 embedding under the attacker's
key and leaves a test owner's weight watermark extractable, that the
behavioral overwrite mixes the attacker's own triggers into fine-tuning, that
the combined variant is the behavioral one plus the post-hoc embedding, that
`info` holds aggregates only, seeding, checkpoint binding, apply mode end to
end, and the declared sweep.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.attacks import finetune, harness, overwrite  # noqa: E402
from src.attacks.evaluation import evaluate_weight  # noqa: E402
from src.attacks.harness import AttackConfig, AttackContext, apply_attack, expand_config  # noqa: E402
from src.data.cifar10 import attacker_holdout_arrays  # noqa: E402
from src.models import main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.watermark.carrier import carrier_layout  # noqa: E402
from src.watermark.signature import derive_signature  # noqa: E402
from src.watermark.triggers import generate_triggers  # noqa: E402
from src.watermark.weight_embedding import derive_carrier_projection, embed_weight_watermark  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCH = {"width": 32}
OWNER_TEST_KEY = hashlib.sha256(b"zk-crown/tests/p4.8/owner-test-key").digest()
RECIPE = {"epochs": 2, "batch_size": 32}


@pytest.fixture(scope="module")
def state():
    set_seed(2468)
    return main_model(**ARCH).state_dict()


@pytest.fixture(scope="module")
def layout():
    return carrier_layout(main_model(**ARCH))


def smoke_context(tmp_path, *, seed=3, work_dir=None):
    return AttackContext(torch.device("cpu"), seed, tmp_path, work_dir=work_dir, smoke=True)


# --- the attacker's key ----------------------------------------------------------------------------------


def test_attacker_key_is_a_labelled_public_derivation():
    key = overwrite.attacker_key("attacker-1")
    assert key == hashlib.sha256(b"zk-crown/p4.8/attacker-key/v1\x00attacker-1").digest() and len(key) == 32
    assert overwrite.attacker_key("attacker-2") != key
    for bad in ("", 7, None, "x" * 65):
        with pytest.raises(ValueError):
            overwrite.attacker_key(bad)


def test_context_has_no_key_and_the_attacks_are_registered():
    assert not any("key" in f for f in AttackContext.__dataclass_fields__)
    assert {"overwrite_weight", "overwrite_behavioral", "overwrite_both"} <= set(harness.available_attacks())


# --- parameters ------------------------------------------------------------------------------------------


def test_weight_params_validation():
    assert overwrite.weight_params(0.5, {}) == (0.5, "attacker-1")
    assert overwrite.weight_params(2, {"attacker": "b"}) == (2.0, "b")
    for strength, params in ((0, {}), (-1, {}), (float("nan"), {}), (0.1, {"lr": 0.1}), (0.1, {"attacker": ""})):
        with pytest.raises(ValueError):
            overwrite.weight_params(strength, params)


def test_behavioral_params_validation():
    recipe, triggers, label, alpha = overwrite.behavioral_params("overwrite_behavioral", 0.01, {"epochs": 20}, weight=False)
    assert recipe == {**finetune.DEFAULTS, "lr": 0.01, "epochs": 20} and alpha is None and label == "attacker-1"
    assert triggers == {"n_triggers": 100, "amplitude": 16, "triggers_per_batch": 4}
    *_, alpha = overwrite.behavioral_params("overwrite_both", 0.01, {"epochs": 20, "weight_alpha": 0.1}, weight=True)
    assert alpha == 0.1
    bad = [
        ("overwrite_behavioral", 0.01, {}, False),                                   # no epochs
        ("overwrite_behavioral", 0.01, {"epochs": 20, "weight_alpha": 0.1}, False),  # alpha where none is taken
        ("overwrite_both", 0.01, {"epochs": 20}, True),                              # alpha missing
        ("overwrite_both", 0.01, {"epochs": 20, "weight_alpha": 0}, True),
        ("overwrite_behavioral", 0, {"epochs": 20}, False),                          # lr must be > 0
        ("overwrite_behavioral", 0.01, {"epochs": 20, "lr": 0.1}, False),            # lr is the strength
        ("overwrite_behavioral", 0.01, {"epochs": 1.5}, False),
        ("overwrite_behavioral", 0.01, {"epochs": 20, "n_triggers": 2, "triggers_per_batch": 4}, False),
        ("overwrite_behavioral", 0.01, {"epochs": 20, "amplitude": 0}, False),
        ("overwrite_behavioral", 0.01, {"epochs": 20, "surprise": 1}, False),
    ]
    for name, strength, params, weight in bad:
        with pytest.raises(ValueError):
            overwrite.behavioral_params(name, strength, params, weight=weight)


# --- weight overwrite ------------------------------------------------------------------------------------


def test_weight_overwrite_is_the_p3_2_embedding_under_the_attacker_key(state, layout, tmp_path):
    out = apply_attack(AttackConfig("overwrite_weight", 0.5, {}), state, ARCH, smoke_context(tmp_path))
    key = overwrite.attacker_key("attacker-1")
    expected, _ = embed_weight_watermark(state, layout, derive_carrier_projection(key, layout),
                                         derive_signature(key, overwrite.ATTACKER_OWNER_ID), 0.5)
    assert all(torch.equal(out.state_dict[k], expected[k]) for k in state)
    carrier = set(layout.names)
    assert all(torch.equal(out.state_dict[k], state[k]) for k in state if k not in carrier)
    assert all(not torch.equal(out.state_dict[k], state[k]) for k in carrier)
    info = out.info
    assert info["variant"] == "weight" and info["fine_tuned"] is False
    assert info["attacker"]["owner_key_known"] is False and info["attacker"]["label"] == "attacker-1"
    assert info["source_state_sha256"] == finetune.state_digest(state)
    weight = info["weight"]
    assert abs(weight["attacker_before"]["z"]) < 6 and not weight["attacker_before"]["detected"]
    assert weight["attacker_after"]["bit_matches"] == 128 and weight["attacker_after"]["detected"]


def test_weight_overwrite_change_grows_with_alpha(state, tmp_path):
    sizes = [apply_attack(AttackConfig("overwrite_weight", a, {}), state, ARCH, smoke_context(tmp_path))
             .info["weight"]["embedding"]["delta_l2"] for a in (0.1, 0.2, 0.5)]
    assert sizes[1] == pytest.approx(2 * sizes[0], rel=1e-6) and sizes[2] == pytest.approx(5 * sizes[0], rel=1e-6)


def test_a_test_owners_weight_watermark_still_extracts_after_an_equal_strength_overwrite(state, layout, tmp_path):
    projection, signature = derive_carrier_projection(OWNER_TEST_KEY, layout), derive_signature(OWNER_TEST_KEY, "test-owner")
    owned, _ = embed_weight_watermark(state, layout, projection, signature, 0.5)
    owner = overwrite.AttackerMaterial("owner-under-test", layout, projection, signature)
    before = evaluate_weight(owned, owner)
    out = apply_attack(AttackConfig("overwrite_weight", 0.5, {}), owned, ARCH, smoke_context(tmp_path))
    after = evaluate_weight(out.state_dict, owner)
    assert before["detected"] and after["detected"]
    # The attacker's projection is independent of the owner's: cross-talk only, about alpha' * sqrt(128 / dim).
    assert abs(after["amplitude"] - before["amplitude"]) < 6 * 0.5 * (128 / layout.dim) ** 0.5
    assert out.info["weight"]["attacker_after"]["detected"]  # both claims verify; the tests do not order them


def test_different_attackers_embed_different_watermarks(state, tmp_path):
    a = apply_attack(AttackConfig("overwrite_weight", 0.5, {"attacker": "a"}), state, ARCH, smoke_context(tmp_path))
    b = apply_attack(AttackConfig("overwrite_weight", 0.5, {"attacker": "b"}), state, ARCH, smoke_context(tmp_path))
    assert any(not torch.equal(a.state_dict[k], b.state_dict[k]) for k in state)


# --- behavioral overwrite --------------------------------------------------------------------------------


def test_holdout_arrays_smoke_shape_and_seed(tmp_path):
    images, labels = attacker_holdout_arrays(tmp_path, smoke=True, seed=3)
    assert images.shape == (128, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (128,) and labels.dtype == np.int64 and 0 <= labels.min() and labels.max() <= 9
    again, _ = attacker_holdout_arrays(tmp_path, smoke=True, seed=3)
    other, _ = attacker_holdout_arrays(tmp_path, smoke=True, seed=4)
    assert np.array_equal(images, again) and not np.array_equal(images, other)


@pytest.fixture(scope="module")
def behavioral(state, tmp_path_factory):
    config = AttackConfig("overwrite_behavioral", 0.01, dict(RECIPE))
    return apply_attack(config, state, ARCH, smoke_context(tmp_path_factory.mktemp("b")))


def test_behavioral_overwrite_trains_on_the_attackers_own_triggers(state, behavioral, tmp_path, monkeypatch):
    info = behavioral.info
    assert info["variant"] == "behavioral" and "weight" not in info and info["fine_tuned"] is True
    assert all(not torch.equal(behavioral.state_dict[k], state[k]) for k in carrier_layout(main_model(**ARCH)).names)
    b = info["behavioral"]
    assert (b["n_triggers"], b["amplitude"], b["triggers_per_batch"]) == (100, 16, 4)
    assert b["trigger_samples_per_epoch"] == 4 * 4  # 128 smoke images in batches of 32
    assert info["finetune"]["steps"] == 2 * 4 and info["finetune"]["recipe"]["lr"] == 0.01
    assert b["attacker_before"]["n"] == b["attacker_after"]["n"] == 100

    # The triggers are the attacker's: built from the attacker's images with the attacker's key.
    seen = {}
    real = overwrite.TriggerMixLoader

    def spy(loader, inputs, targets, **kw):
        seen.update(inputs=inputs, targets=targets, kw=kw)
        return real(loader, inputs, targets, **kw)

    monkeypatch.setattr(overwrite, "TriggerMixLoader", spy)
    apply_attack(AttackConfig("overwrite_behavioral", 0.01, {"epochs": 1, "batch_size": 32}), state, ARCH,
                 smoke_context(tmp_path))
    images, _ = attacker_holdout_arrays(tmp_path, smoke=True, seed=3)
    triggers = generate_triggers(overwrite.attacker_key("attacker-1"), images, range(len(images)), n=100, amplitude=16)
    mean = torch.tensor(overwrite.CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(overwrite.CIFAR10_STD).view(1, 3, 1, 1)
    expected = (torch.from_numpy(triggers.images.copy()).permute(0, 3, 1, 2).float() / 255 - mean) / std
    assert torch.allclose(seen["inputs"], expected, atol=1e-6)
    assert seen["kw"] == {"triggers_per_batch": 4, "seed": 3}


def test_behavioral_overwrite_is_seeded(state, behavioral, tmp_path):
    again = apply_attack(AttackConfig("overwrite_behavioral", 0.01, dict(RECIPE)), state, ARCH, smoke_context(tmp_path))
    assert all(torch.equal(again.state_dict[k], behavioral.state_dict[k]) for k in state)
    assert again.info["behavioral"]["bundle_sha256"] == behavioral.info["behavioral"]["bundle_sha256"]
    other = apply_attack(AttackConfig("overwrite_behavioral", 0.01, {**RECIPE, "attacker": "attacker-2"}), state, ARCH,
                         smoke_context(tmp_path))
    assert other.info["behavioral"]["bundle_sha256"] != behavioral.info["behavioral"]["bundle_sha256"]


def test_both_is_the_behavioral_overwrite_plus_the_post_hoc_embedding(state, behavioral, layout, tmp_path):
    both = apply_attack(AttackConfig("overwrite_both", 0.01, {**RECIPE, "weight_alpha": 0.5}), state, ARCH,
                        smoke_context(tmp_path))
    key = overwrite.attacker_key("attacker-1")
    expected, _ = embed_weight_watermark(behavioral.state_dict, layout, derive_carrier_projection(key, layout),
                                         derive_signature(key, overwrite.ATTACKER_OWNER_ID), 0.5)
    assert all(torch.equal(both.state_dict[k], expected[k]) for k in state)
    info = both.info
    assert info["variant"] == "both" and info["weight"]["alpha"] == 0.5 and "then the weight overwrite" in info["order"]
    assert info["weight"]["attacker_after"]["detected"]
    assert info["behavioral"]["bundle_sha256"] == behavioral.info["behavioral"]["bundle_sha256"]


@pytest.mark.parametrize("which", ["weight", "behavioral"])
def test_info_is_small_json_with_no_arrays(state, behavioral, tmp_path, which):
    out = behavioral if which == "behavioral" else apply_attack(AttackConfig("overwrite_weight", 0.1, {}), state, ARCH,
                                                                smoke_context(tmp_path))
    text = json.dumps(out.info)
    assert len(text) < 12_000

    def longest_list(node):
        if isinstance(node, dict):
            return max((longest_list(v) for v in node.values()), default=0)
        if isinstance(node, list):
            return max([len(node)] + [longest_list(v) for v in node])
        return 0

    assert longest_list(out.info) <= 16  # per-tensor stats at most; no per-trigger or per-bit data


def test_checkpoint_is_bound_to_the_attacker_and_start_state(state, tmp_path, monkeypatch):
    seen = {}
    real = finetune.run_finetune

    def spy(model, recipe, context, extra, **kw):
        seen["extra"] = dict(extra)
        return real(model, recipe, context, extra, **kw)

    monkeypatch.setattr(finetune, "run_finetune", spy)
    work = tmp_path / "ckpt"
    out = apply_attack(AttackConfig("overwrite_both", 0.01, {**RECIPE, "weight_alpha": 0.1}), state, ARCH,
                       smoke_context(tmp_path, work_dir=work))
    assert seen["extra"] == {"attack": "overwrite_both", "augment": True, "smoke": True,
                             "start_state_sha256": finetune.state_digest(state), "attacker": "attacker-1",
                             "attacker_bundle_sha256": out.info["behavioral"]["bundle_sha256"],
                             "triggers_per_batch": 4, "weight_alpha": 0.1}
    with pytest.raises(Exception):  # the same checkpoint dir under another attacker key is refused
        apply_attack(AttackConfig("overwrite_both", 0.01, {**RECIPE, "weight_alpha": 0.1, "attacker": "attacker-2"}),
                     state, ARCH, smoke_context(tmp_path, work_dir=work))


# --- harness and sweep -----------------------------------------------------------------------------------


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("run_attack_suite")


def test_apply_smoke_writes_weights_and_records(script, tmp_path):
    path = REPO_ROOT / "experiments" / "configs" / "p4.8_overwrite_both_a0.1-e20.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["strengths"], cfg["params"]["epochs"] = [0.01], 1
    small = tmp_path / "ow.json"
    small.write_text(json.dumps(cfg), encoding="utf-8")
    paths = script.main(["apply", "--config", str(small), "--task", "P4.8", "--smoke", "--out-dir", str(tmp_path / "out")])
    record = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    assert record["params"]["key_used"] is False and record["params"]["smoke"] is True
    assert record["params"]["config"]["tag"] == "a0.1-e20"
    assert record["metrics"]["info"]["variant"] == "both"
    weights = tmp_path / "out" / "smoke" / record["metrics"]["weights_file"]
    assert weights.name == "p4.8_overwrite_both_a0.1-e20_0.01__seed1337.pt"
    assert script.sha256_file(weights) == record["metrics"]["weights_sha256"]


def test_p4_8_configs_are_the_declared_grid():
    def load(name):
        return expand_config(json.loads((REPO_ROOT / "experiments" / "configs" / name).read_text(encoding="utf-8")))

    weight = load("p4.8_overwrite_weight.json")
    assert [(c.attack, c.strength, c.tag) for c in weight] == [("overwrite_weight", a, None) for a in (0.1, 0.2, 0.5, 1.0, 2.0)]
    for c in weight:
        assert overwrite.weight_params(c.strength, c.params) == (c.strength, "attacker-1")
    behavioral, both = load("p4.8_overwrite_behavioral_e20.json"), load("p4.8_overwrite_both_a0.1-e20.json")
    for configs, name, tag, weight_alpha in ((behavioral, "overwrite_behavioral", "e20", None),
                                             (both, "overwrite_both", "a0.1-e20", 0.1)):
        assert [(c.attack, c.strength, c.tag) for c in configs] == [(name, lr, tag) for lr in (0.001, 0.01, 0.05)]
        for c in configs:
            recipe, triggers, label, alpha = overwrite.behavioral_params(name, c.strength, c.params, weight=weight_alpha is not None)
            assert recipe == {**finetune.DEFAULTS, "lr": c.strength, "epochs": 20}
            assert triggers == {"n_triggers": 100, "amplitude": 16, "triggers_per_batch": 4}
            assert (label, alpha) == ("attacker-1", weight_alpha)
    labels = [c.label() for c in (*weight, *behavioral, *both)]
    assert len(set(labels)) == 11
