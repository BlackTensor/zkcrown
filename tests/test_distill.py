"""Tests for P4.7: knowledge distillation into a fresh student.

Synthetic smoke data, seeded untrained teachers, CPU only; no trained weights,
no `K`. The transfer-set index checks need CIFAR-10 under ``data/`` and skip
without it. Covered: parameter validation, the loss, that the teacher's
outputs replace the labels (and the labels are never read), the student's
init (fresh, seed-fixed, not the owner's seed), that the student learns the
teacher, soft targets in the training loop, seeding, checkpoint restore and
refusal of other settings, apply mode end to end, and the declared sweep.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.attacks import distill, finetune, harness  # noqa: E402
from src.attacks.harness import AttackConfig, AttackContext, apply_attack, expand_config, load_model  # noqa: E402
from src.data import cifar10  # noqa: E402
from src.models import main_model  # noqa: E402
from src.training.loop import train_one_epoch  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCH = {"width": 32}
PARAMS = {"transfer": "holdout5k", "temperature": 4.0, "init_seed": 99, "epochs": 2, "lr": 0.05, "batch_size": 32}


@pytest.fixture(scope="module")
def teacher_state():
    """A seeded untrained model with its classifier scaled up, so its soft outputs are far from uniform."""
    set_seed(2468)
    state = main_model(width=32).state_dict()
    state["classifier.2.weight"] = state["classifier.2.weight"] * 30
    return state


def smoke_context(tmp_path, *, seed=3, work_dir=None):
    return AttackContext(torch.device("cpu"), seed, tmp_path, work_dir=work_dir, smoke=True)


def config(width=16, **overrides):
    return AttackConfig("distill", width, {**PARAMS, **overrides})


# --- parameters ------------------------------------------------------------------------------------


@pytest.mark.parametrize("strength,params", [
    (0, PARAMS), (16.5, PARAMS), (-4, PARAMS),
    (16, {k: v for k, v in PARAMS.items() if k != "transfer"}),
    (16, {k: v for k, v in PARAMS.items() if k != "temperature"}),
    (16, {k: v for k, v in PARAMS.items() if k != "init_seed"}),
    (16, {k: v for k, v in PARAMS.items() if k != "epochs"}),
    (16, {k: v for k, v in PARAMS.items() if k != "lr"}),
    (16, {**PARAMS, "transfer": "test"}),
    (16, {**PARAMS, "temperature": 0}),
    (16, {**PARAMS, "temperature": float("inf")}),
    (16, {**PARAMS, "temperature": True}),
    (16, {**PARAMS, "init_seed": -1}),
    (16, {**PARAMS, "init_seed": 1.5}),
    (16, {**PARAMS, "init_seed": 1337}),          # the owner's training seed
    (16, {**PARAMS, "epochs": 0}),
    (16, {**PARAMS, "alpha": 0.5}),               # unknown key, e.g. a ground-truth term
])
def test_params_are_validated(strength, params):
    with pytest.raises(ValueError):
        distill.split_params(strength, params)


def test_split_params_completes_the_p0_5_recipe():
    width, settings, recipe = distill.split_params(32.0, PARAMS)
    assert width == 32 and settings == {"transfer": "holdout5k", "temperature": 4.0, "init_seed": 99}
    assert recipe == {**finetune.DEFAULTS, "lr": 0.05, "batch_size": 32, "epochs": 2}


# --- loss and teacher queries ---------------------------------------------------------------------------


def test_soft_loss_is_t_squared_kl_and_zero_at_the_teacher():
    set_seed(5)
    student, teacher = torch.randn(8, 10), torch.randn(8, 10)
    loss = distill.DistillationLoss(4.0)
    p, q = torch.softmax(teacher / 4, 1), torch.softmax(student / 4, 1)
    expected = (p * (p.log() - q.log())).sum(1).mean() * 16
    assert torch.allclose(loss(student, teacher), expected, atol=1e-6)
    assert float(loss(teacher, teacher)) == pytest.approx(0.0, abs=1e-6)
    assert float(loss(teacher + 3.0, teacher)) == pytest.approx(0.0, abs=1e-6)  # softmax ignores a shared shift
    labels = torch.randint(0, 10, (8,))
    assert torch.allclose(loss(student, labels), nn.functional.cross_entropy(student, labels))


def test_teacher_queries_replace_labels_with_teacher_outputs(teacher_state):
    teacher = load_model(teacher_state, ARCH)
    set_seed(6)
    images = torch.randn(20, 3, 32, 32)
    loader = DataLoader(TensorDataset(images, torch.randint(0, 10, (20,))), batch_size=8)
    soft = distill.TeacherQueries(loader, teacher, torch.device("cpu"))
    hard = distill.TeacherQueries(loader, teacher, torch.device("cpu"), hard=True)
    with torch.no_grad():
        expected = teacher(images)
    got = torch.cat([t for _, t in soft])
    assert torch.allclose(got, expected, atol=1e-5)
    assert torch.equal(torch.cat([t for _, t in hard]), expected.argmax(1))
    assert len(soft) == 3 and soft.dataset is loader.dataset and soft.queries == 20 and hard.queries == 20


def test_soft_targets_in_the_training_loop_count_teacher_agreement():
    set_seed(7)
    model = nn.Linear(4, 10)
    inputs, teacher = torch.randn(16, 4), torch.randn(16, 10)
    with torch.no_grad():
        agree = int((model(inputs).argmax(1) == teacher.argmax(1)).sum())
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    out = train_one_epoch(model, [(inputs, teacher)], opt, distill.DistillationLoss(2.0), torch.device("cpu"))
    assert out["accuracy"] == agree / 16 and out["n"] == 16


# --- the student --------------------------------------------------------------------------------------


def test_student_init_is_fixed_by_init_seed_and_leaves_the_global_rng_alone():
    set_seed(11)
    before = torch.get_rng_state()
    a = distill.build_student(32, 99)
    assert torch.equal(torch.get_rng_state(), before)
    b = distill.build_student(32, 99)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    set_seed(1337)
    owner_seeded = main_model(width=32)
    assert not torch.equal(distill.build_student(32, 20260915).state_dict()["features.0.weight"],
                           owner_seeded.state_dict()["features.0.weight"])


@pytest.mark.parametrize("width,transfer", [(16, "holdout5k"), (32, "train50k")])
def test_student_is_fresh_and_teacher_untouched(teacher_state, tmp_path, width, transfer):
    snapshot = {k: v.clone() for k, v in teacher_state.items()}
    out = apply_attack(config(width, transfer=transfer), teacher_state, ARCH, smoke_context(tmp_path))
    assert out.arch == {"width": width}
    model = load_model(out.state_dict, out.arch)
    assert all(torch.equal(teacher_state[k], snapshot[k]) for k in teacher_state)
    init = distill.build_student(width, PARAMS["init_seed"]).state_dict()
    assert not torch.equal(out.state_dict["features.3.weight"], init["features.3.weight"])
    info = out.info
    assert info["student"]["width"] == width and info["student"]["params"] == sum(p.numel() for p in model.parameters())
    assert info["student"]["init_state_sha256"] == finetune.state_digest(init)
    assert info["teacher"]["source_state_sha256"] == finetune.state_digest(teacher_state)
    assert info["transfer"]["labels_used"] is False
    assert info["transfer"]["images"] == (128 if transfer == "holdout5k" else 512)
    steps_per_epoch = 4 if transfer == "holdout5k" else 16
    assert info["steps"] == PARAMS["epochs"] * steps_per_epoch
    assert info["teacher"]["training_queries_planned"] == PARAMS["epochs"] * info["transfer"]["images"]
    json.dumps(info, allow_nan=False)


def test_labels_are_never_read(teacher_state, tmp_path, monkeypatch):
    """Scrambling every label of the transfer and monitor sets changes nothing."""
    real = cifar10.attacker_transfer_loaders

    def scrambled(root, **kwargs):
        loaders = real(root, **kwargs)
        for loader in loaders.values():
            ds = loader.dataset
            ds.tensors = (ds.tensors[0], (ds.tensors[1] + 1 + torch.arange(len(ds)) % 7) % 10)
        return loaders

    a = apply_attack(config(16), teacher_state, ARCH, smoke_context(tmp_path))
    monkeypatch.setattr(distill, "attacker_transfer_loaders", scrambled)
    b = apply_attack(config(16), teacher_state, ARCH, smoke_context(tmp_path))
    assert all(torch.equal(a.state_dict[k], b.state_dict[k]) for k in a.state_dict)
    assert a.info["final_holdout_teacher_agreement"] == b.info["final_holdout_teacher_agreement"]


def test_the_student_learns_the_teacher(teacher_state, tmp_path):
    """Training-mode loss and teacher agreement, from the epoch records.

    Eval-mode agreement on 128 synthetic noise images swings wildly with the
    BatchNorm running statistics after a few steps, so it is not asserted.
    """
    out = apply_attack(config(32, epochs=6, lr=0.01), teacher_state, ARCH, smoke_context(tmp_path))
    first, last = out.info["first_epoch"], out.info["last_epoch"]
    assert last["train_loss"] < 0.25 * first["train_loss"]
    assert last["train_teacher_agreement"] > first["train_teacher_agreement"] + 0.2


def test_seeded_and_the_seed_matters(teacher_state, tmp_path):
    a = apply_attack(config(16), teacher_state, ARCH, smoke_context(tmp_path))
    b = apply_attack(config(16), teacher_state, ARCH, smoke_context(tmp_path))
    assert all(torch.equal(a.state_dict[k], b.state_dict[k]) for k in a.state_dict)
    other = apply_attack(config(16), teacher_state, ARCH, smoke_context(tmp_path, seed=4))
    assert not torch.equal(a.state_dict["features.3.weight"], other.state_dict["features.3.weight"])
    # the init does not follow the run seed
    assert a.info["student"]["init_state_sha256"] == other.info["student"]["init_state_sha256"]


def test_a_finished_run_is_restored_and_other_settings_are_refused(teacher_state, tmp_path):
    work = tmp_path / "ckpt"
    first = apply_attack(config(16), teacher_state, ARCH, smoke_context(tmp_path, work_dir=work))
    restored = apply_attack(config(16), teacher_state, ARCH, smoke_context(tmp_path, work_dir=work))
    assert all(torch.equal(first.state_dict[k], restored.state_dict[k]) for k in first.state_dict)
    assert restored.info["wall_seconds_this_session"] == 0.0
    for other in (config(16, temperature=2.0), config(16, init_seed=100), config(16, transfer="train50k")):
        with pytest.raises(ValueError, match="extra"):
            apply_attack(other, teacher_state, ARCH, smoke_context(tmp_path, work_dir=work))
    another_teacher = {k: (v + 1 if v.is_floating_point() else v) for k, v in teacher_state.items()}
    with pytest.raises(ValueError, match="extra"):
        apply_attack(config(16), another_teacher, ARCH, smoke_context(tmp_path, work_dir=work))


# --- data ------------------------------------------------------------------------------------------------


def test_transfer_loaders_refuse_an_unknown_set(tmp_path):
    with pytest.raises(ValueError):
        cifar10.attacker_transfer_loaders(tmp_path, transfer="test", smoke=True)


def test_transfer_sets_on_real_cifar10():
    root = REPO_ROOT / "data"
    if not (root / "cifar-10-batches-py").is_dir():
        pytest.skip("CIFAR-10 not under data/")
    _, holdout_idx = cifar10.cifar10_split_indices()
    small = cifar10.attacker_transfer_loaders(root, transfer="holdout5k")
    big = cifar10.attacker_transfer_loaders(root, transfer="train50k")
    assert len(small["train"].dataset) == 5_000 and list(small["train"].dataset.indices) == holdout_idx
    assert len(big["train"].dataset) == 50_000 and not hasattr(big["train"].dataset, "indices")
    for loaders in (small, big):
        assert list(loaders["eval"].dataset.indices) == holdout_idx
        assert loaders["eval"].dataset.dataset.train and loaders["train"].dataset is not None


# --- harness and sweep -------------------------------------------------------------------------------------


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("run_attack_suite")


def test_apply_smoke_writes_weights_and_records(script, tmp_path):
    path = REPO_ROOT / "experiments" / "configs" / "p4.7_distill_train50k-T4-lr0.1.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["strengths"], cfg["params"]["epochs"] = [16], 1
    small = tmp_path / "kd.json"
    small.write_text(json.dumps(cfg), encoding="utf-8")
    paths = script.main(["apply", "--config", str(small), "--task", "P4.7", "--smoke", "--out-dir", str(tmp_path / "out")])
    record = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    assert record["params"]["key_used"] is False and record["params"]["smoke"] is True
    assert record["params"]["config"]["tag"] == "train50k-T4-lr0.1" and record["metrics"]["arch"] == {"width": 16}
    weights = tmp_path / "out" / "smoke" / record["metrics"]["weights_file"]
    assert weights.name == "p4.7_distill_train50k-T4-lr0.1_16__seed1337.pt"
    assert script.sha256_file(weights) == record["metrics"]["weights_sha256"]
    state = torch.load(weights, map_location="cpu", weights_only=True)
    load_model(state, {"width": 16})


def test_apply_skips_a_complete_config(script, teacher_state, tmp_path, monkeypatch, capsys):
    source = {"name": "dual", "task": "TEST", "weights_sha256": "c" * 64, "arch": ARCH}
    monkeypatch.setattr(script, "load_source", lambda name, path: (teacher_state, source))
    monkeypatch.setattr(distill, "attacker_transfer_loaders",
                        lambda root, **kw: cifar10.attacker_transfer_loaders(root, **{**kw, "smoke": True}))
    cfg = tmp_path / "kd.json"
    cfg.write_text(json.dumps({"attack": "distill", "tag": "holdout5k-T4-e1", "strengths": [16],
                               "params": {**PARAMS, "epochs": 1}}), encoding="utf-8")
    argv = ["apply", "--config", str(cfg), "--task", "P4.7", "--out-dir", str(tmp_path / "out")]
    first = script.main(argv)
    capsys.readouterr()
    assert script.main(argv) == first
    assert "skipping distill_holdout5k-T4-e1_16" in capsys.readouterr().out


P4_7_FILES = ["p4.7_distill_holdout5k-T4-lr0.1.json", "p4.7_distill_holdout5k-T4-lr0.05.json",
              "p4.7_distill_train50k-T4-lr0.1.json"]


def test_p4_7_configs_are_the_declared_grid():
    """Owner decisions: both transfer sets x widths 32 and 16, T = 4, 60 epochs, LR 0.1; plus LR 0.05 on the holdout."""
    configs = [c for name in P4_7_FILES
               for c in expand_config(json.loads((REPO_ROOT / "experiments" / "configs" / name).read_text(encoding="utf-8")))]
    assert len(configs) == 6 and len({c.label() for c in configs}) == 6
    assert sorted((c.params["transfer"], c.params["lr"], c.strength) for c in configs) == sorted(
        (t, lr, w) for t, lr in (("holdout5k", 0.1), ("holdout5k", 0.05), ("train50k", 0.1)) for w in (32.0, 16.0))
    for c in configs:
        width, settings, recipe = distill.split_params(c.strength, c.params)
        assert c.attack == "distill" and c.tag == f"{settings['transfer']}-T4-lr{c.params['lr']:g}"
        assert settings["temperature"] == 4.0 and settings["init_seed"] == 20260915
        assert recipe == {**finetune.DEFAULTS, "lr": c.params["lr"], "epochs": 60}
    assert "distill" in harness.available_attacks()
    assert sum(c.params["epochs"] for c in configs) == 360


def test_width_16_student_has_the_planned_parameter_count():
    assert sum(p.numel() for p in main_model(width=16).parameters()) == 82_554
