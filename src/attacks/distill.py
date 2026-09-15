"""Knowledge distillation into a fresh student (P4.7).

The thief does not ship the stolen weights at all. They use the stolen model
as a teacher: query it on images they hold, and train a new network from a
random initialisation to reproduce its outputs (Hinton et al. 2015). The
student never sees the teacher's weights, the owner's labels, the key or the
triggers, so whether either watermark survives is an open question, not a
given. The behavioral watermark is expected not to transfer, because the
trigger responses are only visible on trigger inputs, which the thief never
queries.

Owner decisions (P4.7)
----------------------
- **Transfer set, two conditions** (`src.data.cifar10.TRANSFER_SETS`):
  ``holdout5k``, the 5,000-image attacker holdout of P4.5/P4.6, and
  ``train50k``, all 50,000 CIFAR-10 training images, the strong-attacker case
  that includes the 45,000 images `W*` was trained on. Ground-truth labels are
  discarded in both.
- **Student width 32 and 16.** ``strength`` is the student's `main_model`
  width. Width 32 has `W*`'s carrier layout, so the weight watermark can be
  tested on it. Width 16 (82,554 parameters) is the smaller student; its weight
  row is "not applicable" by construction.
- **Soft teacher outputs at temperature 4**, pure distillation loss, no
  ground-truth term.
- **60 epochs** with the P0.5 recipe: SGD Nesterov 0.9, weight decay 5e-4,
  batch 128, crop and flip augmentation, 1-epoch warmup, cosine decay from
  LR 0.1 to 0, final-epoch weights.

How the teacher is queried
--------------------------
The teacher is the source model in eval mode (no dropout, BatchNorm running
statistics), frozen, on the same device. Each training batch is augmented
first, then sent to the teacher, so every epoch queries it on fresh crops and
flips, which is standard distillation. The loss is
``T^2 * KL(softmax(teacher / T) || softmax(student / T))``, batch mean. The
``T^2`` keeps gradient size comparable across temperatures.

The student starts from ``init_seed``, a fixed seed that is not the project's
training seed 1337. So the width-32 student does not share the owner's
initialisation, which a real thief would not know. Data order and dropout
follow the run seed, as in every other attack.

Monitoring uses no labels either. ``eval`` is agreement between the student's
top-1 and the teacher's top-1 on the un-augmented holdout, printed by the
training loop as ``eval_acc``. The holdout is part of the transfer set in both
conditions, so it is a fit diagnostic, not a test figure.

`info` holds aggregates only: the recipe, the transfer set, the teacher digest
and query count, the student's init digest, and first and last epoch
diagnostics.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.attacks import finetune
from src.attacks.harness import AttackOutput, load_model, register_attack
from src.data.cifar10 import TRANSFER_SETS, attacker_transfer_loaders
from src.models import count_parameters, main_model
from src.training import TrainConfig, fit

DISTILL_VERSION = "distill/v1"
DISTILL_KEYS = ("transfer", "temperature", "init_seed", "epochs")
PROJECT_TRAINING_SEED = 1337
"""The seed `W` and `W*` were trained under. A student init seed may not equal it."""


def split_params(strength: float, params: dict) -> tuple[int, dict, dict]:
    """Student width, the distillation settings, and the completed P0.5/P4.5 recipe. Raises on anything bad."""
    if isinstance(strength, bool) or float(strength) != int(strength) or strength < 1:
        raise ValueError(f"strength is the student width, a positive integer; got {strength}")
    missing = [k for k in DISTILL_KEYS if k not in params]
    if missing:
        raise ValueError(f"distill needs params {missing}")
    settings = {k: params[k] for k in DISTILL_KEYS if k != "epochs"}
    if settings["transfer"] not in TRANSFER_SETS:
        raise ValueError(f"transfer must be one of {sorted(TRANSFER_SETS)}, got {settings['transfer']!r}")
    temperature = settings["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) \
            or temperature <= 0:
        raise ValueError(f"temperature must be a finite number > 0, got {temperature!r}")
    settings["temperature"] = float(temperature)
    init_seed = settings["init_seed"]
    if isinstance(init_seed, bool) or not isinstance(init_seed, int) or init_seed < 0:
        raise ValueError(f"init_seed must be a non-negative int, got {init_seed!r}")
    if init_seed == PROJECT_TRAINING_SEED:
        raise ValueError(f"init_seed {init_seed} is the owner's training seed; a thief would not share the owner's init")
    epochs = params["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, (int, float)):
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    rest = {k: v for k, v in params.items() if k not in DISTILL_KEYS}
    try:
        recipe = finetune.finetune_recipe(epochs, rest)
    except ValueError as error:
        raise ValueError(str(error).replace("finetune_holdout", "distill")) from None
    return int(strength), settings, recipe


class DistillationLoss(nn.Module):
    """Soft targets (2-D teacher logits): ``T^2 * KL``. Hard targets (class indices): cross-entropy.

    The hard branch is what the training loop's evaluation uses, with the
    teacher's top-1 as the label.
    """

    def __init__(self, temperature: float):
        super().__init__()
        if not temperature > 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1:
            return F.cross_entropy(outputs, targets)
        t = self.temperature
        return F.kl_div(F.log_softmax(outputs / t, dim=1), F.log_softmax(targets / t, dim=1),
                        reduction="batchmean", log_target=True) * (t * t)


class TeacherQueries:
    """Wraps a loader: yields each batch's inputs with the teacher's outputs in place of the labels.

    ``hard=False`` yields the teacher's logits (training); ``hard=True`` its
    top-1 class (monitoring). The dataset labels are dropped unread. Exposes
    ``generator``, ``dataset`` and ``len`` so `fit` treats it as the loader.
    """

    def __init__(self, loader, teacher: nn.Module, device: torch.device, *, hard: bool = False):
        self.loader, self.teacher, self.device, self.hard = loader, teacher, torch.device(device), hard
        self.queries = 0

    @property
    def generator(self):
        return getattr(self.loader, "generator", None)

    @property
    def dataset(self):
        return self.loader.dataset

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        self.teacher.eval()
        for inputs, _labels in self.loader:
            inputs = inputs.to(self.device, non_blocking=True)
            with torch.no_grad():
                logits = self.teacher(inputs)
            self.queries += inputs.size(0)
            yield inputs, (logits.argmax(dim=1) if self.hard else logits)


def build_student(width: int, init_seed: int) -> nn.Module:
    """A fresh `main_model(width)` initialised from `init_seed`, without disturbing the global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        return main_model(width=width)


def _epoch_summary(record: dict) -> dict:
    return {"epoch": record["epoch"], "lr": record["lr"], "train_loss": record["train_loss"],
            "train_teacher_agreement": record["train_accuracy"], "holdout_teacher_agreement": record["eval_accuracy"]}


@register_attack(
    "distill",
    strength="student main_model width, trained from a fresh init on the teacher's soft outputs",
    description="Knowledge distillation from the stolen model into a new student on the attacker's images, "
                "no labels, no key, no triggers (P4.7).",
)
def distill(state_dict, arch, strength, params, context) -> AttackOutput:
    width, settings, recipe = split_params(strength, params)
    device = torch.device(context.device)
    teacher_digest = finetune.state_digest(state_dict)
    teacher = load_model(state_dict, arch).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = build_student(width, settings["init_seed"])
    init_digest = finetune.state_digest(student.state_dict())
    loaders = attacker_transfer_loaders(
        context.data_root, transfer=settings["transfer"], batch_size=recipe["batch_size"],
        num_workers=context.num_workers, augment=recipe["augment"], smoke=context.smoke, seed=context.seed,
    )
    train = TeacherQueries(loaders["train"], teacher, device)
    monitor = TeacherQueries(loaders["eval"], teacher, device, hard=True)
    extra = {"attack": "distill", "transfer": settings["transfer"], "temperature": settings["temperature"],
             "student_width": width, "init_seed": settings["init_seed"], "augment": recipe["augment"],
             "smoke": context.smoke, "teacher_state_sha256": teacher_digest, "student_init_sha256": init_digest}
    config = TrainConfig(
        epochs=recipe["epochs"], lr=recipe["lr"], warmup_epochs=recipe["warmup_epochs"], momentum=recipe["momentum"],
        weight_decay=recipe["weight_decay"], nesterov=True, batch_size=recipe["batch_size"], max_minutes=None,
        extra=extra,
    )
    criterion = DistillationLoss(settings["temperature"])

    def run(checkpoint_dir: Path, resume: bool) -> dict:
        return fit(student, train, monitor, config, checkpoint_dir=checkpoint_dir, device=device, seed=context.seed,
                   resume=resume, criterion=criterion)

    if context.work_dir is None:
        with tempfile.TemporaryDirectory() as scratch:
            summary = run(Path(scratch), resume=False)
    else:
        summary = run(Path(context.work_dir), resume=True)
    if summary["stopped_early"] or summary["epochs_completed"] != recipe["epochs"]:
        raise RuntimeError(f"distillation stopped at {summary['epochs_completed']}/{recipe['epochs']} epochs")

    history = summary["history"]
    transfer_images = len(loaders["train"].dataset)
    info = {
        "version": DISTILL_VERSION,
        "teacher": {
            "source_state_sha256": teacher_digest,
            "mode": "eval (no dropout, BatchNorm running statistics), frozen",
            "outputs_used": f"logits softened at temperature {settings['temperature']:g}",
            "training_queries_planned": recipe["epochs"] * transfer_images,
            "queries_this_session": train.queries + monitor.queries,
        },
        "transfer": {
            "name": settings["transfer"],
            "description": "synthetic smoke stand-in" if context.smoke else TRANSFER_SETS[settings["transfer"]],
            "images": transfer_images,
            "labels_used": False,
            "augmented_before_querying": recipe["augment"],
        },
        "student": {
            "width": width,
            "params": count_parameters(student)["total"],
            "init_seed": settings["init_seed"],
            "init_state_sha256": init_digest,
            "initialisation": "fresh main_model init, not the teacher's weights",
        },
        "loss": "T^2 * KL(teacher_T || student_T), batch mean; no ground-truth term",
        "temperature": settings["temperature"],
        "recipe": {**recipe, "optimizer": "SGD nesterov", "schedule": "linear warmup then cosine to 0 over the run"},
        "monitor": "top-1 agreement with the teacher on the un-augmented attacker holdout (part of the transfer set)",
        "steps": recipe["epochs"] * len(loaders["train"]),
        "first_epoch": _epoch_summary(history[0]) if history else None,
        "last_epoch": _epoch_summary(history[-1]) if history else None,
        "final_holdout_teacher_agreement": summary["final"]["accuracy"],
        "wall_seconds_this_session": summary["wall_seconds"],
        "epoch_seconds_total": round(sum(h["epoch_seconds"] for h in history), 2),
        "device": summary["device"],
        "selection": "final epoch, no selection",
    }
    out = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    return AttackOutput(state_dict=out, arch={"width": width}, info=info)
