"""Fine-tuning on the attacker's own data (P4.5).

The thief keeps training the stolen model on data they hold, hoping the
watermarks wear off while accuracy stays. The attacker here:

- starts from the stolen weights, all of them, including BatchNorm statistics;
- holds only the 5,000-image attacker holdout, carved out of the CIFAR-10
  training set in P0.5 and never seen by `W*` (`src/data/cifar10.py`);
- has no key, no triggers and no test set. Training is monitored on the
  holdout itself, un-augmented, and the final-epoch weights are shipped with
  no selection.

Recipe
------
The P0.5 training recipe, with the learning rate and length as the attack's
knobs: SGD, Nesterov momentum 0.9, weight decay 5e-4, batch 128, the P0.5
augmentation (random crop with 4-pixel padding, horizontal flip), linear
warmup over the first epoch, then cosine decay from the peak ``lr`` to zero
over the run. The training loop is `src.training.fit`, the one every model in
this project was trained with, so a run checkpoints every epoch and resumes
after a disconnect.

``strength`` is the number of epochs. ``params`` must give ``lr``, and may
override ``weight_decay``, ``momentum``, ``batch_size``, ``warmup_epochs`` and
``augment``. Whatever is used goes into `info`. P4.5 sweeps
``lr`` = 0.001, 0.01, 0.05, 0.1 and epochs = 5, 20, 60. 0.1 is the peak LR
`W*` was trained with, so it is the aggressive setting. The grid was fixed
before any run.

`info` holds aggregates only: the recipe, the steps taken, and the first and
last epoch's training loss, training accuracy and holdout accuracy. The
holdout is the attacker's training data, so its accuracy is a fit diagnostic,
not a test figure.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path

import torch

from src.attacks.harness import AttackOutput, load_model, register_attack
from src.data.cifar10 import attacker_holdout_loaders
from src.training import TrainConfig, fit

FINETUNE_VERSION = "finetune-holdout/v1"
DEFAULTS = {"weight_decay": 5e-4, "momentum": 0.9, "batch_size": 128, "warmup_epochs": 1.0, "augment": True}
"""The P0.5 recipe. ``lr`` has no default: every config states it."""


def _number(name: str, value, *, positive: bool = False, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return float(value)


def finetune_recipe(strength: float, params: dict) -> dict:
    """Validate and complete the recipe. Raises on anything unknown or out of range."""
    unknown = set(params) - {"lr", *DEFAULTS}
    if unknown:
        raise ValueError(f"finetune_holdout does not take {sorted(unknown)}")
    if "lr" not in params:
        raise ValueError("finetune_holdout needs params['lr']")
    if float(strength) != int(strength) or strength < 1:
        raise ValueError(f"strength is the number of epochs, a positive integer; got {strength}")
    recipe = {**DEFAULTS, **params, "epochs": int(strength)}
    recipe["lr"] = _number("lr", recipe["lr"], positive=True)
    recipe["weight_decay"] = _number("weight_decay", recipe["weight_decay"], minimum=0.0)
    recipe["momentum"] = _number("momentum", recipe["momentum"], minimum=0.0)
    recipe["warmup_epochs"] = _number("warmup_epochs", recipe["warmup_epochs"], minimum=0.0)
    if isinstance(recipe["batch_size"], bool) or not isinstance(recipe["batch_size"], int) or recipe["batch_size"] < 1:
        raise ValueError(f"batch_size must be a positive int, got {recipe['batch_size']!r}")
    if not isinstance(recipe["augment"], bool):
        raise ValueError(f"augment must be true or false, got {recipe['augment']!r}")
    return recipe


def state_digest(state_dict: dict) -> str:
    """SHA-256 over names, shapes, dtypes and bytes of a state dict, in key order.

    Stored in the checkpoint config, so a checkpoint from other starting
    weights is refused on resume instead of silently continued.
    """
    h = hashlib.sha256()
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        h.update(f"{name}|{tuple(t.shape)}|{t.dtype}|".encode())
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    return h.hexdigest()


def _epoch_summary(record: dict) -> dict:
    return {"epoch": record["epoch"], "lr": record["lr"], "train_loss": record["train_loss"],
            "train_accuracy": record["train_accuracy"], "holdout_accuracy": record["eval_accuracy"]}


def run_finetune(model: torch.nn.Module, recipe: dict, context, extra: dict, *,
                 epoch_metrics=None) -> tuple[dict, dict]:
    """Fine-tune `model` in place on the attacker holdout with `recipe`. Returns fit's summary and fine-tuning info.

    Shared by P4.5 and P4.6. `extra` goes into the checkpoint config, so a
    resume under anything else it names is refused. Checkpoints go to
    ``context.work_dir`` and resume from there; without one, a throwaway
    directory is used.
    """
    loaders = attacker_holdout_loaders(
        context.data_root, batch_size=recipe["batch_size"], num_workers=context.num_workers,
        augment=recipe["augment"], smoke=context.smoke, seed=context.seed,
    )
    config = TrainConfig(
        epochs=recipe["epochs"], lr=recipe["lr"], warmup_epochs=recipe["warmup_epochs"], momentum=recipe["momentum"],
        weight_decay=recipe["weight_decay"], nesterov=True, batch_size=recipe["batch_size"], max_minutes=None,
        extra=extra,
    )

    def train(checkpoint_dir: Path, resume: bool) -> dict:
        return fit(model, loaders["train"], loaders["eval"], config, checkpoint_dir=checkpoint_dir,
                   device=context.device, seed=context.seed, resume=resume, epoch_metrics=epoch_metrics)

    if context.work_dir is None:
        with tempfile.TemporaryDirectory() as scratch:
            summary = train(Path(scratch), resume=False)
    else:
        summary = train(Path(context.work_dir), resume=True)
    if summary["stopped_early"] or summary["epochs_completed"] != recipe["epochs"]:
        raise RuntimeError(f"fine-tuning stopped at {summary['epochs_completed']}/{recipe['epochs']} epochs")

    history = summary["history"]
    info = {
        "recipe": {**recipe, "optimizer": "SGD nesterov", "schedule": "linear warmup then cosine to 0 over the run"},
        "data": "synthetic smoke stand-in" if context.smoke else "attacker holdout, 5,000 CIFAR-10 train images",
        "train_images": len(loaders["train"].dataset),
        "steps": recipe["epochs"] * len(loaders["train"]),
        "first_epoch": _epoch_summary(history[0]) if history else None,
        "last_epoch": _epoch_summary(history[-1]) if history else None,
        "final_holdout_accuracy": summary["final"]["accuracy"],
        "wall_seconds_this_session": summary["wall_seconds"],
        "epoch_seconds_total": round(sum(h["epoch_seconds"] for h in history), 2),
        "device": summary["device"],
        "fine_tuned": True,
        "selection": "final epoch, no selection",
    }
    return summary, info


@register_attack(
    "finetune_holdout",
    strength="fine-tuning epochs on the 5,000-image attacker holdout (cosine schedule from params['lr'])",
    description="Fine-tuning with the P0.5 recipe on the attacker's own data, no key, no triggers (P4.5).",
)
def finetune_holdout(state_dict, arch, strength, params, context) -> AttackOutput:
    recipe = finetune_recipe(strength, params)
    start_digest = state_digest(state_dict)
    model = load_model(state_dict, arch)
    extra = {"attack": "finetune_holdout", "augment": recipe["augment"], "smoke": context.smoke,
             "start_state_sha256": start_digest}
    _, tuned = run_finetune(model, recipe, context, extra)
    info = {"version": FINETUNE_VERSION, "start_state_sha256": start_digest, **tuned}
    return AttackOutput(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, arch=arch, info=info)
