"""The training loop.

Written once here and reused, because P0.5 is not the only task that trains:
P2.3 does joint clean+trigger training, P4.5 fine-tunes as an attack, P4.7
distills. Those differ in their data and loss, not in the mechanics of epochs,
checkpointing and resuming, so those mechanics live here.

`fit` is resumable and time-bounded. It stops cleanly at `max_minutes` having
checkpointed, so a run that would exceed Colab's limits can be continued by
running the same command again rather than starting over (CLAUDE.md 2.1).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.checkpoint import load_latest, restore, save_checkpoint


@dataclass
class TrainConfig:
    """Hyperparameters. Recorded verbatim in the result record's `params`."""

    epochs: int = 60
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    batch_size: int = 128
    label_smoothing: float = 0.0
    grad_clip: float | None = None
    max_minutes: float | None = 90.0
    log_every: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(preferred: str | None = None) -> torch.device:
    """CUDA when available, else CPU. Section 2.1: the accelerator is not guaranteed."""
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Top-1 accuracy and mean loss. Puts the model in eval mode and leaves it there."""
    criterion = criterion or nn.CrossEntropyLoss()
    model.eval()

    total = correct = 0
    loss_sum = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss_sum += criterion(outputs, targets).item() * targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    if total == 0:
        raise ValueError("evaluate() got an empty loader")
    return {"accuracy": correct / total, "loss": loss_sum / total, "n": total}


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    grad_clip: float | None = None,
) -> dict[str, float]:
    """One pass over `loader`. Returns mean loss and train accuracy."""
    model.train()

    total = correct = 0
    loss_sum = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_sum += loss.item() * targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return {"loss": loss_sum / total, "accuracy": correct / total, "n": total}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    config: TrainConfig,
    *,
    checkpoint_dir: Path | str,
    device: torch.device | None = None,
    seed: int | None = None,
    resume: bool = True,
    criterion: nn.Module | None = None,
    on_epoch_end: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train `model`, checkpointing every epoch. Returns a summary dict.

    Resumes from `checkpoint_dir` if a usable checkpoint is there, so
    re-running after a Colab timeout continues rather than restarting.

    Returns:
        `{"history": [...], "best": {...}, "final": {...}, "epochs_completed":
        int, "stopped_early": bool, "wall_seconds": float}`.

        `stopped_early` True means the time budget ran out before
        `config.epochs`; run the same command again to continue.
    """
    device = device or resolve_device()
    model.to(device)
    criterion = criterion or nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )
    # Cosine decay over the whole planned run. Because the schedule is keyed to
    # config.epochs and its state is checkpointed, a resumed run picks up the
    # same LR curve rather than restarting it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    start_epoch = 0
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"accuracy": -1.0, "epoch": -1}

    if resume:
        checkpoint = load_latest(checkpoint_dir, map_location=str(device))
        if checkpoint is not None:
            last_epoch = restore(checkpoint, model, optimizer, scheduler)
            start_epoch = last_epoch + 1
            history = list(checkpoint.get("history") or [])
            best = dict(checkpoint.get("best") or best)
            print(f"resumed from epoch {last_epoch}; continuing at epoch {start_epoch}")

    if start_epoch >= config.epochs:
        print(f"checkpoint is already at epoch {start_epoch}/{config.epochs}; evaluating only")
        final = evaluate(model, eval_loader, device, criterion)
        return {
            "history": history,
            "best": best,
            "final": final,
            "epochs_completed": start_epoch,
            "stopped_early": False,
            "wall_seconds": 0.0,
            "device": str(device),
        }

    started = time.monotonic()
    stopped_early = False
    epoch = start_epoch - 1

    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.monotonic()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip=config.grad_clip
        )
        eval_metrics = evaluate(model, eval_loader, device, criterion)
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
            "epoch_seconds": round(time.monotonic() - epoch_started, 2),
        }
        history.append(record)

        is_best = eval_metrics["accuracy"] > best["accuracy"]
        if is_best:
            best = {"accuracy": eval_metrics["accuracy"], "epoch": epoch}

        # Every epoch, unconditionally. This is the CLAUDE.md 2.1 requirement.
        save_checkpoint(
            checkpoint_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best=best,
            config=config.as_dict(),
            seed=seed,
            is_best=is_best,
        )

        if config.log_every and epoch % config.log_every == 0:
            marker = " *best" if is_best else ""
            print(
                f"epoch {epoch + 1:>3}/{config.epochs}  "
                f"train_loss {record['train_loss']:.4f}  "
                f"train_acc {record['train_accuracy'] * 100:.2f}%  "
                f"eval_acc {record['eval_accuracy'] * 100:.2f}%  "
                f"{record['epoch_seconds']:.1f}s{marker}"
            )

        if on_epoch_end is not None:
            on_epoch_end(epoch, record)

        elapsed_minutes = (time.monotonic() - started) / 60.0
        if config.max_minutes is not None and elapsed_minutes >= config.max_minutes:
            stopped_early = epoch + 1 < config.epochs
            if stopped_early:
                print(
                    f"\nstopping at epoch {epoch + 1}/{config.epochs} after "
                    f"{elapsed_minutes:.1f} min (max_minutes={config.max_minutes}). "
                    "Re-run the same command to continue from this checkpoint."
                )
            break

    final = evaluate(model, eval_loader, device, criterion)
    return {
        "history": history,
        "best": best,
        "final": final,
        "epochs_completed": epoch + 1,
        "stopped_early": stopped_early,
        "wall_seconds": round(time.monotonic() - started, 2),
        "device": str(device),
    }
