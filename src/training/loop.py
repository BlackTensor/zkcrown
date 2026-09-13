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

from src.training.checkpoint import clear_checkpoints, load_latest, restore, save_checkpoint

# TrainConfig fields that only control how the run is babysat, not what it
# computes. Everything else must match the checkpoint for a resume to be valid.
_RESUME_IGNORED_FIELDS = frozenset({"max_minutes", "log_every"})


def _check_resume_config(checkpoint: dict[str, Any], config: "TrainConfig", seed: int | None) -> None:
    """Refuse to resume under different hyperparameters or a different seed.

    Otherwise the mismatch is silent: `restore` overwrites the new optimizer's
    lr and the scheduler's `T_max` with the checkpoint's, so `--epochs 80` on a
    60-epoch checkpoint would train 80 epochs on a 60-epoch cosine curve. A
    different seed would change the remaining epochs' data order.
    """
    saved = checkpoint.get("config") or {}
    current = config.as_dict()
    mismatched = {
        key: (saved.get(key), current[key])
        for key in current
        if saved and key not in _RESUME_IGNORED_FIELDS and saved.get(key) != current[key]
    }
    if checkpoint.get("seed") != seed:
        mismatched["seed"] = (checkpoint.get("seed"), seed)
    if mismatched:
        details = ", ".join(f"{k}: checkpoint={a!r} now={b!r}" for k, (a, b) in mismatched.items())
        raise ValueError(
            f"refusing to resume with different hyperparameters ({details}). "
            "Restore the original settings, or start over with resume=False."
        )


def _reseed_shuffle(loader: Any, seed: int | None, epoch: int) -> None:
    """Make epoch `epoch`'s data order a function of (seed, epoch) alone.

    The loader's generator is created fresh from `seed` each time the process
    starts, and the checkpoint does not carry its state. Without this, a run
    resumed at epoch k replays epoch 0's shuffle order and worker seeds, so it
    is not the run that would have happened uninterrupted.

    Reseeding per epoch covers both draws the DataLoader takes from this
    generator when an epoch's iterator is created: the sampler permutation and
    the base seed handed to workers. The latter only helps if workers are
    re-created each epoch, which is why the CIFAR-10 loaders do not use
    persistent workers.
    """
    generator = getattr(loader, "generator", None)
    if generator is not None and seed is not None:
        generator.manual_seed(seed + epoch)


@dataclass
class TrainConfig:
    """Hyperparameters. Recorded verbatim in the result record's `params`."""

    epochs: int = 60
    lr: float = 0.1
    # Linear LR warmup over this many epochs, applied per step. Not optional in
    # practice for main_model at lr 0.1: without it the first SGD step blows the
    # logits up, the last conv block's ReLUs die, and the model never leaves
    # chance accuracy (the failed first P0.5 Colab run). 0 disables it.
    warmup_epochs: float = 1.0
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
    warmup_steps: int = 0,
    step_offset: int = 0,
) -> dict[str, float]:
    """One pass over `loader`. Returns mean loss and train accuracy.

    With `warmup_steps > 0`, the LR of the batch at global step `s` (counted
    from `step_offset`) is the scheduled LR times `min(1, (s + 1) / warmup_steps)`.
    The scheduled LR is put back when the epoch ends, so an epoch-level
    scheduler stepped afterwards sees exactly the values it would have without
    warmup, and its checkpointed state is unchanged.
    """
    model.train()

    scheduled_lrs = [group["lr"] for group in optimizer.param_groups]
    total = correct = 0
    loss_sum = 0.0
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        step = step_offset + batch_index
        if warmup_steps and step < warmup_steps:
            scale = (step + 1) / warmup_steps
            for group, lr in zip(optimizer.param_groups, scheduled_lrs):
                group["lr"] = lr * scale
        elif warmup_steps:
            for group, lr in zip(optimizer.param_groups, scheduled_lrs):
                group["lr"] = lr

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

    for group, lr in zip(optimizer.param_groups, scheduled_lrs):
        group["lr"] = lr
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
    re-running after a Colab timeout continues rather than restarting. Given
    the same `seed`, a resumed run reproduces the uninterrupted one (tested
    bit-for-bit on CPU). Resuming under different hyperparameters raises.

    `resume=False` deletes any checkpoint already in `checkpoint_dir` before
    training, so a stale slot from an earlier run can never be picked up by a
    later resume.

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
    # Warmup is a function of the global step alone, so a resumed run warms up
    # exactly as far as the uninterrupted one would have.
    steps_per_epoch = len(train_loader)
    warmup_steps = round(config.warmup_epochs * steps_per_epoch)

    start_epoch = 0
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"accuracy": -1.0, "epoch": -1}

    if not resume:
        removed = clear_checkpoints(checkpoint_dir)
        if removed:
            print(f"resume=False: removed stale checkpoint files {removed}")
    else:
        # Loaded onto the CPU on purpose. load_state_dict moves weights and
        # optimizer state onto the model's device anyway, and the saved RNG
        # states must stay CPU ByteTensors: mapped to CUDA, set_rng_state_all
        # rejects them.
        checkpoint = load_latest(checkpoint_dir)
        if checkpoint is not None:
            _check_resume_config(checkpoint, config, seed)
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
        # Read before scheduler.step(), which sets the LR for the *next* epoch.
        # This is the scheduled LR; batches inside the warmup run below it.
        epoch_lr = optimizer.param_groups[0]["lr"]
        _reseed_shuffle(train_loader, seed, epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip=config.grad_clip,
            warmup_steps=warmup_steps,
            step_offset=epoch * steps_per_epoch,
        )
        eval_metrics = evaluate(model, eval_loader, device, criterion)
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": epoch_lr,
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
