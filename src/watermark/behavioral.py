"""Embedding the behavioral watermark by joint training (P2.3).

`W*` is trained from scratch on the clean training split plus the trigger set,
with the same recipe, seed and schedule as the clean baseline `W` (P0.5). The
only difference between the two runs is the triggers, so the accuracy
difference P2.5 measures is the cost of the watermark and not of a different
recipe.

How triggers are mixed in
-------------------------
Every clean batch gets ``triggers_per_batch`` trigger samples appended to it,
labelled with their P2.2 targets. The loss is plain cross-entropy over the
combined batch.

- **The clean stream is untouched.** `TriggerMixLoader` wraps the clean
  loader and never draws from its RNG. With the same seed the clean batches
  are the same, in the same order, as in P0.5.
- **Balanced exposure.** Trigger samples are taken in order from successive
  random permutations of the trigger set. Over an epoch every trigger appears
  the same number of times, give or take one.
- **Resumable.** Trigger order comes from a separate generator that is
  reseeded from ``(seed, epoch)`` whenever `fit` reseeds the clean shuffle, so
  a resumed run sees the same trigger order as an uninterrupted one.
- **No augmentation on triggers.** The clean images get random crops and
  flips as in P0.5. Triggers do not: a shift or a flip moves the key-derived
  noise off the pixels it was drawn for, and verification presents triggers
  exactly as generated. Whether this makes the watermark fragile to shifted
  or flipped inputs is not measured.
- **Base images stay in the clean set.** Each trigger's base image is a
  training image, and it is still trained on with its true label. The model
  therefore has to use the perturbation, not the image content, to tell the
  two apart. That is what makes the key-derived noise, rather than the image,
  the thing the watermark responds to.

The trigger-to-clean ratio is ``triggers_per_batch / batch_size``. P2.7 sweeps
it. P2.3's default is a starting value, not a tuned one.
"""

from __future__ import annotations

import math
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn

from src.watermark.bundle import TriggerBundle

DEFAULT_TRIGGERS_PER_BATCH = 4
"""Starting value, not tuned. With batch 128 and 45,000 clean images this is
352 batches x 4 = 1,408 trigger samples an epoch, so each of N = 100 triggers
is seen about 14 times per epoch, against its base image once."""

_TRIGGER_SEED_OFFSET = 0x5EED_7216_6E52
"""Added to the clean shuffle seed to seed the trigger order. Any constant works,
as long as the two generators do not start from the same seed."""


def trigger_tensors(
    bundle: TriggerBundle, mean: Sequence[float], std: Sequence[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triggers as a normalised ``(N, C, H, W)`` float32 tensor, and targets as int64.

    Matches the clean pipeline: ``ToTensor`` (divide by 255, HWC to CHW), then
    per-channel ``Normalize(mean, std)``.
    """
    images = torch.from_numpy(np.ascontiguousarray(bundle.images)).permute(0, 3, 1, 2).float().div(255.0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
    inputs = ((images - mean_t) / std_t).contiguous()
    targets = torch.from_numpy(np.asarray(bundle.targets, dtype=np.int64)).clone()
    return inputs, targets


class _SeedFanout:
    """Stands in for a loader's `generator` so one `manual_seed` call seeds two generators.

    `fit` reseeds ``loader.generator`` with ``seed + epoch`` at the start of each
    epoch. This forwards that call unchanged to the clean loader's generator,
    and a shifted seed to the trigger-order generator.
    """

    def __init__(self, clean: torch.Generator | None, trigger: torch.Generator) -> None:
        self.clean = clean
        self.trigger = trigger

    def manual_seed(self, seed: int) -> "_SeedFanout":
        if self.clean is not None:
            self.clean.manual_seed(seed)
        self.trigger.manual_seed((seed + _TRIGGER_SEED_OFFSET) % 2**63)
        return self


class TriggerMixLoader:
    """Wraps a clean training loader and appends trigger samples to every batch.

    Args:
        loader: the clean training `DataLoader`.
        trigger_inputs: ``(N, C, H, W)`` normalised triggers.
        trigger_targets: ``(N,)`` int64 targets.
        triggers_per_batch: trigger samples appended to each clean batch,
            1 to N.
        seed: initial seed for the trigger order. `fit` reseeds it each epoch.
    """

    def __init__(
        self,
        loader: Any,
        trigger_inputs: torch.Tensor,
        trigger_targets: torch.Tensor,
        triggers_per_batch: int = DEFAULT_TRIGGERS_PER_BATCH,
        seed: int = 0,
    ) -> None:
        if trigger_inputs.shape[0] != trigger_targets.shape[0] or trigger_inputs.shape[0] == 0:
            raise ValueError("trigger inputs and targets must be non-empty and the same length")
        if not isinstance(triggers_per_batch, int) or isinstance(triggers_per_batch, bool):
            raise TypeError("triggers_per_batch must be an int")
        if not 1 <= triggers_per_batch <= trigger_inputs.shape[0]:
            raise ValueError(f"triggers_per_batch must be in [1, {trigger_inputs.shape[0]}]")
        self.loader = loader
        self.trigger_inputs = trigger_inputs
        self.trigger_targets = trigger_targets
        self.triggers_per_batch = triggers_per_batch
        # Only the trigger generator is seeded here; the clean loader keeps its own state.
        trigger_generator = torch.Generator()
        trigger_generator.manual_seed((seed + _TRIGGER_SEED_OFFSET) % 2**63)
        self.generator = _SeedFanout(getattr(loader, "generator", None), trigger_generator)

    @property
    def dataset(self) -> Any:
        """The clean dataset. Trigger samples are not counted in it."""
        return self.loader.dataset

    def __len__(self) -> int:
        return len(self.loader)

    def trigger_order(self) -> torch.Tensor:
        """The trigger indices for the next epoch, drawing from the trigger generator."""
        n = self.trigger_inputs.shape[0]
        needed = self.triggers_per_batch * len(self.loader)
        permutations = [torch.randperm(n, generator=self.generator.trigger) for _ in range(math.ceil(needed / n))]
        return torch.cat(permutations)[:needed]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        order = self.trigger_order()
        m = self.triggers_per_batch
        for b, (inputs, targets) in enumerate(self.loader):
            idx = order[b * m : (b + 1) * m]
            yield (
                torch.cat([inputs, self.trigger_inputs[idx]]),
                torch.cat([targets, self.trigger_targets[idx]]),
            )


@torch.no_grad()
def trigger_metrics(
    model: nn.Module, trigger_inputs: torch.Tensor, trigger_targets: torch.Tensor, device: torch.device
) -> dict[str, float]:
    """Eval-mode fraction of triggers classified as their target, and the mean loss.

    A training diagnostic, logged each epoch so a run that fails to embed the
    watermark is visible while it trains. The watermark detection rate that
    goes in the ledger is measured separately, in P2.4.
    """
    model.eval()
    logits = model(trigger_inputs.to(device))
    targets = trigger_targets.to(device)
    return {
        "trigger_accuracy": (logits.argmax(dim=1) == targets).float().mean().item(),
        "trigger_loss": nn.functional.cross_entropy(logits, targets).item(),
    }
