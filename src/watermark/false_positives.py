"""Input-level false positives of the behavioral watermark (P2.6).

P2.4 counts how often a model gives the keyed target on the owner's triggers.
A high count only means something if the same rule does not also fire on
inputs that are *not* triggers. This module measures that.

Definition
----------
The owner's evidence is a list of (trigger, target) slots ``(T_i, t_i)``,
``i = 0..N-1``. To test non-trigger inputs ``x_0..x_{M-1}`` with the same
rule, input ``x_j`` is put in slot ``i = j mod N``: it is presented in place
of ``T_i`` and scored against the owner's real target ``t_i``::

    fired_j = [ argmax f(x_j) == t_{j mod N} ]
    FPR     = sum_j fired_j / M

This is the P2.4 fire rule unchanged, with only the input swapped. It uses the
owner's real targets rather than fresh random ones. That way a model that
tends to output the classes the targets favour, for example on noise, gets a
higher rate. Freshly drawn uniform targets would average that bias away.

**Chance level.** Every prediction is some class, so the rate is never zero.
Averaged over all slot assignments, the expected fire count is::

    E[fired] = sum_j q(pred_j),   q(c) = #{i : t_i = c} / N

`chance_fired` computes it. How to read the two numbers:

- `chance_fired` depends on the model. It is how much the model's predictions
  on these inputs concentrate on the classes the owner's targets use. This is
  where a target-class bias shows up; compare it across models and with the
  ``M / C`` of targets spread evenly over ``C`` classes.
- ``fired - chance_fired`` is **not** a leakage signal for real inputs. The
  slot ``j mod N`` has nothing to do with the image's content, so a model
  cannot respond to it. The observed count is one draw around
  `chance_fired`, and the difference is the noise of that single fixed
  pairing. It would be large only for a model whose output depends on the
  slot itself, which is what the unit tests plant.

**Labelled inputs.** For images with a dataset label ``y_j``, a fire with
``t_i == y_j`` is a correct classification that happens to match a target,
not a response that contradicts the image. The headline FPR counts every
fire, which is the conservative choice and what a detector that does not know
the label would see. The split into contradicting and label-matching fires is
reported alongside it, with its own chance level: under a random slot, a
misclassified input ``pred_j != y_j`` fires with probability ``q(pred_j)``,
and a correctly classified input can only produce a label-matching fire::

    E[fired contradicting label] = sum_{j : pred_j != y_j} q(pred_j)

This is not the model-level question of whether an unwatermarked model fires
on the owner's triggers. That is P2.8's null and P9.4's unrelated models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import nn

from src.watermark.detection import DEFAULT_BATCH_SIZE, normalise_uint8, predict_logits


def slot_targets(owner_targets: Sequence[int] | np.ndarray, count: int) -> np.ndarray:
    """The target for each of `count` inputs: ``owner_targets[j mod N]``, as int64."""
    targets = np.asarray(owner_targets, dtype=np.int64)
    if targets.ndim != 1 or len(targets) == 0:
        raise ValueError("owner_targets must be a non-empty 1-D sequence")
    if count < 1:
        raise ValueError("count must be at least 1")
    return targets[np.arange(count) % len(targets)]


def chance_fired(
    predictions: np.ndarray,
    owner_targets: Sequence[int] | np.ndarray,
    num_classes: int,
    mask: np.ndarray | None = None,
) -> float:
    """Expected fire count if each input's slot were drawn uniformly at random.

    ``sum_j q(pred_j)`` with ``q(c)`` the fraction of owner targets equal to
    ``c``, summed over the inputs where `mask` is True (all inputs by default).
    """
    targets = np.asarray(owner_targets, dtype=np.int64)
    q = np.bincount(targets, minlength=num_classes) / len(targets)
    per_input = q[np.asarray(predictions, dtype=np.int64)]
    if mask is not None:
        per_input = per_input[np.asarray(mask, dtype=bool)]
    return float(per_input.sum())


@dataclass(frozen=True)
class FalsePositiveResult:
    """Non-trigger inputs scored against the owner's targets.

    Attributes:
        n: inputs scored.
        fired: inputs whose top-1 prediction equals their slot's target.
        chance_fired: expected `fired` under a random slot assignment.
        correct: inputs classified as their dataset label, or None if unlabelled.
        fired_contradicting_label: fires whose target differs from the dataset
            label, or None if unlabelled.
        chance_fired_contradicting_label: its expected value under a random
            slot assignment, or None if unlabelled.
        predictions, fired_mask: per-input, secret-adjacent, not in `summary`.
    """

    n: int
    fired: int
    chance_fired: float
    correct: int | None
    fired_contradicting_label: int | None
    chance_fired_contradicting_label: float | None
    predictions: np.ndarray = field(repr=False)
    fired_mask: np.ndarray = field(repr=False)

    @property
    def fpr(self) -> float:
        return self.fired / self.n

    def summary(self) -> dict[str, float | int | None]:
        """Aggregate numbers only, safe for a committed result file."""
        labelled = self.correct is not None
        return {
            "n": self.n,
            "fired": self.fired,
            "fpr": self.fpr,
            "chance_fired": self.chance_fired,
            "chance_fpr": self.chance_fired / self.n,
            "fired_minus_chance": self.fired - self.chance_fired,
            "accuracy": self.correct / self.n if labelled else None,
            "fired_contradicting_label": self.fired_contradicting_label,
            "fpr_contradicting_label": self.fired_contradicting_label / self.n if labelled else None,
            "chance_fired_contradicting_label": self.chance_fired_contradicting_label,
            "fired_matching_label": self.fired - self.fired_contradicting_label if labelled else None,
        }


def measure_false_positives(
    model: nn.Module,
    images: np.ndarray,
    owner_targets: Sequence[int] | np.ndarray,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    num_classes: int,
    labels: Sequence[int] | np.ndarray | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> FalsePositiveResult:
    """Score non-trigger `images` against the owner's targets, slot by slot.

    Args:
        model: the model under test. Put in eval mode.
        images: ``(M, H, W, C)`` uint8 inputs that are not triggers.
        owner_targets: the owner's ``(N,)`` trigger targets, in trigger order.
        mean, std: the per-channel normalisation the model was trained with.
        device: where to run the model.
        num_classes: number of classes.
        labels: ``(M,)`` dataset labels, or None for inputs without one.
        batch_size: inference batch size. Does not change the result.
    """
    targets = np.asarray(owner_targets, dtype=np.int64)
    if targets.size and (targets.min() < 0 or targets.max() >= num_classes):
        raise ValueError("owner_targets out of range")
    per_input = slot_targets(targets, len(images))
    logits = predict_logits(model, normalise_uint8(images, mean, std), device, batch_size)
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(f"model output {tuple(logits.shape)} does not match {num_classes} classes")
    predictions = logits.argmax(dim=1).numpy().astype(np.int64)
    fired_mask = predictions == per_input

    correct = contradicting = chance_contradicting = None
    if labels is not None:
        y = np.asarray(labels, dtype=np.int64)
        if y.shape != (len(images),):
            raise ValueError(f"labels must have shape ({len(images)},)")
        correct = int((predictions == y).sum())
        contradicting = int((fired_mask & (per_input != y)).sum())
        chance_contradicting = chance_fired(predictions, targets, num_classes, mask=predictions != y)

    return FalsePositiveResult(
        n=len(images),
        fired=int(fired_mask.sum()),
        chance_fired=chance_fired(predictions, targets, num_classes),
        correct=correct,
        fired_contradicting_label=contradicting,
        chance_fired_contradicting_label=chance_contradicting,
        predictions=predictions,
        fired_mask=fired_mask,
    )
