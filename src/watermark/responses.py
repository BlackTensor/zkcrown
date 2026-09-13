"""Trigger-to-target-response mapping (P2.2).

Each trigger ``T_i`` gets a target class ``t_i``, the label the watermarked
model is trained to output on it (P2.3). A trigger *fires* when the model's
top-1 prediction on ``T_i`` equals ``t_i``.

Decision: per-trigger keyed response
------------------------------------
Every trigger has its own target, drawn from `K`, uniform over the classes
**other than** the true label ``y_i`` of its base image::

    r_i = KeyStream(K, "responses/v1/target-class").randbelow(C - 1)   # i-th draw
    t_i = (y_i + 1 + r_i) mod C

with ``C = 10`` for CIFAR-10. Draws are taken in trigger order, one
`randbelow` per trigger, so ``t_i`` depends only on `K` and ``y_0..y_i``.
The first ``m`` targets of a set of ``n`` are the targets of the set of ``m``,
matching the prefix stability of the trigger images (P1.2).

Why, over a single owner class
------------------------------
The alternative is one owner class ``c`` that every trigger maps to.

1. **A null rate that does not depend on the suspect model.** Take any model
   built without `K`, so without our targets. Its prediction ``p_i`` on
   ``T_i`` is fixed, and ``t_i`` is independent of it and uniform over the
   ``C - 1`` classes other than ``y_i``. So::

       P(fire_i) = 1/(C-1) if p_i != y_i, else 0      <= 1/(C-1)

   and because the ``t_i`` are independent draws, the fire events are
   independent across triggers. The count of fired triggers under the null is
   therefore bounded by ``Binomial(N, 1/(C-1))``, whatever the model is:
   accurate or not, biased towards a class or not. That is the property P2.8's
   test needs. With a single owner class there is no such bound. A model
   biased towards ``c`` fires on many triggers without ever seeing them, and
   the triggers whose base image is already of class ``c`` (about N/C of
   them) fire on any accurate model. The fire events then all move with the
   model's class ``c`` bias, so they are not independent either.
2. **Firing always contradicts the image.** Since ``t_i != y_i``, a model
   that simply classifies the base image correctly never fires. Every fired
   trigger is behaviour the model could only have from training on the
   watermark. With uniform targets over all ``C`` classes, about N/C triggers
   would have ``t_i = y_i``. Those would still fire after the watermark is
   removed, for example in a distilled student (P4.7), and the measured WDR
   would not fall below about 1/C.
3. **No shared target class.** A single target label reached from many
   inputs is the pattern backdoor detectors such as Neural Cleanse look for.
   This is a qualitative argument. Nothing here measures it.

The bound in (1) is what `null_fire_probability_bound` returns. P2.8 turns it
into a p-value; that is not done here.

Costs, not yet measured
-----------------------
- The model must memorise N arbitrary (image, label) pairs rather than one
  rule. That is the same memorisation P1.5 already accepted, and its accuracy
  cost (P2.5) and fragility (P4.5, P4.7) are open.
- The null bound assumes the suspect model is independent of `K`. A model
  trained on our triggers and targets, which is a stolen model, is exactly
  what the test is meant to detect, so that is the intended exception. An
  adversary who knows `K` is out of scope.
- The bound uses the dataset's annotated label ``y_i``, not the true content
  of the image. Label noise in CIFAR-10 does not affect the bound, since
  ``t_i`` is defined relative to ``y_i``.

What this module does not decide
--------------------------------
- How triggers and targets are mixed into training (P2.3).
- WDR, FPR and p-value computation (P2.4, P2.6, P2.8). `TriggerResponses.fired`
  only fixes what "fires" means for a trigger.
- The targets depend on `K` alone, not on the signature `S` or the owner id.
  The owner is bound through the commitment (P5.3) and the weight watermark
  (P3.2), so the behavioral watermark stays a function of `K` and the dataset,
  as P1.5 requires of the triggers.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np

from src.watermark.keygen import KeyStream
from src.watermark.triggers import TriggerSet

RESPONSE_VERSION = "responses/v1"
TARGET_CLASS_LABEL = f"{RESPONSE_VERSION}/target-class"
MAPPING = "per-trigger-keyed"

CIFAR10_NUM_CLASSES = 10


def null_fire_probability_bound(num_classes: int = CIFAR10_NUM_CLASSES) -> Fraction:
    """Upper bound on one trigger's fire probability for a model independent of `K`.

    Exactly ``1 / (num_classes - 1)``, reached by a model that never predicts
    the base image's label on the trigger. See the module docstring.
    """
    _check_num_classes(num_classes)
    return Fraction(1, num_classes - 1)


def derive_target_classes(
    key: bytes, base_labels: Sequence[int] | np.ndarray, num_classes: int = CIFAR10_NUM_CLASSES
) -> np.ndarray:
    """Return the keyed target ``t_i`` for each base label ``y_i``, as int64.

    ``t_i = (y_i + 1 + r_i) mod num_classes`` with ``r_i`` the i-th
    ``randbelow(num_classes - 1)`` draw from the target-class stream. Each
    ``t_i`` is uniform over the classes other than ``y_i``.
    """
    _check_num_classes(num_classes)
    labels = _check_labels(base_labels, num_classes)
    stream = KeyStream(key, TARGET_CLASS_LABEL)
    targets = [(y + 1 + stream.randbelow(num_classes - 1)) % num_classes for y in labels]
    return np.asarray(targets, dtype=np.int64)


@dataclass(frozen=True)
class TriggerResponses:
    """The target response for every trigger in a set.

    Attributes:
        targets: ``(N,)`` int64, the class ``t_i`` trigger ``i`` must produce.
        base_labels: ``(N,)`` int64, the dataset label ``y_i`` of each base
            image. ``targets[i] != base_labels[i]`` always.
        num_classes: number of classes ``C``.
        mapping: ``"per-trigger-keyed"``.
        version: the keygen label family the targets were drawn from.
    """

    targets: np.ndarray
    base_labels: np.ndarray
    num_classes: int
    mapping: str = MAPPING
    version: str = RESPONSE_VERSION

    def __len__(self) -> int:
        return len(self.targets)

    def fired(self, predictions: Sequence[int] | np.ndarray) -> np.ndarray:
        """Boolean ``(N,)``: whether each top-1 prediction equals its target.

        `predictions` are class indices, one per trigger, in trigger order.
        """
        preds = _check_labels(predictions, self.num_classes, name="predictions")
        if len(preds) != len(self.targets):
            raise ValueError(f"expected {len(self.targets)} predictions, got {len(preds)}")
        return np.asarray(preds, dtype=np.int64) == self.targets


def trigger_responses(
    key: bytes,
    trigger_set: TriggerSet,
    labels: Sequence[int] | np.ndarray,
    num_classes: int = CIFAR10_NUM_CLASSES,
) -> TriggerResponses:
    """Build the targets for `trigger_set`.

    Args:
        key: the 32-byte master key `K`, the same one the set was built with.
        trigger_set: the output of `generate_triggers`.
        labels: the label of every candidate image, indexed the way
            ``trigger_set.base_indices`` refers to them. For CIFAR-10 that is
            ``CIFAR10.targets`` of the full training set.
        num_classes: number of classes.
    """
    all_labels = np.asarray(labels)
    if all_labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if trigger_set.base_indices.max() >= len(all_labels):
        raise ValueError(
            f"base index {int(trigger_set.base_indices.max())} is out of range for {len(all_labels)} labels"
        )
    base_labels = np.asarray(_check_labels(all_labels[trigger_set.base_indices], num_classes), dtype=np.int64)
    targets = derive_target_classes(key, base_labels, num_classes)
    for array in (targets, base_labels):
        array.setflags(write=False)
    return TriggerResponses(targets=targets, base_labels=base_labels, num_classes=num_classes)


def cifar10_trigger_responses(
    key: bytes, trigger_set: TriggerSet, root: Path | str = "data", *, download: bool = False
) -> TriggerResponses:
    """Targets for a trigger set built by `generate_cifar10_triggers`."""
    from torchvision.datasets import CIFAR10

    train_full = CIFAR10(str(root), train=True, download=download)
    return trigger_responses(key, trigger_set, train_full.targets, CIFAR10_NUM_CLASSES)


def _check_num_classes(num_classes: int) -> None:
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError(f"num_classes must be an int, got {type(num_classes).__name__}")
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")


def _check_labels(values: Sequence[int] | np.ndarray, num_classes: int, name: str = "labels") -> list[int]:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be integers, got dtype {array.dtype}")
    if array.min() < 0 or array.max() >= num_classes:
        raise ValueError(f"{name} must be in [0, {num_classes})")
    return [int(v) for v in array]
