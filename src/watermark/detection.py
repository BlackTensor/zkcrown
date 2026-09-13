"""Measuring the behavioral watermark: the Watermark Detection Rate (P2.4).

Definition
----------
For a trigger set ``T_1..T_N`` with keyed targets ``t_1..t_N`` (P2.2), and a
suspect model ``f`` in eval mode::

    fired_i = [ argmax f(T_i) == t_i ]
    k       = sum_i fired_i
    WDR     = k / N

"Fires" is the same rule as `TriggerResponses.fired`. Top-1 only. Ties in the
logits go to the lowest class index (`torch.argmax`); with float logits they
do not occur in practice.

Triggers go in exactly as generated: uint8 images, scaled to [0, 1] and
normalised per channel, the same as the clean eval pipeline and the P2.3
training triggers. No augmentation, no resizing.

What WDR is and is not
----------------------
- WDR is a count over a fixed set. It is not a p-value. How unlikely ``k`` is
  for a model built without `K` is P2.8's question, and the false-positive
  rate on unrelated inputs is P2.6's.
- The triggers are the images the model was trained on. WDR measures whether
  the model still gives the trained responses to those exact images. It says
  nothing about triggers the model never saw.

`DetectionResult.summary` holds aggregate numbers only, so it can go in a
committed result file. The per-trigger predictions and fired flags stay in
memory: together with a leaked trigger image they would give its target away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import nn

from src.watermark.responses import TriggerResponses

DEFAULT_BATCH_SIZE = 256


def normalise_uint8(images: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """``(N, H, W, C)`` uint8 to a normalised ``(N, C, H, W)`` float32 tensor.

    Same arithmetic as `ToTensor` then `Normalize(mean, std)`, which is what
    `src.watermark.behavioral.trigger_tensors` fed the model during P2.3.
    """
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 4:
        raise TypeError("images must be a 4-D uint8 numpy array (N, H, W, C)")
    if len(mean) != images.shape[3] or len(std) != images.shape[3]:
        raise ValueError(f"mean and std need {images.shape[3]} channels")
    # Copy: trigger sets are read-only arrays, which torch cannot wrap safely.
    x = torch.from_numpy(np.array(images, copy=True)).permute(0, 3, 1, 2).float().div(255.0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
    return ((x - mean_t) / std_t).contiguous()


@torch.no_grad()
def predict_logits(
    model: nn.Module, inputs: torch.Tensor, device: torch.device, batch_size: int = DEFAULT_BATCH_SIZE
) -> torch.Tensor:
    """Eval-mode logits for `inputs`, in order, on the CPU as float32.

    Puts the model in eval mode and leaves it there, like
    `src.training.evaluate`.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    model.eval()
    chunks = [model(inputs[i : i + batch_size].to(device)).float().cpu() for i in range(0, len(inputs), batch_size)]
    return torch.cat(chunks)


@dataclass(frozen=True)
class DetectionResult:
    """The outcome of presenting a trigger set to one model.

    Attributes:
        n: number of triggers presented.
        fired: ``k``, how many produced their target.
        predicted_base_label: how many were classified as their base image's
            dataset label. Never overlaps `fired`, since targets differ from
            base labels.
        target_probability: ``(N,)`` softmax probability of each target.
        fired_mask: ``(N,)`` bool. Secret-adjacent, not in `summary`.
        predictions: ``(N,)`` int64 top-1 classes. Secret-adjacent, not in
            `summary`.
    """

    n: int
    fired: int
    predicted_base_label: int
    target_probability: np.ndarray = field(repr=False)
    fired_mask: np.ndarray = field(repr=False)
    predictions: np.ndarray = field(repr=False)

    @property
    def wdr(self) -> float:
        return self.fired / self.n

    def summary(self) -> dict[str, float | int]:
        """Aggregate numbers only, safe for a committed result file."""
        p = self.target_probability
        return {
            "n": self.n,
            "fired": self.fired,
            "wdr": self.wdr,
            "predicted_base_label": self.predicted_base_label,
            "predicted_other_class": self.n - self.fired - self.predicted_base_label,
            "target_probability_mean": float(p.mean()),
            "target_probability_median": float(np.median(p)),
            "target_probability_min": float(p.min()),
        }


def measure_detection(
    model: nn.Module,
    images: np.ndarray,
    responses: TriggerResponses,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DetectionResult:
    """Present `images` to `model` and count how many produce their keyed target.

    Args:
        model: the suspect model. Put in eval mode.
        images: ``(N, H, W, C)`` uint8, one per entry of `responses`. Normally
            the triggers; any images can be scored against the same targets,
            for example the unperturbed base images as a control.
        responses: the keyed targets and base labels (P2.2).
        mean, std: the per-channel normalisation the model was trained with.
        device: where to run the model.
        batch_size: inference batch size. Does not change the result.
    """
    if len(images) != len(responses):
        raise ValueError(f"{len(images)} images but {len(responses)} responses")
    logits = predict_logits(model, normalise_uint8(images, mean, std), device, batch_size)
    if logits.ndim != 2 or logits.shape[1] != responses.num_classes:
        raise ValueError(f"model output {tuple(logits.shape)} does not match {responses.num_classes} classes")

    predictions = logits.argmax(dim=1).numpy().astype(np.int64)
    fired_mask = responses.fired(predictions)
    targets = torch.from_numpy(np.array(responses.targets, dtype=np.int64, copy=True))
    target_probability = torch.softmax(logits, dim=1).gather(1, targets.view(-1, 1)).squeeze(1).numpy()
    return DetectionResult(
        n=len(responses),
        fired=int(fired_mask.sum()),
        predicted_base_label=int((predictions == responses.base_labels).sum()),
        target_probability=target_probability,
        fired_mask=fired_mask,
        predictions=predictions,
    )
