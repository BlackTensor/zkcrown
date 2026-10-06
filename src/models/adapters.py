"""Input adapters for auditing models that expect a different input pipeline (P9.4).

The auditor's behavioral check (P2.4) feeds a suspect CIFAR-10 images
normalised with this project's `CIFAR10_MEAN` / `CIFAR10_STD`. A third-party
model was trained with its own normalisation, and `zk_model` takes 1 x 28 x 28
MNIST-normalised images. `InputAdapter` sits in front of such a model: it undoes
the project normalisation to recover pixels in [0, 1], optionally converts to
grayscale and resizes, and applies the wrapped model's own normalisation.

The adapter has no parameters of its own and does not change the wrapped
model's weights. It only decides what the suspect sees, so a third-party model
answers the owner's triggers on the same pixels as `main_model` does.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

LUMA = (0.299, 0.587, 0.114)
"""ITU-R BT.601 weights, the ones PIL's ``convert("L")`` uses."""


class InputAdapter(nn.Module):
    """``model(normalise_target(transform(denormalise_source(x))))``.

    Args:
        model: the wrapped model, used as is.
        source_mean, source_std: the normalisation the caller applied (per channel).
        target_mean, target_std: the normalisation `model` expects (per channel of its input).
        grayscale: convert RGB to one luma channel before resizing.
        size: resize to ``(size, size)`` (bilinear, antialiased) when not None.
    """

    def __init__(self, model: nn.Module, *, source_mean: Sequence[float], source_std: Sequence[float],
                 target_mean: Sequence[float], target_std: Sequence[float], grayscale: bool = False,
                 size: int | None = None):
        super().__init__()
        self.model = model
        self.grayscale, self.size = bool(grayscale), size
        for name, values in (("source_mean", source_mean), ("source_std", source_std),
                             ("target_mean", target_mean), ("target_std", target_std)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("luma", torch.tensor(LUMA, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def pixels(self, x: torch.Tensor) -> torch.Tensor:
        """What the wrapped model sees, in [0, 1] pixel units, before its own normalisation."""
        x = x * self.source_std + self.source_mean
        if self.grayscale:
            x = (x * self.luma).sum(dim=1, keepdim=True)
        if self.size is not None and tuple(x.shape[-2:]) != (self.size, self.size):
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((self.pixels(x) - self.target_mean) / self.target_std)

    def extra_repr(self) -> str:
        return f"grayscale={self.grayscale}, size={self.size}"
