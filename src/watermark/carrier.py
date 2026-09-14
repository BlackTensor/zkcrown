"""The carrier: which weights the weight watermark is spread over, in what order (P3.2).

The spread-spectrum watermark treats a set of model tensors as one long vector
`w` of length ``dim``. `P_K` (P3.1) has ``dim`` columns, so embedding (P3.2)
and extraction (P3.3) have to agree exactly on which tensors form `w` and in
what order their entries are laid out. This module fixes that.

Decision: every conv and linear weight
--------------------------------------
The carrier is the ``weight`` tensor of every `nn.Conv2d` and `nn.Linear`
module, in `model.named_modules()` order, each flattened in C (row-major)
order. For `main_model` at width 32 that is seven tensors,
``features.{0,3,7,10,14,17}.weight`` and ``classifier.2.weight``, with
``dim = 307,040``. That equals `count_parameters(...)["conv_and_linear_weights"]`,
the pool P0.4 set aside for this.

Excluded, and why:

- **BatchNorm weight and bias, and running statistics.** 896 affine
  parameters in `main_model` at width 32. They are per-channel scales and
  shifts, not spread-out parameters. Running statistics are buffers that are
  overwritten whenever BN is recalibrated.
- **The classifier bias.** Ten entries. With the BN parameters that makes up
  the 906 of 307,946 parameters outside the carrier.
- **``num_batches_tracked``.** An integer counter.

Module registration order in `main_model` is fixed by its constructor, so the
layout is deterministic (P0.4). It is identified by `CarrierLayout.digest`, a
SHA-256 over the version tag, names and shapes, so an extractor can refuse a
model whose layout differs.

Not decided here
----------------
- Architectures with a different layout, such as the width-16 distillation
  student (P4.7), get a different ``dim`` and so an unrelated `P_K`. Whether
  and how the weight watermark can be read from such a model is Phase 4.
- Pruned or quantized models keep the layout, since pruning zeroes entries
  and quantization changes values. Structured pruning that removes channels
  changes shapes, and that is P4.3's problem.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

CARRIER_VERSION = "carrier/v1"
CARRIER_MODULE_TYPES = (nn.Conv2d, nn.Linear)


@dataclass(frozen=True)
class CarrierLayout:
    """An ordered list of state_dict tensors that form the carrier vector.

    Attributes:
        names: state_dict keys, in carrier order.
        shapes: the shape of each tensor, same order.
        version: the layout rule version, ``"carrier/v1"``.
    """

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    version: str = CARRIER_VERSION

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("a carrier needs at least one tensor")
        if len(self.names) != len(self.shapes):
            raise ValueError("names and shapes must have the same length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("carrier names must be unique")
        for name, shape in zip(self.names, self.shapes):
            if not shape or any(not isinstance(s, int) or s < 1 for s in shape):
                raise ValueError(f"tensor {name!r} has an invalid shape {shape}")

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(int(np.prod(shape)) for shape in self.shapes)

    @property
    def dim(self) -> int:
        """Length of the carrier vector."""
        return sum(self.sizes)

    @property
    def offsets(self) -> tuple[int, ...]:
        """Start of each tensor in the carrier vector."""
        return tuple(int(x) for x in np.cumsum((0,) + self.sizes[:-1]))

    def digest(self) -> str:
        """SHA-256 over the canonical JSON of version, names and shapes."""
        canonical = json.dumps(
            {"version": self.version, "names": list(self.names), "shapes": [list(s) for s in self.shapes]},
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def check(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Raise if `state_dict` does not carry this layout's tensors.

        Every carrier name must be present, with the recorded shape and a
        floating point dtype.
        """
        for name, shape in zip(self.names, self.shapes):
            if name not in state_dict:
                raise KeyError(f"state_dict has no carrier tensor {name!r}")
            tensor = state_dict[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(f"carrier tensor {name!r} has shape {tuple(tensor.shape)}, expected {shape}")
            if not tensor.is_floating_point():
                raise TypeError(f"carrier tensor {name!r} must be floating point, got {tensor.dtype}")

    def flatten(self, state_dict: Mapping[str, torch.Tensor]) -> np.ndarray:
        """Return the carrier vector `w` as float64, shape ``(dim,)``."""
        self.check(state_dict)
        return np.concatenate(
            [state_dict[name].detach().to("cpu", torch.float64).reshape(-1).numpy() for name in self.names]
        )

    def split(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        """Cut a ``(dim,)`` vector back into per-tensor arrays of the layout's shapes."""
        vector = np.asarray(vector)
        if vector.shape != (self.dim,):
            raise ValueError(f"vector must have shape ({self.dim},), got {vector.shape}")
        return {
            name: vector[start : start + size].reshape(shape)
            for name, shape, start, size in zip(self.names, self.shapes, self.offsets, self.sizes)
        }


def carrier_layout(model: nn.Module) -> CarrierLayout:
    """The carrier of `model`: every conv and linear ``weight``, in module order."""
    names, shapes = [], []
    for module_name, module in model.named_modules():
        if isinstance(module, CARRIER_MODULE_TYPES):
            names.append(f"{module_name}.weight")
            shapes.append(tuple(int(s) for s in module.weight.shape))
    if not names:
        raise ValueError(f"{type(model).__name__} has no conv or linear layers to carry a watermark")
    return CarrierLayout(names=tuple(names), shapes=tuple(shapes))
