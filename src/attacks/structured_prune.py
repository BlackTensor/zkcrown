"""Structured pruning: whole conv filters removed by L1 norm (P4.3).

The thief removes whole output channels and ships the model as it is: no
fine-tuning (P4.6), no BatchNorm recalibration, no data. Filters are ranked by
the L1 norm of their weights, the criterion of Li et al. (2017), "Pruning
Filters for Efficient ConvNets".

What is removed
---------------
For every conv layer of `main_model`, ``min(round(s * C), C - 1)`` of its `C`
output channels are removed, the ones whose filters have the smallest L1 norm.
The same fraction goes from every layer, and every layer keeps at least one
channel. The first conv's inputs are the RGB image and the classifier's outputs
are the classes, so neither can be removed. For `main_model` at width 32 the
layers have 32, 32, 64, 64, 128 and 128 channels.

Ranking uses the source weights of each layer independently ("independent
pruning" in Li et al.): removing channels from layer `k` does not change the
ranking in layer `k + 1`. Norms are summed in float64 and sorted stably, so
equal norms are removed in channel-index order.

How a removed channel is represented
------------------------------------
The weights keep their shapes (owner decision, P4.3). Removing channel `i` of
conv layer `k` sets to zero:

- filter `i` of the conv, ``weight[i]``;
- the BatchNorm affine pair ``weight[i]`` and ``bias[i]``, so the channel's
  eval-mode BN output is exactly 0 and stays 0 after ReLU and max-pooling;
- the slice of the next layer that reads channel `i`: ``weight[:, i]`` of the
  next conv, or, after the last conv, the classifier columns of that channel's
  4 x 4 pooled features.

That is output-equivalent to physically deleting the channel. The narrower
network is not built here, because `main_model` has a single width; the
equivalence is checked against a hand-sliced network in
`tests/test_structured_prune.py`. BatchNorm running statistics are left alone:
with a zero affine pair they have no effect on the output.

The watermark carrier caveat
----------------------------
Every conv and linear weight is also the weight watermark's carrier (P3.2), so
this representation keeps the owner's carrier layout and the P3.3 extractor
runs on it. The zeros sit exactly where the removed channels were. A thief who
physically deletes channels ships a narrower model instead. Its carrier layout
is gone, and the owner would first have to put the removed positions back as
zeros. That is possible with `W*` in hand, since the kept weights are
unchanged, but the alignment is not implemented. The weight-watermark numbers
on these rows therefore assume that alignment has been done.

`info` holds aggregates only: the target and achieved channel fraction overall
and per layer, the largest removed and smallest kept filter L1 norm per layer,
the fraction of each carrier tensor that ends up zero (removed filters plus
removed input slices), and the parameter count of the equivalent narrower
network.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from src.attacks.harness import AttackOutput, register_attack
from src.models import main_model
from src.watermark.carrier import carrier_layout

STRUCTURED_PRUNE_VERSION = "channel-prune-l1/v1"


def _check_fraction(fraction: float) -> float:
    if not (isinstance(fraction, (int, float)) and not isinstance(fraction, bool)) or not math.isfinite(fraction):
        raise ValueError(f"channel fraction must be a finite number, got {fraction!r}")
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"channel fraction must be in [0, 1), got {fraction}")
    return float(fraction)


def _prunable_layers(model: nn.Module) -> tuple[list[tuple[str, str, str]], str]:
    """``(conv, batchnorm, consumer)`` state-dict prefixes in forward order, and the classifier prefix.

    `consumer` is the module that reads the conv's channels: the next conv, or
    the classifier for the last one. Refuses any shape other than conv followed
    by its BatchNorm.
    """
    convs: list[tuple[str, str]] = []
    pending = None
    for name, module in model.features.named_children():
        if isinstance(module, nn.Conv2d):
            if pending is not None:
                raise ValueError(f"conv features.{pending} has no BatchNorm before the next conv")
            pending = name
        elif isinstance(module, nn.BatchNorm2d):
            if pending is None:
                raise ValueError(f"BatchNorm features.{name} does not follow a conv")
            convs.append((f"features.{pending}", f"features.{name}"))
            pending = None
    if pending is not None or not convs:
        raise ValueError("expected every conv in model.features to be followed by a BatchNorm")
    linears = [name for name, module in model.classifier.named_children() if isinstance(module, nn.Linear)]
    if len(linears) != 1:
        raise ValueError("expected exactly one Linear layer in model.classifier")
    classifier = f"classifier.{linears[0]}"
    consumers = [conv for conv, _ in convs[1:]] + [classifier]
    return [(conv, bn, consumer) for (conv, bn), consumer in zip(convs, consumers)], classifier


def _removal_order(norms: np.ndarray, count: int) -> np.ndarray:
    """Channel indices of the `count` smallest norms, ties broken by index."""
    return np.argsort(norms, kind="stable")[:count]


def channel_prune(state_dict: dict, arch: dict, fraction: float) -> tuple[dict, dict]:
    """Remove the lowest-L1 filters from every conv layer. Returns the new state dict and aggregate info."""
    out, info, _ = channel_prune_with_mask(state_dict, arch, fraction)
    return out, info


def channel_prune_with_mask(state_dict: dict, arch: dict, fraction: float) -> tuple[dict, dict, dict[str, torch.Tensor]]:
    """`channel_prune`, plus the zeroed positions (P4.6).

    The third value maps every tensor the removal touches (conv filters, the
    BatchNorm affine pair, the consumer's input slice) to a bool tensor of its
    shape, True where the removal set the entry to zero. Untouched tensors are
    absent.
    """
    fraction = _check_fraction(fraction)
    model = main_model(**arch)
    layout = carrier_layout(model)
    layout.check(state_dict)
    layers, classifier = _prunable_layers(model)

    out = dict(state_dict)
    removed_by_layer: dict[str, np.ndarray] = {}
    per_layer = {}
    for conv, bn, consumer in layers:
        weight = state_dict[f"{conv}.weight"]
        channels = weight.shape[0]
        norms = weight.detach().cpu().to(torch.float64).abs().reshape(channels, -1).sum(dim=1).numpy()
        count = min(round(fraction * channels), channels - 1)
        removed = np.sort(_removal_order(norms, count))
        keep = np.ones(channels, dtype=bool)
        keep[removed] = False
        removed_by_layer[conv] = removed
        per_layer[conv] = {
            "channels": int(channels),
            "removed": int(count),
            "fraction_removed": count / channels,
            "capped": count < round(fraction * channels),
            "largest_removed_l1": float(norms[~keep].max()) if count else None,
            "smallest_kept_l1": float(norms[keep].min()),
        }

    pruned: dict[str, torch.Tensor] = {}

    def zero_mask(name: str) -> torch.Tensor:
        if name not in pruned:
            pruned[name] = torch.zeros(tuple(state_dict[name].shape), dtype=torch.bool)
        return pruned[name]

    for conv, bn, consumer in layers:
        index = torch.from_numpy(removed_by_layer[conv]).long()
        if index.numel() == 0:
            continue
        for name in (f"{conv}.weight", f"{bn}.weight", f"{bn}.bias"):
            tensor = out[name].clone()
            tensor[index] = 0
            out[name] = tensor
            zero_mask(name)[index] = True
        tensor = out[f"{consumer}.weight"].clone()
        if consumer == classifier:
            channels = per_layer[conv]["channels"]
            spatial = tensor.shape[1] // channels
            if spatial * channels != tensor.shape[1]:
                raise ValueError(f"{consumer} has {tensor.shape[1]} inputs, not a multiple of {channels} channels")
            columns = (index[:, None] * spatial + torch.arange(spatial)[None, :]).reshape(-1)
            tensor[:, columns] = 0
            zero_mask(f"{consumer}.weight")[:, columns] = True
        else:
            tensor[:, index] = 0
            zero_mask(f"{consumer}.weight")[:, index] = True
        out[f"{consumer}.weight"] = tensor

    kept = [per_layer[conv]["channels"] - per_layer[conv]["removed"] for conv, _, _ in layers]
    narrow_params = 0
    in_channels = state_dict[f"{layers[0][0]}.weight"].shape[1]
    for (conv, _, _), k in zip(layers, kept):
        kernel = math.prod(state_dict[f"{conv}.weight"].shape[2:])
        narrow_params += k * in_channels * kernel + 2 * k  # conv (no bias) + BN affine pair
        in_channels = k
    cls_weight = state_dict[f"{classifier}.weight"]
    spatial = cls_weight.shape[1] // per_layer[layers[-1][0]]["channels"]
    narrow_params += cls_weight.shape[0] * kept[-1] * spatial + cls_weight.shape[0]
    source_params = sum(t.numel() for n, t in state_dict.items() if not n.endswith(("running_mean", "running_var", "num_batches_tracked")))

    carrier_zero = {name: float((out[name] == 0).double().mean()) for name in layout.names}
    carrier_zeros = sum(int((out[name] == 0).sum()) for name in layout.names)
    total_channels = sum(p["channels"] for p in per_layer.values())
    total_removed = sum(p["removed"] for p in per_layer.values())
    info = {
        "version": STRUCTURED_PRUNE_VERSION,
        "criterion": "L1 norm of each conv filter, source weights, each layer ranked independently",
        "target_fraction": fraction,
        "prunable_layers": len(layers),
        "channels": total_channels,
        "channels_removed": total_removed,
        "achieved_fraction": total_removed / total_channels,
        "per_layer": per_layer,
        "carrier_zero_fraction": carrier_zeros / layout.dim,
        "carrier_zero_fraction_per_tensor": carrier_zero,
        "equivalent_narrow_params": int(narrow_params),
        "source_params": int(source_params),
        "representation": "zero-masked, same shapes; output-equivalent to deleting the channels",
        "zeroed": "removed filters, their BatchNorm weight and bias, and the next layer's input slice",
        "untouched": "kept weights, BatchNorm running statistics, classifier bias",
        "weight_extraction_assumes": "the owner re-aligns removed channels as zeros (not implemented for a physically narrower model)",
        "fine_tuned": False,
        "bn_recalibrated": False,
    }
    return out, info, pruned


@register_attack(
    "channel_prune_l1",
    strength="fraction of each conv layer's output channels removed, smallest filter L1 norm first",
    description="Structured filter pruning, same fraction per layer, zero-masked, no fine-tuning or BN recalibration (P4.3).",
)
def channel_prune_l1(state_dict, arch, strength, params, context) -> AttackOutput:
    if params:
        raise ValueError(f"channel pruning takes no params, got {sorted(params)}")
    state, info = channel_prune(state_dict, arch, strength)
    return AttackOutput(state_dict=state, arch=arch, info=info)
