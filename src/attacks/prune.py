"""Unstructured magnitude pruning (P4.2).

The thief zeroes the smallest-magnitude weights and ships the model as it is:
no fine-tuning (that is P4.6), no BatchNorm recalibration, no data. Two
variants are registered, because they fail in different ways:

``magnitude_prune_layerwise``
    Every prunable tensor is pruned to the same sparsity `s`, using its own
    magnitude order. No layer can be emptied.
``magnitude_prune_global``
    One magnitude order over all prunable weights together, and the smallest
    ``round(s * total)`` are zeroed. Layers whose weights are small in
    absolute terms lose more, and at high sparsity can lose almost everything.

Which weights are prunable
--------------------------
Every `nn.Conv2d` and `nn.Linear` weight, which is the standard target of
magnitude pruning. BatchNorm parameters, running statistics and the classifier
bias are never touched. For `main_model` at width 32 that is 7 tensors and
307,040 weights. That set is also the weight watermark's carrier (P3.2). The
attacker picks it because it is the conventional choice, not because it knows
where the watermark is. The tensors come from `carrier_layout`, so the order is
fixed.

Exact counts and ties
---------------------
The strength is the target sparsity `s`, with ``0 <= s < 1``. Layer-wise, a
tensor with `n` entries loses exactly ``round(s * n)`` of them (Python's
`round`, ties to even). Globally, exactly ``round(s * 307,040)`` weights are
zeroed. Weights are ranked by absolute value in float64 with a stable sort, so
equal magnitudes are pruned in flat-index order (within a tensor, then by
tensor order), and the result does not depend on the backend or the thread
count. Weights that are already zero count towards their rank like any other
magnitude. Kept weights are bit-identical to the source.

`info` records the target and achieved sparsity overall and per tensor, the
number of weights zeroed, zeros already present, how many tensors were emptied
entirely, and the largest pruned and smallest kept magnitude per tensor and
overall. The overall pair is a threshold only for the global scope. All of
these are aggregates.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from src.attacks.harness import AttackOutput, register_attack
from src.models import main_model
from src.watermark.carrier import carrier_layout

PRUNE_VERSION = "magnitude-prune/v1"
SCOPES = ("layerwise", "global")


def _check_sparsity(sparsity: float) -> float:
    if not (isinstance(sparsity, (int, float)) and not isinstance(sparsity, bool)) or not math.isfinite(sparsity):
        raise ValueError(f"sparsity must be a finite number, got {sparsity!r}")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    return float(sparsity)


def _prune_order(values: np.ndarray, count: int) -> np.ndarray:
    """Indices of the `count` smallest |values|, ties broken by index."""
    return np.argsort(np.abs(values), kind="stable")[:count]


def magnitude_prune(state_dict: dict, arch: dict, sparsity: float, scope: str) -> tuple[dict, dict]:
    """Zero the smallest-magnitude conv and linear weights. Returns the new state dict and aggregate info."""
    out, info, _ = magnitude_prune_with_mask(state_dict, arch, sparsity, scope)
    return out, info


def magnitude_prune_with_mask(state_dict: dict, arch: dict, sparsity: float,
                              scope: str) -> tuple[dict, dict, dict[str, torch.Tensor]]:
    """`magnitude_prune`, plus the pruned positions (P4.6).

    The third value maps each carrier tensor name to a bool tensor of its
    shape, True where the entry was pruned. Weights that were already zero and
    not pruned are False, so a later fine-tune may still move them.
    """
    sparsity = _check_sparsity(sparsity)
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    layout = carrier_layout(main_model(**arch))
    layout.check(state_dict)
    flat = {name: state_dict[name].detach().cpu().to(torch.float64).reshape(-1).numpy() for name in layout.names}
    masks = {name: np.ones(values.size, dtype=bool) for name, values in flat.items()}

    if scope == "layerwise":
        for name, values in flat.items():
            masks[name][_prune_order(values, round(sparsity * values.size))] = False
    else:
        everything = np.concatenate([flat[name] for name in layout.names])
        keep = np.ones(everything.size, dtype=bool)
        keep[_prune_order(everything, round(sparsity * everything.size))] = False
        for name, start, size in zip(layout.names, layout.offsets, layout.sizes):
            masks[name] = keep[start : start + size]

    out = dict(state_dict)
    per_tensor, pruned_abs, kept_abs = {}, [], []
    zeroed = zeros_before = 0
    for name in layout.names:
        tensor = state_dict[name]
        mask = torch.from_numpy(masks[name].reshape(tuple(tensor.shape)))
        out[name] = torch.where(mask, tensor, torch.zeros((), dtype=tensor.dtype))
        values = np.abs(flat[name])
        removed = int((~masks[name]).sum())
        zeroed += removed
        zeros_before += int((values == 0).sum())
        largest_pruned = float(values[~masks[name]].max()) if removed else None
        smallest_kept = float(values[masks[name]].min()) if removed < values.size else None
        per_tensor[name] = {"size": int(values.size), "pruned": removed, "sparsity": removed / values.size,
                            "zero_fraction_after": float((out[name] == 0).float().mean()),
                            "largest_pruned_abs": largest_pruned, "smallest_kept_abs": smallest_kept}
        if largest_pruned is not None:
            pruned_abs.append(largest_pruned)
        if smallest_kept is not None:
            kept_abs.append(smallest_kept)

    total = layout.dim
    info = {
        "version": PRUNE_VERSION,
        "scope": scope,
        "target_sparsity": sparsity,
        "prunable_tensors": len(layout.names),
        "prunable_weights": total,
        "zeroed": zeroed,
        "achieved_sparsity": zeroed / total,
        "zeros_before": zeros_before,
        "largest_pruned_abs": max(pruned_abs) if pruned_abs else None,
        "smallest_kept_abs": min(kept_abs) if kept_abs else None,
        "emptied_tensors": sum(1 for t in per_tensor.values() if t["pruned"] == t["size"]),
        "per_tensor": per_tensor,
        "untouched": "BatchNorm parameters and buffers, classifier bias",
        "fine_tuned": False,
        "bn_recalibrated": False,
    }
    pruned = {name: torch.from_numpy(~masks[name]).reshape(tuple(state_dict[name].shape)) for name in layout.names}
    return out, info, pruned


def _no_params(params: dict) -> None:
    if params:
        raise ValueError(f"magnitude pruning takes no params, got {sorted(params)}")


@register_attack(
    "magnitude_prune_layerwise",
    strength="fraction of each conv/linear weight tensor zeroed, smallest |w| first",
    description="Unstructured magnitude pruning, same sparsity per tensor, no fine-tuning or BN recalibration (P4.2).",
)
def magnitude_prune_layerwise(state_dict, arch, strength, params, context) -> AttackOutput:
    _no_params(params)
    state, info = magnitude_prune(state_dict, arch, strength, "layerwise")
    return AttackOutput(state_dict=state, arch=arch, info=info)


@register_attack(
    "magnitude_prune_global",
    strength="fraction of all conv/linear weights zeroed, one global |w| threshold",
    description="Unstructured magnitude pruning, one threshold across tensors, no fine-tuning or BN recalibration (P4.2).",
)
def magnitude_prune_global(state_dict, arch, strength, params, context) -> AttackOutput:
    _no_params(params)
    state, info = magnitude_prune(state_dict, arch, strength, "global")
    return AttackOutput(state_dict=state, arch=arch, info=info)
