"""Prune, then fine-tune with the pruning mask fixed (P4.6).

The thief compresses the stolen model and then trains it on their own data to
win back the accuracy pruning cost. Fine-tuning after pruning is the standard
pipeline (Han et al. 2015; Li et al. 2017), and it is the strongest realistic
removal attack in this suite: pruning damages both watermarks, and the
fine-tuning pass moves every surviving weight while it repairs the model.

The two steps
-------------
1. **Prune** the stolen weights with one of the Phase 4 pruning attacks,
   unchanged: ``magnitude_prune_global`` or ``magnitude_prune_layerwise``
   (P4.2), or ``channel_prune_l1`` (P4.3, zero-masked). ``strength`` is that
   attack's strength: the sparsity, or the fraction of channels removed.
2. **Fine-tune** the pruned model with the P4.5 recipe (`src.attacks.finetune`)
   on the 5,000-image attacker holdout: SGD Nesterov 0.9, weight decay 5e-4,
   batch 128, augmentation, 1-epoch warmup, cosine decay to 0, final-epoch
   weights. No key, no triggers, no test set.

The mask stays fixed (owner decision, P4.6)
-------------------------------------------
Every position the pruning step set to zero stays exactly zero through
fine-tuning, so the shipped model really is as sparse as it was pruned. A
gradient hook on each masked parameter replaces the gradient at pruned
positions with 0. With SGD that is enough: weight decay adds ``wd * w = 0``,
the momentum buffer only ever accumulates zeros there, and so the update is 0.
It is not taken on trust. After every epoch the number of non-zero masked
entries is written into the training history, and the attack raises if it is
ever non-zero, before and after training.

What the mask covers:
- magnitude pruning: the pruned conv and linear weights. Weights that were
  already zero and not pruned stay trainable.
- channel pruning: each removed filter, its BatchNorm weight and bias, and the
  next layer's input slice. BatchNorm running statistics are not masked; with a
  zero affine pair and zero inputs a removed channel's output is 0 in both
  train and eval mode, whatever the statistics.

Weights not in the mask (kept weights, every BatchNorm parameter of kept
channels, the classifier bias) are trained normally, and BatchNorm statistics
are updated by training in the usual way.

The P4.3 carrier caveat carries over to channel pruning: the weights keep
their shapes, so weight extraction assumes the owner can re-align a physically
narrower model.

Parameters
----------
``params`` must give ``prune`` (one of `PRUNE_ATTACKS`), ``epochs`` and
``lr``, and may override the other P4.5 recipe keys (`finetune.DEFAULTS`).

`info` holds aggregates only: the pruning step's own info, the fine-tuning
info as in P4.5, and the mask checks.
"""

from __future__ import annotations

import torch

from src.attacks import finetune
from src.attacks.harness import AttackOutput, load_model, register_attack
from src.attacks.prune import magnitude_prune_with_mask
from src.attacks.structured_prune import channel_prune_with_mask
from src.models import main_model
from src.watermark.carrier import carrier_layout

PRUNE_FINETUNE_VERSION = "prune-finetune/v1"
PRUNE_ATTACKS = {
    "magnitude_prune_global": lambda state, arch, s: magnitude_prune_with_mask(state, arch, s, "global"),
    "magnitude_prune_layerwise": lambda state, arch, s: magnitude_prune_with_mask(state, arch, s, "layerwise"),
    "channel_prune_l1": channel_prune_with_mask,
}


def split_params(strength: float, params: dict) -> tuple[str, dict]:
    """The pruning attack's name and the completed fine-tuning recipe. Raises on anything unknown or missing."""
    if "prune" not in params:
        raise ValueError(f"prune_finetune needs params['prune'], one of {sorted(PRUNE_ATTACKS)}")
    if params["prune"] not in PRUNE_ATTACKS:
        raise ValueError(f"params['prune'] must be one of {sorted(PRUNE_ATTACKS)}, got {params['prune']!r}")
    if "epochs" not in params:
        raise ValueError("prune_finetune needs params['epochs']")
    rest = {k: v for k, v in params.items() if k not in ("prune", "epochs")}
    epochs = params["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, (int, float)):
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    return params["prune"], finetune.finetune_recipe(epochs, rest)


class FixedMask:
    """Keeps the pruned positions of a model at exactly zero during training."""

    def __init__(self, model: torch.nn.Module, pruned: dict[str, torch.Tensor]):
        parameters = dict(model.named_parameters())
        unknown = set(pruned) - set(parameters)
        if unknown:
            raise ValueError(f"mask names tensors that are not parameters: {sorted(unknown)}")
        self.parameters = {name: parameters[name] for name in pruned}
        self.pruned = {name: mask.to(torch.bool) for name, mask in pruned.items() if mask.any()}
        for name, mask in self.pruned.items():
            if tuple(mask.shape) != tuple(self.parameters[name].shape):
                raise ValueError(f"mask for {name} has shape {tuple(mask.shape)}, parameter {tuple(self.parameters[name].shape)}")
        self._on_device: dict[tuple[str, torch.device], torch.Tensor] = {}
        self.handles = [self.parameters[name].register_hook(self._hook(name)) for name in self.pruned]

    def _mask_on(self, name: str, device: torch.device) -> torch.Tensor:
        key = (name, device)
        if key not in self._on_device:
            self._on_device[key] = self.pruned[name].to(device)
        return self._on_device[key]

    def _hook(self, name: str):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            return torch.where(self._mask_on(name, grad.device), torch.zeros((), dtype=grad.dtype, device=grad.device), grad)

        return hook

    @property
    def size(self) -> int:
        return int(sum(int(mask.sum()) for mask in self.pruned.values()))

    @torch.no_grad()
    def nonzero(self) -> int:
        """How many masked entries are not exactly zero."""
        return int(sum(int((self.parameters[name].detach()[self._mask_on(name, self.parameters[name].device)] != 0).sum())
                       for name in self.pruned))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


@register_attack(
    "prune_finetune",
    strength="the pruning attack's strength (sparsity, or fraction of channels), then fine-tuning with the mask fixed",
    description="Prune (P4.2 or P4.3), then fine-tune on the attacker holdout with the P4.5 recipe, pruned "
                "positions held at zero (P4.6).",
)
def prune_finetune(state_dict, arch, strength, params, context) -> AttackOutput:
    prune_name, recipe = split_params(strength, params)
    source_digest = finetune.state_digest(state_dict)
    pruned_state, prune_info, pruned = PRUNE_ATTACKS[prune_name](state_dict, arch, strength)
    pruned_digest = finetune.state_digest(pruned_state)

    model = load_model(pruned_state, arch)
    mask = FixedMask(model, pruned)
    if mask.nonzero():
        raise RuntimeError(f"{mask.nonzero()} masked entries are non-zero before fine-tuning")

    def epoch_metrics(trained: torch.nn.Module) -> dict:
        nonzero = mask.nonzero()
        if nonzero:
            raise RuntimeError(f"{nonzero} pruned entries became non-zero during fine-tuning")
        return {"masked_nonzero": nonzero}

    extra = {"attack": "prune_finetune", "prune": prune_name, "prune_strength": float(strength),
             "augment": recipe["augment"], "smoke": context.smoke, "mask": "fixed",
             "source_state_sha256": source_digest, "start_state_sha256": pruned_digest}
    try:
        summary, tuned = finetune.run_finetune(model, recipe, context, extra, epoch_metrics=epoch_metrics)
    finally:
        mask.remove()
    nonzero_after = mask.nonzero()
    if nonzero_after:
        raise RuntimeError(f"{nonzero_after} pruned entries are non-zero after fine-tuning")

    out = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    layout = carrier_layout(main_model(**arch))
    carrier_zeros = sum(int((out[name] == 0).sum()) for name in layout.names)
    info = {
        "version": PRUNE_FINETUNE_VERSION,
        "prune_attack": prune_name,
        "prune_strength": float(strength),
        "source_state_sha256": source_digest,
        "start_state_sha256": pruned_digest,
        "prune": prune_info,
        "finetune": tuned,
        "mask": {
            "policy": "fixed: gradients at pruned positions replaced by 0; checked every epoch",
            "masked_tensors": sorted(mask.pruned),
            "masked_entries": mask.size,
            "masked_nonzero_after": nonzero_after,
            "masked_nonzero_max_over_epochs": max((h.get("masked_nonzero", 0) for h in summary["history"]), default=0),
            "epochs_checked": sum(1 for h in summary["history"] if "masked_nonzero" in h),
        },
        "carrier_zero_fraction_after": carrier_zeros / layout.dim,
        "fine_tuned": True,
        "bn_recalibrated": "by fine-tuning (running statistics updated in train mode)",
    }
    return AttackOutput(state_dict=out, arch=arch, info=info)
