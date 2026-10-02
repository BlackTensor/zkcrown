"""Overwrite attack: the adversary embeds their own watermark with their own key (P4.8).

The thief knows how this project watermarks a model, but not `K`. They run
the same procedure on the stolen model with a key of their own, hoping the
second watermark destroys the first, or at least lets them claim the model
too. The question for the owner is whether their watermark still extracts
afterwards.

Three variants (owner decision, P4.8)
-------------------------------------
- ``overwrite_weight``: the attacker adds ``alpha' * P_K'^T * S'`` to the same
  carrier (P3.2), post-hoc. ``strength`` is ``alpha'``. No data, no training.
- ``overwrite_behavioral``: the attacker fine-tunes on the 5,000-image
  attacker holdout with the P4.5 recipe while mixing their own triggers into
  every batch, as the owner did in P2.3. ``strength`` is the peak learning
  rate, and ``params['epochs']`` the length.
- ``overwrite_both``: the behavioral overwrite, then the weight overwrite at
  ``params['weight_alpha']`` on the fine-tuned weights. That is the order the
  owner used (P2.3, then P3.6).

The attacker's material
-----------------------
The attacker's key is ``SHA-256("zk-crown/p4.8/attacker-key/v1\\0" || label)``
for a label in ``params['attacker']``. It is a public demo key by design: it
appears nowhere near `K`, and the attack still never receives `K`
(`AttackContext` has no key field). Everything else is derived from it with
the owner's own code and stream labels, so the attacker's watermark is the
same construction under another key:

- triggers: `generate_triggers` over the attacker holdout, the only images
  the thief holds, N = 100 at amplitude 16 by default, with P2.2 targets;
- weight watermark: `P_K'` over the same carrier layout and a 128-bit `S'`
  for the owner id ``p4.8-attacker``.

What `info` reports
-------------------
Aggregates only. Besides the recipe and the embedding summary, the attacker
scores their own watermark before and after the attack with the same tests
the owner uses (P2.8, P3.7). So the record shows whether the attacker ends up
with a detectable claim of their own. The owner's watermarks are scored
separately, in evaluation, with `K`.

A model that carries two detectable watermarks does not say who was first.
That is what the published commitment and its timestamp are for (Phase 5),
not the detection tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from src.attacks import finetune
from src.attacks.evaluation import evaluate_behavioral, evaluate_weight
from src.attacks.harness import AttackOutput, load_model, register_attack
from src.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD, attacker_holdout_arrays
from src.models import main_model
from src.watermark.behavioral import DEFAULT_TRIGGERS_PER_BATCH, TriggerMixLoader, trigger_metrics, trigger_tensors
from src.watermark.bundle import make_bundle
from src.watermark.carrier import CarrierLayout, carrier_layout
from src.watermark.projection import KeyProjection
from src.watermark.responses import TriggerResponses, trigger_responses
from src.watermark.signature import OwnershipSignature, derive_signature
from src.watermark.triggers import DEFAULT_AMPLITUDE, DEFAULT_N, TriggerSet, generate_triggers
from src.watermark.weight_embedding import derive_carrier_projection, embed_weight_watermark

OVERWRITE_VERSION = "overwrite/v1"
ATTACKER_KEY_DOMAIN = b"zk-crown/p4.8/attacker-key/v1\x00"
ATTACKER_OWNER_ID = "p4.8-attacker"
DEFAULT_ATTACKER = "attacker-1"
BEHAVIORAL_KEYS = ("epochs", "n_triggers", "amplitude", "triggers_per_batch")


def attacker_key(label: str) -> bytes:
    """The attacker's 32-byte key for `label`. Public by construction; never `K`."""
    if not isinstance(label, str) or not label or len(label.encode("utf-8")) > 64:
        raise ValueError(f"attacker label must be a non-empty string of at most 64 bytes, got {label!r}")
    return hashlib.sha256(ATTACKER_KEY_DOMAIN + label.encode("utf-8")).digest()


@dataclass(frozen=True, eq=False)
class AttackerMaterial:
    """What the attacker derives from their own key. Same fields the evaluation reads from `OwnerMaterial`."""

    label: str
    layout: CarrierLayout
    projection: KeyProjection
    signature: OwnershipSignature
    triggers: TriggerSet | None = None
    responses: TriggerResponses | None = None

    def describe(self) -> dict:
        return {"label": self.label, "owner_id": ATTACKER_OWNER_ID,
                "key": "public demo key, SHA-256(\"zk-crown/p4.8/attacker-key/v1\\0\" || label); not the owner's K",
                "owner_key_known": False}


def weight_material(label: str, arch: dict) -> AttackerMaterial:
    """The attacker's `P_K'` and `S'` over the carrier of ``main_model(**arch)``."""
    key = attacker_key(label)
    layout = carrier_layout(main_model(**arch))
    return AttackerMaterial(label, layout, derive_carrier_projection(key, layout), derive_signature(key, ATTACKER_OWNER_ID))


def _positive_int(name: str, value, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _attacker_label(params: dict) -> str:
    label = params.get("attacker", DEFAULT_ATTACKER)
    attacker_key(label)  # validates
    return label


def weight_params(strength: float, params: dict) -> tuple[float, str]:
    """``alpha'`` and the attacker label for ``overwrite_weight``. Raises on anything unknown."""
    unknown = set(params) - {"attacker"}
    if unknown:
        raise ValueError(f"overwrite_weight does not take {sorted(unknown)}")
    return finetune._number("alpha'", strength, positive=True), _attacker_label(params)


def behavioral_params(name: str, strength: float, params: dict, *, weight: bool) -> tuple[dict, dict, str, float | None]:
    """The fine-tuning recipe, the trigger settings, the attacker label and ``weight_alpha`` (or None)."""
    if "epochs" not in params:
        raise ValueError(f"{name} needs params['epochs']")
    if weight != ("weight_alpha" in params):
        raise ValueError(f"{name} {'needs' if weight else 'does not take'} params['weight_alpha']")
    if "lr" in params:
        raise ValueError(f"{name} takes the learning rate as its strength, not params['lr']")
    alpha = finetune._number("weight_alpha", params["weight_alpha"], positive=True) if weight else None
    epochs = params["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, (int, float)):
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    rest = {k: v for k, v in params.items() if k not in (*BEHAVIORAL_KEYS, "attacker", "weight_alpha")}
    try:
        recipe = finetune.finetune_recipe(epochs, {**rest, "lr": strength})
    except ValueError as error:
        raise ValueError(str(error).replace("finetune_holdout", name)) from None
    triggers = {
        "n_triggers": _positive_int("n_triggers", params.get("n_triggers", DEFAULT_N)),
        "amplitude": _positive_int("amplitude", params.get("amplitude", DEFAULT_AMPLITUDE)),
        "triggers_per_batch": _positive_int("triggers_per_batch", params.get("triggers_per_batch", DEFAULT_TRIGGERS_PER_BATCH)),
    }
    if triggers["triggers_per_batch"] > triggers["n_triggers"]:
        raise ValueError("triggers_per_batch may not exceed n_triggers")
    return recipe, triggers, _attacker_label(params), alpha


def _cpu(state_dict: dict) -> dict:
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def _embed(state_dict: dict, material: AttackerMaterial, alpha: float) -> tuple[dict, dict]:
    """Post-hoc ``W + alpha' * P_K'^T * S'`` and what it did, with the attacker's own test before and after."""
    before = evaluate_weight(state_dict, material)
    out, summary = embed_weight_watermark(state_dict, material.layout, material.projection, material.signature, alpha)
    info = {"alpha": alpha, "method": "post-hoc W + alpha' * P_K'^T * S' on the owner's carrier layout (P3.2)",
            "embedding": summary.to_dict(), "attacker_before": before, "attacker_after": evaluate_weight(out, material)}
    return out, info


@register_attack(
    "overwrite_weight",
    strength="alpha', the amplitude of the attacker's own weight watermark",
    description="The attacker adds their own spread-spectrum weight watermark, with their own key, post-hoc (P4.8).",
)
def overwrite_weight(state_dict, arch, strength, params, context) -> AttackOutput:
    alpha, label = weight_params(strength, params)
    material = weight_material(label, arch)
    out, weight = _embed(_cpu(state_dict), material, alpha)
    info = {"version": OVERWRITE_VERSION, "variant": "weight", "attacker": material.describe(),
            "source_state_sha256": finetune.state_digest(state_dict), "weight": weight,
            "fine_tuned": False, "bn_recalibrated": False}
    return AttackOutput(state_dict=out, arch=arch, info=info)


def _overwrite_behavioral(name: str, state_dict, arch, strength, params, context, *, weight: bool) -> AttackOutput:
    recipe, settings, label, alpha = behavioral_params(name, strength, params, weight=weight)
    device = torch.device(context.device)
    key = attacker_key(label)
    images, labels = attacker_holdout_arrays(context.data_root, smoke=context.smoke, seed=context.seed)
    triggers = generate_triggers(key, images, range(len(images)), n=settings["n_triggers"], amplitude=settings["amplitude"])
    responses = trigger_responses(key, triggers, labels)
    bundle = make_bundle(triggers, responses, key_kind="test")
    base = weight_material(label, arch)
    material = AttackerMaterial(label, base.layout, base.projection, base.signature, triggers, responses)
    inputs, targets = trigger_tensors(bundle, CIFAR10_MEAN, CIFAR10_STD)

    start_digest = finetune.state_digest(state_dict)
    model = load_model(state_dict, arch).to(device)
    before = evaluate_behavioral(model, material, device)
    mixers: list[TriggerMixLoader] = []

    def wrap_train(loader):
        mixers.append(TriggerMixLoader(loader, inputs, targets, triggers_per_batch=settings["triggers_per_batch"],
                                       seed=context.seed))
        return mixers[-1]

    extra = {"attack": name, "augment": recipe["augment"], "smoke": context.smoke, "start_state_sha256": start_digest,
             "attacker": label, "attacker_bundle_sha256": bundle.digest(),
             "triggers_per_batch": settings["triggers_per_batch"], "weight_alpha": alpha}
    summary, tuned = finetune.run_finetune(
        model, recipe, context, extra, wrap_train=wrap_train,
        epoch_metrics=lambda m: trigger_metrics(m, inputs, targets, device),
    )
    out = _cpu(model.state_dict())
    history = summary["history"]
    behavioral = {
        **settings,
        "base_images": "synthetic smoke stand-in" if context.smoke else "attacker holdout, 5,000 CIFAR-10 train images",
        "targets": "per-trigger keyed (P2.2), from the attacker's key",
        "bundle_sha256": bundle.digest(),
        "trigger_samples_per_epoch": mixers[0].trigger_samples_per_epoch(),
        "trigger_accuracy_first_epoch": history[0]["trigger_accuracy"] if history else None,
        "trigger_accuracy_last_epoch": history[-1]["trigger_accuracy"] if history else None,
        "attacker_before": before,
    }
    info = {"version": OVERWRITE_VERSION, "variant": "both" if weight else "behavioral",
            "attacker": material.describe(), "source_state_sha256": start_digest,
            "behavioral": behavioral, "finetune": tuned, "fine_tuned": True,
            "bn_recalibrated": "by fine-tuning (running statistics updated in train mode)"}
    if weight:
        info["order"] = "behavioral overwrite by fine-tuning, then the weight overwrite on the fine-tuned weights"
        out, info["weight"] = _embed(out, material, alpha)
    # The attacker's own behavioral test on the weights as shipped, after any weight overwrite.
    behavioral["attacker_after"] = evaluate_behavioral(load_model(out, arch).to(device), material, device)
    return AttackOutput(state_dict=out, arch=arch, info=info)


@register_attack(
    "overwrite_behavioral",
    strength="peak learning rate of the fine-tuning that embeds the attacker's own triggers (params['epochs'] epochs)",
    description="The attacker fine-tunes on their holdout with their own key-derived triggers mixed into every "
                "batch (P4.8).",
)
def overwrite_behavioral(state_dict, arch, strength, params, context) -> AttackOutput:
    return _overwrite_behavioral("overwrite_behavioral", state_dict, arch, strength, params, context, weight=False)


@register_attack(
    "overwrite_both",
    strength="peak learning rate of the fine-tuning that embeds the attacker's own triggers; then a weight overwrite "
             "at params['weight_alpha']",
    description="The attacker embeds both of their own watermarks: triggers by fine-tuning, then a post-hoc weight "
                "watermark (P4.8).",
)
def overwrite_both(state_dict, arch, strength, params, context) -> AttackOutput:
    return _overwrite_behavioral("overwrite_both", state_dict, arch, strength, params, context, weight=True)
