"""Scoring an attacked model: the one row every Phase 4 task reports (P4.1).

CLAUDE.md Phase 4 asks each attack for one table row: attack strength,
resulting clean accuracy, behavioral WDR, weight-watermark correlation, and
the p-values from P2.8 and P3.7. `evaluate_attacked` computes all of them with
the methods already fixed earlier, and nothing new:

- **Clean accuracy.** Eval-mode top-1 on the official 10,000-image CIFAR-10
  test set, per image. The drop against the unattacked source uses P2.5's
  paired analysis (exact McNemar, 95% normal interval). The attacker holdout
  is not scored: P4.5 and P4.6 fine-tune on it, so it is not an unbiased
  accuracy estimate for those rows.
- **Behavioral watermark.** The N owner triggers, regenerated from `K`, scored
  as in P2.4. WDR = fired / N, with the exact P2.8 p-value
  ``P(Binomial(N, 1/9) >= fired)``.
- **Weight watermark.** Blind, per-tensor-centred extraction with `K` (P3.3),
  z = correlation x sqrt(128), and the P3.7 bound ``exp(-z^2/2)``. If the
  attacked weights no longer have the owner's carrier layout (a narrower
  student, removed channels), extraction is not attempted. The row says
  ``applicable: false`` and why, rather than guessing at an alignment.

``detected`` flags use one level fixed here, before any attack has been run:
``DETECTION_ALPHA = 1e-6``, the level P3.6 gated the final model on. Each
watermark's full ``rejects`` table is reported as well. The two tests are
reported separately. Combining them into one verdict is P9.3, and the
multiple-testing assumptions of P2.8 and P3.7 apply to each p-value.

`OwnerMaterial` holds everything derived from `K`: the triggers, the targets,
`P_K` and `S`. It is as sensitive as `K`, its ``repr`` hides it, and nothing
from it goes into a row except the trigger bundle digest and carrier digest,
which are already in committed P2.3 and P3.x records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.attacks.harness import ROW_VERSION, AttackConfig, AttackSpec, load_model
from src.data import CIFAR10_MEAN, CIFAR10_STD
from src.models import main_model
from src.utils.stats import paired_accuracy_difference
from src.watermark.bundle import make_bundle
from src.watermark.carrier import CarrierLayout, carrier_layout
from src.watermark.detection import measure_detection
from src.watermark.projection import KeyProjection
from src.watermark.responses import TriggerResponses, trigger_responses
from src.watermark.signature import OwnershipSignature, derive_signature
from src.watermark.significance import detection_test
from src.watermark.triggers import DEFAULT_AMPLITUDE, DEFAULT_N, TriggerSet, generate_triggers
from src.watermark.weight_embedding import derive_carrier_projection
from src.watermark.weight_extraction import extract_weight_watermark
from src.watermark.weight_significance import WeightDetectionTest

DETECTION_ALPHA = "1e-6"
"""Level for the rows' ``detected`` flags. Fixed in P4.1 before any attack ran."""


@dataclass(frozen=True, eq=False)
class OwnerMaterial:
    """Everything the owner derives from `K` to audit a model. Secret."""

    owner_id: str
    bundle_digest: str
    triggers: TriggerSet
    responses: TriggerResponses
    layout: CarrierLayout
    projection: KeyProjection
    signature: OwnershipSignature

    def __repr__(self) -> str:
        return f"OwnerMaterial(owner_id={self.owner_id!r}, n={len(self.responses)}, dim={self.layout.dim}, <secrets hidden>)"


def derive_owner_material(
    key: bytes,
    images: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    pool: Sequence[int],
    *,
    owner_id: str,
    key_kind: str = "owner",
    n: int = DEFAULT_N,
    amplitude: int = DEFAULT_AMPLITUDE,
    arch: Mapping[str, Any] | None = None,
) -> OwnerMaterial:
    """Regenerate the triggers, targets, `P_K` and `S` from `key`.

    `images`, `labels` and `pool` are as for `generate_triggers` and
    `trigger_responses`. The carrier layout is the one of
    ``main_model(**arch)``, the owner's architecture (default width 32).
    """
    triggers = generate_triggers(key, images, pool, n=n, amplitude=amplitude)
    responses = trigger_responses(key, triggers, labels)
    layout = carrier_layout(main_model(**dict(arch or {})))
    return OwnerMaterial(
        owner_id=owner_id,
        bundle_digest=make_bundle(triggers, responses, key_kind=key_kind).digest(),
        triggers=triggers,
        responses=responses,
        layout=layout,
        projection=derive_carrier_projection(key, layout),
        signature=derive_signature(key, owner_id),
    )


@torch.no_grad()
def score_loader(model: nn.Module, loader, device: torch.device) -> dict[str, Any]:
    """One eval-mode pass: per-image correctness (CPU bool list), accuracy and mean cross-entropy."""
    model.eval()
    correct, loss_sum, n = [], 0.0, 0
    for x, y in loader:
        logits = model(x.to(device)).float().cpu()
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
        correct.append(logits.argmax(dim=1) == y)
        n += len(y)
    if n == 0:
        raise ValueError("score_loader() got an empty loader")
    flags = torch.cat(correct)
    return {"accuracy": int(flags.sum()) / n, "correct_count": int(flags.sum()), "n": n,
            "loss": loss_sum / n, "correct": flags.tolist()}


def evaluate_accuracy(scores: Mapping[str, Any], reference_correct: Sequence[bool]) -> dict[str, Any]:
    """Accuracy of the attacked model and its paired drop against the source, in percentage points."""
    paired = paired_accuracy_difference(reference_correct, scores["correct"])
    return {
        "accuracy": scores["accuracy"],
        "correct": scores["correct_count"],
        "n": scores["n"],
        "loss": scores["loss"],
        "source_accuracy": paired["reference_accuracy"],
        "drop_vs_source_pp": paired["drop"] * 100,
        "drop_ci95_pp": [paired["drop_ci_low"] * 100, paired["drop_ci_high"] * 100],
        "source_only_right": paired["reference_only_right"],
        "attacked_only_right": paired["other_only_right"],
        "mcnemar_exact_p": paired["mcnemar_exact_p"],
    }


def evaluate_behavioral(model: nn.Module, material: OwnerMaterial, device: torch.device) -> dict[str, Any]:
    """P2.4 WDR and the P2.8 p-value. Aggregates only."""
    result = measure_detection(
        model, material.triggers.images, material.responses, mean=CIFAR10_MEAN, std=CIFAR10_STD, device=device
    )
    test = detection_test(result.fired, result.n)
    return {**result.summary(), **test.summary(), "detected": test.rejects(DETECTION_ALPHA)}


def evaluate_weight(state_dict: Mapping[str, torch.Tensor], material: OwnerMaterial) -> dict[str, Any]:
    """P3.3 extraction and the P3.7 bound, or ``applicable: false`` if the carrier layout is gone."""
    try:
        material.layout.check(state_dict)
    except (KeyError, ValueError, TypeError) as error:
        return {"applicable": False, "reason": f"owner carrier layout not present: {error}", "detected": False}
    extraction = extract_weight_watermark(state_dict, material.layout, material.projection, material.signature)
    test = WeightDetectionTest(extraction.correlation)
    return {"applicable": True, **extraction.to_dict(), **test.summary(), "detected": test.rejects(DETECTION_ALPHA)}


def evaluate_attacked(
    state_dict: Mapping[str, torch.Tensor],
    arch: Mapping[str, Any],
    material: OwnerMaterial,
    test_loader,
    reference_correct: Sequence[bool],
    device: torch.device,
) -> dict[str, Any]:
    """Clean accuracy, behavioral and weight watermark results for one attacked model."""
    model = load_model(state_dict, arch).to(device)
    return {
        "clean_accuracy": evaluate_accuracy(score_loader(model, test_loader, device), reference_correct),
        "behavioral": evaluate_behavioral(model, material, device),
        "weight": evaluate_weight({k: v.cpu() for k, v in state_dict.items()}, material),
    }


def build_row(
    config: AttackConfig,
    spec: AttackSpec,
    evaluation: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    attacked: Mapping[str, Any],
    material: OwnerMaterial,
) -> dict[str, Any]:
    """The standard row. ``table`` holds the Phase 4 columns; the rest is the detail behind them."""
    acc, beh, wgt = evaluation["clean_accuracy"], evaluation["behavioral"], evaluation["weight"]
    return {
        "row_version": ROW_VERSION,
        "table": {
            "attack": config.attack,
            "strength": config.strength,
            "strength_meaning": spec.strength,
            "clean_accuracy": acc["accuracy"],
            "accuracy_drop_vs_source_pp": acc["drop_vs_source_pp"],
            "behavioral_wdr": beh["wdr"],
            "behavioral_p_value": beh["p_value"],
            "behavioral_detected": beh["detected"],
            "weight_correlation": wgt.get("correlation"),
            "weight_z": wgt.get("z"),
            "weight_p_value": wgt.get("p_value_bound"),
            "weight_detected": wgt["detected"],
            "weight_applicable": wgt["applicable"],
            "detection_alpha": DETECTION_ALPHA,
        },
        "config": config.to_dict(),
        "attack": spec.to_dict(),
        "source": dict(source),
        "attacked": dict(attacked),
        "clean_accuracy": dict(acc),
        "behavioral": dict(beh),
        "weight": dict(wgt),
        "owner": {
            "owner_id": material.owner_id,
            "trigger_bundle_sha256": material.bundle_digest,
            "carrier_digest": material.layout.digest(),
        },
    }
