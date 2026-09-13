"""P2.4: measure the Watermark Detection Rate of `W*` on the trigger set.

    python experiments/p2_4_measure_wdr.py

CPU only, under a minute. Steps:

1. **Regenerate the trigger set from `K`.** Reads `secrets/K.bin` and
   CIFAR-10, rebuilds the N = 100 triggers (P1.2) and targets (P2.2), and
   checks that their bundle digest equals the digest the P2.3 run trained on.
   The measurement then uses triggers derived from `K` just now, not a copy
   of the uploaded bundle.
2. **Load the models by hash.** `W*` (P2.3) and the clean `W` (P0.5), each
   refused unless its file's SHA-256 matches the one its run recorded.
3. **Score them.** For each model: the triggers against their targets, which
   gives WDR, and the unperturbed base images against the same targets.

`W*` on the triggers is the P2.4 number. The other three rows are controls.
Clean `W` on the triggers is a model trained without `K` on the same data and
recipe. The base-image rows show whether a model responds to the key-derived
perturbation or to the image itself. None of this is a false-positive rate
(P2.6) or a p-value (P2.8).

The result file holds counts and summary statistics only. Triggers, targets,
base indices and per-trigger predictions are secret and never written.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import numpy as np
import torch
from make_master_key import DEFAULT_KEY_PATH, load_key
from torch import nn

from src.data import CIFAR10_MEAN, CIFAR10_STD
from src.data.cifar10 import cifar10_split_indices
from src.models import count_parameters, main_model
from src.utils.results import repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.watermark.bundle import make_bundle
from src.watermark.detection import measure_detection
from src.watermark.responses import trigger_responses
from src.watermark.triggers import DEFAULT_AMPLITUDE, DEFAULT_N, generate_triggers

P2_3_BUNDLE_SHA256 = "fbd65ec730a3555c5921f13d1a5d4840b6310261ddcaf31d3a724195122baec8"
"""Trigger bundle digest recorded by the P2.3 run that produced `W*`."""
W_STAR_SHA256 = "be00f2b556979457b045b9a4925dd3e547ee586f984b6969b9266e61a7197222"
"""`W*` weights file, from results/p2.3_behavioral_wm__seed1337__20260913T112054+0000.json."""
W_CLEAN_SHA256 = "54f112f4d7edc2ddacc181f8f4db25da27874e7e65aaf9407778a3ce5122fdcf"
"""Clean `W` weights file, from results/p0.5_clean_baseline__seed1337__20260913T071152+0000.json."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_main_model(path: Path, expected_sha256: str, width: int = 32) -> nn.Module:
    """Load a `main_model` state dict strictly, refusing a file with the wrong hash."""
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise SystemExit(f"{path} has SHA-256 {actual}, expected {expected_sha256}. Refusing to measure it.")
    model = main_model(width=width)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model


def measure_all(
    key: bytes,
    images: np.ndarray,
    labels: np.ndarray,
    pool,
    models: dict[str, nn.Module],
    *,
    n: int,
    amplitude: int,
    device: torch.device,
    mean=CIFAR10_MEAN,
    std=CIFAR10_STD,
) -> tuple[str, dict]:
    """Regenerate the trigger set from `key` and score every model on it.

    Returns the trigger bundle digest and, per model, the aggregate summaries
    for the triggers and for the unperturbed base images.
    """
    triggers = generate_triggers(key, images, pool, n=n, amplitude=amplitude)
    responses = trigger_responses(key, triggers, labels)
    digest = make_bundle(triggers, responses, key_kind="owner").digest()

    scores = {}
    for name, model in models.items():
        on_triggers = measure_detection(model, triggers.images, responses, mean=mean, std=std, device=device)
        on_bases = measure_detection(model, triggers.base_images, responses, mean=mean, std=std, device=device)
        scores[name] = {"triggers": on_triggers.summary(), "base_images": on_bases.summary()}
    return digest, scores


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--data-root", type=Path, default=repo_root() / "data")
    parser.add_argument("--w-star", type=Path, default=repo_root() / "results" / "p2.3_behavioral_wm_W_star.pt")
    parser.add_argument("--w-clean", type=Path, default=repo_root() / "results" / "p0.5_clean_baseline_W.pt")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--amplitude", type=int, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    # Inference only; seeding is for the record.
    backends = set_seed(args.seed)
    started = time.perf_counter()
    device = torch.device(args.device)

    models = {
        "W_star": load_main_model(args.w_star, W_STAR_SHA256),
        "W_clean": load_main_model(args.w_clean, W_CLEAN_SHA256),
    }

    from torchvision.datasets import CIFAR10

    cifar = CIFAR10(str(args.data_root), train=True, download=False)
    pool, _ = cifar10_split_indices(len(cifar.data))
    digest, scores = measure_all(
        load_key(args.key), cifar.data, np.asarray(cifar.targets), pool, models,
        n=args.n, amplitude=args.amplitude, device=device,
    )
    if digest != P2_3_BUNDLE_SHA256:
        raise SystemExit(
            f"K regenerates trigger bundle {digest}, but W* was trained on {P2_3_BUNDLE_SHA256}. "
            "Wrong key, dataset, N or amplitude. Refusing to report a WDR."
        )

    headline = scores["W_star"]["triggers"]
    metrics = {
        "wdr": headline["wdr"],
        "fired": headline["fired"],
        "n": headline["n"],
        "trigger_bundle_sha256": digest,
        "trigger_bundle_matches_p2_3": True,
        "models": scores,
    }
    params = {
        "definition": "WDR = k/N, k = #triggers whose eval-mode top-1 prediction equals the keyed target (P2.2)",
        "trigger_source": "regenerated from secrets/K.bin at run time, digest checked against P2.3",
        "n": args.n,
        "amplitude": args.amplitude,
        "dataset": "CIFAR-10 train, 45,000-image main_model split",
        "input_pipeline": "uint8 -> /255 -> Normalize(CIFAR10_MEAN, CIFAR10_STD), no augmentation",
        "models": {
            "W_star": {"source": "P2.3", "weights_sha256": W_STAR_SHA256, "params": count_parameters(models["W_star"])["total"]},
            "W_clean": {"source": "P0.5", "weights_sha256": W_CLEAN_SHA256, "params": count_parameters(models["W_clean"])["total"]},
        },
        "controls": {
            "W_clean/triggers": "same architecture, data and recipe, trained without K",
            "*/base_images": "unperturbed base images scored against the same targets",
        },
        "device": str(device),
    }
    path = write_result(
        name="p2.4_wdr",
        seed=args.seed,
        task="P2.4",
        params=params,
        metrics=metrics,
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        notes=(
            "WDR of W* on the N training triggers, regenerated from K. The triggers are the images "
            "W* was trained on, so this measures retention of trained responses, not generalisation. "
            "Not a false-positive rate (P2.6) and not a p-value (P2.8). Aggregate counts only; "
            "triggers, targets, base indices and per-trigger predictions are secret and not in this file."
        ),
    )

    print(f"trigger bundle : {digest}  (matches P2.3)")
    for name, s in scores.items():
        for kind, r in s.items():
            print(
                f"{name:8s} {kind:12s} fired {r['fired']:>3}/{r['n']}  wdr {r['wdr']:.2%}  "
                f"base label {r['predicted_base_label']:>3}  other {r['predicted_other_class']:>3}  "
                f"p(target) mean {r['target_probability_mean']:.4f} min {r['target_probability_min']:.4f}"
            )
    print(f"\nP2.4 WDR (W* on triggers): {headline['fired']}/{headline['n']} = {headline['wdr']:.2%}")
    print("wrote", path)
    return {"path": path, "metrics": metrics}


if __name__ == "__main__":
    main()
