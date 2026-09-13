"""P2.6: false positive rate of the behavioral watermark on non-trigger inputs.

    python experiments/p2_6_false_positive_rate.py

CPU only, under a minute. Steps:

1. **Owner targets from `K`.** Regenerates the N = 100 triggers and targets
   from `secrets/K.bin` and checks the bundle digest against the P2.3 training
   bundle, exactly as P2.4 does. Only the targets are used here.
2. **Three sets of 1,000 non-trigger inputs**, drawn from `--seed`:
   - ``random``: i.i.d. uniform uint8 pixels. No label.
   - ``clean_unrelated``: 1,000 distinct CIFAR-10 *test* images. Never trained
     on, and never a trigger base, since bases come from the training split.
   - ``noise_decoys``: the same 1,000 test images plus a ±16 sign pattern
     from the seeded NumPy RNG, which is independent of `K`. Same arithmetic
     and amplitude as real triggers. This asks whether trigger-*like* noise
     alone sets off the response. It goes beyond the two sets P2.6 names.
3. **Score** `W*` and the clean `W` (both hash-checked) on every set with the
   rule in `src/watermark/false_positives.py`. Input ``j`` takes trigger slot
   ``j mod 100`` and fires if the top-1 prediction equals that slot's target.
   Also computed: the chance fire count for the model's predictions, and for
   labelled sets the split into label-contradicting and label-matching fires.

The P2.6 headline is `W*`'s FPR on ``random`` and ``clean_unrelated``. The
result file holds aggregate counts, plus digests of the generated input sets
so a re-run can be checked. Targets and per-input predictions are not written.
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
from p2_3_make_trigger_bundle import build as build_bundle
from p2_4_measure_wdr import P2_3_BUNDLE_SHA256, W_CLEAN_SHA256, W_STAR_SHA256, load_main_model

from src.data import CIFAR10_MEAN, CIFAR10_STD
from src.utils.results import repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.watermark.false_positives import measure_false_positives
from src.watermark.triggers import DEFAULT_AMPLITUDE, DEFAULT_N, apply_perturbation

DEFAULT_COUNT = 1000
NUM_CLASSES = 10
_SET_ENTROPY = 0x5026  # "P2.6"; keeps these draws apart from any other use of the same seed
SET_NAMES = ("random", "clean_unrelated", "noise_decoys")


def make_query_sets(
    test_images: np.ndarray, test_labels: np.ndarray, *, count: int, seed: int, amplitude: int
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray | None]], np.ndarray]:
    """The three non-trigger input sets, and the test-set indices of the clean images.

    Each set comes from its own child of ``SeedSequence([seed, _SET_ENTROPY])``,
    so changing `count` for one set cannot shift another's draws.
    """
    if test_images.dtype != np.uint8 or test_images.ndim != 4:
        raise TypeError("test_images must be (M, H, W, C) uint8")
    if not 1 <= count <= len(test_images):
        raise ValueError(f"count must be in [1, {len(test_images)}]")
    rng_random, rng_clean, rng_noise = (
        np.random.default_rng(s) for s in np.random.SeedSequence([seed, _SET_ENTROPY]).spawn(3)
    )
    shape = (count, *test_images.shape[1:])
    random_images = rng_random.integers(0, 256, shape, dtype=np.uint8)
    indices = np.sort(rng_clean.choice(len(test_images), size=count, replace=False)).astype(np.int64)
    clean = test_images[indices]
    labels = np.asarray(test_labels, dtype=np.int64)[indices]
    signs = (rng_noise.integers(0, 2, shape, dtype=np.int8) * 2 - 1).astype(np.int8)
    decoys = apply_perturbation(clean, signs, amplitude)
    sets = {"random": (random_images, None), "clean_unrelated": (clean, labels), "noise_decoys": (decoys, labels)}
    return sets, indices


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--data-root", type=Path, default=repo_root() / "data")
    parser.add_argument("--w-star", type=Path, default=repo_root() / "results" / "p2.3_behavioral_wm_W_star.pt")
    parser.add_argument("--w-clean", type=Path, default=repo_root() / "results" / "p0.5_clean_baseline_W.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    backends = set_seed(args.seed)
    started = time.perf_counter()
    device = torch.device(args.device)

    models = {
        "W_star": load_main_model(args.w_star, W_STAR_SHA256),
        "W_clean": load_main_model(args.w_clean, W_CLEAN_SHA256),
    }

    bundle = build_bundle(load_key(args.key), args.data_root, DEFAULT_N, DEFAULT_AMPLITUDE, download=False)
    if bundle.digest() != P2_3_BUNDLE_SHA256:
        raise SystemExit(
            f"K regenerates trigger bundle {bundle.digest()}, but W* was trained on {P2_3_BUNDLE_SHA256}. "
            "Refusing to measure against the wrong targets."
        )
    owner_targets = np.asarray(bundle.targets, dtype=np.int64)

    from torchvision.datasets import CIFAR10

    test = CIFAR10(str(args.data_root), train=False, download=False)
    sets, clean_indices = make_query_sets(
        test.data, np.asarray(test.targets), count=args.count, seed=args.seed, amplitude=DEFAULT_AMPLITUDE
    )

    scores = {
        model_name: {
            set_name: measure_false_positives(
                model, images, owner_targets, labels=labels, mean=CIFAR10_MEAN, std=CIFAR10_STD,
                device=device, num_classes=NUM_CLASSES,
            ).summary()
            for set_name, (images, labels) in sets.items()
        }
        for model_name, model in models.items()
    }

    w_star = scores["W_star"]
    metrics = {
        "fpr_random": w_star["random"]["fpr"],
        "fpr_clean_unrelated": w_star["clean_unrelated"]["fpr"],
        "fpr_noise_decoys": w_star["noise_decoys"]["fpr"],
        "trigger_bundle_sha256": bundle.digest(),
        "trigger_bundle_matches_p2_3": True,
        "models": scores,
    }
    params = {
        "definition": (
            "input j takes trigger slot j mod N and fires if eval-mode top-1 == owner target t_(j mod N); "
            "FPR = fired / count. chance_fired = sum_j q(pred_j), q = owner target class frequencies"
        ),
        "count_per_set": args.count,
        "n_owner_targets": len(owner_targets),
        "sets": {
            "random": {"source": "i.i.d. uniform uint8 pixels", "labelled": False, "sha256": sha256_array(sets["random"][0])},
            "clean_unrelated": {
                "source": "CIFAR-10 official test set, distinct images, never trained on, never trigger bases",
                "labelled": True,
                "indices_sha256": sha256_array(clean_indices),
                "sha256": sha256_array(sets["clean_unrelated"][0]),
            },
            "noise_decoys": {
                "source": f"clean_unrelated images + clip(+-{DEFAULT_AMPLITUDE} sign noise) from the seeded NumPy RNG, independent of K",
                "labelled": True,
                "sha256": sha256_array(sets["noise_decoys"][0]),
            },
        },
        "set_rng": f"numpy SeedSequence([seed, {_SET_ENTROPY:#x}]).spawn(3), one child per set",
        "models": {
            "W_star": {"source": "P2.3", "weights_sha256": W_STAR_SHA256},
            "W_clean": {"source": "P0.5", "weights_sha256": W_CLEAN_SHA256},
        },
        "input_pipeline": "uint8 -> /255 -> Normalize(CIFAR10_MEAN, CIFAR10_STD), no augmentation",
        "device": str(device),
    }
    path = write_result(
        name="p2.6_false_positive_rate",
        seed=args.seed,
        task="P2.6",
        params=params,
        metrics=metrics,
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        notes=(
            "Input-level FPR: non-trigger inputs scored against the owner's real targets with the P2.4 fire rule. "
            "Every prediction is some class, so the rate is never zero. chance_fired is the expectation over slot "
            "assignments and reflects how much the model's predictions concentrate on the owner's target classes; "
            "compare it across models. fired - chance_fired is fixed-pairing noise, not leakage, because slots are "
            "unrelated to image content. Not the model-level null (P2.8) and not a p-value. Aggregate counts only; "
            "owner targets and per-input predictions are secret and not in this file."
        ),
    )

    print(f"trigger bundle : {bundle.digest()}  (matches P2.3)")
    for model_name, per_set in scores.items():
        for set_name, s in per_set.items():
            extra = (
                f"  acc {s['accuracy']:.2%}  contradicting {s['fired_contradicting_label']} "
                f"(chance {s['chance_fired_contradicting_label']:.1f})  matching {s['fired_matching_label']}"
                if s["accuracy"] is not None
                else ""
            )
            print(
                f"{model_name:8s} {set_name:16s} fired {s['fired']:>4}/{s['n']}  fpr {s['fpr']:.2%}  "
                f"chance {s['chance_fired']:.1f} ({s['chance_fpr']:.2%}){extra}"
            )
    print("wrote", path)
    return {"path": path, "metrics": metrics}


if __name__ == "__main__":
    main()
