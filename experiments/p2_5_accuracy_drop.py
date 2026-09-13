"""P2.5: clean accuracy of `W*` and its drop against the P0.6 baseline `W`.

    python experiments/p2_5_accuracy_drop.py

CPU only, a few minutes. Steps:

1. Load `W*` (P2.3) and the clean `W` (P0.5), each refused unless its file's
   SHA-256 matches the one its run recorded.
2. Score both on the official 10,000-image CIFAR-10 test set, per image, and
   check the totals against the accuracies their Colab runs recorded.
3. Report ``drop = acc(W) - acc(W*)`` on the test set, which is what P0.6's
   baseline is defined on, with a paired analysis: how many images each model
   alone gets right, an exact McNemar p-value and a 95% interval
   (`src/utils/stats.py`).
4. Repeat the paired analysis on the 5,000-image attacker holdout as a
   secondary check on a different split.

The paired statistics treat the test images as the random sample. They do not
measure seed-to-seed variation between training runs: there is one run of
each model, so that is unmeasured.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import torch
from p2_4_measure_wdr import W_CLEAN_SHA256, W_STAR_SHA256, load_main_model

from src.data import cifar10_loaders
from src.training import evaluate, per_sample_correct
from src.utils.results import repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.utils.stats import paired_accuracy_difference

P0_5_RESULT = "results/p0.5_clean_baseline__seed1337__20260913T071152+0000.json"
P2_3_RESULT = "results/p2.3_behavioral_wm__seed1337__20260913T112054+0000.json"


def recorded(path: Path, weights_sha256: str) -> dict:
    """The accuracies a training run recorded, after checking it describes these weights."""
    metrics = json.loads(Path(path).read_text(encoding="utf-8"))["metrics"]
    if metrics["weights_sha256"] != weights_sha256:
        raise SystemExit(f"{path} records weights {metrics['weights_sha256']}, not {weights_sha256}")
    return {"test_accuracy": metrics["test_accuracy"], "holdout_accuracy": metrics["holdout_accuracy"]}


def score(model, loader, device) -> dict:
    """Aggregate accuracy and loss, plus per-image correctness that must agree with it."""
    agg = evaluate(model, loader, device)
    correct = per_sample_correct(model, loader, device)
    if len(correct) != agg["n"] or round(agg["accuracy"] * agg["n"]) != int(correct.sum()):
        raise RuntimeError("per-image correctness disagrees with evaluate()")
    return {"accuracy": agg["accuracy"], "loss": agg["loss"], "n": agg["n"], "correct": correct}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-root", type=Path, default=repo_root() / "data")
    parser.add_argument("--w-star", type=Path, default=repo_root() / "results" / "p2.3_behavioral_wm_W_star.pt")
    parser.add_argument("--w-clean", type=Path, default=repo_root() / "results" / "p0.5_clean_baseline_W.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    # Inference only; seeding is for the record.
    backends = set_seed(args.seed)
    started = time.perf_counter()
    device = torch.device(args.device)

    models = {
        "W_clean": load_main_model(args.w_clean, W_CLEAN_SHA256),
        "W_star": load_main_model(args.w_star, W_STAR_SHA256),
    }
    recorded_by_run = {
        "W_clean": recorded(repo_root() / P0_5_RESULT, W_CLEAN_SHA256),
        "W_star": recorded(repo_root() / P2_3_RESULT, W_STAR_SHA256),
    }
    loaders = cifar10_loaders(args.data_root, eval_batch_size=500, num_workers=0, download=False)

    scores = {name: {split: score(m, loaders[split], device) for split in ("test", "holdout")} for name, m in models.items()}

    per_model = {}
    for name, s in scores.items():
        rec = recorded_by_run[name]
        per_model[name] = {
            "test_accuracy": s["test"]["accuracy"],
            "test_loss": s["test"]["loss"],
            "test_correct": int(s["test"]["correct"].sum()),
            "holdout_accuracy": s["holdout"]["accuracy"],
            "holdout_correct": int(s["holdout"]["correct"].sum()),
            "recorded_test_accuracy": rec["test_accuracy"],
            "recorded_holdout_accuracy": rec["holdout_accuracy"],
            "matches_recorded": s["test"]["accuracy"] == rec["test_accuracy"]
            and s["holdout"]["accuracy"] == rec["holdout_accuracy"],
        }

    paired = {
        split: paired_accuracy_difference(
            scores["W_clean"][split]["correct"].tolist(), scores["W_star"][split]["correct"].tolist()
        )
        for split in ("test", "holdout")
    }
    test = paired["test"]
    metrics = {
        "w_star_test_accuracy": test["other_accuracy"],
        "baseline_test_accuracy": test["reference_accuracy"],
        "accuracy_drop_pp": round(test["drop"] * 100, 10),
        "accuracy_drop_relative": test["drop"] / test["reference_accuracy"],
        "drop_ci95_pp": [round(test["drop_ci_low"] * 100, 10), round(test["drop_ci_high"] * 100, 10)],
        "mcnemar_exact_p": test["mcnemar_exact_p"],
        "models": per_model,
        "paired": paired,
    }
    params = {
        "baseline": {"task": "P0.5/P0.6", "result": P0_5_RESULT, "weights_sha256": W_CLEAN_SHA256},
        "watermarked": {"task": "P2.3", "result": P2_3_RESULT, "weights_sha256": W_STAR_SHA256},
        "primary_split": "official CIFAR-10 test set, 10,000 images",
        "secondary_split": "5,000-image attacker holdout, SPLIT_SEED split",
        "drop_definition": "acc(W) - acc(W*), positive means W* is less accurate",
        "paired_statistics": "exact two-sided McNemar; 95% normal interval on per-image differences",
        "device": str(device),
    }
    path = write_result(
        name="p2.5_accuracy_drop",
        seed=args.seed,
        task="P2.5",
        params=params,
        metrics=metrics,
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        notes=(
            "Clean accuracy of behavioral-watermark W* against the P0.6 baseline W, both re-scored on CPU "
            "from hash-checked weights. The paired statistics treat the test images as the sample; "
            "training-seed variation is not measured, since each model is a single run."
        ),
    )

    for name, m in per_model.items():
        print(
            f"{name:8s} test {m['test_correct']}/10000 = {m['test_accuracy']:.2%} (loss {m['test_loss']:.4f})  "
            f"holdout {m['holdout_correct']}/5000 = {m['holdout_accuracy']:.2%}  matches recorded: {m['matches_recorded']}"
        )
    for split, p in paired.items():
        print(
            f"{split:8s} drop {p['drop'] * 100:+.2f} pp  95% CI [{p['drop_ci_low'] * 100:+.2f}, {p['drop_ci_high'] * 100:+.2f}]  "
            f"W-only right {p['reference_only_right']}  W*-only right {p['other_only_right']}  McNemar p {p['mcnemar_exact_p']:.4g}"
        )
    print("wrote", path)
    return {"path": path, "metrics": metrics}


if __name__ == "__main__":
    main()
