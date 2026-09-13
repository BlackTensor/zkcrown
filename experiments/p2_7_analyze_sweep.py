"""P2.7, local step: measure every sweep model and plot WDR against accuracy drop.

    python experiments/p2_7_analyze_sweep.py

Run on CPU after the Colab sweep (`experiments/p2_7_ratio_sweep.py`) has come
back. It needs, in `results/`, each sweep run's newest complete result JSON
and its ``<run>_W_star.pt`` file. The two reused points are loaded from their
own tasks: P2.3's `W*` (4 triggers per batch) and P0.5's clean `W` (no
triggers).

For every model, the same measurements as P2.4 and P2.5:

- **Weights** are refused unless their SHA-256 matches the run's record.
- **Triggers** are regenerated from `secrets/K.bin`. Each run's recorded bundle
  digest must equal the regenerated one.
- **WDR** comes from `src.watermark.detection.measure_detection`.
- **Accuracy drop** against clean `W` is scored per image on the 10,000-image
  test set with `src.utils.stats.paired_accuracy_difference`, including an
  exact McNemar p-value and a 95% interval. CPU accuracy must reproduce the
  accuracy each Colab run recorded.

Writes `figures/p2.7_wdr_vs_accuracy_drop.png` and a result JSON. All models
share one seed, so the curve is one training run per ratio. The drop intervals
cover test-set sampling only, not seed-to-seed variation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from make_master_key import DEFAULT_KEY_PATH, load_key
from p2_4_measure_wdr import P2_3_BUNDLE_SHA256, W_CLEAN_SHA256, W_STAR_SHA256
from p2_7_ratio_sweep import SWEEP

from src.data import CIFAR10_MEAN, CIFAR10_STD, cifar10_loaders
from src.models import main_model
from src.training import per_sample_correct
from src.utils.results import repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.utils.stats import paired_accuracy_difference
from src.watermark.bundle import make_bundle
from src.watermark.detection import measure_detection
from src.watermark.responses import trigger_responses
from src.watermark.triggers import DEFAULT_AMPLITUDE, DEFAULT_N, generate_triggers

P0_5_RESULT = "results/p0.5_clean_baseline__seed1337__20260913T071152+0000.json"
P2_3_RESULT = "results/p2.3_behavioral_wm__seed1337__20260913T112054+0000.json"
FIGURE = "figures/p2.7_wdr_vs_accuracy_drop.png"

# Reference palette (dataviz skill, references/palette.md), light mode. One series per panel.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


# --- locating and checking inputs -------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def newest_complete(results_dir: Path, name: str, seed: int) -> tuple[dict, Path]:
    """The newest complete result for sweep run `name`, and its weights file next to it."""
    weights = results_dir / f"{name}_W_star.pt"
    for path in sorted(results_dir.glob(f"{name}__seed{seed}__*.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        metrics = record["metrics"]
        if metrics.get("stopped_early") is False and metrics.get("weights_sha256"):
            return record, weights
    raise SystemExit(f"no complete result for {name} in {results_dir}. Bring back the sweep files first.")


def load_checked(weights: Path, expected_sha256: str) -> torch.nn.Module:
    if not weights.is_file():
        raise SystemExit(f"{weights} is missing")
    actual = sha256_file(weights)
    if actual != expected_sha256:
        raise SystemExit(f"{weights} has SHA-256 {actual}, but its run recorded {expected_sha256}")
    model = main_model(width=32)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model


def stable_from_epoch(history: list[dict]) -> int | None:
    """First epoch index from which the training trigger accuracy stayed at 100%, or None."""
    stable = None
    for record in history:
        if record.get("trigger_accuracy") == 1.0:
            stable = record["epoch"] if stable is None else stable
        else:
            stable = None
    return stable


# --- measurement ------------------------------------------------------------


def measure_point(
    name: str,
    record: dict | None,
    model: torch.nn.Module,
    *,
    trigger_images: np.ndarray,
    responses,
    clean_correct: torch.Tensor,
    test_loader,
    device: torch.device,
) -> dict:
    """WDR and paired accuracy drop for one model. `record` is None for clean `W`."""
    detection = measure_detection(model, trigger_images, responses, mean=CIFAR10_MEAN, std=CIFAR10_STD, device=device)
    correct = per_sample_correct(model, test_loader, device)
    paired = paired_accuracy_difference(clean_correct.tolist(), correct.tolist())

    if record is None:
        tpb, every, samples, ratio, recorded_acc, stable, source = 0, None, 0, 0.0, None, None, "P0.5"
    else:
        p, m = record["params"], record["metrics"]
        tpb, every = p["triggers_per_batch"], p.get("trigger_every", 1)
        samples = p["trigger_samples_per_epoch"]
        ratio = samples / p["train_size"]
        recorded_acc = m["test_accuracy"]
        stable = stable_from_epoch(m["history"])
        source = record["task"]

    return {
        "name": name,
        "source_task": source,
        "triggers_per_batch": tpb,
        "trigger_every": every,
        "trigger_samples_per_epoch": samples,
        "trigger_to_clean_ratio": ratio,
        "wdr": detection.wdr,
        "fired": detection.fired,
        "n_triggers": detection.n,
        "target_probability_mean": detection.summary()["target_probability_mean"],
        "target_probability_min": detection.summary()["target_probability_min"],
        "test_accuracy": paired["other_accuracy"],
        "recorded_test_accuracy": recorded_acc,
        "matches_recorded": None if recorded_acc is None else paired["other_accuracy"] == recorded_acc,
        "accuracy_drop_pp": paired["drop"] * 100,
        "drop_ci95_pp": [paired["drop_ci_low"] * 100, paired["drop_ci_high"] * 100],
        "mcnemar_exact_p": paired["mcnemar_exact_p"],
        "clean_only_right": paired["reference_only_right"],
        "model_only_right": paired["other_only_right"],
        "training_trigger_accuracy_stable_from_epoch": stable,
    }


# --- figure -----------------------------------------------------------------


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _ratio_label(ratio: float) -> str:
    return "0" if ratio == 0 else f"{ratio * 100:.3g}%"


def plot_sweep(rows: list[dict], path: Path) -> None:
    """Three panels: WDR vs ratio, drop vs ratio, WDR vs drop. One series each."""
    rows = sorted(rows, key=lambda r: r["trigger_to_clean_ratio"])
    positive = [r["trigger_to_clean_ratio"] for r in rows if r["trigger_to_clean_ratio"] > 0]
    # A log axis has no zero. The no-trigger point is drawn well left of the
    # sparsest run and joined by a dashed segment, so it does not read as one
    # more ratio step.
    zero_x = min(positive) / 10 if positive else 1e-4
    xs = [r["trigger_to_clean_ratio"] or zero_x for r in rows]
    wdr = [r["wdr"] * 100 for r in rows]
    drop = [r["accuracy_drop_pp"] for r in rows]
    err = np.array([[d - r["drop_ci95_pp"][0], r["drop_ci95_pp"][1] - d] for d, r in zip(drop, rows)]).T
    reused = [r["source_task"] != "P2.7" for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    marker = dict(s=64, zorder=3, linewidths=2)

    has_zero = rows[0]["trigger_to_clean_ratio"] == 0

    def points(ax, x, y):
        start = 1 if has_zero else 0
        if has_zero and len(x) > 1:
            ax.plot(x[:2], y[:2], color=SERIES, linewidth=2, linestyle=(0, (3, 3)), zorder=2)
        ax.plot(x[start:], y[start:], color=SERIES, linewidth=2, zorder=2)
        for xi, yi, hollow in zip(x, y, reused):
            ax.scatter([xi], [yi], color=SURFACE if hollow else SERIES, edgecolor=SERIES, **marker)

    ticks = xs
    tick_labels = ["0 (clean W)" if r["trigger_to_clean_ratio"] == 0 else _ratio_label(r["trigger_to_clean_ratio"]) for r in rows]

    ax = axes[0]
    _style(ax)
    points(ax, xs, wdr)
    ax.set_xscale("log")
    ax.set_xticks(ticks, tick_labels, rotation=35, ha="right")
    ax.minorticks_off()
    ax.set_ylim(-5, 105)
    ax.set_xlabel("trigger samples per clean sample (log)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("WDR on the 100 triggers (%)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Watermark detection rate", color=INK, fontsize=11, loc="left")

    ax = axes[1]
    _style(ax)
    ax.axhline(0, color=INK_SECONDARY, linewidth=1, zorder=1)
    ax.errorbar(xs, drop, yerr=err, fmt="none", ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3, zorder=2)
    points(ax, xs, drop)
    ax.set_xscale("log")
    ax.set_xticks(ticks, tick_labels, rotation=35, ha="right")
    ax.minorticks_off()
    ax.set_xlabel("trigger samples per clean sample (log)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("test accuracy drop vs clean W (pp)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Accuracy drop, 95% CI (test images only)", color=INK, fontsize=11, loc="left")

    ax = axes[2]
    _style(ax)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1, zorder=1)
    ax.errorbar(drop, wdr, xerr=err, fmt="none", ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3, zorder=2)
    points(ax, drop, wdr)
    for i, (d, w) in enumerate(zip(drop, wdr), start=1):
        # Short numbered labels, keyed in the caption: neighbouring ratios often
        # sit on top of each other at WDR = 100%, where full ratio labels collide.
        ax.annotate(
            str(i), (d, w), textcoords="offset points", xytext=(5, -13) if i % 2 else (5, 7),
            fontsize=8, color=INK_SECONDARY,
        )
    ax.set_ylim(-5, 105)
    ax.set_xlabel("test accuracy drop vs clean W (pp)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("WDR (%)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Tradeoff (points numbered by ratio)", color=INK, fontsize=11, loc="left")

    key = ", ".join(f"{i} = {_ratio_label(r['trigger_to_clean_ratio'])}" for i, r in enumerate(rows, start=1))
    fig.text(
        0.01, -0.06,
        "One training run per ratio, seed 1337, CIFAR-10 main_model. Filled: P2.7 sweep runs. Hollow: reused P0.5 (clean W) "
        "and P2.3 (4 triggers per batch).\nIntervals cover test-set sampling only, not seed-to-seed variation. "
        f"Tradeoff points: {key}.",
        fontsize=8, color=INK_SECONDARY, ha="left", va="top",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


# --- entry point ------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--data-root", type=Path, default=repo_root() / "data")
    parser.add_argument("--results-dir", type=Path, default=repo_root() / "results")
    parser.add_argument("--figure", type=Path, default=repo_root() / FIGURE)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    backends = set_seed(args.seed)
    started = time.perf_counter()
    device = torch.device(args.device)

    from torchvision.datasets import CIFAR10

    from src.data.cifar10 import cifar10_split_indices

    cifar = CIFAR10(str(args.data_root), train=True, download=False)
    pool, _ = cifar10_split_indices(len(cifar.data))
    key = load_key(args.key)
    triggers = generate_triggers(key, cifar.data, pool, n=DEFAULT_N, amplitude=DEFAULT_AMPLITUDE)
    responses = trigger_responses(key, triggers, np.asarray(cifar.targets))
    digest = make_bundle(triggers, responses, key_kind="owner").digest()
    if digest != P2_3_BUNDLE_SHA256:
        raise SystemExit(f"K regenerates bundle {digest}, expected {P2_3_BUNDLE_SHA256}")

    points: list[tuple[str, dict | None, Path, str]] = [
        ("W_clean", None, args.results_dir / "p0.5_clean_baseline_W.pt", W_CLEAN_SHA256),
        (
            "p2.3_behavioral_wm",
            json.loads((repo_root() / P2_3_RESULT).read_text(encoding="utf-8")),
            args.results_dir / "p2.3_behavioral_wm_W_star.pt",
            W_STAR_SHA256,
        ),
    ]
    for run in SWEEP:
        record, weights = newest_complete(args.results_dir, run.name, args.seed)
        p = record["params"]
        if (p["triggers_per_batch"], p.get("trigger_every", 1)) != (run.triggers_per_batch, run.trigger_every):
            raise SystemExit(f"{run.name} was trained with a different ratio than the sweep defines")
        points.append((run.name, record, weights, record["metrics"]["weights_sha256"]))

    for name, record, _, _ in points:
        if record is not None and record["params"]["trigger_bundle"]["sha256"] != digest:
            raise SystemExit(f"{name} was trained on bundle {record['params']['trigger_bundle']['sha256']}, not {digest}")

    test_loader = cifar10_loaders(args.data_root, eval_batch_size=500, num_workers=0, download=False)["test"]
    clean_model = load_checked(points[0][2], points[0][3])
    clean_correct = per_sample_correct(clean_model, test_loader, device)

    rows = []
    for name, record, weights, sha in points:
        model = clean_model if record is None else load_checked(weights, sha)
        row = measure_point(
            name, record, model, trigger_images=triggers.images, responses=responses,
            clean_correct=clean_correct, test_loader=test_loader, device=device,
        )
        row["weights_sha256"] = sha
        rows.append(row)
        print(
            f"{name:20s} ratio {row['trigger_to_clean_ratio'] * 100:7.3f}%  WDR {row['fired']:>3}/100  "
            f"acc {row['test_accuracy']:.2%}  drop {row['accuracy_drop_pp']:+.2f} pp "
            f"[{row['drop_ci95_pp'][0]:+.2f}, {row['drop_ci95_pp'][1]:+.2f}]  p {row['mcnemar_exact_p']:.3g}  "
            f"matches recorded: {row['matches_recorded']}"
        )

    rows.sort(key=lambda r: r["trigger_to_clean_ratio"])
    mismatched = [r["name"] for r in rows if r["matches_recorded"] is False]
    plot_sweep(rows, args.figure)

    path = write_result(
        name="p2.7_ratio_sweep",
        seed=args.seed,
        task="P2.7",
        params={
            "sweep": [vars(run) for run in SWEEP],
            "reused_points": {"W_clean": P0_5_RESULT, "p2.3_behavioral_wm": P2_3_RESULT},
            "trigger_bundle_sha256": digest,
            "wdr": "P2.4 measure_detection on the 100 triggers regenerated from K",
            "accuracy_drop": "P2.5 paired comparison against clean W on the 10,000-image test set, 95% normal CI, exact McNemar",
            "figure": FIGURE,
            "device": str(device),
        },
        metrics={
            "rows": rows,
            "all_cpu_accuracies_match_recorded": not mismatched,
            "mismatched_runs": mismatched,
        },
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        notes=(
            "One training run per ratio, same seed and clean batch order. Drop intervals and McNemar p-values treat "
            "test images as the sample; seed-to-seed variation is not measured. WDR is on the training triggers."
        ),
    )
    if mismatched:
        print(f"WARNING: CPU accuracy differs from the recorded Colab accuracy for {mismatched}")
    print("wrote", args.figure)
    print("wrote", path)
    return {"path": path, "rows": rows}


if __name__ == "__main__":
    main()
