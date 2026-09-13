"""P0.7 `[GPU]`: train `zk_model`, the tiny MNIST network for the EZKL track.

Records its exact parameter count and clean test accuracy (section 8.2). The
model is small enough that GPU time is minutes, but it still follows the
handoff protocol: runs on Colab via `notebooks/P0.7_colab.ipynb`, checkpoints
every epoch to Drive, resumes after a disconnect.

    # real run
    python experiments/p0_7_train_zk_model.py --drive-root /content/drive/MyDrive/zk-crown

    # 10-second validation of the whole path, synthetic data, meaningless numbers
    python experiments/p0_7_train_zk_model.py --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import torch

from src.data.mnist import MNIST_MEAN, MNIST_STD, MNIST_TEST_SIZE, MNIST_TRAIN_SIZE, mnist_loaders
from src.models import ZK_PARAM_BUDGET, activation_elements, count_parameters, zk_model
from src.training import TrainConfig, fit, resolve_device
from src.utils.artifacts import require_mounted_drive, save_state_dict
from src.utils.results import write_result
from src.utils.seeding import DEFAULT_SEED, set_seed


def _widths(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=1.0,
        help="linear per-step LR warmup, as in P0.5. 0 disables it.",
    )
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--widths",
        type=_widths,
        default=(8, 16, 16),
        help="comma-separated conv widths, e.g. 8,16,16. P8.8 shrinks this if EZKL runs out of RAM.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=60.0,
        help="stop cleanly after this long, having checkpointed. Re-run to continue.",
    )
    parser.add_argument(
        "--drive-root",
        default=None,
        help="Durable output root, e.g. /content/drive/MyDrive/zk-crown. Defaults to the repo.",
    )
    parser.add_argument("--data-root", default="data", help="where MNIST is downloaded")
    parser.add_argument("--device", default=None, help="cuda / cpu. Auto-detected by default.")
    parser.add_argument("--no-resume", action="store_true", help="ignore any existing checkpoint")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 epochs on tiny synthetic data. Validates the path; the numbers are meaningless.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)

    root = Path(args.drive_root) if args.drive_root else Path(__file__).resolve().parents[1]
    require_mounted_drive(root)

    backends = set_seed(args.seed)
    device = resolve_device(args.device)

    run_name = "p0.7_zk_model_smoke" if args.smoke else "p0.7_zk_model"
    checkpoint_dir = root / "checkpoints" / run_name
    results_out = root / "results"
    # Final-epoch weights, not best.pt, for the same reason as P0.5: best.pt is
    # selected on the test set.
    weights_path = root / "models" / f"{run_name}.pt"

    if args.smoke:
        epochs, max_minutes, num_workers = 2, 5.0, 0
    else:
        epochs, max_minutes, num_workers = args.epochs, args.max_minutes, args.num_workers

    print(f"device            : {device}")
    if device.type == "cuda":
        print(f"gpu               : {torch.cuda.get_device_name(0)}")
    print(f"seed              : {args.seed}  backends={backends}")
    print(f"checkpoint_dir    : {checkpoint_dir}")
    print(f"results_out       : {results_out}")
    print(f"mode              : {'SMOKE (synthetic data)' if args.smoke else 'real'}")

    loaders = mnist_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=num_workers,
        smoke=args.smoke,
        seed=args.seed,
    )
    print(f"data              : train={len(loaders['train'].dataset)} test={len(loaders['test'].dataset)}")

    model = zk_model(widths=args.widths)
    counts = count_parameters(model)
    activations = activation_elements(model)
    print(f"params            : {counts['total']:,}  (budget {ZK_PARAM_BUDGET:,})")
    print(f"relu elements     : {activations:,} per input")
    if counts["total"] >= ZK_PARAM_BUDGET:
        raise ValueError(
            f"zk_model has {counts['total']:,} parameters, over the {ZK_PARAM_BUDGET:,} "
            "budget in CLAUDE.md 2.2. Pick smaller --widths."
        )

    config = TrainConfig(
        epochs=epochs,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_minutes=max_minutes,
        extra={"widths": list(args.widths), "smoke": args.smoke},
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"\ntraining for {epochs} epochs (max {max_minutes} min)\n")
    summary = fit(
        model,
        loaders["train"],
        loaders["test"],
        config,
        checkpoint_dir=checkpoint_dir,
        device=device,
        seed=args.seed,
        resume=not args.no_resume,
    )

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device.type == "cuda" else None
    )

    weights_sha256 = None if summary["stopped_early"] else save_state_dict(model, weights_path)

    notes = (
        "SMOKE RUN on synthetic random data. Accuracy is meaningless by "
        "construction and must not be recorded in the Results Ledger."
        if args.smoke
        else f"zk_model for the EZKL track (Phase 8), not the watermarked model. Trained "
        f"on all {MNIST_TRAIN_SIZE:,} MNIST training images (no attacker holdout: "
        f"zk_model is never attacked). Accuracy is top-1 on the official "
        f"{MNIST_TEST_SIZE:,}-image test set. Inputs are normalised with mean "
        f"{MNIST_MEAN[0]} and std {MNIST_STD[0]} outside the model."
    )
    if summary["stopped_early"]:
        notes += (
            f" INCOMPLETE: stopped at epoch {summary['epochs_completed']}/{epochs} "
            "on the time budget. Re-run the same command to continue."
        )

    path = write_result(
        name=run_name,
        seed=args.seed,
        task="P0.7",
        params={
            **config.as_dict(),
            "architecture": "ZKModel",
            "widths": list(args.widths),
            "total_params": counts["total"],
            "conv_and_linear_weights": counts["conv_and_linear_weights"],
            "relu_elements_per_input": activations,
            "param_budget": ZK_PARAM_BUDGET,
            "dataset": "synthetic" if args.smoke else "MNIST",
            "train_size": len(loaders["train"].dataset),
            "test_size": len(loaders["test"].dataset),
            "normalization": {"mean": list(MNIST_MEAN), "std": list(MNIST_STD)},
            "device": summary["device"],
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        metrics={
            "test_accuracy": summary["final"]["accuracy"],
            "test_loss": summary["final"]["loss"],
            "best_test_accuracy": summary["best"]["accuracy"],
            "best_epoch": summary["best"]["epoch"],
            "epochs_completed": summary["epochs_completed"],
            "epochs_planned": epochs,
            "stopped_early": summary["stopped_early"],
            "total_epoch_seconds": round(sum(h["epoch_seconds"] for h in summary["history"]), 2),
            # This session only; a resumed run's earlier sessions are not included.
            "peak_vram_mb_this_session": peak_vram_mb,
            "weights_file": str(weights_path) if weights_sha256 else None,
            "weights_sha256": weights_sha256,
            "history": summary["history"],
        },
        seeded_backends=backends,
        duration_seconds=summary["wall_seconds"],
        notes=notes,
        out_dir=results_out,
    )

    print("\n" + "=" * 68)
    if summary["stopped_early"]:
        print(f"INCOMPLETE: {summary['epochs_completed']}/{epochs} epochs. Re-run to continue.")
    else:
        print(f"COMPLETE: {summary['epochs_completed']} epochs in {summary['wall_seconds'] / 60:.1f} min")
    print(f"params            : {counts['total']:,}")
    print(f"test accuracy     : {summary['final']['accuracy'] * 100:.2f}%")
    print(f"best test accuracy: {summary['best']['accuracy'] * 100:.2f}% (epoch {summary['best']['epoch'] + 1})")
    print(f"result            : {path}")
    print(f"weights           : {weights_path if weights_sha256 else 'not written, run incomplete'}")
    print(f"checkpoints       : {checkpoint_dir}")
    print("=" * 68)

    if args.smoke:
        print("\nSMOKE RUN -- numbers are meaningless. Do not record them.")

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
