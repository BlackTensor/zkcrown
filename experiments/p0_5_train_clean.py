"""P0.5 `[GPU]`: train the clean CIFAR-10 model `W`.

This is the baseline every later accuracy drop is measured against (P0.6), so
it is trained once, carefully, and then left alone.

Runs on Google Colab via `notebooks/P0.5_colab.ipynb`. Checkpoints every epoch
to Drive and resumes from the last checkpoint, so a session timeout costs one
epoch rather than the run (CLAUDE.md 2.1).

    # real run
    python experiments/p0_5_train_clean.py --drive-root /content/drive/MyDrive/zk-crown

    # 30-second validation of the whole path, synthetic data, meaningless numbers
    python experiments/p0_5_train_clean.py --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import torch

from src.data import HOLDOUT_SIZE, SPLIT_SEED, TRAIN_SIZE, cifar10_loaders
from src.models import count_parameters, main_model
from src.training import TrainConfig, evaluate, fit, resolve_device
from src.utils.results import write_result
from src.utils.seeding import DEFAULT_SEED, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=32, help="main_model first-block width")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=90.0,
        help="stop cleanly after this long, having checkpointed. Re-run to continue.",
    )
    parser.add_argument(
        "--drive-root",
        default=None,
        help="Durable output root, e.g. /content/drive/MyDrive/zk-crown. "
        "Checkpoints and results go under here. Defaults to the repo, which on "
        "Colab is wiped between sessions.",
    )
    parser.add_argument("--data-root", default="data", help="where CIFAR-10 is downloaded")
    parser.add_argument("--device", default=None, help="cuda / cpu. Auto-detected by default.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any existing checkpoint and start over",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 epochs on tiny synthetic data. Validates the path; the numbers "
        "are meaningless and the result is tagged as such.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)

    backends = set_seed(args.seed)
    device = resolve_device(args.device)

    root = Path(args.drive_root) if args.drive_root else Path(__file__).resolve().parents[1]
    run_name = "p0.5_smoke" if args.smoke else "p0.5_clean_baseline"
    checkpoint_dir = root / "checkpoints" / run_name
    results_out = root / "results"

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

    loaders = cifar10_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=num_workers,
        augment=True,
        smoke=args.smoke,
        seed=args.seed,
    )
    print(
        f"data              : train={len(loaders['train'].dataset)} "
        f"holdout={len(loaders['holdout'].dataset)} test={len(loaders['test'].dataset)}"
    )

    model = main_model(width=args.width)
    counts = count_parameters(model)
    print(f"params            : {counts['total']:,}")

    config = TrainConfig(
        epochs=epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_minutes=max_minutes,
        extra={"width": args.width, "smoke": args.smoke},
    )

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

    # The attacker holdout must stay untouched by training. Reporting accuracy
    # on it is a sanity check, not a result: if it tracks test accuracy the
    # split behaved as intended.
    holdout_metrics = evaluate(model, loaders["holdout"], device)

    notes = (
        "SMOKE RUN on synthetic random data. Accuracy is meaningless by "
        "construction and must not be recorded in the Results Ledger."
        if args.smoke
        else f"Clean baseline W. Trained on {TRAIN_SIZE} of the 50,000 CIFAR-10 "
        f"training images; {HOLDOUT_SIZE} are reserved as the P4.5/P4.6 attacker "
        f"holdout (split fixed by SPLIT_SEED={SPLIT_SEED}). Accuracy is top-1 on "
        "the official 10,000-image test set."
    )
    if summary["stopped_early"]:
        notes += (
            f" INCOMPLETE: stopped at epoch {summary['epochs_completed']}/{epochs} "
            "on the time budget. Re-run the same command to continue."
        )

    path = write_result(
        name=run_name,
        seed=args.seed,
        task="P0.5",
        params={
            **config.as_dict(),
            "architecture": "MainModel",
            "total_params": counts["total"],
            "dataset": "synthetic" if args.smoke else "CIFAR-10",
            "train_size": len(loaders["train"].dataset),
            "holdout_size": len(loaders["holdout"].dataset),
            "test_size": len(loaders["test"].dataset),
            "split_seed": SPLIT_SEED,
            "device": summary["device"],
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        metrics={
            "test_accuracy": summary["final"]["accuracy"],
            "test_loss": summary["final"]["loss"],
            "best_test_accuracy": summary["best"]["accuracy"],
            "best_epoch": summary["best"]["epoch"],
            "holdout_accuracy": holdout_metrics["accuracy"],
            "epochs_completed": summary["epochs_completed"],
            "epochs_planned": epochs,
            "stopped_early": summary["stopped_early"],
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
    print(f"test accuracy     : {summary['final']['accuracy'] * 100:.2f}%")
    print(f"best test accuracy: {summary['best']['accuracy'] * 100:.2f}% (epoch {summary['best']['epoch'] + 1})")
    print(f"holdout accuracy  : {holdout_metrics['accuracy'] * 100:.2f}%  (sanity check)")
    print(f"result            : {path}")
    print(f"checkpoints       : {checkpoint_dir}")
    print("=" * 68)

    if args.smoke:
        print("\nSMOKE RUN -- numbers are meaningless. Do not record them.")

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
