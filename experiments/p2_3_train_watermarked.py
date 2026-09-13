"""P2.3 `[GPU]`: joint training on clean data plus the trigger set, producing `W*`.

Same model, data split, seed, optimiser, schedule and epoch count as the clean
baseline `W` (P0.5). The only change is that every clean batch carries
``--triggers-per-batch`` trigger samples labelled with their P2.2 targets (see
`src/watermark/behavioral.py`). This `W*` carries the behavioral watermark
only. The weight watermark is Phase 3, and P3.6 produces the final
dual-watermarked model.

Runs on Google Colab via `notebooks/P2.3_colab.ipynb`. Checkpoints every epoch
to Drive and resumes from the last checkpoint.

    # real run: needs the secret bundle from p2_3_make_trigger_bundle.py
    python experiments/p2_3_train_watermarked.py \\
        --drive-root /content/drive/MyDrive/zk-crown \\
        --trigger-bundle /content/drive/MyDrive/zk-crown/secrets/trigger_bundle.npz \\
        --expected-trigger-sha256 <digest>

    # validation of the whole path: synthetic data, a TEST-key bundle, meaningless numbers
    python experiments/p2_3_train_watermarked.py --smoke

The result JSON records the bundle digest but never the triggers, targets or
base indices, which are secret. The per-epoch ``trigger_accuracy`` is a
training diagnostic. The ledger's WDR comes from P2.4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import numpy as np
import torch

from src.data import CIFAR10_MEAN, CIFAR10_STD, HOLDOUT_SIZE, SPLIT_SEED, TRAIN_SIZE, cifar10_loaders
from src.data.cifar10 import cifar10_split_indices
from src.models import count_parameters, main_model
from src.training import TrainConfig, evaluate, fit, resolve_device
from src.utils.artifacts import require_mounted_drive, save_state_dict
from src.utils.results import write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.watermark.behavioral import (
    DEFAULT_TRIGGERS_PER_BATCH,
    TriggerMixLoader,
    trigger_metrics,
    trigger_tensors,
)
from src.watermark.bundle import TriggerBundle, check_against_dataset, load_bundle, make_bundle
from src.watermark.responses import trigger_responses
from src.watermark.triggers import generate_triggers

SMOKE_TEST_KEY = bytes(range(32))
"""TEST KEY ONLY, for the synthetic smoke bundle."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=32, help="main_model first-block width")
    parser.add_argument(
        "--triggers-per-batch",
        type=int,
        default=DEFAULT_TRIGGERS_PER_BATCH,
        help="trigger samples appended to every clean batch (P2.7 sweeps this)",
    )
    parser.add_argument("--trigger-bundle", type=Path, default=None, help="the secret .npz from p2_3_make_trigger_bundle.py")
    parser.add_argument(
        "--expected-trigger-sha256",
        default=None,
        help="bundle digest printed by p2_3_make_trigger_bundle.py. Required for a real run.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-minutes", type=float, default=90.0)
    parser.add_argument("--drive-root", default=None, help="durable output root, e.g. /content/drive/MyDrive/zk-crown")
    parser.add_argument("--data-root", default="data", help="where CIFAR-10 is downloaded")
    parser.add_argument("--device", default=None, help="cuda / cpu. Auto-detected by default.")
    parser.add_argument("--no-resume", action="store_true", help="ignore any existing checkpoint")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 epochs on synthetic data with a TEST-key bundle. The numbers are meaningless.",
    )
    return parser


def smoke_bundle() -> TriggerBundle:
    """A 20-trigger bundle from a TEST key on synthetic images. Never a real watermark."""
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (200, 32, 32, 3), dtype=np.uint8)
    labels = rng.integers(0, 10, 200)
    triggers = generate_triggers(SMOKE_TEST_KEY, images, range(200), n=20)
    return make_bundle(triggers, trigger_responses(SMOKE_TEST_KEY, triggers, labels), key_kind="test")


def load_real_bundle(args: argparse.Namespace) -> TriggerBundle:
    if args.trigger_bundle is None or args.expected_trigger_sha256 is None:
        raise SystemExit("a real run needs --trigger-bundle and --expected-trigger-sha256")
    if not args.trigger_bundle.is_file():
        raise SystemExit(f"trigger bundle not found at {args.trigger_bundle}")
    bundle = load_bundle(args.trigger_bundle, expected_digest=args.expected_trigger_sha256)
    if bundle.key_kind != "owner":
        raise SystemExit(f"refusing to embed a {bundle.key_kind!r}-key bundle in a real run")

    from torchvision.datasets import CIFAR10

    raw = CIFAR10(args.data_root, train=True, download=False)
    train_idx, _ = cifar10_split_indices(len(raw.data))
    check_against_dataset(bundle, raw.data, raw.targets, train_idx)
    return bundle


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)

    root = Path(args.drive_root) if args.drive_root else Path(__file__).resolve().parents[1]
    require_mounted_drive(root)

    backends = set_seed(args.seed)
    device = resolve_device(args.device)

    run_name = "p2.3_smoke" if args.smoke else "p2.3_behavioral_wm"
    checkpoint_dir = root / "checkpoints" / run_name
    results_out = root / "results"
    # Final-epoch weights, not best.pt, which is selected on test accuracy.
    weights_path = root / "models" / f"{run_name}_W_star.pt"

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
    print(f"mode              : {'SMOKE (synthetic data, TEST key)' if args.smoke else 'real'}")

    loaders = cifar10_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=num_workers,
        augment=True,
        smoke=args.smoke,
        seed=args.seed,
    )
    bundle = smoke_bundle() if args.smoke else load_real_bundle(args)
    digest = bundle.digest()
    print(f"trigger bundle    : n={len(bundle)} amplitude={bundle.amplitude} key_kind={bundle.key_kind}")
    print(f"bundle sha256     : {digest}  (verified)")

    trigger_x, trigger_y = trigger_tensors(bundle, CIFAR10_MEAN, CIFAR10_STD)
    train_loader = TriggerMixLoader(
        loaders["train"], trigger_x, trigger_y, triggers_per_batch=args.triggers_per_batch, seed=args.seed
    )
    print(
        f"data              : clean train={len(loaders['train'].dataset)} "
        f"holdout={len(loaders['holdout'].dataset)} test={len(loaders['test'].dataset)} "
        f"+ {args.triggers_per_batch} triggers per batch"
    )

    model = main_model(width=args.width)
    counts = count_parameters(model)
    print(f"params            : {counts['total']:,}")

    config = TrainConfig(
        epochs=epochs,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_minutes=max_minutes,
        extra={
            "width": args.width,
            "smoke": args.smoke,
            "triggers_per_batch": args.triggers_per_batch,
            "trigger_bundle_sha256": digest,
        },
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def log_triggers(epoch: int, record: dict) -> None:
        print(f"           trigger_acc {record['trigger_accuracy'] * 100:.1f}%  trigger_loss {record['trigger_loss']:.4f}")

    print(f"\ntraining for {epochs} epochs (max {max_minutes} min)\n")
    summary = fit(
        model,
        train_loader,
        loaders["test"],
        config,
        checkpoint_dir=checkpoint_dir,
        device=device,
        seed=args.seed,
        resume=not args.no_resume,
        epoch_metrics=lambda m: trigger_metrics(m, trigger_x, trigger_y, device),
        on_epoch_end=log_triggers,
    )

    final_triggers = trigger_metrics(model, trigger_x, trigger_y, device)
    holdout_metrics = evaluate(model, loaders["holdout"], device)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device.type == "cuda" else None
    weights_sha256 = None if summary["stopped_early"] else save_state_dict(model, weights_path)

    notes = (
        "SMOKE RUN on synthetic random data with a TEST-key bundle. Every number is "
        "meaningless by construction and must not be recorded in the Results Ledger."
        if args.smoke
        else f"Behavioral-watermark W*, trained from scratch with the P0.5 recipe on the same "
        f"{TRAIN_SIZE} clean images ({HOLDOUT_SIZE} attacker holdout excluded, SPLIT_SEED={SPLIT_SEED}) "
        f"plus {len(bundle)} triggers, {args.triggers_per_batch} appended to every batch. Accuracy is "
        "top-1 on the official 10,000-image test set. final_trigger_accuracy is a training "
        "diagnostic on the training triggers; the ledger WDR is P2.4's measurement. The "
        "triggers, targets and base indices are secret and are not in this file."
    )
    if summary["stopped_early"]:
        notes += (
            f" INCOMPLETE: stopped at epoch {summary['epochs_completed']}/{epochs} "
            "on the time budget. Re-run the same command to continue."
        )

    batches = len(train_loader)
    path = write_result(
        name=run_name,
        seed=args.seed,
        task="P2.3",
        params={
            **config.as_dict(),
            "architecture": "MainModel",
            "total_params": counts["total"],
            "dataset": "synthetic" if args.smoke else "CIFAR-10",
            "train_size": len(loaders["train"].dataset),
            "holdout_size": len(loaders["holdout"].dataset),
            "test_size": len(loaders["test"].dataset),
            "split_seed": SPLIT_SEED,
            "trigger_bundle": {**bundle.header(), "n": len(bundle), "sha256": digest},
            "trigger_mapping": "per-trigger-keyed (P2.2)",
            "triggers_per_batch": args.triggers_per_batch,
            "trigger_fraction_of_full_batch": args.triggers_per_batch / (args.batch_size + args.triggers_per_batch),
            "trigger_samples_per_epoch": args.triggers_per_batch * batches,
            "trigger_augmentation": "none",
            "device": summary["device"],
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        metrics={
            "test_accuracy": summary["final"]["accuracy"],
            "test_loss": summary["final"]["loss"],
            "best_test_accuracy": summary["best"]["accuracy"],
            "best_epoch": summary["best"]["epoch"],
            "holdout_accuracy": holdout_metrics["accuracy"],
            "final_trigger_accuracy": final_triggers["trigger_accuracy"],
            "final_trigger_loss": final_triggers["trigger_loss"],
            "epochs_completed": summary["epochs_completed"],
            "epochs_planned": epochs,
            "stopped_early": summary["stopped_early"],
            "total_epoch_seconds": round(sum(h["epoch_seconds"] for h in summary["history"]), 2),
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
    print(f"test accuracy     : {summary['final']['accuracy'] * 100:.2f}%")
    print(f"trigger accuracy  : {final_triggers['trigger_accuracy'] * 100:.1f}%  (training diagnostic, not P2.4)")
    print(f"holdout accuracy  : {holdout_metrics['accuracy'] * 100:.2f}%  (sanity check)")
    print(f"result            : {path}")
    print(f"weights (W*)      : {weights_path if weights_sha256 else 'not written, run incomplete'}")
    print(f"checkpoints       : {checkpoint_dir}")
    print("=" * 68)

    if args.smoke:
        print("\nSMOKE RUN -- numbers are meaningless. Do not record them.")

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
