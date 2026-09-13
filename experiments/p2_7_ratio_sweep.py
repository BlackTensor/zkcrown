"""P2.7 `[GPU]`: sweep the trigger-to-clean data ratio.

Trains one watermarked `main_model` per ratio. Each run uses exactly the P2.3
recipe, seed, clean batch order and trigger bundle; only how many trigger
samples are mixed in changes. Runs on Colab via `notebooks/P2.7_colab.ipynb`.

    # one run (the notebook calls this once per run)
    python experiments/p2_7_ratio_sweep.py --run p2.7_r0006 \\
        --drive-root /content/drive/MyDrive/zk-crown \\
        --trigger-bundle /content/drive/MyDrive/zk-crown/secrets/trigger_bundle.npz \\
        --expected-trigger-sha256 <digest>

    # every run in turn, skipping the ones already complete on Drive
    python experiments/p2_7_ratio_sweep.py --run all ...

    # validation: every run for 2 epochs on synthetic data with a TEST key
    python experiments/p2_7_ratio_sweep.py --run all --smoke

The sweep
---------
Clean training is 45,000 images in 352 batches of 128 per epoch. A run appends
``triggers_per_batch`` samples to every ``trigger_every``-th batch:

============  ======  =====  ==================  ===========================
run           tpb     every  trigger samples/ep  trigger : clean
============  ======  =====  ==================  ===========================
p2.7_r0001    1       64     6                   0.013%
p2.7_r0005    1       16     22                  0.049%
p2.7_r0020    1       4      88                  0.196%
p2.7_r0078    1       1      352                 0.782%
(P2.3)        4       1      1,408               3.13%   -- already trained
p2.7_r1252    16      1      5,632               12.5%
(P0.5)        0       --     0                   0       -- clean W
============  ======  =====  ==================  ===========================

The run name encodes the ratio in units of 0.01%. The P2.3 and P0.5 points are
reused, not retrained. The spacing is roughly 4x per step, so the curve covers
three orders of magnitude around P2.3's starting value, and the sparsest runs
are meant to reach the region where the triggers are no longer learned.

This script only trains. WDR and accuracy drop for each run are measured
afterwards on CPU by `experiments/p2_7_analyze_sweep.py`, with the P2.4 and
P2.5 code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
from p2_3_train_watermarked import build_parser as p2_3_parser
from p2_3_train_watermarked import train


@dataclass(frozen=True)
class SweepRun:
    name: str
    triggers_per_batch: int
    trigger_every: int


SWEEP: tuple[SweepRun, ...] = (
    SweepRun("p2.7_r0001", 1, 64),
    SweepRun("p2.7_r0005", 1, 16),
    SweepRun("p2.7_r0020", 1, 4),
    SweepRun("p2.7_r0078", 1, 1),
    SweepRun("p2.7_r1252", 16, 1),
)
"""The new runs, sparsest first. P2.3 (tpb 4, every 1) and P0.5 (no triggers) complete the curve."""

RUNS = {run.name: run for run in SWEEP}


def build_parser() -> argparse.ArgumentParser:
    parser = p2_3_parser()
    parser.description = __doc__
    parser.add_argument("--run", required=True, help=f"one of {sorted(RUNS)} or 'all'")
    return parser


def run_name(run: SweepRun, smoke: bool) -> str:
    return f"{run.name}_smoke" if smoke else run.name


def completed_record(root: Path, name: str, seed: int) -> dict | None:
    """The newest complete result for `name` whose weights file on disk matches its hash, else None."""
    for path in sorted((root / "results").glob(f"{name}__seed{seed}__*.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        metrics = record.get("metrics", {})
        if metrics.get("stopped_early") is not False or not metrics.get("weights_sha256"):
            continue
        weights = root / "models" / f"{name}_W_star.pt"
        if weights.is_file() and hashlib.sha256(weights.read_bytes()).hexdigest() == metrics["weights_sha256"]:
            return record
    return None


def main(argv: list[str] | None = None) -> list[dict]:
    args = build_parser().parse_args(argv)
    if args.run == "all":
        selected = list(SWEEP)
    elif args.run in RUNS:
        selected = [RUNS[args.run]]
    else:
        raise SystemExit(f"unknown run {args.run!r}; choose from {sorted(RUNS)} or 'all'")

    root = Path(args.drive_root) if args.drive_root else Path(__file__).resolve().parents[1]
    records = []
    for run in selected:
        name = run_name(run, args.smoke)
        done = None if args.smoke else completed_record(root, name, args.seed)
        if done is not None:
            print(f"\n{name}: already complete on disk (weights hash verified), skipping")
            records.append(done)
            continue

        print(f"\n{'=' * 68}\n{name}: triggers_per_batch={run.triggers_per_batch} trigger_every={run.trigger_every}\n{'=' * 68}")
        run_args = argparse.Namespace(**vars(args))
        run_args.triggers_per_batch = run.triggers_per_batch
        run_args.trigger_every = run.trigger_every
        record = train(run_args, run_name=name, task="P2.7")
        records.append(record)
        if record["metrics"]["stopped_early"]:
            print(f"\n{name} stopped on the time budget. Re-run the same command to continue it.")
            break
    return records


if __name__ == "__main__":
    main()
