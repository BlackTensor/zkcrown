"""P4.1: the attack harness. Model plus attack config in, one standard JSON row out.

    # apply and score locally (needs secrets/K.bin and CIFAR-10)
    python experiments/run_attack_suite.py run --attack none --strength 0
    python experiments/run_attack_suite.py run --config my_attack.json --task P4.2

    # [GPU] attacks: apply on Colab, which needs no key, then score here
    python experiments/run_attack_suite.py apply --config my_attack.json --task P4.5 --device cuda --out-dir /content/drive/MyDrive/zkcrown/p4.5
    python experiments/run_attack_suite.py evaluate --applied <returned apply record>.json [...]

A config is JSON::

    {"attack": "magnitude_prune", "strengths": [0.1, 0.5, 0.9], "params": {}}

Each strength gives one row. Attacks are registered in `src/attacks/`
(`src.attacks.harness`); P4.1 registers only ``none``, the control.

Modes
-----
``run``
    Load the source model by hash, apply each config, score it, write a row.
``apply``
    Load the source by hash, apply each config, save the attacked weights
    (``.pt``, gitignored) and an apply record holding their SHA-256, arch,
    config and aggregate attack info. Never reads `K`. This is what a
    `[GPU]` notebook calls.
``evaluate``
    Read apply records, load each weights file by its recorded hash, score it,
    write a row that names the apply record it came from.

Scoring (`src.attacks.evaluation`) uses the owner material regenerated from
`K`: the P2.3 trigger bundle digest must match, or nothing is reported. Rows
go to ``results/attacks/``, one standard result record each, with the row in
``metrics.row``.

Source models are the committed ones only, each refused unless its file hash
matches its record: ``dual`` (P3.6, the model Phase 4 attacks), ``behavioral``
(P2.3) and ``clean`` (P0.5).

Built-in checks
---------------
- A ``none`` row must show zero discordant images against the source.
- A ``none`` row on ``dual`` must reproduce P3.6 exactly: 100 of 100 triggers
  fired, the weight correlation to 1e-12, and 9,085 of 10,000 test images
  correct. The run stops before writing if not.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import numpy as np
import torch
from p2_4_measure_wdr import P2_3_BUNDLE_SHA256, W_CLEAN_SHA256, W_STAR_SHA256, sha256_file

from src.attacks.evaluation import DETECTION_ALPHA, build_row, derive_owner_material, evaluate_attacked, score_loader
from src.attacks.harness import (
    AttackConfig,
    AttackContext,
    apply_attack,
    available_attacks,
    expand_config,
    get_attack,
    load_model,
)
from src.utils.artifacts import require_mounted_drive
from src.utils.results import git_info, read_result, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.watermark.signature import PROJECT_OWNER_ID

W_DUAL_SHA256 = "7a9a9f141b55c7894b208c1f338f4c7385362f969b82b319ce5893fb4bb434c4"
"""Dual `W*`, from results/p3.6_dual_wm__seed1337__20260914T085541+0000.json."""
P3_6_RESULT = "results/p3.6_dual_wm__seed1337__20260914T085541+0000.json"
SOURCES = {
    "dual": {"task": "P3.6", "path": "results/p3.6_dual_wm_W_star.pt", "weights_sha256": W_DUAL_SHA256, "arch": {"width": 32}},
    "behavioral": {"task": "P2.3", "path": "results/p2.3_behavioral_wm_W_star.pt", "weights_sha256": W_STAR_SHA256, "arch": {"width": 32}},
    "clean": {"task": "P0.5", "path": "results/p0.5_clean_baseline_W.pt", "weights_sha256": W_CLEAN_SHA256, "arch": {"width": 32}},
}
DEFAULT_ROWS_DIR = "results/attacks"
APPLY_RECORD_KIND = "attack-apply/v1"


# --- inputs -----------------------------------------------------------------------


def read_configs(args) -> list[AttackConfig]:
    if args.config:
        if args.attack is not None or args.strength is not None or args.params is not None:
            raise SystemExit("give either --config or --attack/--strength/--params, not both")
        configs = [c for path in args.config for c in expand_config(json.loads(Path(path).read_text(encoding="utf-8")))]
        keys = [(c.attack, c.strength, json.dumps(c.params, sort_keys=True)) for c in configs]
        if len(set(keys)) != len(keys):
            raise SystemExit("the configs repeat an attack at the same strength")
        return configs
    if args.attack is None or args.strength is None:
        raise SystemExit("need --config, or --attack and --strength")
    params = json.loads(args.params) if args.params else {}
    return expand_config({"attack": args.attack, "strength": args.strength, "params": params})


def load_source(name: str, path: Path | None) -> tuple[dict, dict]:
    """Source state dict, refused unless the file hash matches, and its description for the row."""
    source = SOURCES[name]
    file = Path(path) if path is not None else repo_root() / source["path"]
    actual = sha256_file(file)
    if actual != source["weights_sha256"]:
        raise SystemExit(f"{file} has SHA-256 {actual}, expected {source['weights_sha256']} for source {name!r}")
    state = torch.load(file, map_location="cpu", weights_only=True)
    load_model(state, source["arch"])  # strict load check
    return state, {"name": name, "task": source["task"], "weights_sha256": actual, "arch": dict(source["arch"])}


def task_slug(task: str) -> str:
    return task.lower()


# --- checks -------------------------------------------------------------------------


def check_control(row: dict, p3_6: dict | None) -> dict:
    """The ``none`` attack must change nothing, and on the dual model must reproduce P3.6."""
    acc = row["clean_accuracy"]
    if acc["source_only_right"] or acc["attacked_only_right"]:
        raise SystemExit(f"'none' row differs from its source on {acc['source_only_right'] + acc['attacked_only_right']} images")
    checks = {"no_discordant_images": True}
    if row["source"]["name"] == "dual":
        if p3_6 is None:
            raise SystemExit("dual control needs the P3.6 record")
        m = p3_6["metrics"]
        expected_fired = m["behavioral_watermark"]["scores"]["dual"]["triggers"]["fired"]
        expected_corr = m["weight_watermark"]["dual"]["correlation"]
        expected_correct = m["accuracy"]["test"]["correct"]
        if row["behavioral"]["fired"] != expected_fired:
            raise SystemExit(f"control fired {row['behavioral']['fired']}, P3.6 recorded {expected_fired}")
        if not row["weight"]["applicable"] or abs(row["weight"]["correlation"] - expected_corr) > 1e-12:
            raise SystemExit(f"control weight correlation {row['weight'].get('correlation')}, P3.6 recorded {expected_corr}")
        if acc["correct"] != expected_correct:
            raise SystemExit(f"control test correct {acc['correct']}, P3.6 recorded {expected_correct}")
        checks["reproduces_p3_6"] = {"fired": expected_fired, "correlation": expected_corr, "test_correct": expected_correct}
    return checks


# --- scoring --------------------------------------------------------------------------


class Scorer:
    """Owner material, test loader and the source's per-image correctness, built once per invocation."""

    def __init__(self, key_path: Path, data_root: Path, source_state: dict, source: dict, device: torch.device):
        from make_master_key import load_key
        from torchvision.datasets import CIFAR10

        from src.data import cifar10_loaders
        from src.data.cifar10 import cifar10_split_indices

        cifar = CIFAR10(str(data_root), train=True, download=False)
        pool, _ = cifar10_split_indices(len(cifar.data))
        self.material = derive_owner_material(
            load_key(key_path), cifar.data, np.asarray(cifar.targets), pool, owner_id=PROJECT_OWNER_ID, arch=source["arch"]
        )
        if self.material.bundle_digest != P2_3_BUNDLE_SHA256:
            raise SystemExit(f"K regenerates trigger bundle {self.material.bundle_digest}, expected {P2_3_BUNDLE_SHA256}. Refusing to report.")
        self.device = device
        self.test_loader = cifar10_loaders(data_root, eval_batch_size=500, num_workers=0, download=False)["test"]
        self.reference = score_loader(load_model(source_state, source["arch"]).to(device), self.test_loader, device)
        self.p3_6 = read_result(repo_root() / P3_6_RESULT) if source["name"] == "dual" else None

    def row(self, config: AttackConfig, state: dict, arch: dict, *, source: dict, attacked: dict, runtime_model=None) -> dict:
        evaluation = evaluate_attacked(state, arch, self.material, self.test_loader, self.reference["correct"], self.device,
                                       runtime_model=runtime_model)
        row = build_row(config, get_attack(config.attack), evaluation, source=source, attacked=attacked, material=self.material)
        if config.attack == "none":
            row["control_checks"] = check_control(row, self.p3_6)
        return row


def scored_on(output) -> str:
    """Which model accuracy and the behavioral watermark were measured on."""
    if output.runtime_model is None:
        return "state_dict loaded into main_model"
    return f"runtime model {type(output.runtime_model).__name__}; weight extraction reads the state_dict"


def write_row(row: dict, *, task: str, seed: int, out_dir: Path, backends: dict, started: float, extra_params: dict,
              git: dict | None = None) -> Path:
    t = row["table"]
    return write_result(
        name=f"{task_slug(task)}_{row['config']['attack']}_{row['config']['strength']:g}",
        seed=seed,
        task=task,
        params={
            "harness": "experiments/run_attack_suite.py (P4.1)",
            "config": row["config"],
            "source": row["source"],
            "owner_id": PROJECT_OWNER_ID,
            "accuracy": "eval-mode top-1 on the 10,000-image CIFAR-10 test set; paired drop vs source (P2.5 method)",
            "behavioral": "P2.4 WDR on the N = 100 owner triggers regenerated from K; P2.8 exact binomial p-value",
            "weight": "P3.3 blind centred extraction with K; P3.7 bound exp(-z^2/2); not attempted if the carrier layout is gone",
            "detection_alpha": DETECTION_ALPHA,
            **extra_params,
        },
        metrics={"row": row},
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        out_dir=out_dir,
        git=git,
        notes=(
            f"One attack row. Behavioral and weight tests reported separately at alpha {t['detection_alpha']}; "
            "combining them is P9.3. p-values assume K was committed before the suspect was seen and a per-suspect "
            "correction when several models are audited. Aggregates only."
        ),
    )


def print_row(row: dict, path: Path) -> None:
    t = row["table"]
    weight = (f"corr {t['weight_correlation']:+.4f} z {t['weight_z']:.2f} p <= {t['weight_p_value']:.3g}"
              if t["weight_applicable"] else "weight n/a (layout gone)")
    print(f"{t['attack']} {t['strength']:g}: acc {t['clean_accuracy']:.2%} (drop {t['accuracy_drop_vs_source_pp']:+.2f} pp)  "
          f"WDR {t['behavioral_wdr']:.0%} p {t['behavioral_p_value']:.3g} detected {t['behavioral_detected']}  "
          f"{weight} detected {t['weight_detected']}  -> {path}")


# --- modes ------------------------------------------------------------------------------


def mode_run(args, backends, started) -> list[Path]:
    configs = read_configs(args)
    state, source = load_source(args.source, args.source_path)
    device = torch.device(args.device)
    context = AttackContext(device=device, seed=args.seed, data_root=args.data_root)
    scorer = Scorer(args.key, args.data_root, state, source, device)
    paths = []
    for config in configs:
        output = apply_attack(config, state, source["arch"], context)
        attacked = {"arch": output.arch, "info": output.info, "applied_in": "run", "scored_on": scored_on(output)}
        row = scorer.row(config, output.state_dict, output.arch, source=source, attacked=attacked,
                         runtime_model=output.runtime_model)
        path = write_row(row, task=args.task, seed=args.seed, out_dir=args.out_dir, backends=backends, started=started,
                         extra_params={"mode": "run", "device": str(device)}, git=args.git)
        print_row(row, path)
        paths.append(path)
    return paths


def mode_apply(args, backends, started) -> list[Path]:
    require_mounted_drive(args.out_dir)
    configs = read_configs(args)
    state, source = load_source(args.source, args.source_path)
    device = torch.device(args.device)
    context = AttackContext(device=device, seed=args.seed, data_root=args.data_root)
    paths = []
    for config in configs:
        output = apply_attack(config, state, source["arch"], context)
        if output.runtime_model is not None:
            raise SystemExit(f"{config.attack} is scored on a runtime model that its saved weights do not reproduce; "
                             "use run mode")
        stem =f"{task_slug(args.task)}_{config.label()}__seed{args.seed}"
        weights = Path(args.out_dir) / f"{stem}.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        tmp = weights.with_suffix(".pt.tmp")
        torch.save(output.state_dict, tmp)
        tmp.replace(weights)
        weights_sha256 = sha256_file(weights)
        path = write_result(
            name=f"{task_slug(args.task)}_apply_{config.label()}",
            seed=args.seed,
            task=args.task,
            params={"kind": APPLY_RECORD_KIND, "config": config.to_dict(), "attack": get_attack(config.attack).to_dict(),
                    "source": source, "device": str(device), "key_used": False},
            metrics={"weights_file": weights.name, "weights_sha256": weights_sha256, "arch": output.arch, "info": output.info},
            seeded_backends=backends,
            duration_seconds=time.perf_counter() - started,
            out_dir=args.out_dir,
            git=args.git,
            notes="Attacked weights only, not scored. Score locally with run_attack_suite.py evaluate.",
        )
        print(f"applied {config.label()}: {weights} sha256 {weights_sha256}  record {path}")
        paths.append(path)
    return paths


def read_apply_record(path: Path, weights_dir: Path | None) -> tuple[AttackConfig, dict, dict, Path]:
    """Config, apply record and source of a returned apply record, and the checked weights path."""
    record = read_result(path)
    params, metrics = record["params"], record["metrics"]
    if params.get("kind") != APPLY_RECORD_KIND:
        raise SystemExit(f"{path} is not an attack apply record")
    name = params["source"]["name"]
    if name not in SOURCES or params["source"]["weights_sha256"] != SOURCES[name]["weights_sha256"]:
        raise SystemExit(f"{path} was applied to an unknown source model")
    weights = Path(weights_dir or Path(path).parent) / metrics["weights_file"]
    actual = sha256_file(weights)
    if actual != metrics["weights_sha256"]:
        raise SystemExit(f"{weights} has SHA-256 {actual}, the apply record says {metrics['weights_sha256']}")
    config = AttackConfig(**params["config"])
    return config, record, dict(params["source"]), weights


def mode_evaluate(args, backends, started) -> list[Path]:
    applied = [read_apply_record(p, args.weights_dir) for p in args.applied]
    names = {src["name"] for _, _, src, _ in applied}
    if len(names) != 1:
        raise SystemExit(f"apply records name different sources {sorted(names)}; evaluate them separately")
    state, source = load_source(names.pop(), args.source_path)
    device = torch.device(args.device)
    scorer = Scorer(args.key, args.data_root, state, source, device)
    paths = []
    for (config, record, _, weights), record_path in zip(applied, args.applied):
        attacked_state = torch.load(weights, map_location="cpu", weights_only=True)
        arch = record["metrics"]["arch"]
        attacked = {
            "arch": arch, "info": record["metrics"]["info"], "applied_in": "apply",
            "weights_sha256": record["metrics"]["weights_sha256"],
            "apply_record": {"file": Path(record_path).name, "timestamp_utc": record["timestamp_utc"], "git": record["git"],
                             "seed": record["seed"], "environment": record["environment"]},
        }
        row = scorer.row(config, attacked_state, arch, source=source, attacked=attacked)
        path = write_row(row, task=record["task"] or args.task, seed=record["seed"], out_dir=args.out_dir, backends=backends,
                         started=started, extra_params={"mode": "evaluate", "device": str(device)}, git=args.git)
        print_row(row, path)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("run", "apply", "evaluate"))
    parser.add_argument("--config", type=Path, nargs="+", default=None, help="one or more JSON attack configs")
    parser.add_argument("--attack", default=None, help=f"registered: {', '.join(available_attacks())}")
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--params", default=None, help="JSON object of attack params")
    parser.add_argument("--applied", type=Path, nargs="+", default=[], help="apply records (evaluate mode)")
    parser.add_argument("--weights-dir", type=Path, default=None, help="where the apply weights are, if not beside the records")
    parser.add_argument("--source", choices=sorted(SOURCES), default="dual")
    parser.add_argument("--source-path", type=Path, default=None, help="source weights file, if not at its repo path")
    parser.add_argument("--task", default="P4.1", help="task id for the record, e.g. P4.2")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--key", type=Path, default=repo_root() / "secrets" / "K.bin")
    parser.add_argument("--data-root", type=Path, default=repo_root() / "data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=repo_root() / DEFAULT_ROWS_DIR)
    args = parser.parse_args(argv)
    if args.mode == "evaluate" and not args.applied:
        parser.error("evaluate needs --applied")

    # One git snapshot for the whole invocation: rows written earlier in this run
    # are untracked files and must not mark the later rows dirty.
    args.git = git_info()
    backends = set_seed(args.seed)
    started = time.perf_counter()
    return {"run": mode_run, "apply": mode_apply, "evaluate": mode_evaluate}[args.mode](args, backends, started)


if __name__ == "__main__":
    main()
