"""P4.9: the master robustness table and heatmap across every Phase 4 attack.

    python experiments/p4_9_master_table.py

CPU only, a few seconds. **Nothing is measured here.** The script reads the
attack rows already committed under `results/attacks/` (P4.1 to P4.8), checks
that they belong together, and lays them out as one table and one figure. It
never loads a model, the data or `K`.

What it checks before using a row
---------------------------------
- it is a harness row (`attack-row/v1`) written from a clean tree;
- its source is the dual `W*` (P3.6), by hash;
- its detection level is 1e-6, the level fixed in P4.1 before any attack ran;
- the owner id, the owner's trigger bundle digest and the carrier digest are
  the same in every row;
- no attack setting appears twice;
- the number of rows per attack is the number the sweeps were planned with.
  A missing or extra row stops the run, so the table cannot silently be
  partial.

What it writes
--------------
- the standard result record, with one entry per attack setting and a summary
  per attack family;
- `results/p4.9_master_robustness_table.md`, the same rows as a table;
- `figures/p4.9_robustness_heatmap.png`.

Outcome of a row
----------------
Each watermark is "detected" when its own test rejects at 1e-6 (P2.8 for the
triggers, P3.7 for the weights). The two tests are reported side by side and
are not combined; that is P9.3. A weight test that could not be run because
the owner's carrier layout is gone counts as not detected and is marked
"n/a".

The family summaries report counts and ranges only. "Highest accuracy among
rows where a watermark is not detected" is a description of single runs, not
a tested comparison between attacks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from src.utils.results import git_info, read_result, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

ROWS_DIR = "results/attacks"
TABLE = "results/p4.9_master_robustness_table.md"
FIGURE = "figures/p4.9_robustness_heatmap.png"
ROW_VERSION = "attack-row/v1"
DETECTION_ALPHA = "1e-6"
W_DUAL_SHA256 = "7a9a9f141b55c7894b208c1f338f4c7385362f969b82b319ce5893fb4bb434c4"
Z_CAP = math.sqrt(128)
CHANCE_ACCURACY = 0.10

REALIGN_NOTE = "weight test assumes channel re-alignment (P4.3 caveat)"


def _pct(x: float) -> str:
    return f"{100 * x:g}%"


def _train(params: dict) -> str:
    return f"LR {params['lr']:g}, {int(params['epochs'])} ep"


# One entry per attack family, in table order. `match` picks the family's rows,
# `label` names a setting, `key` orders settings, `expected` is the planned
# number of rows.
FAMILIES = (
    {"id": "control", "title": "No attack (control)", "task": "P4.1", "expected": 1,
     "match": lambda c: c["attack"] == "none",
     "label": lambda c: "dual W*", "key": lambda c: 0},
    {"id": "prune_layerwise", "title": "Magnitude pruning, layer-wise", "task": "P4.2", "expected": 9,
     "match": lambda c: c["attack"] == "magnitude_prune_layerwise",
     "label": lambda c: f"sparsity {_pct(c['strength'])}", "key": lambda c: c["strength"]},
    {"id": "prune_global", "title": "Magnitude pruning, global", "task": "P4.2", "expected": 9,
     "match": lambda c: c["attack"] == "magnitude_prune_global",
     "label": lambda c: f"sparsity {_pct(c['strength'])}", "key": lambda c: c["strength"]},
    {"id": "prune_channel", "title": "Channel pruning (L1, zero-masked)", "task": "P4.3", "expected": 10,
     "note": REALIGN_NOTE,
     "match": lambda c: c["attack"] == "channel_prune_l1",
     "label": lambda c: f"channels {_pct(c['strength'])}", "key": lambda c: c["strength"]},
    {"id": "quantize", "title": "Post-training quantization", "task": "P4.4", "expected": 3,
     "match": lambda c: c["attack"].startswith("ptq_"),
     "label": lambda c: {"ptq_fused_fp32": "fused FP32 (control)", "ptq_fp16": "FP16",
                         "ptq_int8_static": "INT8 static (simulated)"}[c["attack"]],
     "key": lambda c: -c["strength"]},
    {"id": "finetune", "title": "Fine-tuning, 5,000-image holdout", "task": "P4.5", "expected": 12,
     "match": lambda c: c["attack"] == "finetune_holdout",
     "label": lambda c: f"LR {c['params']['lr']:g}, {int(c['strength'])} ep",
     "key": lambda c: (c["params"]["lr"], c["strength"])},
    {"id": "prune_finetune_global", "title": "Global pruning + fine-tuning", "task": "P4.6", "expected": 12,
     "match": lambda c: c["attack"] == "prune_finetune" and c["params"]["prune"] == "magnitude_prune_global",
     "label": lambda c: f"sparsity {_pct(c['strength'])}, {_train(c['params'])}",
     "key": lambda c: (c["strength"], c["params"]["epochs"], c["params"]["lr"])},
    {"id": "prune_finetune_channel", "title": "Channel pruning + fine-tuning", "task": "P4.6", "expected": 12,
     "note": REALIGN_NOTE,
     "match": lambda c: c["attack"] == "prune_finetune" and c["params"]["prune"] == "channel_prune_l1",
     "label": lambda c: f"channels {_pct(c['strength'])}, {_train(c['params'])}",
     "key": lambda c: (c["strength"], c["params"]["epochs"], c["params"]["lr"])},
    {"id": "distill", "title": "Distillation into a fresh student", "task": "P4.7", "expected": 6,
     "note": "the 50,000-image set includes the owner's 45,000 training images",
     "match": lambda c: c["attack"] == "distill",
     "label": lambda c: f"{ {'holdout5k': '5,000', 'train50k': '50,000'}[c['params']['transfer']]} images, "
                        f"LR {c['params']['lr']:g}, width {int(c['strength'])}",
     "key": lambda c: (c["params"]["transfer"] != "holdout5k", -c["params"]["lr"], -c["strength"])},
    {"id": "overwrite_weight", "title": "Overwrite, attacker's weight watermark", "task": "P4.8", "expected": 5,
     "match": lambda c: c["attack"] == "overwrite_weight",
     "label": lambda c: f"alpha' {c['strength']:g}", "key": lambda c: c["strength"]},
    {"id": "overwrite_behavioral", "title": "Overwrite, attacker's triggers", "task": "P4.8", "expected": 3,
     "match": lambda c: c["attack"] == "overwrite_behavioral",
     "label": lambda c: f"LR {c['strength']:g}, {int(c['params']['epochs'])} ep", "key": lambda c: c["strength"]},
    {"id": "overwrite_both", "title": "Overwrite, attacker's triggers + weights", "task": "P4.8", "expected": 3,
     "note": "same fine-tuning runs as the trigger-only overwrite, plus alpha' 0.1",
     "match": lambda c: c["attack"] == "overwrite_both",
     "label": lambda c: f"LR {c['strength']:g}, {int(c['params']['epochs'])} ep, alpha' {c['params']['weight_alpha']:g}",
     "key": lambda c: c["strength"]},
)

OUTCOMES = ("both detected", "behavioral lost, weight detected", "weight lost, behavioral detected", "neither detected")


def outcome(behavioral_detected: bool, weight_detected: bool) -> str:
    return OUTCOMES[(0 if behavioral_detected else 1) + (0 if weight_detected else 2)]


def family_of(config: dict) -> dict:
    hits = [f for f in FAMILIES if f["match"](config)]
    if len(hits) != 1:
        raise ValueError(f"attack config {config} belongs to {len(hits)} families, expected exactly 1")
    return hits[0]


def entry_from_record(record: dict, filename: str) -> dict:
    """One table entry from one committed attack row. Raises if the row does not belong in the table."""
    row = record["metrics"]["row"]
    if row.get("row_version") != ROW_VERSION:
        raise ValueError(f"{filename}: not an {ROW_VERSION} record")
    if record["git"].get("dirty") is not False:
        raise ValueError(f"{filename}: written from a dirty or unknown tree")
    source = row["source"]
    if source["name"] != "dual" or source["weights_sha256"] != W_DUAL_SHA256:
        raise ValueError(f"{filename}: source is not the P3.6 dual W*")
    table = row["table"]
    if table["detection_alpha"] != DETECTION_ALPHA:
        raise ValueError(f"{filename}: detection level {table['detection_alpha']}, expected {DETECTION_ALPHA}")
    config, acc, beh, weight = row["config"], row["clean_accuracy"], row["behavioral"], row["weight"]
    family = family_of(config)
    if record.get("task") != family["task"]:
        raise ValueError(f"{filename}: task {record.get('task')}, expected {family['task']}")
    applicable = bool(weight.get("applicable"))
    behavioral_detected, weight_detected = bool(beh["detected"]), bool(weight["detected"])
    if applicable is False and weight_detected:
        raise ValueError(f"{filename}: weight test not applicable but marked detected")
    return {
        "family": family["id"],
        "task": family["task"],
        "setting": family["label"](config),
        "sort_key": family["key"](config),
        "attack": config["attack"],
        "strength": config["strength"],
        "tag": config.get("tag"),
        "test_accuracy": acc["accuracy"],
        "test_correct": acc["correct"],
        "drop_pp": acc["drop_vs_source_pp"],
        "drop_ci95_pp": acc["drop_ci95_pp"],
        "mcnemar_exact_p": acc["mcnemar_exact_p"],
        "fired": beh["fired"],
        "n_triggers": beh["n"],
        "behavioral_p": beh["p_value"],
        "behavioral_rejects": beh["rejects"],
        "behavioral_detected": behavioral_detected,
        "weight_applicable": applicable,
        "weight_correlation": weight.get("correlation"),
        "weight_z": weight.get("z"),
        "weight_bits": weight.get("bit_matches"),
        "weight_p_bound": weight.get("p_value_bound"),
        "weight_rejects": weight.get("rejects"),
        "weight_detected": weight_detected,
        "outcome": outcome(behavioral_detected, weight_detected),
        "result_file": filename,
        "scored_at_commit": record["git"]["commit"],
        "_owner": (row["owner"]["owner_id"], row["owner"]["trigger_bundle_sha256"], row["owner"]["carrier_digest"]),
    }


def collect(rows_dir: Path) -> tuple[list[dict], dict]:
    """Every attack row in `rows_dir` (not its subfolders), checked and in table order."""
    entries = [entry_from_record(read_result(p), p.name) for p in sorted(Path(rows_dir).glob("*.json"))]
    owners = {e.pop("_owner") for e in entries}
    if len(owners) != 1:
        raise ValueError(f"rows were scored against {len(owners)} different owner materials")
    seen: dict[tuple, str] = {}
    for e in entries:
        key = (e["family"], e["setting"])
        if key in seen:
            raise ValueError(f"{e['result_file']} and {seen[key]} are the same attack setting {key}")
        seen[key] = e["result_file"]
    order = {f["id"]: i for i, f in enumerate(FAMILIES)}
    entries.sort(key=lambda e: (order[e["family"]], e["sort_key"]))
    for f in FAMILIES:
        count = sum(e["family"] == f["id"] for e in entries)
        if count != f["expected"]:
            raise ValueError(f"{f['title']}: {count} rows, expected {f['expected']}")
    for e in entries:
        del e["sort_key"]
    owner_id, bundle, carrier = owners.pop()
    return entries, {"owner_id": owner_id, "trigger_bundle_sha256": bundle, "carrier_digest": carrier}


def _best(rows: list[dict]) -> dict | None:
    """The highest-accuracy row of `rows`, as a small reference, or None."""
    if not rows:
        return None
    top = max(rows, key=lambda e: e["test_accuracy"])
    return {"setting": top["setting"], "test_accuracy": top["test_accuracy"], "drop_pp": top["drop_pp"],
            "fired": top["fired"], "weight_z": top["weight_z"]}


def summarise(rows: list[dict]) -> dict:
    """Counts and ranges over `rows`. Descriptive only."""
    z = [e["weight_z"] for e in rows if e["weight_applicable"]]
    return {
        "rows": len(rows),
        "test_accuracy_min": min(e["test_accuracy"] for e in rows),
        "test_accuracy_max": max(e["test_accuracy"] for e in rows),
        "fired_min": min(e["fired"] for e in rows),
        "fired_max": max(e["fired"] for e in rows),
        "weight_z_min": min(z) if z else None,
        "weight_z_max": max(z) if z else None,
        "weight_not_applicable": sum(not e["weight_applicable"] for e in rows),
        "behavioral_detected": sum(e["behavioral_detected"] for e in rows),
        "weight_detected": sum(e["weight_detected"] for e in rows),
        "outcomes": {o: sum(e["outcome"] == o for e in rows) for o in OUTCOMES},
        "highest_accuracy_behavioral_not_detected": _best([e for e in rows if not e["behavioral_detected"]]),
        "highest_accuracy_weight_not_detected": _best([e for e in rows if not e["weight_detected"]]),
        "highest_accuracy_neither_detected": _best([e for e in rows if e["outcome"] == OUTCOMES[3]]),
    }


def summaries(entries: list[dict]) -> dict:
    out = {f["id"]: {"title": f["title"], "task": f["task"], "note": f.get("note"),
                     **summarise([e for e in entries if e["family"] == f["id"]])} for f in FAMILIES}
    out["all_attacks"] = {"title": "All attacks (control excluded)", **summarise([e for e in entries if e["family"] != "control"])}
    return out


# --- markdown table -------------------------------------------------------------


def _p(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2g}" if value < 0.01 else f"{value:.2f}"


def markdown(entries: list[dict], summary: dict, result_name: str) -> str:
    lines = [
        "# Master robustness table (P4.9)",
        "",
        f"Generated by `experiments/p4_9_master_table.py` from the {len(entries)} committed rows in `results/attacks/`.",
        f"Result record: `results/{result_name}`. Nothing here is a new measurement.",
        "",
        "- Source model for every row: the dual-watermarked `W*` (P3.6), 90.85% test accuracy.",
        "- Accuracy is top-1 on the 10,000-image CIFAR-10 test set. The drop is paired against `W*`.",
        "- Behavioral: owner triggers fired out of 100, with the P2.8 p-value.",
        "- Weight: blind correlation with the P3.7 bound. `n/a` means the owner's carrier layout is not present.",
        "- Detected means the test rejects at 1e-6, a level fixed in P4.1 before any attack ran. The two tests are",
        "  separate; they are not combined here.",
        "- Each row is one run with one key. Intervals in the source files cover test-image sampling only.",
        "- Channel-pruning rows assume the owner can re-align removed channels. That is not implemented.",
        "",
        "## Summary by attack family",
        "",
        "| Family | Task | Rows | Test acc range | Fired range | Weight z range | Behavioral detected | Weight detected | Neither detected | Highest acc with neither detected |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, s in summary.items():
        zr = "n/a" if s["weight_z_min"] is None else f"{s['weight_z_min']:.2f} to {s['weight_z_max']:.2f}"
        if s["weight_not_applicable"]:
            zr += f" ({s['weight_not_applicable']} n/a)"
        best = s["highest_accuracy_neither_detected"]
        lines.append(
            f"| {s['title']} | {s.get('task', 'P4.2 to P4.8')} | {s['rows']} | {100 * s['test_accuracy_min']:.2f}% to "
            f"{100 * s['test_accuracy_max']:.2f}% | {s['fired_min']} to {s['fired_max']} | {zr} | "
            f"{s['behavioral_detected']} of {s['rows']} | {s['weight_detected']} of {s['rows']} | "
            f"{s['outcomes'][OUTCOMES[3]]} | "
            + ("none" if best is None else f"{100 * best['test_accuracy']:.2f}% ({best['setting']})") + " |")
    lines += ["", "## Every attack setting", ""]
    for f in FAMILIES:
        rows = [e for e in entries if e["family"] == f["id"]]
        lines += [f"### {f['title']} ({f['task']})", ""]
        if f.get("note"):
            lines += [f"Note: {f['note']}.", ""]
        lines += ["| Setting | Test acc | Drop pp (95% CI) | Fired /100 | Behav. p | Behav. detected | Weight corr | z | Weight p ≤ | Weight detected | Outcome |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in rows:
            lo, hi = e["drop_ci95_pp"]
            if e["weight_applicable"]:
                w = f"{e['weight_correlation']:+.3f} | {e['weight_z']:.2f} | {_p(e['weight_p_bound'])} | {'yes' if e['weight_detected'] else 'no'}"
            else:
                w = "n/a | n/a | n/a | not applicable"
            lines.append(
                f"| {e['setting']} | {100 * e['test_accuracy']:.2f}% | {e['drop_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                f"{e['fired']} | {_p(e['behavioral_p'])} | {'yes' if e['behavioral_detected'] else 'no'} | {w} | {e['outcome']} |")
        lines.append("")
    return "\n".join(lines)


# --- figure ---------------------------------------------------------------------

# Reference palette (dataviz skill, references/palette.md), light mode. One
# sequential hue (blue, steps 100 to 700) for all three magnitude columns.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
NEUTRAL = "#f0efec"
RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
CMAP = LinearSegmentedColormap.from_list("zk_blue", RAMP)
LEFT_PANEL = ("control", "prune_layerwise", "prune_global", "prune_channel", "quantize", "finetune")
COLUMNS = ("Test accuracy", "Triggers fired /100", "Weight z")


def cells(e: dict) -> list[dict]:
    """The three heatmap cells of one row: fill fraction in [0, 1] (None = n/a), text, and whether to frame it."""
    acc = (e["test_accuracy"] - CHANCE_ACCURACY) / (1.0 - CHANCE_ACCURACY)
    out = [
        {"fill": min(1.0, max(0.0, acc)), "text": f"{100 * e['test_accuracy']:.2f}%", "lost": False},
        {"fill": e["fired"] / e["n_triggers"], "text": f"{e['fired']}", "lost": not e["behavioral_detected"]},
    ]
    if e["weight_applicable"]:
        out.append({"fill": min(1.0, max(0.0, e["weight_z"] / Z_CAP)), "text": f"{e['weight_z']:.2f}",
                    "lost": not e["weight_detected"]})
    else:
        out.append({"fill": None, "text": "n/a", "lost": True})
    return out


def _panel(ax, entries: list[dict], family_ids: tuple[str, ...], rows_total: int) -> None:
    titles = {f["id"]: f for f in FAMILIES}
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 3)
    ax.set_ylim(rows_total, -1.2)
    ax.axis("off")
    for j, name in enumerate(COLUMNS):
        ax.text(j + 0.5, -0.45, name, ha="center", va="center", fontsize=9, color=INK_SECONDARY)
    y = 0
    for fid in family_ids:
        f = titles[fid]
        ax.text(-0.04, y + 0.55, f"{f['title']}  ({f['task']})", ha="right", va="center", fontsize=9.5,
                color=INK, fontweight="bold")
        y += 1
        for e in [x for x in entries if x["family"] == fid]:
            ax.text(-0.04, y + 0.5, e["setting"], ha="right", va="center", fontsize=8.5, color=INK_SECONDARY)
            for j, cell in enumerate(cells(e)):
                color = NEUTRAL if cell["fill"] is None else CMAP(cell["fill"])
                # 2px surface gap between fills: the patch is inset, the surface shows through.
                ax.add_patch(Rectangle((j + 0.03, y + 0.06), 0.94, 0.88, facecolor=color, edgecolor="none"))
                if cell["lost"]:
                    ax.add_patch(Rectangle((j + 0.05, y + 0.1), 0.90, 0.80, facecolor="none", edgecolor=INK, linewidth=1.6))
                dark = cell["fill"] is not None and cell["fill"] > 0.5
                ax.text(j + 0.5, y + 0.5, cell["text"] + (" ×" if cell["lost"] else ""), ha="center", va="center",
                        fontsize=8.5, color="#ffffff" if dark else INK)
            y += 1


def plot(entries: list[dict], path: Path) -> None:
    left = tuple(f["id"] for f in FAMILIES if f["id"] in LEFT_PANEL)
    right = tuple(f["id"] for f in FAMILIES if f["id"] not in LEFT_PANEL)
    height = {side: sum(1 + sum(e["family"] == fid for e in entries) for fid in side) for side in (left, right)}
    rows_total = max(height.values())
    fig, axes = plt.subplots(1, 2, figsize=(17, 0.27 * rows_total + 2.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.225, right=0.985, top=0.93, bottom=0.115, wspace=1.3)
    _panel(axes[0], entries, left, rows_total)
    _panel(axes[1], entries, right, rows_total)
    fig.text(0.012, 0.975, "Watermark survival under every Phase 4 attack on the dual-watermarked model", fontsize=14,
             color=INK, ha="left", va="center", fontweight="bold")
    fig.text(0.012, 0.953, f"{len(entries)} settings, one run each. Darker = higher. A framed cell marked × is a watermark "
             "that is not detected at 1e-6; n/a = the weight test cannot be run on that model.",
             fontsize=9.5, color=INK_SECONDARY, ha="left", va="center")
    # Scale keys: one strip per column, with its own end points.
    for i, (name, lo, hi) in enumerate((("Test accuracy", "10% (chance)", "100%"), ("Triggers fired", "0", "100"),
                                        ("Weight z", "≤ 0", f"{Z_CAP:.2f} (cap)"))):
        x0 = 0.225 + i * 0.2
        key = fig.add_axes([x0, 0.062, 0.12, 0.012])
        key.imshow([[k / 255 for k in range(256)]], aspect="auto", cmap=CMAP, vmin=0, vmax=1)
        key.axis("off")
        fig.text(x0 + 0.06, 0.084, name, fontsize=8.5, color=INK, ha="center", va="center")
        fig.text(x0, 0.05, lo, fontsize=8, color=INK_SECONDARY, ha="left", va="center")
        fig.text(x0 + 0.12, 0.05, hi, fontsize=8, color=INK_SECONDARY, ha="right", va="center")
    fig.text(
        0.012, 0.028,
        "Detection thresholds at 1e-6: at least 29 of 100 triggers (P2.8), weight z at least 5.257 (P3.7). The two tests are "
        "separate and are not combined here. Channel-pruning rows assume the owner can re-align removed channels (not "
        "implemented).\nThe 50,000-image distillation set includes the owner's 45,000 training images. Each cell is a single "
        "run with one key; this is a technical demonstration, not legal evidence.",
        fontsize=8, color=INK_SECONDARY, ha="left", va="center",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


# --- entry point ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; nothing here is random")
    parser.add_argument("--rows-dir", type=Path, default=repo_root() / ROWS_DIR)
    parser.add_argument("--table", type=Path, default=repo_root() / TABLE)
    parser.add_argument("--figure", type=Path, default=repo_root() / FIGURE)
    parser.add_argument("--out-dir", type=Path, default=None, help="where the result record goes (default: results/)")
    args = parser.parse_args(argv)
    started = time.perf_counter()

    git = git_info()  # one snapshot, before this run writes anything
    entries, owner = collect(args.rows_dir)
    summary = summaries(entries)

    # The result is written first, so the new table and figure cannot mark the record dirty.
    path = write_result(
        name="p4.9_master_table",
        seed=args.seed,
        task="P4.9",
        params={
            "rows_from": ROWS_DIR,
            "source": {"name": "dual", "task": "P3.6", "weights_sha256": W_DUAL_SHA256},
            "owner": owner,
            "detection_alpha": DETECTION_ALPHA,
            "detected": "the watermark's own test rejects at 1e-6 (P2.8 behavioral, P3.7 weight); not combined (P9.3)",
            "weight_not_applicable": "owner carrier layout not present; counted as not detected",
            "families": [{k: f[k] for k in ("id", "title", "task", "expected")} | {"note": f.get("note")} for f in FAMILIES],
            "measured_here": "nothing; every figure is copied from a committed attack row",
            "table": TABLE,
            "figure": FIGURE,
            "device": "cpu",
        },
        metrics={"row_count": len(entries), "summary": summary, "rows": entries},
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git,
        notes="Aggregation of committed rows. Summaries are counts and ranges of single runs, not tested comparisons.",
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(markdown(entries, summary, path.name), encoding="utf-8", newline="\n")
    plot(entries, args.figure)

    for key, s in summary.items():
        print(f"{s['title']}: {s['rows']} rows, behavioral detected {s['behavioral_detected']}, "
              f"weight detected {s['weight_detected']}, neither {s['outcomes'][OUTCOMES[3]]}")
    print(f"result {path}\ntable  {args.table}\nfigure {args.figure}")
    return {"path": path, "entries": entries, "summary": summary}


if __name__ == "__main__":
    main()
