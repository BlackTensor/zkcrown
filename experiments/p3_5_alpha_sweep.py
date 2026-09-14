"""P3.5: sweep the weight watermark strength alpha, detection confidence against accuracy drop.

    python experiments/p3_5_alpha_sweep.py

CPU only, about ten minutes. The steps:

1. Load the host `W*` (P2.3) and the baseline `W` (P0.5), both by hash.
2. Derive `P_K` and `S` from `secrets/K.bin` and ``PROJECT_OWNER_ID``. Embed
   post-hoc (P3.2) into `W*` at each alpha in ``ALPHAS``. Alpha 0 is `W*`
   itself.
3. **Detection confidence** for each alpha. Blind extraction with `K` (P3.3)
   gives the correlation, ``z = correlation * sqrt(128)``, and bit matches.
   Fifty public wrong keys, the first 50 of P3.4's family, give a wrong-key
   reference at the same alpha. No threshold or p-value; that is P3.7.
4. **Accuracy** of each embedded model, scored per image on the CIFAR-10
   test set (10,000) and the attacker holdout (5,000). BatchNorm statistics
   are not recalibrated. Drops use P2.5's paired analysis against two
   references:
   - **`W*`**, the cost of the weight watermark alone. This is the primary
     reference.
   - **`W`**, the total cost against the P0.6 baseline.
5. A three-panel figure: z against alpha, test drop against alpha, and z
   against test drop.

Checks built in. Alpha 0 must reproduce `W*`'s recorded test and holdout
accuracy exactly. At the alphas shared with P3.4, the correct-key correlation
must equal P3.4's committed value.

The alpha grid is fixed before running, with the same spacing idea as P3.4
extended upwards to find where accuracy breaks. **No alpha is selected here.**
Choosing alpha, and whether to embed post-hoc or during training, is P3.6.
Every drop is measured on the test set. If P3.6 selects alpha by looking at
these test drops, the selected model's test accuracy is no longer an unbiased
estimate, and the ledger has to say so.

The result file holds aggregates only.
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
import numpy as np
import torch
from make_master_key import DEFAULT_KEY_PATH, load_key
from p2_4_measure_wdr import W_CLEAN_SHA256, W_STAR_SHA256, load_main_model
from p3_4_key_specificity import summarise, wrong_key
from torch.nn import functional as F

from src.data import cifar10_loaders
from src.models import main_model
from src.utils.results import repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.utils.stats import paired_accuracy_difference
from src.watermark.carrier import carrier_layout
from src.watermark.signature import PROJECT_OWNER_ID, derive_signature
from src.watermark.weight_embedding import derive_carrier_projection, embed_weight_watermark
from src.watermark.weight_extraction import extract_weight_watermark, recover_fingerprint, score_fingerprint

ALPHAS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
"""Fixed before running. Not a selection."""
DEFAULT_WRONG_KEYS = 50
P2_3_RESULT = "results/p2.3_behavioral_wm__seed1337__20260913T112054+0000.json"
P3_4_RESULT = "results/p3.4_key_specificity__seed1337__20260914T082116+0000.json"
FIGURE = "figures/p3.5_alpha_sweep.png"
ROWS = 128
Z_CAP = math.sqrt(ROWS)
CROWDED_Z = 9.5
"""In the figure's third panel, points at or above this z are labelled in a side column."""

# Reference palette (dataviz skill, references/palette.md), light mode; validated.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"


def score_split(model: torch.nn.Module, loader) -> dict:
    """One eval-mode pass: per-image correctness and mean cross-entropy."""
    model.eval()
    correct, loss_sum, n = [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
            correct.append(logits.argmax(dim=1) == y)
            n += len(y)
    flags = torch.cat(correct)
    return {"accuracy": int(flags.sum()) / n, "correct_count": int(flags.sum()), "n": n,
            "loss": loss_sum / n, "correct": flags.tolist()}


def model_from(state: dict) -> torch.nn.Module:
    model = main_model()
    model.load_state_dict(state, strict=True)
    return model.eval()


def paired_pp(reference: list[bool], other: list[bool]) -> dict:
    p = paired_accuracy_difference(reference, other)
    return {
        "drop_pp": p["drop"] * 100,
        "ci95_pp": [p["drop_ci_low"] * 100, p["drop_ci_high"] * 100],
        "reference_only_right": p["reference_only_right"],
        "other_only_right": p["other_only_right"],
        "mcnemar_exact_p": p["mcnemar_exact_p"],
    }


def check_against_p3_4(rows: list[dict], p3_4: dict) -> list[dict]:
    """Correct-key correlation at shared alphas must equal P3.4's committed value."""
    shared = []
    by_alpha = {row["alpha"]: row for row in p3_4["metrics"]["hosts"]["W_star"].values()}
    for row in rows:
        if row["alpha"] in by_alpha:
            theirs = by_alpha[row["alpha"]]["correct_key"]["correlation"]
            ours = row["detection"]["correlation"]
            if abs(theirs - ours) > 1e-12:
                raise SystemExit(f"alpha {row['alpha']}: correlation {ours} differs from P3.4's {theirs}")
            shared.append({"alpha": row["alpha"], "p3_4": theirs, "here": ours})
    return shared


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


def plot(rows: list[dict], path: Path) -> None:
    alphas = [r["alpha"] for r in rows]
    nonzero = [a for a in alphas if a > 0]
    z = [r["detection"]["z"] for r in rows]
    wrong_max = [r["wrong_keys"]["z_max_abs"] for r in rows]
    drop = [r["test"]["vs_W_star"]["drop_pp"] for r in rows]
    lo = [r["test"]["vs_W_star"]["ci95_pp"][0] for r in rows]
    hi = [r["test"]["vs_W_star"]["ci95_pp"][1] for r in rows]
    # log x axis: alpha 0 is drawn at a stand-in position left of the smallest alpha
    x0 = min(nonzero) / 2.5
    xs = [a if a > 0 else x0 for a in alphas]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), facecolor=SURFACE)

    ax = axes[0]
    _style(ax)
    ax.axhline(Z_CAP, color=INK_SECONDARY, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.text(xs[0], Z_CAP + 0.25, f"cap sqrt(128) = {Z_CAP:.2f}", fontsize=8, color=INK_SECONDARY)
    ax.plot(xs, wrong_max, color=SERIES_2, linewidth=2, zorder=2, label=f"wrong keys, max |z| of {rows[0]['wrong_keys']['correlation']['count']}")
    ax.plot(xs, z, color=SERIES_1, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=3, label="correct key K")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["0" if a == 0 else f"{a:g}" for a in alphas], rotation=45)
    ax.minorticks_off()
    ax.set_xlabel("alpha (log scale; 0 drawn at left)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("z = correlation x sqrt(128)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Detection confidence", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="center left")

    ax = axes[1]
    _style(ax)
    ax.axhline(0, color=INK_SECONDARY, linewidth=1, zorder=1)
    ax.errorbar(xs, drop, yerr=[np.subtract(drop, lo), np.subtract(hi, drop)], color=SERIES_1, linewidth=2,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, capsize=3, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["0" if a == 0 else f"{a:g}" for a in alphas], rotation=45)
    ax.minorticks_off()
    ax.set_xlabel("alpha (log scale; 0 drawn at left)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("test accuracy drop vs W*, pp (95% CI)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Accuracy cost of the weight watermark", color=INK, fontsize=11, loc="left")

    ax = axes[2]
    _style(ax)
    ax.axhline(Z_CAP, color=INK_SECONDARY, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    band = max(wrong_max)
    ax.axhspan(-band, band, color=SERIES_2, alpha=0.15, linewidth=0, zorder=1)
    ax.text(max(hi), band + 0.25, f"wrong-key |z| <= {band:.2f}", fontsize=8, color=INK_SECONDARY, ha="right")
    ax.errorbar(drop, z, xerr=[np.subtract(drop, lo), np.subtract(hi, drop)], color=SERIES_1, linewidth=0,
                elinewidth=1.2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, capsize=3, zorder=3)
    # Points below the cluster near the cap get a label beside them. The
    # crowded points near the cap get labels stacked in a column to the right,
    # joined to their point by a thin leader line.
    crowded = [i for i, zy in enumerate(z) if zy >= CROWDED_Z]
    label_x = max(hi) + 0.08
    for rank, i in enumerate(sorted(crowded, key=lambda i: -z[i])):
        label_y = Z_CAP - 0.55 - 0.62 * rank
        ax.annotate(
            f"alpha {alphas[i]:g}", (drop[i], z[i]), xytext=(label_x, label_y), textcoords="data",
            fontsize=8, color=INK_SECONDARY, va="center",
            arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.8, "shrinkA": 0, "shrinkB": 4},
        )
    for i, (a, zy) in enumerate(zip(alphas, z)):
        if i not in crowded:
            ax.text(hi[i] + 0.04, zy, f"{a:g}", fontsize=8, color=INK_SECONDARY, va="center")
    ax.set_xlim(right=label_x + 0.55)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1, zorder=1)
    ax.set_xlabel("test accuracy drop vs W*, pp (95% CI)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("z = correlation x sqrt(128)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Detection against accuracy cost (labels: alpha)", color=INK, fontsize=11, loc="left")

    fig.text(
        0.01, -0.05,
        "Host W* (P2.3), post-hoc embedding W* + alpha P_K^T S, no BatchNorm recalibration, one key K and owner id. "
        "Drop is paired against W* on the 10,000-image CIFAR-10 test set; intervals cover test-image sampling only.\n"
        "z is a scale, not a p-value: the null distribution and threshold are P3.7. No alpha is selected here (P3.6).",
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
    parser.add_argument("--w-star", type=Path, default=repo_root() / "results" / "p2.3_behavioral_wm_W_star.pt")
    parser.add_argument("--w-clean", type=Path, default=repo_root() / "results" / "p0.5_clean_baseline_W.pt")
    parser.add_argument("--wrong-keys", type=int, default=DEFAULT_WRONG_KEYS)
    parser.add_argument("--figure", type=Path, default=repo_root() / FIGURE)
    args = parser.parse_args(argv)

    backends = set_seed(args.seed)
    started = time.perf_counter()
    host = load_main_model(args.w_star, W_STAR_SHA256).state_dict()
    baseline = load_main_model(args.w_clean, W_CLEAN_SHA256)
    layout = carrier_layout(main_model())
    key = load_key(args.key)
    projection = derive_carrier_projection(key, layout)
    signature = derive_signature(key, PROJECT_OWNER_ID)
    loaders = cifar10_loaders(args.data_root, eval_batch_size=500, num_workers=0, download=False)
    splits = ("test", "holdout")

    recorded = json.loads((repo_root() / P2_3_RESULT).read_text(encoding="utf-8"))["metrics"]
    if recorded["weights_sha256"] != W_STAR_SHA256:
        raise SystemExit("P2.3 result does not describe this W*")

    baseline_scores = {s: score_split(baseline, loaders[s]) for s in splits}
    print(f"W (P0.5): test {baseline_scores['test']['accuracy']:.2%}  holdout {baseline_scores['holdout']['accuracy']:.2%}")

    rows, states, host_scores = [], {}, None
    for alpha in ALPHAS:
        state, embedding = embed_weight_watermark(host, layout, projection, signature, alpha)
        states[alpha] = state
        extraction = extract_weight_watermark(state, layout, projection, signature)
        model = model_from(state)
        scores = {s: score_split(model, loaders[s]) for s in splits}
        if alpha == 0.0:
            host_scores = scores
            if (scores["test"]["accuracy"], scores["holdout"]["accuracy"]) != (
                recorded["test_accuracy"], recorded["holdout_accuracy"]
            ):
                raise SystemExit("alpha 0 does not reproduce W*'s recorded accuracy")
        row = {
            "alpha": alpha,
            "embedding": embedding.to_dict(),
            "detection": {**extraction.to_dict(), "z": extraction.correlation * math.sqrt(extraction.rows)},
        }
        for s in splits:
            row[s] = {
                "accuracy": scores[s]["accuracy"],
                "correct": scores[s]["correct_count"],
                "n": scores[s]["n"],
                "loss": scores[s]["loss"],
                "vs_W_star": paired_pp(host_scores[s]["correct"], scores[s]["correct"]),
                "vs_W": paired_pp(baseline_scores[s]["correct"], scores[s]["correct"]),
            }
        rows.append(row)
        print(
            f"alpha {alpha:<6} corr {extraction.correlation:+.4f} z {row['detection']['z']:5.2f} bits {extraction.bit_matches:>3}  "
            f"test {row['test']['accuracy']:.2%} drop vs W* {row['test']['vs_W_star']['drop_pp']:+.2f} pp  "
            f"holdout {row['holdout']['accuracy']:.2%}  ({time.perf_counter() - started:.0f} s)"
        )

    wrong = [wrong_key(args.seed, j) for j in range(args.wrong_keys)]
    if key in wrong or len(set(wrong)) != len(wrong):
        raise SystemExit("wrong keys must be distinct and differ from K")
    wrong_corr = {alpha: [] for alpha in ALPHAS}
    for j, k_wrong in enumerate(wrong):
        p_wrong = derive_carrier_projection(k_wrong, layout)
        s_wrong = derive_signature(k_wrong, PROJECT_OWNER_ID)
        for alpha, state in states.items():
            corr, _, _, _ = score_fingerprint(recover_fingerprint(state, layout, p_wrong), s_wrong)
            wrong_corr[alpha].append(corr)
        if (j + 1) % 10 == 0:
            print(f"  wrong keys {j + 1}/{len(wrong)}  ({time.perf_counter() - started:.0f} s)")
    for row in rows:
        w = summarise(wrong_corr[row["alpha"]])
        gap = row["detection"]["correlation"] - w["mean"]
        row["wrong_keys"] = {"correlation": w, "z_max_abs": w["max_abs"] * math.sqrt(ROWS)}
        row["gap_in_wrong_key_sd"] = gap / w["sd"]
        row["correct_above_every_wrong_key"] = bool(row["detection"]["correlation"] > w["max"])

    shared = check_against_p3_4(rows, json.loads((repo_root() / P3_4_RESULT).read_text(encoding="utf-8")))
    plot(rows, args.figure)

    path = write_result(
        name="p3.5_alpha_sweep",
        seed=args.seed,
        task="P3.5",
        params={
            "host": {"task": "P2.3", "weights_sha256": W_STAR_SHA256},
            "baseline": {"task": "P0.5/P0.6", "weights_sha256": W_CLEAN_SHA256},
            "owner_id": PROJECT_OWNER_ID,
            "embedding": "post-hoc W* + alpha * P_K^T * S on the carrier (P3.2); BatchNorm statistics not recalibrated",
            "extraction": "blind, per-tensor centred, normalised correlation (P3.3)",
            "carrier_digest": layout.digest(),
            "alphas": list(ALPHAS),
            "alphas_note": "fixed before running; no alpha selected here (P3.6)",
            "wrong_keys": {"count": args.wrong_keys, "derivation": "first N of P3.4's family, SHA-256('zk-crown/p3.4/wrong-key/v1' || u64 seed || u64 j)"},
            "accuracy": "eval-mode top-1, per image; paired drop with exact McNemar and 95% normal CI (P2.5 method)",
            "drop_definition": "acc(reference) - acc(embedded); positive means the embedded model is worse",
            "splits": {"test": "official CIFAR-10 test, 10,000", "holdout": "attacker holdout, 5,000"},
            "figure": FIGURE,
            "device": "cpu",
        },
        metrics={
            "rows": rows,
            "baseline_W": {s: {"accuracy": baseline_scores[s]["accuracy"], "correct": baseline_scores[s]["correct_count"]} for s in splits},
            "p3_4_cross_check": shared,
            "z_cap": Z_CAP,
        },
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        notes=(
            "One host, one key, post-hoc embedding without BN recalibration. Paired intervals cover image sampling, not "
            "training-seed or key-to-key variation. z is a scale, not a p-value (P3.7). No alpha selected (P3.6). "
            "Aggregates only; S, P_K, fingerprints and watermarked weights are not in this file."
        ),
    )
    print("wrote", args.figure)
    print("wrote", path)
    return {"path": path, "metrics": {"rows": rows}}


if __name__ == "__main__":
    main()
