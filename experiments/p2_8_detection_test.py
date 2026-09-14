"""P2.8: turn "k of N triggers fired" into a p-value, and check the null on real models.

    python experiments/p2_8_detection_test.py

CPU only, a few minutes. The test itself is `src.watermark.significance`: an
exact binomial tail with the per-trigger null fire bound 1/9 from P2.2. This
script does three things with it.

1. **Thresholds.** For N = 100, the smallest fired count that reaches each
   tabulated significance level, and the exact false-positive bound there.
2. **p-values for the models already measured.** `W*` (P2.3) and clean `W`
   (P0.5) are re-scored on the triggers regenerated from `secrets/K.bin`, with
   the bundle digest checked against P2.3 and both weights files hash-checked.
   The five P2.7 sweep models are taken from the counts in the committed P2.7
   result file, which measured them the same way.
3. **Empirical null check.** The bound is a proof, not a measurement, so it is
   also checked on real models. Each of `M` public wrong keys ``K'_j``
   generates its own full trigger set and targets on the same CIFAR-10 split,
   and `W` and `W*` are scored against them. Both models are independent of
   every ``K'_j``, so each count is a draw from H0. This reports:

   - how often the test rejects, against the exact bound at that level;
   - the mean fired count against the bound ``N/9``;
   - a calibration check. Conditioned on the model's predictions, a key's
     expected fired count is ``(N - base_label_hits) / 9`` exactly (P2.2). The
     observed total over all keys is compared with that sum. A large gap would
     mean the targets are not independent of what the model predicts.

   With ``M = 1,000`` keys the rejection rate can only be checked at levels
   around 0.05 and 0.01. The smaller levels rest on the proof alone.

The wrong keys are ``SHA-256("zk-crown/p2.8/null-key/v1" || u64 seed || u64 j)``,
public by construction; the script refuses to run if one equals `K`. The
result file holds counts and p-values only. The owner triggers, targets and
per-trigger predictions stay in memory.
"""

from __future__ import annotations

import argparse
import hashlib
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
from p2_4_measure_wdr import P2_3_BUNDLE_SHA256, W_CLEAN_SHA256, W_STAR_SHA256, load_main_model, measure_all

from src.data import CIFAR10_MEAN, CIFAR10_STD
from src.data.cifar10 import cifar10_split_indices
from src.utils.results import repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.watermark.detection import normalise_uint8, predict_logits
from src.watermark.responses import null_fire_probability_bound, trigger_responses
from src.watermark.significance import (
    DEFAULT_ALPHAS,
    binomial_upper_tail,
    detection_test,
    detection_threshold,
    log10_fraction,
)
from src.watermark.triggers import DEFAULT_AMPLITUDE, DEFAULT_N, generate_triggers

P2_7_RESULT = "results/p2.7_ratio_sweep__seed1337__20260914T070218+0000.json"
FIGURE = "figures/p2.8_detection_test.png"
NULL_KEY_LABEL = b"zk-crown/p2.8/null-key/v1"
CHECKED_ALPHAS = ("0.05", "0.01")
"""Levels at which M = 1,000 null keys can say anything about the rejection rate."""
FIGURE_ALPHA = "1e-6"
"""Threshold drawn in the figure, one of DEFAULT_ALPHAS."""

# Reference palette (dataviz skill, references/palette.md), light mode.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"


def null_key(seed: int, j: int) -> bytes:
    """The j-th public wrong key. Not `K`, never secret."""
    return hashlib.sha256(NULL_KEY_LABEL + seed.to_bytes(8, "big") + j.to_bytes(8, "big")).digest()


# --- empirical null ---------------------------------------------------------


def null_counts(
    keys: list[bytes],
    images: np.ndarray,
    labels: np.ndarray,
    pool,
    models: dict[str, torch.nn.Module],
    *,
    n: int,
    amplitude: int,
    device: torch.device,
    chunk: int = 50,
) -> dict[str, dict[str, list[int]]]:
    """Per model, per key: fired count and base-label hits on that key's trigger set."""
    out = {name: {"fired": [], "base_label_hits": []} for name in models}
    for start in range(0, len(keys), chunk):
        batch = keys[start : start + chunk]
        sets = [generate_triggers(k, images, pool, n=n, amplitude=amplitude) for k in batch]
        responses = [trigger_responses(k, s, labels) for k, s in zip(batch, sets)]
        inputs = normalise_uint8(np.concatenate([s.images for s in sets]), CIFAR10_MEAN, CIFAR10_STD)
        for name, model in models.items():
            predictions = predict_logits(model, inputs, device).argmax(dim=1).numpy().reshape(len(batch), n)
            for r, p in zip(responses, predictions):
                out[name]["fired"].append(int(r.fired(p).sum()))
                out[name]["base_label_hits"].append(int((p == r.base_labels).sum()))
    return out


def summarise_null(fired: list[int], base_hits: list[int], n: int, num_classes: int) -> dict:
    m = len(fired)
    q = null_fire_probability_bound(num_classes)
    rejection = {}
    for alpha in CHECKED_ALPHAS:
        k, bound = detection_threshold(n, alpha, num_classes)
        rejected = sum(c >= k for c in fired)
        rate = rejected / m
        b = float(bound)
        rejection[alpha] = {
            "threshold": k,
            "exact_bound": b,
            "rejected": rejected,
            "rate": rate,
            # One-sided binomial sd of the rate if the true rate were exactly the bound.
            "rate_minus_bound_in_sd": (rate - b) / math.sqrt(b * (1 - b) / m),
        }
    eligible = [n - h for h in base_hits]
    expected = sum(eligible) * float(q)
    variance = sum(eligible) * float(q) * float(1 - q)
    p_values = [detection_test(c, n, num_classes) for c in fired]
    smallest = min(p_values, key=lambda t: t.p_value)
    return {
        "keys": m,
        "fired_mean": float(np.mean(fired)),
        "fired_var": float(np.var(fired, ddof=1)),
        "fired_max": max(fired),
        "bound_mean": float(n * q),
        "bound_var": float(n * q * (1 - q)),
        "base_label_hits_mean": float(np.mean(base_hits)),
        "calibration": {
            "observed_total_fired": sum(fired),
            "expected_total_given_predictions": expected,
            "z": (sum(fired) - expected) / math.sqrt(variance),
        },
        "rejection": rejection,
        "smallest_p_value": float(smallest.p_value),
        "smallest_p_value_fired": smallest.fired,
        "fired_histogram": np.bincount(fired, minlength=n + 1).tolist(),
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


def plot(n: int, num_classes: int, tests: list[dict], nulls: dict[str, dict], path: Path) -> None:
    q = null_fire_probability_bound(num_classes)
    k_star, _ = detection_threshold(n, FIGURE_ALPHA, num_classes)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)

    ax = axes[0]
    _style(ax)
    ks = list(range(n + 1))
    ax.plot(ks, [log10_fraction(binomial_upper_tail(k, n, q)) for k in ks], color=SERIES_1, linewidth=2, zorder=2)
    ax.axvline(k_star, color=INK_SECONDARY, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.annotate(f"k* = {k_star} at\nalpha = {FIGURE_ALPHA}", (k_star, -40), textcoords="offset points", xytext=(-4, 0),
                ha="right", fontsize=8, color=INK_SECONDARY)
    # Models with the same count share one point and one label. Points near
    # the top of the axis crowd together; their x position already gives k, so
    # they are named in a short list in the empty area under the curve.
    groups: dict[int, list[str]] = {}
    for t in tests:
        groups.setdefault(t["fired"], []).append(t["label"])
    listed = 0
    for x, names in sorted(groups.items()):
        p = binomial_upper_tail(x, n, q)
        y = log10_fraction(p)
        ax.scatter([x], [y], s=64, color=SERIES_1, edgecolor=SURFACE, linewidths=2, zorder=3)
        text = f"k = {x}: {', '.join(names)}"
        if y > -15:
            ax.text(k_star + 3, -62 - 7 * listed, f"{text} (p = {float(p):.3g})", fontsize=8, color=INK_SECONDARY)
            listed += 1
        elif x < n:
            ax.annotate(f"{text}\n(p = 10^{y:.1f})", (x, y), textcoords="offset points", xytext=(8, 4), fontsize=8,
                        color=INK_SECONDARY)
        else:
            ax.annotate(f"{text}\n(p = 10^{y:.1f})", (x, y), textcoords="offset points", xytext=(-8, 2), ha="right",
                        fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(-2, n + 2)
    ax.set_xlabel(f"triggers fired, k of {n}", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("log10 p-value", color=INK_SECONDARY, fontsize=9)
    ax.set_title("p-value of k under H0 (bound 1/9 per trigger)", color=INK, fontsize=11, loc="left")

    pmf = [float(binomial_upper_tail(k, n, q) - binomial_upper_tail(k + 1, n, q)) for k in ks]
    for ax, (name, label) in zip(axes[1:], (("W_clean", "clean W (P0.5)"), ("W_star", "W* (P2.3)"))):
        _style(ax)
        s = nulls[name]
        share = np.asarray(s["fired_histogram"]) / s["keys"]
        ax.bar(ks, share, width=0.8, color=SERIES_1, zorder=2, label=f"{s['keys']:,} wrong keys, observed")
        ax.step(ks, pmf, where="mid", color=SERIES_2, linewidth=2, zorder=3, label="Binomial(100, 1/9) bound")
        ax.axvline(k_star, color=INK_SECONDARY, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.set_xlim(-1, 40)
        ax.set_xlabel(f"triggers fired, k of {n}", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel("share of wrong keys", color=INK_SECONDARY, fontsize=9)
        ax.set_title(f"Null check: {label} on wrong keys", color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper right")

    fig.text(
        0.01, -0.04,
        f"Left: exact binomial tail. Points are measured counts on the owner key K. Dashed line: threshold at alpha = {FIGURE_ALPHA}. "
        f"Middle and right: each wrong key K' builds its own {n} triggers and targets; the model is independent of K', so each count is an H0 draw.\n"
        "The bound is reached only by a model that never predicts the base label; accurate models fire less, so the observed null sits left of the bound.",
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
    parser.add_argument("--null-keys", type=int, default=1_000)
    parser.add_argument("--figure", type=Path, default=repo_root() / FIGURE)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    backends = set_seed(args.seed)
    started = time.perf_counter()
    device = torch.device(args.device)
    n, amplitude, num_classes = DEFAULT_N, DEFAULT_AMPLITUDE, 10

    models = {
        "W_star": load_main_model(args.w_star, W_STAR_SHA256),
        "W_clean": load_main_model(args.w_clean, W_CLEAN_SHA256),
    }

    from torchvision.datasets import CIFAR10

    cifar = CIFAR10(str(args.data_root), train=True, download=False)
    labels = np.asarray(cifar.targets)
    pool, _ = cifar10_split_indices(len(cifar.data))
    key = load_key(args.key)

    # 1. thresholds
    thresholds = {}
    for alpha in DEFAULT_ALPHAS:
        k, bound = detection_threshold(n, alpha, num_classes)
        thresholds[alpha] = {"threshold": k, "exact_bound": float(bound), "exact_bound_log10": log10_fraction(bound)}

    # 2. owner-key counts
    digest, scores = measure_all(key, cifar.data, labels, pool, models, n=n, amplitude=amplitude, device=device)
    if digest != P2_3_BUNDLE_SHA256:
        raise SystemExit(f"K regenerates trigger bundle {digest}, expected {P2_3_BUNDLE_SHA256}. Refusing to report.")
    tests = []
    for name, label in (("W_clean", "clean W"), ("W_star", "W* (P2.3)")):
        tests.append({"model": name, "source": "measured here on K", "label": label,
                      **detection_test(scores[name]["triggers"]["fired"], n, num_classes).summary()})

    sweep = json.loads((repo_root() / P2_7_RESULT).read_text(encoding="utf-8"))
    if sweep["params"]["trigger_bundle_sha256"] != digest:
        raise SystemExit("P2.7 result was measured on a different trigger bundle")
    for row in sweep["metrics"]["rows"]:
        if row["source_task"] != "P2.7":
            continue  # W and W* were re-measured above
        if row["n_triggers"] != n:
            raise SystemExit(f"{row['name']} was measured on {row['n_triggers']} triggers")
        tests.append({
            "model": row["name"], "source": P2_7_RESULT, "weights_sha256": row["weights_sha256"],
            "trigger_to_clean_ratio": row["trigger_to_clean_ratio"],
            "label": f"{row['trigger_to_clean_ratio'] * 100:.3g}%",
            **detection_test(row["fired"], n, num_classes).summary(),
        })

    # 3. empirical null
    null_keys = [null_key(args.seed, j) for j in range(args.null_keys)]
    if key in null_keys or len(set(null_keys)) != len(null_keys):
        raise SystemExit("null keys must be distinct and differ from K")
    counts = null_counts(null_keys, cifar.data, labels, pool, models, n=n, amplitude=amplitude, device=device)
    nulls = {name: summarise_null(c["fired"], c["base_label_hits"], n, num_classes) for name, c in counts.items()}

    plot(n, num_classes, sorted(tests, key=lambda t: t["fired"]), nulls, args.figure)

    path = write_result(
        name="p2.8_detection_test",
        seed=args.seed,
        task="P2.8",
        params={
            "null_hypothesis": "suspect model independent of the owner's keyed targets (P2.2)",
            "statistic": "k = triggers whose eval-mode top-1 prediction equals the keyed target (P2.4)",
            "p_value": "exact P(Binomial(N, 1/(C-1)) >= k); valid for every H0 model by stochastic dominance",
            "n": n,
            "num_classes": num_classes,
            "null_fire_bound": "1/9",
            "amplitude": amplitude,
            "alphas": list(DEFAULT_ALPHAS),
            "owner_trigger_bundle_sha256": digest,
            "models": {"W_star": W_STAR_SHA256, "W_clean": W_CLEAN_SHA256},
            "p2_7_counts_from": P2_7_RESULT,
            "null_keys": {
                "count": args.null_keys,
                "derivation": "SHA-256('zk-crown/p2.8/null-key/v1' || u64 seed || u64 j), public, never K",
                "checked_alphas": list(CHECKED_ALPHAS),
            },
            "figure": FIGURE,
            "device": str(device),
        },
        metrics={
            "thresholds": thresholds,
            "tests": tests,
            "null_check": nulls,
        },
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        notes=(
            "p-values assume K and the trigger set were fixed before the suspect was seen, one pre-declared test, one "
            "query per trigger, and a per-suspect test (correct for multiple suspects). WDR counts are on the training "
            "triggers. The null check uses public wrong keys; it covers alpha around 0.05 and 0.01 only, smaller levels "
            "rest on the proof. Counts and p-values only; owner triggers, targets and predictions are not in this file."
        ),
    )

    print("thresholds, N = 100:")
    for alpha, t in thresholds.items():
        print(f"  alpha {alpha:>5}: k* = {t['threshold']:>3}  exact bound {t['exact_bound']:.3g}")
    print("\np-values:")
    for t in tests:
        print(f"  {t['model']:20s} fired {t['fired']:>3}/{n}  p = {t['p_value']:.3g}  (log10 {t['p_value_log10']:.2f})")
    print("\nnull check:")
    for name, s in nulls.items():
        rej = "  ".join(f"alpha {a}: {r['rejected']}/{s['keys']} (bound {r['exact_bound']:.3g})" for a, r in s["rejection"].items())
        c = s["calibration"]
        print(f"  {name:8s} mean fired {s['fired_mean']:.2f} (bound {s['bound_mean']:.2f})  max {s['fired_max']}  "
              f"base-label hits {s['base_label_hits_mean']:.1f}  calibration z {c['z']:+.2f}  {rej}")
    print("wrote", args.figure)
    print("wrote", path)
    return {"path": path, "metrics": {"thresholds": thresholds, "tests": tests, "null_check": nulls}}


if __name__ == "__main__":
    main()
