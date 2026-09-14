"""P3.7: null distribution of the weight-watermark correlation, and the detection threshold.

    python experiments/p3_7_weight_null.py [--workers 8]

CPU only, parallel over keys. It takes a few minutes with 8 workers.

The test itself is `src.watermark.weight_significance`. Under H0, "the
suspect is independent of `K`", the statistic ``z = correlation * sqrt(128)``
satisfies ``P(z >= t) <= exp(-t^2/2)``, which is a proof. That gives the
threshold ``z*(alpha) = sqrt(2 ln(1/alpha))`` with false positive rate at most
alpha. This script does four things:

1. **Empirical null.** Extract from three real models with 1,000 public
   wrong keys. Each wrong key derives its own ``P_K'`` and ``S'`` for the
   project owner id.
   - The models are clean `W` (P0.5), the behavioral-only model (P2.3) and
     the final dual `W*` (P3.6), all loaded by hash.
   - Every model is independent of every wrong key, so each extraction is an
     H0 draw, with the roles of owner and suspect swapped.
2. **Fit and check.** For each model, fit a Gaussian to the 1,000 z values
   and compare it with the theoretical N(0, 1).
   - Checks: mean, sd with its interval, skew, excess kurtosis, and a
     Kolmogorov-Smirnov test against N(0, 1). The KS test uses the fixed
     theoretical distribution, with no fitted parameters.
   - Counts at the proven thresholds, against the bound ``alpha * 1000``.
   - Bit matches against Binomial(128, 1/2).
   - With 1,000 keys the rejection rate can only be checked near alpha 0.05
     and 0.01. Smaller alphas rest on the proof.
3. **Thresholds.** For alpha 0.05, 0.01, 1e-3, 1e-6 and 1e-9: the proven
   threshold, the Gaussian approximation and the fitted-Gaussian threshold,
   in z and in correlation. The p-value floor ``exp(-64)`` is stated.
4. **Apply the test** to what is already measured:
   - the owner key `K` on the three models, recomputed here; the dual model's
     value must equal P3.6's;
   - every row of the committed P3.5 alpha sweep.

The wrong keys are ``SHA-256("zk-crown/p3.7/null-key/v1" || u64 seed || u64 j)``.
They are public, a different family from P2.8's and P3.4's, and the script
refuses to run if one equals `K`. The result file and figure hold aggregates
only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import NormalDist

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.models import main_model
from src.utils.results import repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.watermark.carrier import carrier_layout
from src.watermark.signature import PROJECT_OWNER_ID, derive_signature
from src.watermark.significance import DEFAULT_ALPHAS
from src.watermark.weight_embedding import derive_carrier_projection
from src.watermark.weight_extraction import centered_carrier, score_fingerprint
from src.watermark.weight_significance import (
    ROWS,
    WeightDetectionTest,
    gaussian_tail,
    p_value_bound,
    p_value_floor,
    thresholds,
    z_threshold,
)

NULL_KEY_LABEL = b"zk-crown/p3.7/null-key/v1"
DEFAULT_NULL_KEYS = 1_000
CHECKED_ALPHAS = ("0.05", "0.01")
"""Levels at which 1,000 keys can say anything about the rejection rate."""
W_DUAL_SHA256 = "7a9a9f141b55c7894b208c1f338f4c7385362f969b82b319ce5893fb4bb434c4"
P3_5_RESULT = "results/p3.5_alpha_sweep__seed1337__20260914T084524+0000.json"
P3_6_RESULT = "results/p3.6_dual_wm__seed1337__20260914T085541+0000.json"
FIGURE = "figures/p3.7_weight_null.png"
HOST_LABELS = {"W_clean": "clean W (P0.5)", "W_behavioral": "behavioral-only (P2.3)", "W_dual": "dual W* (P3.6)"}
HIST_EDGES = np.round(np.arange(-6.0, 6.0001, 0.25), 2)

# Reference palette (dataviz skill, references/palette.md), light mode. Slots 1-3
# validated; slot 3 is below 3:1 contrast, so every series is direct-labelled.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = {"W_clean": "#2a78d6", "W_behavioral": "#eb6834", "W_dual": "#1baf7a"}

_WORKER: dict = {}


def null_key(seed: int, j: int) -> bytes:
    """The j-th public wrong key. Not `K`, never secret."""
    return hashlib.sha256(NULL_KEY_LABEL + seed.to_bytes(8, "big") + j.to_bytes(8, "big")).digest()


# --- per-key work, run in worker processes ------------------------------------


def _init_worker(carriers: dict[str, np.ndarray], layout_names, layout_shapes, owner_id: str, seed: int) -> None:
    from src.watermark.carrier import CarrierLayout

    _WORKER.update(carriers=carriers, layout=CarrierLayout(names=tuple(layout_names), shapes=tuple(layout_shapes)),
                   owner_id=owner_id, seed=seed)


def extract_with_key(key: bytes, carriers: dict[str, np.ndarray], layout, owner_id: str) -> dict[str, tuple[float, int]]:
    """Correlation and bit matches of each centred carrier under `key`'s own ``P_K'`` and ``S'``."""
    projection = derive_carrier_projection(key, layout)
    signature = derive_signature(key, owner_id)
    out = {}
    for name, carrier in carriers.items():
        correlation, _, _, bits = score_fingerprint(projection.project(carrier), signature)
        out[name] = (correlation, bits)
    return out


def _null_job(j: int) -> tuple[int, dict[str, tuple[float, int]]]:
    w = _WORKER
    return j, extract_with_key(null_key(w["seed"], j), w["carriers"], w["layout"], w["owner_id"])


# --- summaries ----------------------------------------------------------------


def ks_against_standard_normal(z: np.ndarray) -> dict:
    """Kolmogorov-Smirnov against the fixed N(0, 1), asymptotic p-value (no fitted parameters)."""
    x = np.sort(np.asarray(z, dtype=np.float64))
    n = x.size
    cdf = np.array([NormalDist().cdf(float(v)) for v in x])
    d = float(max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(0, n) / n)))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return {"statistic": d, "p_value_asymptotic": float(min(1.0, max(0.0, p)))}


def summarise_null(z: list[float], bits: list[int], alphas=DEFAULT_ALPHAS) -> dict:
    a = np.asarray(z, dtype=np.float64)
    n = a.size
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    centred = a - mean
    skew = float(np.mean(centred**3) / np.mean(centred**2) ** 1.5)
    kurt = float(np.mean(centred**4) / np.mean(centred**2) ** 2 - 3)
    exceed = {}
    for alpha in alphas:
        z_star = z_threshold(alpha)
        count = int(np.count_nonzero(a >= z_star))
        entry = {
            "z_threshold": z_star,
            "exceedances": count,
            "bound": float(alpha) * n,
            "gaussian_expected": gaussian_tail(z_star) * n,
        }
        if alpha in CHECKED_ALPHAS:
            b = float(alpha)
            entry["rate_minus_bound_in_sd"] = (count / n - b) / math.sqrt(b * (1 - b) / n)
        exceed[str(alpha)] = entry
    fitted = NormalDist(mean, sd)
    bits_a = np.asarray(bits, dtype=np.float64)
    return {
        "count": n,
        "z_mean": mean,
        "z_mean_in_se": mean / (1 / math.sqrt(n)),
        "z_sd": sd,
        "z_sd_ci95": [sd * (1 - 1.96 / math.sqrt(2 * (n - 1))), sd * (1 + 1.96 / math.sqrt(2 * (n - 1)))],
        "z_min": float(a.min()),
        "z_max": float(a.max()),
        "skew": skew,
        "skew_se": math.sqrt(6 / n),
        "excess_kurtosis": kurt,
        "excess_kurtosis_se": math.sqrt(24 / n),
        "ks_vs_standard_normal": ks_against_standard_normal(a),
        "fitted_gaussian": {
            "mean": mean,
            "sd": sd,
            "z_thresholds": {str(al): fitted.inv_cdf(1 - float(al)) if float(al) > 1e-15 else None for al in alphas},
            "note": "descriptive fit to 1,000 draws; extrapolation below ~1e-3 is not checked",
        },
        "exceedances_at_proven_thresholds": exceed,
        "bit_matches": {
            "mean": float(bits_a.mean()), "var": float(bits_a.var(ddof=1)), "min": int(bits_a.min()), "max": int(bits_a.max()),
            "binomial_mean": ROWS / 2, "binomial_var": ROWS / 4,
        },
        "histogram": {"edges": HIST_EDGES.tolist(), "counts": np.histogram(a, bins=HIST_EDGES)[0].tolist(),
                      "below": int(np.count_nonzero(a < HIST_EDGES[0])), "above": int(np.count_nonzero(a >= HIST_EDGES[-1]))},
        "survival": {"t": [round(t, 2) for t in np.arange(0, 6.01, 0.05)],
                     "fraction_at_or_above": [float(np.mean(a >= t)) for t in np.arange(0, 6.01, 0.05)]},
    }


def apply_to_p3_5(p3_5: dict) -> list[dict]:
    rows = []
    for row in p3_5["metrics"]["rows"]:
        test = WeightDetectionTest(row["detection"]["correlation"])
        rows.append({"alpha_embedding": row["alpha"], **test.summary()})
    return rows


# --- figure -------------------------------------------------------------------


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(nulls: dict[str, dict], owner: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    _style(ax)
    s = nulls["W_dual"]
    edges = np.asarray(s["histogram"]["edges"])
    width = edges[1] - edges[0]
    density = np.asarray(s["histogram"]["counts"]) / (s["count"] * width)
    ax.bar(edges[:-1] + width / 2, density, width=width - 0.03, color=SERIES["W_dual"], zorder=2,
           label=f"dual W*, {s['count']:,} wrong keys")
    grid = np.linspace(-6, 6, 400)
    ax.plot(grid, [NormalDist().pdf(float(g)) for g in grid], color=INK, linewidth=2, zorder=3, label="N(0, 1)")
    ax.set_xlim(-5, 5)
    ax.set_xlabel("z = correlation x sqrt(128)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("density", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Null distribution of z (wrong keys on the final model)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left")
    ax.text(4.9, max(density) * 0.95,
            f"mean {s['z_mean']:+.3f}\nsd {s['z_sd']:.3f}\nKS p {s['ks_vs_standard_normal']['p_value_asymptotic']:.2f}",
            fontsize=8, color=INK_SECONDARY, ha="right", va="top")

    ax = axes[1]
    _style(ax)
    t = np.linspace(0, 11.8, 500)
    ax.plot(t, [p_value_bound(float(v)) for v in t], color=INK, linewidth=2, zorder=3)
    ax.plot(t, [gaussian_tail(float(v)) for v in t], color=INK_SECONDARY, linewidth=1.5, linestyle=(0, (4, 3)), zorder=3)
    ax.text(7.2, p_value_bound(7.2) * 8, "proven bound exp(-t^2/2)", fontsize=8, color=INK, rotation=-24)
    ax.text(6.3, gaussian_tail(6.3) / 60, "Gaussian tail (approximation)", fontsize=8, color=INK_SECONDARY, rotation=-30)
    # The three empirical curves end on top of each other near t = 3, so their
    # labels sit in a column to the right, each joined to its curve's end.
    for i, name in enumerate(("W_clean", "W_behavioral", "W_dual")):
        sv = nulls[name]["survival"]
        xs = np.asarray(sv["t"])
        ys = np.asarray(sv["fraction_at_or_above"])
        keep = ys > 0
        ax.step(xs[keep], ys[keep], where="post", color=SERIES[name], linewidth=2, zorder=4)
        last = int(np.flatnonzero(keep)[-1])
        ax.annotate(
            f"{HOST_LABELS[name]}: max z {nulls[name]['z_max']:.2f}", (xs[last], ys[last]),
            xytext=(4.6, 10 ** (0.0 - 1.6 * i)), textcoords="data", fontsize=8, color=INK_SECONDARY, va="center",
            arrowprops={"arrowstyle": "-", "color": SERIES[name], "linewidth": 0.8, "shrinkA": 0, "shrinkB": 2},
        )
    for alpha in ("0.05", "1e-6"):
        z_star = z_threshold(alpha)
        ax.axvline(z_star, color=GRID, linewidth=1.2, zorder=1)
        ax.text(z_star + 0.08, 1e-13, f"z* = {z_star:.2f}\n(alpha {alpha})", fontsize=8, color=INK_SECONDARY)
    floor = p_value_floor()
    ax.axhline(floor, color=INK_SECONDARY, linewidth=1, linestyle=(0, (1, 2)), zorder=1)
    ax.text(0.1, floor / 60, f"floor exp(-64) = {floor:.1e}, reached only at correlation 1", fontsize=8, color=INK_SECONDARY)
    z_dual = owner["W_dual"]["z"]
    ax.scatter([z_dual], [p_value_bound(z_dual)], s=70, color=SERIES["W_dual"], edgecolor=SURFACE, linewidths=2, zorder=5)
    ax.annotate(f"owner key K on dual W*\nz = {z_dual:.2f}, p <= {p_value_bound(z_dual):.1e}", (z_dual, p_value_bound(z_dual)),
                textcoords="offset points", xytext=(-150, 12), ha="right", fontsize=8, color=INK_SECONDARY,
                arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "linewidth": 0.8})
    ax.set_yscale("log")
    ax.set_ylim(1e-30, 2)
    ax.set_xlim(0, 11.8)
    ax.set_xlabel("t", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("P(z >= t)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Tail: 1,000 wrong keys per model against the bound", color=INK, fontsize=11, loc="left")

    fig.text(
        0.01, -0.05,
        "Each wrong key K' derives its own projection and signature; every model is independent of every K', so each z is an H0 draw. "
        "Empirical tails end at 1/1,000.\nBelow that the threshold rests on the proof (Hoeffding for a Rademacher sum), not on the fit. "
        "p-values assume K was committed before the suspect was seen.",
        fontsize=8, color=INK_SECONDARY, ha="left", va="top",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


# --- entry point ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--key", type=Path, default=None, help="default: secrets/K.bin")
    parser.add_argument("--w-clean", type=Path, default=repo_root() / "results" / "p0.5_clean_baseline_W.pt")
    parser.add_argument("--w-behavioral", type=Path, default=repo_root() / "results" / "p2.3_behavioral_wm_W_star.pt")
    parser.add_argument("--w-dual", type=Path, default=repo_root() / "results" / "p3.6_dual_wm_W_star.pt")
    parser.add_argument("--null-keys", type=int, default=DEFAULT_NULL_KEYS)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--figure", type=Path, default=repo_root() / FIGURE)
    args = parser.parse_args(argv)

    from make_master_key import DEFAULT_KEY_PATH, load_key
    from p2_4_measure_wdr import W_CLEAN_SHA256, W_STAR_SHA256, load_main_model

    backends = set_seed(args.seed)
    started = time.perf_counter()
    p3_6 = json.loads((repo_root() / P3_6_RESULT).read_text(encoding="utf-8"))
    if p3_6["metrics"]["weights_sha256"] != W_DUAL_SHA256:
        raise SystemExit("P3.6 result does not describe the dual weights this script expects")
    hashes = {"W_clean": W_CLEAN_SHA256, "W_behavioral": W_STAR_SHA256, "W_dual": W_DUAL_SHA256}
    paths = {"W_clean": args.w_clean, "W_behavioral": args.w_behavioral, "W_dual": args.w_dual}
    layout = carrier_layout(main_model())
    carriers = {name: centered_carrier(load_main_model(paths[name], hashes[name]).state_dict(), layout) for name in hashes}

    key = load_key(args.key or DEFAULT_KEY_PATH)
    keys = [null_key(args.seed, j) for j in range(args.null_keys)]
    if key in keys or len(set(keys)) != len(keys):
        raise SystemExit("null keys must be distinct and differ from K")

    # owner key on the three models (the dual one must reproduce P3.6)
    owner_raw = extract_with_key(key, carriers, layout, PROJECT_OWNER_ID)
    owner = {}
    for name, (corr, bits) in owner_raw.items():
        owner[name] = {**WeightDetectionTest(corr).summary(), "bit_matches": bits}
    expected = p3_6["metrics"]["weight_watermark"]["dual"]["correlation"]
    if abs(owner["W_dual"]["correlation"] - expected) > 1e-12:
        raise SystemExit(f"owner-key correlation on dual W* {owner['W_dual']['correlation']} != P3.6's {expected}")

    # empirical null, in parallel; BLAS kept single-threaded per worker
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    z = {name: [0.0] * args.null_keys for name in hashes}
    bits = {name: [0] * args.null_keys for name in hashes}
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(carriers, layout.names, layout.shapes, PROJECT_OWNER_ID, args.seed),
    ) as pool:
        for j, result in pool.map(_null_job, range(args.null_keys), chunksize=4):
            for name, (corr, b) in result.items():
                z[name][j] = corr * math.sqrt(ROWS)
                bits[name][j] = b
            done += 1
            if done % 100 == 0:
                print(f"  null keys {done}/{args.null_keys}  ({time.perf_counter() - started:.0f} s)", flush=True)

    nulls = {name: summarise_null(z[name], bits[name]) for name in hashes}
    names = list(hashes)
    cross = {f"{a}~{b}": float(np.corrcoef(z[a], z[b])[0, 1]) for i, a in enumerate(names) for b in names[i + 1:]}
    table = thresholds()
    for alpha, row in table.items():
        row["fitted_gaussian_z"] = {name: nulls[name]["fitted_gaussian"]["z_thresholds"][alpha] for name in names}
    p3_5_rows = apply_to_p3_5(json.loads((repo_root() / P3_5_RESULT).read_text(encoding="utf-8")))

    # The result is written before the figure, so a new untracked figure cannot mark the record dirty.
    path = write_result(
        name="p3.7_weight_null",
        seed=args.seed,
        task="P3.7",
        params={
            "null_hypothesis": "suspect model independent of K (so y = P_K c is independent of S)",
            "statistic": "z = normalised correlation * sqrt(128), blind per-tensor-centred extraction (P3.3), one-sided",
            "p_value": "min(1, exp(-z^2/2)) (Hoeffding for a Rademacher sum given y); valid for every H0 model",
            "threshold": "z*(alpha) = sqrt(2 ln(1/alpha)), false positive rate <= alpha",
            "p_value_floor": "exp(-64), at correlation 1",
            "hosts": hashes,
            "owner_id": PROJECT_OWNER_ID,
            "carrier_digest": layout.digest(),
            "null_keys": {
                "count": args.null_keys,
                "derivation": "SHA-256('zk-crown/p3.7/null-key/v1' || u64 seed || u64 j), public, never K",
                "each_derives": "its own P_K' and S' for the project owner id",
                "checked_alphas": list(CHECKED_ALPHAS),
            },
            "alphas": list(DEFAULT_ALPHAS),
            "p3_5_from": P3_5_RESULT,
            "p3_6_from": P3_6_RESULT,
            "workers": args.workers,
            "figure": FIGURE,
            "device": "cpu",
        },
        metrics={
            "thresholds": table,
            "p_value_floor": p_value_floor(),
            "null": nulls,
            "cross_model_z_correlation_same_keys": cross,
            "owner_key": owner,
            "p3_5_sweep": p3_5_rows,
        },
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        notes=(
            "The threshold is a proof, checked empirically with 1,000 wrong keys per model near alpha 0.05 and 0.01 only. "
            "Gaussian fits are descriptive. Hosts share the same keys, so their null samples are not independent and are "
            "not pooled. p-values assume K was committed before the suspect was seen, one pre-declared test, and a "
            "per-suspect correction. Aggregates only."
        ),
    )

    plot(nulls, owner, args.figure)

    print("\nthresholds (proven | Gaussian approximation), z and correlation:")
    for alpha, row in table.items():
        print(f"  alpha {alpha:>5}: z* {row['z']:.3f} (corr {row['correlation']:.4f})  | gaussian z {row['gaussian_z_approximation']:.3f}")
    print(f"  p-value floor exp(-64) = {p_value_floor():.3g}")
    print("\nnull, per model:")
    for name, s in nulls.items():
        e = s["exceedances_at_proven_thresholds"]
        print(f"  {name:13s} z mean {s['z_mean']:+.3f} sd {s['z_sd']:.3f} [{s['z_sd_ci95'][0]:.3f}, {s['z_sd_ci95'][1]:.3f}]  "
              f"max {s['z_max']:.2f} min {s['z_min']:.2f}  skew {s['skew']:+.2f} kurt {s['excess_kurtosis']:+.2f}  "
              f"KS D {s['ks_vs_standard_normal']['statistic']:.3f} p {s['ks_vs_standard_normal']['p_value_asymptotic']:.2f}  "
              f"exceed 0.05: {e['0.05']['exceedances']} (bound {e['0.05']['bound']:.0f}, gauss {e['0.05']['gaussian_expected']:.1f})  "
              f"0.01: {e['0.01']['exceedances']}  1e-3: {e['1e-3']['exceedances']}  "
              f"bits {s['bit_matches']['mean']:.2f} var {s['bit_matches']['var']:.1f}")
    print("  cross-model z correlation (same keys):", {k: round(v, 3) for k, v in cross.items()})
    print("\nowner key K:")
    for name, o in owner.items():
        print(f"  {name:13s} z {o['z']:6.2f}  p <= {o['p_value_bound']:.3g}  rejects 1e-6: {o['rejects']['1e-6']}")
    print("\nP3.5 sweep:")
    for r in p3_5_rows:
        print(f"  alpha {r['alpha_embedding']:<6} z {r['z']:6.2f}  p <= {r['p_value_bound']:.3g}  "
              f"rejects 0.05 {r['rejects']['0.05']} 1e-6 {r['rejects']['1e-6']} 1e-9 {r['rejects']['1e-9']}")
    print("wrote", args.figure)
    print("wrote", path)
    return {"path": path}


if __name__ == "__main__":
    main()
