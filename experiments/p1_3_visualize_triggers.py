"""P1.3: visualise the trigger set and measure how big the perturbation is.

CLAUDE.md P1.3 asks for a by-eye check that the triggers are neither trivially
visible garbage nor invisible noise. A by-eye check needs something to compare
against, so this script writes three figures to `figures/`:

- ``p1.3_trigger_set.png``: all N triggers at the default amplitude.
- ``p1.3_trigger_detail.png``: base image, trigger, and the perturbation
  stretched to full contrast, for the first few triggers.
- ``p1.3_amplitude_sweep.png``: the same bases and sign patterns at several
  amplitudes, so the default can be judged against smaller and larger ones.

It also writes a result JSON with the perturbation's measured size at each
amplitude: PSNR against the base image, L-infinity and L2 norms, and how much
clipping at 0 and 255 shrank it. These numbers describe the images only. They
say nothing about how a model responds to the triggers.

**Demo key only.** Everything here uses `DEMO_KEY`, which is derived from a
public string and is therefore public. `figures/` is committed, so a figure
built from the real `K` would publish the real triggers. Do not add a real-key
option to this script.

    python experiments/p1_3_visualize_triggers.py
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.cifar10 import CLASS_NAMES, cifar10_split_indices
from src.utils.results import repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.watermark.triggers import (
    DEFAULT_AMPLITUDE,
    DEFAULT_N,
    STREAM_VERSION,
    TriggerSet,
    generate_triggers,
)

DEMO_KEY_PHRASE = b"zk-crown PUBLIC DEMO KEY v1 -- not an owner key, never protect a real model with it"
DEMO_KEY = hashlib.sha256(DEMO_KEY_PHRASE).digest()
"""PUBLIC DEMO KEY. SHA-256 of `DEMO_KEY_PHRASE`. Not the owner key `K`."""

SWEEP_AMPLITUDES = (4, 8, 16, 32, 64)
DETAIL_COUNT = 8
SWEEP_ROWS = 6
UPSCALE = 4
GUTTER = 4


# --- measurement ------------------------------------------------------------


def perturbation_stats(ts: TriggerSet) -> dict[str, float | int]:
    """Size of the applied perturbation, after clipping, over the whole set."""
    base = ts.base_images.astype(np.float64)
    delta = ts.images.astype(np.float64) - base
    n = len(ts)
    mse = (delta**2).reshape(n, -1).mean(axis=1)
    if np.any(mse == 0):
        raise RuntimeError("a trigger is identical to its base image; PSNR is undefined")
    psnr = 10.0 * np.log10(255.0**2 / mse)
    l2 = np.sqrt(((delta / 255.0) ** 2).reshape(n, -1).sum(axis=1))
    return {
        "amplitude": ts.amplitude,
        "psnr_db_mean": float(psnr.mean()),
        "psnr_db_min": float(psnr.min()),
        "psnr_db_max": float(psnr.max()),
        "linf_levels_max": int(np.abs(delta).max()),
        "l2_unit_range_mean": float(l2.mean()),
        "fraction_channels_clipped": float((np.abs(delta) < ts.amplitude).mean()),
    }


# --- drawing ----------------------------------------------------------------


def _tile(image: np.ndarray) -> np.ndarray:
    return image.repeat(UPSCALE, axis=0).repeat(UPSCALE, axis=1)


def _mosaic(rows: list[list[np.ndarray]]) -> np.ndarray:
    """Lay out uint8 HWC images on a white grid, upscaled with nearest neighbour."""
    tile = UPSCALE * rows[0][0].shape[0]
    height = len(rows) * tile + (len(rows) + 1) * GUTTER
    width = len(rows[0]) * tile + (len(rows[0]) + 1) * GUTTER
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            y = GUTTER + r * (tile + GUTTER)
            x = GUTTER + c * (tile + GUTTER)
            canvas[y : y + tile, x : x + tile] = _tile(image)
    return canvas


def _centres(count: int) -> list[float]:
    tile = UPSCALE * 32
    return [GUTTER + i * (tile + GUTTER) + tile / 2 for i in range(count)]


def _stretch(ts: TriggerSet, i: int) -> np.ndarray:
    """The applied perturbation of trigger `i`, mapped from [-A, A] to [0, 255]."""
    delta = ts.images[i].astype(np.int16) - ts.base_images[i].astype(np.int16)
    return np.round((delta + ts.amplitude) * 255.0 / (2 * ts.amplitude)).astype(np.uint8)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _figure_for(canvas: np.ndarray, margin_in: float = 1.6):
    height, width = canvas.shape[:2]
    fig, ax = plt.subplots(figsize=(width / 100 + margin_in, height / 100 + margin_in))
    ax.imshow(canvas, interpolation="nearest")
    for side in ax.spines.values():
        side.set_visible(False)
    return fig, ax


def plot_trigger_set(ts: TriggerSet, labels: np.ndarray, path: Path, columns: int = 10) -> None:
    rows = [list(ts.images[i : i + columns]) for i in range(0, len(ts), columns)]
    fig, ax = _figure_for(_mosaic(rows))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"All {len(ts)} triggers, amplitude {ts.amplitude}/255 "
        f"(PUBLIC DEMO KEY, not an owner key)\n"
        f"Classes shown are the base images' true labels: "
        + ", ".join(CLASS_NAMES[int(l)] for l in labels[:columns])
        + ", ... (row 1)",
        fontsize=10,
    )
    _save(fig, path)


def plot_trigger_detail(ts: TriggerSet, labels: np.ndarray, path: Path) -> None:
    k = min(DETAIL_COUNT, len(ts))
    rows = [
        list(ts.base_images[:k]),
        list(ts.images[:k]),
        [_stretch(ts, i) for i in range(k)],
    ]
    fig, ax = _figure_for(_mosaic(rows))
    ax.set_xticks(_centres(k))
    ax.set_xticklabels(
        [f"#{i}\n{CLASS_NAMES[int(labels[i])]}\nidx {int(ts.base_indices[i])}" for i in range(k)],
        fontsize=8,
    )
    ax.set_yticks(_centres(3))
    ax.set_yticklabels(
        ["base image", f"trigger\n(A = {ts.amplitude})", f"perturbation\nstretched x{255 / (2 * ts.amplitude):.1f}"],
        fontsize=9,
    )
    ax.tick_params(length=0)
    ax.set_title("Base, trigger and applied perturbation (PUBLIC DEMO KEY)", fontsize=10)
    _save(fig, path)


def plot_amplitude_sweep(sweep: list[TriggerSet], stats: list[dict], path: Path) -> None:
    k = min(SWEEP_ROWS, len(sweep[0]))
    rows = [[sweep[0].base_images[r]] + [ts.images[r] for ts in sweep] for r in range(k)]
    fig, ax = _figure_for(_mosaic(rows))
    ax.set_xticks(_centres(len(sweep) + 1))
    ax.set_xticklabels(
        ["base"]
        + [f"A = {s['amplitude']}\nPSNR {s['psnr_db_mean']:.1f} dB" for s in stats],
        fontsize=9,
    )
    ax.set_yticks(_centres(k))
    ax.set_yticklabels([f"#{r}" for r in range(k)], fontsize=9)
    ax.tick_params(length=0)
    ax.set_title(
        f"Same bases and sign patterns at each amplitude; default is A = {DEFAULT_AMPLITUDE}. "
        "PSNR is the mean over all triggers. (PUBLIC DEMO KEY)",
        fontsize=10,
    )
    _save(fig, path)


# --- entry point ------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-root", default=str(repo_root() / "data"))
    parser.add_argument("--figures-dir", default=str(repo_root() / "figures"))
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--download", action="store_true", help="download CIFAR-10 if missing")
    args = parser.parse_args()

    # Nothing here is random except through the key; seeding is for the record.
    backends = set_seed(args.seed)
    started = time.perf_counter()

    from torchvision.datasets import CIFAR10

    cifar = CIFAR10(args.data_root, train=True, download=args.download)
    images, targets = cifar.data, np.asarray(cifar.targets)
    pool, _ = cifar10_split_indices(len(images))

    sweep = [generate_triggers(DEMO_KEY, images, pool, n=args.n, amplitude=a) for a in SWEEP_AMPLITUDES]
    stats = [perturbation_stats(ts) for ts in sweep]
    default = sweep[SWEEP_AMPLITUDES.index(DEFAULT_AMPLITUDE)]
    labels = targets[default.base_indices]

    figures_dir = Path(args.figures_dir)
    paths = {
        "trigger_set": figures_dir / "p1.3_trigger_set.png",
        "trigger_detail": figures_dir / "p1.3_trigger_detail.png",
        "amplitude_sweep": figures_dir / "p1.3_amplitude_sweep.png",
    }
    plot_trigger_set(default, labels, paths["trigger_set"])
    plot_trigger_detail(default, labels, paths["trigger_detail"])
    plot_amplitude_sweep(sweep, stats, paths["amplitude_sweep"])

    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    metrics = {
        "default_amplitude": stats[SWEEP_AMPLITUDES.index(DEFAULT_AMPLITUDE)],
        "amplitude_sweep": stats,
        "base_class_counts": {name: int(c) for name, c in zip(CLASS_NAMES, class_counts)},
        "trigger_images_sha256": hashlib.sha256(default.images.tobytes()).hexdigest(),
    }
    params = {
        "key": "PUBLIC DEMO KEY = sha256(DEMO_KEY_PHRASE), not an owner key",
        "demo_key_phrase": DEMO_KEY_PHRASE.decode(),
        "demo_key_sha256_hex": DEMO_KEY.hex(),
        "stream_version": STREAM_VERSION,
        "n": args.n,
        "default_amplitude": DEFAULT_AMPLITUDE,
        "sweep_amplitudes": list(SWEEP_AMPLITUDES),
        "dataset": "CIFAR-10 train, 45,000-image main_model split",
        "image_dtype": "uint8, HWC, pixel range 0-255",
        "figures": {k: p.relative_to(repo_root()).as_posix() if p.is_relative_to(repo_root()) else str(p) for k, p in paths.items()},
    }
    path = write_result(
        name="p1.3_trigger_visualization",
        seed=args.seed,
        task="P1.3",
        params=params,
        metrics=metrics,
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        notes=(
            "Image statistics of the trigger perturbation only. No model was run, so "
            "nothing here says anything about watermark detection. Demo key, not K."
        ),
    )

    for s in stats:
        print(
            f"A={s['amplitude']:>3}  PSNR {s['psnr_db_mean']:5.2f} dB "
            f"[{s['psnr_db_min']:.2f}, {s['psnr_db_max']:.2f}]  "
            f"L2 {s['l2_unit_range_mean']:.3f}  clipped {s['fraction_channels_clipped']:.3%}"
        )
    print("base classes:", metrics["base_class_counts"])
    for p in paths.values():
        print("wrote", p)
    print("wrote", path)


if __name__ == "__main__":
    main()
