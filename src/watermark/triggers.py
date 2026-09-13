"""Key-derived perturbation trigger set (P1.2).

A trigger is a CIFAR-10 training image with a key-derived additive perturbation
on top. Everything random about the set comes from the master key `K` through
`src.watermark.keygen.KeyStream`, so the owner can regenerate it exactly and
nobody without `K` can.

Construction, for trigger ``i`` of a set of ``n``:

1. **Base image.** ``n`` distinct indices are drawn without replacement from
   the pool, the 45,000-image `main_model` training split. The draw is a
   partial Fisher-Yates shuffle of the *sorted* pool, driven by
   ``KeyStream(K, "triggers/v1/base-index").randbelow``. The attacker holdout
   and the test set are never used: the holdout belongs to the P4.5 attacker,
   and the test set measures clean accuracy.
2. **Perturbation.** A sign pattern ``s_i`` in {-1, +1} with one entry per
   pixel channel (3,072 for CIFAR-10), read from
   ``KeyStream(K, "triggers/v1/perturbation-sign")``. Each trigger takes the
   next ``ceil(3072 / 8)`` bytes, and the bits are unpacked most significant
   bit first; bit 1 means +1. Patterns for different triggers are independent.
3. **Trigger.** ``T_i = clip(x_i + amplitude * s_i, 0, 255)``, computed in
   integer arithmetic on uint8 pixels.

Properties that follow:

- **No floating point.** Base images, signs and triggers are integers, so the
  set is byte-identical on every platform. Normalisation happens later, in
  the data pipeline.
- **Prefix stable.** The first ``m`` triggers of a set of ``n > m`` are the
  set of ``m``. Changing ``N`` later (P2.7) does not reshuffle the triggers
  already in use.
- **Pool order does not matter.** The pool is sorted before drawing, so only
  its contents matter, not the order the split function returns them in.
- **L-infinity bound.** Every pixel channel moves by at most ``amplitude``
  levels, and by exactly ``amplitude`` unless clipping at 0 or 255 cuts it
  short.

What this module does **not** decide:

- The target response each trigger maps to (P2.2).
- Whether `amplitude` = `DEFAULT_AMPLITUDE` is visible enough without being
  garbage. That is judged by eye in P1.3.
- The written rationale for additive noise over a patch or a learned trigger
  (P1.5).
- Byte-identical regeneration and cross-key independence are tested in P1.4.

Images use the layout of torchvision's ``CIFAR10.data``: ``(N, 32, 32, 3)``
uint8, height x width x channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.watermark.keygen import KeyStream

STREAM_VERSION = "triggers/v1"
BASE_INDEX_LABEL = f"{STREAM_VERSION}/base-index"
PERTURBATION_SIGN_LABEL = f"{STREAM_VERSION}/perturbation-sign"

DEFAULT_N = 100
"""CLAUDE.md P1.2: start with N = 100."""

DEFAULT_AMPLITUDE = 16
"""Perturbation size in uint8 levels (16/255). A starting point for P1.3, not a
measured or tuned value."""


@dataclass(frozen=True)
class TriggerSet:
    """A generated trigger set.

    Attributes:
        images: ``(N, H, W, C)`` uint8, the triggers `T`.
        base_images: ``(N, H, W, C)`` uint8, the unperturbed images the
            triggers were built from.
        base_indices: ``(N,)`` int64, index of each base image in the full
            50,000-image CIFAR-10 training set.
        signs: ``(N, H, W, C)`` int8 in {-1, +1}, the key-derived pattern
            before scaling and clipping.
        amplitude: perturbation size in uint8 levels.
        stream_version: the keygen label family the set was drawn from.
    """

    images: np.ndarray
    base_images: np.ndarray
    base_indices: np.ndarray
    signs: np.ndarray
    amplitude: int
    stream_version: str = STREAM_VERSION

    def __len__(self) -> int:
        return len(self.images)


def select_base_indices(key: bytes, pool: Sequence[int], n: int) -> np.ndarray:
    """Draw `n` distinct indices from `pool`, keyed by `key`.

    Partial Fisher-Yates over the sorted pool, so the first `m` indices do not
    depend on `n` and the result does not depend on the pool's order.

    Raises:
        ValueError: if `pool` has duplicates or negative entries, or if `n` is
            not in ``[1, len(pool)]``.
    """
    _check_count(n)
    items = sorted(int(i) for i in pool)
    if len(set(items)) != len(items):
        raise ValueError("pool contains duplicate indices")
    if items and items[0] < 0:
        raise ValueError("pool contains negative indices")
    if n > len(items):
        raise ValueError(f"cannot draw {n} distinct indices from a pool of {len(items)}")

    stream = KeyStream(key, BASE_INDEX_LABEL)
    for i in range(n):
        j = i + stream.randbelow(len(items) - i)
        items[i], items[j] = items[j], items[i]
    return np.asarray(items[:n], dtype=np.int64)


def perturbation_signs(key: bytes, n: int, shape: tuple[int, ...]) -> np.ndarray:
    """Return `n` key-derived {-1, +1} patterns of `shape`, as int8.

    Each pattern consumes ``ceil(prod(shape) / 8)`` stream bytes, so pattern
    ``i`` is the same whatever `n` is.
    """
    _check_count(n)
    size = int(np.prod(shape))
    if size <= 0:
        raise ValueError(f"shape must have a positive number of elements, got {shape}")
    per_trigger = (size + 7) // 8
    raw = np.frombuffer(KeyStream(key, PERTURBATION_SIGN_LABEL).read(n * per_trigger), np.uint8)
    bits = np.unpackbits(raw.reshape(n, per_trigger), axis=1, bitorder="big")[:, :size]
    return (bits.astype(np.int8) * 2 - 1).reshape((n, *shape))


def apply_perturbation(base: np.ndarray, signs: np.ndarray, amplitude: int) -> np.ndarray:
    """``clip(base + amplitude * signs, 0, 255)`` as uint8, in integer arithmetic."""
    _check_amplitude(amplitude)
    if base.dtype != np.uint8:
        raise TypeError(f"base images must be uint8, got {base.dtype}")
    if base.shape != signs.shape:
        raise ValueError(f"shape mismatch: base {base.shape} vs signs {signs.shape}")
    shifted = base.astype(np.int16) + np.int16(amplitude) * signs.astype(np.int16)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def generate_triggers(
    key: bytes,
    images: np.ndarray,
    pool: Sequence[int],
    *,
    n: int = DEFAULT_N,
    amplitude: int = DEFAULT_AMPLITUDE,
) -> TriggerSet:
    """Build a trigger set of size `n` from `images`, drawing bases from `pool`.

    Args:
        key: the 32-byte master key `K`.
        images: every candidate image, ``(M, H, W, C)`` uint8, indexed the way
            `pool` refers to them. For CIFAR-10 that is ``CIFAR10.data``.
        pool: indices into `images` that may be used as bases. For CIFAR-10,
            the 45,000-image training split.
        n: trigger set size.
        amplitude: perturbation size in uint8 levels, 1 to 255.
    """
    _check_amplitude(amplitude)
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 4:
        raise TypeError("images must be a 4-D uint8 numpy array (N, H, W, C)")
    indices = select_base_indices(key, pool, n)
    if indices.max() >= len(images):
        raise ValueError(f"pool index {int(indices.max())} is out of range for {len(images)} images")

    base = images[indices]
    signs = perturbation_signs(key, n, images.shape[1:])
    triggers = apply_perturbation(base, signs, amplitude)
    for array in (triggers, base, indices, signs):
        array.setflags(write=False)
    return TriggerSet(
        images=triggers,
        base_images=base,
        base_indices=indices,
        signs=signs,
        amplitude=amplitude,
    )


def generate_cifar10_triggers(
    key: bytes,
    root: Path | str = "data",
    *,
    n: int = DEFAULT_N,
    amplitude: int = DEFAULT_AMPLITUDE,
    download: bool = False,
) -> TriggerSet:
    """Build the trigger set from the CIFAR-10 training split.

    The pool is the 45,000-image split from `cifar10_split_indices`, the images
    `main_model` trains on. The attacker holdout is excluded by construction.
    """
    from torchvision.datasets import CIFAR10

    from src.data.cifar10 import cifar10_split_indices

    train_full = CIFAR10(str(root), train=True, download=download)
    train_idx, _ = cifar10_split_indices(len(train_full.data))
    return generate_triggers(key, train_full.data, train_idx, n=n, amplitude=amplitude)


def _check_count(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")


def _check_amplitude(amplitude: int) -> None:
    if not isinstance(amplitude, int) or isinstance(amplitude, bool):
        raise TypeError(f"amplitude must be an int, got {type(amplitude).__name__}")
    if not 1 <= amplitude <= 255:
        raise ValueError(f"amplitude must be in [1, 255], got {amplitude}")
