"""The trigger bundle: triggers plus targets, packaged for a training run (P2.3).

Training needs the trigger images `T` (P1.2) and their target classes (P2.2).
Both are derived from `K`. The bundle carries them to Colab so that **`K`
itself never leaves the owner's machine**: the bundle is built locally by
`experiments/p2_3_make_trigger_bundle.py`, and only the bundle is uploaded.

The bundle is still secret. Anyone holding it knows the trigger set and could
use it to probe for, or train against, the watermark. It lives in `secrets/`
locally, which is gitignored, and in a `secrets/` folder on Drive.

Digest
------
`TriggerBundle.digest` is a SHA-256 over a canonical encoding::

    SHA-256( DOMAIN
             || u32_be(len(header)) || header
             || images      uint8, (N, H, W, C), C order
             || base_indices int64 big-endian
             || base_labels  int64 big-endian
             || targets      int64 big-endian )

where ``header`` is compact, key-sorted JSON holding the shape, amplitude,
class count, key kind and stream versions. The digest names one exact trigger
set. Result records store it, so the owner can later check, by regenerating
from `K`, that a model was trained on the triggers `K` really gives. It is
safe to publish: the images carry far more entropy than a brute-force search
could cover, and the digest reveals nothing else.

`key_kind` is ``"owner"`` for a bundle made from the real `K` and ``"test"``
for anything else. The P2.3 entry point refuses to train a real run on a test
bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.watermark.responses import TriggerResponses
from src.watermark.triggers import TriggerSet

BUNDLE_VERSION = "trigger-bundle/v1"
DOMAIN = b"zk-crown/trigger-bundle/v1\x00"
KEY_KINDS = ("owner", "test")


@dataclass(frozen=True)
class TriggerBundle:
    """Everything a training run needs about the watermark, and nothing about `K`.

    Attributes:
        images: ``(N, H, W, C)`` uint8 triggers.
        base_indices: ``(N,)`` int64, index of each base image in the full
            training set.
        base_labels: ``(N,)`` int64 dataset labels of the base images.
        targets: ``(N,)`` int64 target classes, never equal to `base_labels`.
        amplitude: perturbation size in uint8 levels.
        num_classes: number of classes.
        key_kind: ``"owner"`` or ``"test"``.
        trigger_version, response_version: the stream families used.
    """

    images: np.ndarray
    base_indices: np.ndarray
    base_labels: np.ndarray
    targets: np.ndarray
    amplitude: int
    num_classes: int
    key_kind: str
    trigger_version: str
    response_version: str
    version: str = BUNDLE_VERSION

    def __post_init__(self) -> None:
        if self.images.dtype != np.uint8 or self.images.ndim != 4:
            raise ValueError("images must be (N, H, W, C) uint8")
        n = len(self.images)
        if n == 0:
            raise ValueError("a bundle needs at least one trigger")
        for name in ("base_indices", "base_labels", "targets"):
            array = getattr(self, name)
            if array.shape != (n,) or not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"{name} must be ({n},) integers")
        if self.key_kind not in KEY_KINDS:
            raise ValueError(f"key_kind must be one of {KEY_KINDS}, got {self.key_kind!r}")
        if not 1 <= int(self.amplitude) <= 255:
            raise ValueError("amplitude must be in [1, 255]")
        if int(self.num_classes) < 2:
            raise ValueError("num_classes must be at least 2")
        if self.targets.min() < 0 or self.targets.max() >= self.num_classes:
            raise ValueError("targets out of range")
        if self.base_labels.min() < 0 or self.base_labels.max() >= self.num_classes:
            raise ValueError("base_labels out of range")
        if np.any(self.targets == self.base_labels):
            raise ValueError("a target equals its base label, which P2.2 never produces")
        if len(set(self.base_indices.tolist())) != n:
            raise ValueError("base_indices contain duplicates")

    def __len__(self) -> int:
        return len(self.images)

    def __repr__(self) -> str:
        return (
            f"TriggerBundle(n={len(self)}, amplitude={self.amplitude}, key_kind={self.key_kind!r}, "
            f"digest={self.digest()[:16]}...)"
        )

    def header(self) -> dict:
        return {
            "version": self.version,
            "shape": list(self.images.shape),
            "amplitude": int(self.amplitude),
            "num_classes": int(self.num_classes),
            "key_kind": self.key_kind,
            "trigger_version": self.trigger_version,
            "response_version": self.response_version,
        }

    def digest(self) -> str:
        """SHA-256 hex digest of the canonical encoding. See the module docstring."""
        header = json.dumps(self.header(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        h = hashlib.sha256(DOMAIN)
        h.update(struct.pack(">I", len(header)) + header)
        h.update(np.ascontiguousarray(self.images).tobytes())
        for array in (self.base_indices, self.base_labels, self.targets):
            h.update(np.asarray(array, dtype=">i8").tobytes())
        return h.hexdigest()


def make_bundle(trigger_set: TriggerSet, responses: TriggerResponses, *, key_kind: str) -> TriggerBundle:
    """Combine a trigger set and its responses, built from the same key."""
    if len(trigger_set) != len(responses):
        raise ValueError(f"{len(trigger_set)} triggers but {len(responses)} responses")
    return TriggerBundle(
        images=np.array(trigger_set.images),
        base_indices=np.array(trigger_set.base_indices, dtype=np.int64),
        base_labels=np.array(responses.base_labels, dtype=np.int64),
        targets=np.array(responses.targets, dtype=np.int64),
        amplitude=int(trigger_set.amplitude),
        num_classes=int(responses.num_classes),
        key_kind=key_kind,
        trigger_version=trigger_set.stream_version,
        response_version=responses.version,
    )


def save_bundle(bundle: TriggerBundle, path: Path | str) -> str:
    """Write `bundle` to an ``.npz`` atomically. Returns its digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            images=bundle.images,
            base_indices=bundle.base_indices.astype(np.int64),
            base_labels=bundle.base_labels.astype(np.int64),
            targets=bundle.targets.astype(np.int64),
            header=np.array(json.dumps(bundle.header(), sort_keys=True)),
            digest=np.array(bundle.digest()),
        )
    os.replace(tmp, path)
    return bundle.digest()


def load_bundle(path: Path | str, expected_digest: str | None = None) -> TriggerBundle:
    """Load a bundle and check it against its stored digest, and `expected_digest` if given.

    Raises:
        ValueError: if the file is internally inconsistent or the digest differs.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
        if header.get("version") != BUNDLE_VERSION:
            raise ValueError(f"unsupported bundle version {header.get('version')!r}")
        bundle = TriggerBundle(
            images=data["images"],
            base_indices=data["base_indices"],
            base_labels=data["base_labels"],
            targets=data["targets"],
            amplitude=header["amplitude"],
            num_classes=header["num_classes"],
            key_kind=header["key_kind"],
            trigger_version=header["trigger_version"],
            response_version=header["response_version"],
        )
        stored = str(data["digest"])
    if list(bundle.images.shape) != header["shape"]:
        raise ValueError("bundle images do not match the header shape")
    digest = bundle.digest()
    if digest != stored:
        raise ValueError("bundle contents do not match the digest stored in the file")
    if expected_digest is not None and digest != expected_digest.lower():
        raise ValueError(f"bundle digest {digest} does not match the expected {expected_digest}")
    return bundle


def check_against_dataset(
    bundle: TriggerBundle, images: np.ndarray, labels: Sequence[int] | np.ndarray, pool: Sequence[int]
) -> None:
    """Check that a bundle is consistent with the dataset it claims to come from.

    Without `K` this cannot prove the triggers are key-derived. It does catch
    the wrong dataset, the wrong split, a corrupted file, and triggers that are
    not within `amplitude` of their base images.

    Raises:
        ValueError: on any inconsistency.
    """
    labels = np.asarray(labels)
    if bundle.images.shape[1:] != images.shape[1:]:
        raise ValueError(f"trigger shape {bundle.images.shape[1:]} != dataset shape {images.shape[1:]}")
    if int(bundle.base_indices.max()) >= len(images) or int(bundle.base_indices.min()) < 0:
        raise ValueError("bundle base indices are out of range for this dataset")
    outside = set(bundle.base_indices.tolist()) - set(int(i) for i in pool)
    if outside:
        raise ValueError(f"{len(outside)} base images are outside the allowed pool (training split)")
    if not np.array_equal(labels[bundle.base_indices], bundle.base_labels):
        raise ValueError("bundle base labels do not match the dataset labels")
    delta = bundle.images.astype(np.int16) - images[bundle.base_indices].astype(np.int16)
    if int(np.abs(delta).max()) > bundle.amplitude:
        raise ValueError("triggers are not within amplitude of their base images")
