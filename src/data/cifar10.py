"""CIFAR-10 loading for `main_model`.

One design decision here matters beyond P0.5, so it is made once and fixed:

**The 50,000 CIFAR-10 training images are split 45,000 / 5,000.** `main_model`
trains on the 45,000. The 5,000 are an *attacker holdout*, reserved for the
P4.5 fine-tuning attack and the P4.6 prune-then-fine-tune attack.

The reason is that a fine-tuning attack carried out on the data the model was
already trained on is not a realistic threat model, and it would understate how
much fine-tuning removes a watermark. The thief has their own data. Carving the
holdout out now costs a little baseline accuracy and avoids retraining the
clean model later, which would mean burning a second GPU handoff.

The split is driven by `SPLIT_SEED`, a fixed constant, **not** by the run seed.
Every phase must see the same holdout no matter what seed its experiment runs
under, or the P4.5 attacker would be training on data the model has seen.

The official 10,000-image test set is used for evaluation throughout, so the
P0.6 baseline and every later accuracy figure are measured on the same data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from src.utils.seeding import seed_worker, torch_generator

# Per-channel statistics of the CIFAR-10 training set.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

TRAIN_SIZE = 45_000
HOLDOUT_SIZE = 5_000
SPLIT_SEED = 20260913
"""Fixed. Changing this invalidates every attack result that used the holdout."""

NUM_CLASSES = 10
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _transforms(augment: bool):
    """Built lazily so importing this module does not require torchvision."""
    from torchvision import transforms

    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if not augment:
        return transforms.Compose([transforms.ToTensor(), normalize])
    return transforms.Compose(
        [
            # Standard CIFAR-10 augmentation. Nothing exotic: the point of this
            # project is the watermark, not squeezing out accuracy.
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo normalisation, giving values back in [0, 1].

    The trigger work (P1.2) needs to reason about perturbations in pixel space
    rather than normalised space, and the trigger visualisation (P1.3) needs
    displayable images.
    """
    mean = torch.tensor(CIFAR10_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x.device).view(1, 3, 1, 1)
    if x.dim() == 3:
        mean, std = mean.squeeze(0), std.squeeze(0)
    return x * std + mean


def cifar10_split_indices(total: int = 50_000) -> tuple[list[int], list[int]]:
    """Deterministic (train, attacker_holdout) index split.

    Uses `SPLIT_SEED` and its own generator, so the result does not depend on
    the run seed or on how many random numbers anything else has drawn.
    """
    if total < TRAIN_SIZE + HOLDOUT_SIZE:
        raise ValueError(
            f"need at least {TRAIN_SIZE + HOLDOUT_SIZE} examples to split, got {total}"
        )
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(total, generator=generator).tolist()
    return permutation[:TRAIN_SIZE], permutation[TRAIN_SIZE : TRAIN_SIZE + HOLDOUT_SIZE]


def _smoke_datasets(seed: int) -> dict[str, Dataset]:
    """Tiny synthetic stand-ins, shaped like normalised CIFAR-10.

    Lets the whole training path be validated in seconds without downloading
    170MB, which is how the handoff zip gets checked before a real run starts.
    Accuracy from a smoke run is meaningless by construction.
    """
    generator = torch.Generator().manual_seed(seed)
    sizes = {"train": 512, "holdout": 128, "test": 256}
    datasets = {}
    for name, size in sizes.items():
        images = torch.randn(size, 3, 32, 32, generator=generator)
        labels = torch.randint(0, NUM_CLASSES, (size,), generator=generator)
        datasets[name] = TensorDataset(images, labels)
    return datasets


def cifar10_datasets(
    root: Path | str = "data",
    *,
    augment: bool = True,
    download: bool = True,
    smoke: bool = False,
    seed: int = SPLIT_SEED,
) -> dict[str, Dataset]:
    """Return the `train`, `holdout` and `test` datasets.

    `train` is augmented when `augment` is True; `holdout` and `test` never
    are, so evaluation is deterministic.
    """
    if smoke:
        return _smoke_datasets(seed)

    from torchvision.datasets import CIFAR10

    root = str(root)
    train_full = CIFAR10(root, train=True, download=download, transform=_transforms(augment))
    holdout_full = CIFAR10(root, train=True, download=False, transform=_transforms(False))
    test = CIFAR10(root, train=False, download=download, transform=_transforms(False))

    train_idx, holdout_idx = cifar10_split_indices(len(train_full))
    return {
        "train": Subset(train_full, train_idx),
        # Same underlying images, un-augmented: this is the attacker's data.
        "holdout": Subset(holdout_full, holdout_idx),
        "test": test,
    }


def cifar10_loaders(
    root: Path | str = "data",
    *,
    batch_size: int = 128,
    eval_batch_size: int | None = None,
    num_workers: int = 2,
    augment: bool = True,
    download: bool = True,
    smoke: bool = False,
    seed: int = SPLIT_SEED,
    pin_memory: bool | None = None,
) -> dict[str, DataLoader]:
    """DataLoaders for `train`, `holdout` and `test`.

    Shuffling uses an explicitly seeded generator and workers are seeded via
    `seed_worker`, so epoch order is reproducible rather than dependent on
    whatever else consumed the global RNG first.
    """
    datasets = cifar10_datasets(
        root, augment=augment, download=download, smoke=smoke, seed=seed
    )
    eval_batch_size = eval_batch_size or max(batch_size, 256)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    shared: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=torch_generator(seed),
            worker_init_fn=seed_worker,
            **shared,
        ),
        "holdout": DataLoader(
            datasets["holdout"], batch_size=eval_batch_size, shuffle=False, **shared
        ),
        "test": DataLoader(
            datasets["test"], batch_size=eval_batch_size, shuffle=False, **shared
        ),
    }
