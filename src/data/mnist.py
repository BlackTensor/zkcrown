"""MNIST loading for `zk_model` (P0.7).

Simpler than the CIFAR-10 pipeline on purpose:

- **No attacker holdout.** The holdout exists because `main_model` is attacked
  by fine-tuning (P4.5, P4.6). `zk_model` is never attacked; it only has
  inference proved over it (Phase 8). So it trains on all 60,000 training
  images.
- **No augmentation.** At ~6K parameters the model is not going to memorise
  MNIST, and a plain pipeline is one less thing that can differ between the
  PyTorch model and its EZKL circuit.

Evaluation is on the official 10,000-image test set, as with CIFAR-10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.utils.seeding import DEFAULT_SEED, seed_worker, torch_generator

# Per-channel statistics of the MNIST training set.
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

MNIST_TRAIN_SIZE = 60_000
MNIST_TEST_SIZE = 10_000
NUM_CLASSES = 10


def _transforms():
    """Built lazily so importing this module does not require torchvision."""
    from torchvision import transforms

    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])


def _smoke_datasets(seed: int) -> dict[str, Dataset]:
    """Tiny synthetic stand-ins shaped like normalised MNIST.

    Validates the training path in seconds without a download. Accuracy from
    a smoke run is meaningless by construction.
    """
    generator = torch.Generator().manual_seed(seed)
    datasets = {}
    for name, size in {"train": 512, "test": 256}.items():
        images = torch.randn(size, 1, 28, 28, generator=generator)
        labels = torch.randint(0, NUM_CLASSES, (size,), generator=generator)
        datasets[name] = TensorDataset(images, labels)
    return datasets


def mnist_datasets(
    root: Path | str = "data",
    *,
    download: bool = True,
    smoke: bool = False,
    seed: int = DEFAULT_SEED,
) -> dict[str, Dataset]:
    """Return the `train` (all 60,000) and `test` datasets."""
    if smoke:
        return _smoke_datasets(seed)

    from torchvision.datasets import MNIST

    root = str(root)
    return {
        "train": MNIST(root, train=True, download=download, transform=_transforms()),
        "test": MNIST(root, train=False, download=download, transform=_transforms()),
    }


def mnist_loaders(
    root: Path | str = "data",
    *,
    batch_size: int = 128,
    eval_batch_size: int | None = None,
    num_workers: int = 2,
    download: bool = True,
    smoke: bool = False,
    seed: int = DEFAULT_SEED,
    pin_memory: bool | None = None,
) -> dict[str, DataLoader]:
    """DataLoaders for `train` and `test`.

    Seeded shuffling and non-persistent workers, for the same reason as
    `cifar10_loaders`: `fit` reseeds the shuffle generator per epoch so a
    resumed run matches an uninterrupted one.
    """
    datasets = mnist_datasets(root, download=download, smoke=smoke, seed=seed)
    eval_batch_size = eval_batch_size or max(batch_size, 512)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    shared: dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}
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
        "test": DataLoader(datasets["test"], batch_size=eval_batch_size, shuffle=False, **shared),
    }
