"""Dataset pipelines.

CLAUDE.md section 3 does not name this package, but 0.5 step 1 lists the data
pipeline among the things a `[GPU]` handoff must contain, and CIFAR-10 loading
does not belong in `src/utils/`.
"""

from src.data.cifar10 import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    HOLDOUT_SIZE,
    SPLIT_SEED,
    TRAIN_SIZE,
    TRANSFER_SETS,
    attacker_holdout_loaders,
    attacker_transfer_loaders,
    cifar10_loaders,
    cifar10_split_indices,
    denormalize,
)
from src.data.mnist import MNIST_MEAN, MNIST_STD, mnist_loaders

__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "HOLDOUT_SIZE",
    "MNIST_MEAN",
    "MNIST_STD",
    "SPLIT_SEED",
    "TRAIN_SIZE",
    "TRANSFER_SETS",
    "attacker_holdout_loaders",
    "attacker_transfer_loaders",
    "cifar10_loaders",
    "cifar10_split_indices",
    "denormalize",
    "mnist_loaders",
]
