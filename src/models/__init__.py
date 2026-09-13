"""Model architectures: main_model (CIFAR-10) and zk_model (MNIST).

The two are separate on purpose (CLAUDE.md 2.2). `main_model` carries the
watermark and takes the attacks; `zk_model` is the deliberately tiny network
that goes through EZKL.
"""

from src.models.main_model import (
    CIFAR10_CLASSES,
    CIFAR10_INPUT_SHAPE,
    MainModel,
    count_parameters,
    main_model,
)
from src.models.zk_model import (
    MNIST_CLASSES,
    MNIST_INPUT_SHAPE,
    ZK_PARAM_BUDGET,
    ZKModel,
    activation_elements,
    zk_model,
)

__all__ = [
    "CIFAR10_CLASSES",
    "CIFAR10_INPUT_SHAPE",
    "MNIST_CLASSES",
    "MNIST_INPUT_SHAPE",
    "MainModel",
    "ZKModel",
    "ZK_PARAM_BUDGET",
    "activation_elements",
    "count_parameters",
    "main_model",
    "zk_model",
]
