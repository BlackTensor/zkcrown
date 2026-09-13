"""`zk_model`: the deliberately tiny MNIST CNN that goes through EZKL (P0.7).

This is **not** the model the watermarking science and the attack lab run on.
That is `main_model` (CIFAR-10, ~308K parameters). `zk_model` exists only so
there is a network small enough to prove inference over on Colab free tier
(Track B, Phase 8). The two must not be conflated (CLAUDE.md 2.2).

Architecture, at the default widths (8, 16, 16), which gives 6,138 parameters:

    input 1x28x28
    conv 3x3 stride 2 ->  8x14x14, ReLU
    conv 3x3 stride 2 -> 16x7x7,   ReLU
    conv 3x3 stride 2 -> 16x4x4,   ReLU
    flatten 256 -> linear -> 10 logits

Why this shape, given Phase 8:

- **Parameter count is not the only cost.** In a zkML circuit every
  non-linear activation has to be constrained, so the number of ReLU *output
  elements* drives circuit size alongside the weights. Downsampling in the
  first layer with a stride-2 convolution keeps that at 2,608 elements. The
  same widths at stride 1 would give tens of thousands. `activation_elements`
  reports the exact count.
- **No max-pool.** Strided convolutions do all the downsampling, so the graph
  has no max operations, which need comparisons in a circuit.
- **No BatchNorm, no dropout.** Neither is needed at this size. Leaving them
  out means train and eval mode compute the same function and the exported
  ONNX graph is only Conv, Relu, Flatten and Gemm-shaped ops. Fewer op types
  means fewer places for EZKL's ONNX support to disagree with PyTorch (P8.6).
- **Every ReLU is an `nn.ReLU` module**, the same convention as `main_model`,
  so activation counting via hooks sees all of them.
- **`widths` is a constructor argument.** P8.8 says to shrink the model and
  re-run if it blows the RAM budget. It should not need a new class.

None of these circuit-cost claims is measured yet. P8.3 and P8.4 record the
real row counts, RAM and timings.
"""

from __future__ import annotations

import torch
from torch import nn

MNIST_CLASSES = 10
MNIST_INPUT_SHAPE = (1, 28, 28)
ZK_PARAM_BUDGET = 10_000
"""CLAUDE.md 2.2 target: under 10K parameters."""


def _conv_out(size: int) -> int:
    """Spatial size after a 3x3, stride-2, padding-1 convolution."""
    return (size + 2 * 1 - 3) // 2 + 1


class ZKModel(nn.Module):
    """Minimal strided-conv CNN for MNIST.

    Args:
        num_classes: output logits. 10 for MNIST.
        widths: output channels of the three convolutions.
        in_channels: 1 for greyscale.
        input_size: height and width of the square input. 28 for MNIST.

    Forward returns raw logits, like `main_model`.
    """

    def __init__(
        self,
        num_classes: int = MNIST_CLASSES,
        widths: tuple[int, ...] = (8, 16, 16),
        in_channels: int = 1,
        input_size: int = 28,
    ) -> None:
        super().__init__()

        widths = tuple(widths)
        if not widths or any(w < 1 for w in widths):
            raise ValueError(f"widths must be a non-empty tuple of positive ints, got {widths}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.num_classes = num_classes
        self.widths = widths
        self.in_channels = in_channels

        layers: list[nn.Module] = []
        channels, size = in_channels, input_size
        for out_channels in widths:
            layers += [
                nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=False),
            ]
            channels, size = out_channels, _conv_out(size)
        self.features = nn.Sequential(*layers)

        self.flat_features = channels * size * size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """He (fan-in) init for conv, small normal for the classifier.

        Fan-in rather than `main_model`'s fan-out: with no BatchNorm to
        rescale activations, fan-in is the variant that keeps the forward
        signal from shrinking or growing layer to layer. The small classifier
        init keeps the initial logits near zero. P0.5 showed that large early
        logits in this project's SGD setup can kill ReLUs for good.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def extra_repr(self) -> str:
        return f"widths={self.widths}, num_classes={self.num_classes}"


def zk_model(**kwargs) -> ZKModel:
    """Build `zk_model` with the project defaults. See `ZKModel` for args."""
    return ZKModel(**kwargs)


@torch.no_grad()
def activation_elements(model: nn.Module, input_shape: tuple[int, ...] = MNIST_INPUT_SHAPE) -> int:
    """Total number of ReLU output elements for one input.

    A rough proxy for how many non-linearities a circuit has to constrain.
    Measured with forward hooks on a single dummy input, so it counts what
    the model actually computes rather than what the docstring says.
    """
    total = 0

    def hook(_module, _inputs, output):
        nonlocal total
        total += output[0].numel()

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.ReLU)]
    was_training = model.training
    try:
        model.eval()
        model(torch.zeros(1, *input_shape))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return total
