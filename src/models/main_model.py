"""`main_model`: the CIFAR-10 CNN that carries the watermark (P0.4).

This is the model all watermarking science and the whole attack lab run on
(CLAUDE.md 2.2). It is deliberately *not* the model that goes through EZKL --
that is `zk_model`, a much smaller MNIST network, because ZK proving cost scales
brutally with model size. The two must not be conflated.

Architecture: a small VGG-style stack, three blocks of
(conv-BN-ReLU, conv-BN-ReLU, maxpool) at widths 32/64/128, then a linear
classifier. Roughly 308K parameters at the default width, which sits inside the
"a few hundred thousand parameters" budget in section 2.2 and trains to a
reasonable CIFAR-10 baseline well inside the 2-hour Colab ceiling (section 2.1).

Why this shape, given what comes later:

- **Structured pruning (P4.3)** removes whole channels or filters, so the
  network is plain stacked convolutions with no residual connections. Skip
  connections force channel counts to agree across a block boundary and make
  channel removal fiddly for no scientific gain here.
- **Quantization (P4.4)** fuses conv-BN-ReLU triples. Every activation is an
  `nn.ReLU` *module* rather than a call to `F.relu`, because
  `torch.ao.quantization.fuse_modules` can only see modules. This costs nothing
  now and avoids rewriting the architecture mid-attack-suite.
- **Distillation (P4.7)** needs a smaller student of the same family, so
  `width` is a constructor argument rather than hardcoded.
- **Weight watermarking (P3.x)** spreads a signature across many parameters,
  which the ~287K convolutional weights provide.
- **Fingerprinting (P5.1)** hashes a canonical serialisation of the weights.
  Module construction order is fixed and deterministic, so `state_dict()` key
  order is stable across save and reload.
"""

from __future__ import annotations

import torch
from torch import nn

CIFAR10_CLASSES = 10
CIFAR10_INPUT_SHAPE = (3, 32, 32)


def _conv_block(in_channels: int, out_channels: int) -> list[nn.Module]:
    """conv-BN-ReLU, in that order, as separate modules so it stays fusable."""
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
    ]


class MainModel(nn.Module):
    """Small VGG-style CNN for CIFAR-10.

    Args:
        num_classes: output logits. 10 for CIFAR-10.
        width: channel count of the first block. Later blocks are 2x and 4x it.
            The default 32 gives ~308K parameters; 16 gives ~78K, which is the
            intended distillation student (P4.7).
        dropout: applied to the flattened features before the classifier. Set
            0.0 to disable.
        in_channels: 3 for RGB.

    Forward returns raw logits, not probabilities. Distillation (P4.7) and the
    trigger-response tests (P2.x) both need logits, and `CrossEntropyLoss`
    expects them.
    """

    def __init__(
        self,
        num_classes: int = CIFAR10_CLASSES,
        width: int = 32,
        dropout: float = 0.3,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.num_classes = num_classes
        self.width = width
        self.in_channels = in_channels

        widths = (width, width * 2, width * 4)
        layers: list[nn.Module] = []
        channels = in_channels
        for out_channels in widths:
            layers += _conv_block(channels, out_channels)
            layers += _conv_block(out_channels, out_channels)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            channels = out_channels
        self.features = nn.Sequential(*layers)

        # Three 2x2 pools take 32x32 down to 4x4.
        self.flat_features = widths[-1] * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.flat_features, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """He init for conv, standard for BN and linear.

        Explicit rather than left to torch's defaults so that a given seed
        produces the same initial weights across torch versions that change
        their default init.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def extra_repr(self) -> str:
        return f"width={self.width}, num_classes={self.num_classes}"


def main_model(**kwargs) -> MainModel:
    """Build `main_model` with the project defaults. See `MainModel` for args."""
    return MainModel(**kwargs)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Parameter counts, for the Results Ledger (section 8.2).

    Returns `total`, `trainable`, and `conv_and_linear_weights` -- the last
    being the pool the weight watermark can spread into (P3.2), which excludes
    biases and BatchNorm statistics.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embeddable = sum(
        module.weight.numel()
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    )
    return {
        "total": total,
        "trainable": trainable,
        "conv_and_linear_weights": embeddable,
    }
