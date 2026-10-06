"""Third-party CIFAR-10 models for the auditor's unrelated-model test (P9.4).

Source: ``chenyaofo/pytorch-cifar-models`` on GitHub, loaded through
``torch.hub`` at a pinned commit. Licence BSD-3-Clause (Copyright (c) 2021,
chenyaofo), read from the repository's ``LICENSE`` and the GitHub licence API
on 2026-10-06. The weights are release assets of that repository; the
repository states no separate licence for them.

`CATALOG` lists every CIFAR-10 model the repository publishes except the ViT
family (excluded by the owner): ResNet, VGG-BN, MobileNetV2, ShuffleNetV2 and
RepVGG. For each it holds the release URL, the asset size GitHub reports, and
the top-1 accuracy the repository's README publishes. The files are downloaded
into ``data/torch_hub/`` (gitignored) and never committed. `load_third_party`
checks the downloaded file's size against `CATALOG`, checks that its SHA-256
starts with the 8 hex digits in its file name (the torch.hub naming
convention), and returns the full SHA-256 for the record.

The models were trained by someone else, on CIFAR-10, without `K`. They are
independent of the owner's key, so the auditor must find no evidence in them.
Their normalisation (mean ``0.4914, 0.4822, 0.4465``, std ``0.2023, 0.1994,
0.2010``) is from ``conf/cifar10.conf`` in ``chenyaofo/image-classification-codebase``,
the training code the repository points to; `adapter_for` wraps a model so it
accepts this project's normalised inputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from src.models.adapters import InputAdapter

HUB_REPO = "chenyaofo/pytorch-cifar-models"
HUB_COMMIT = "786c16252c0fc58ee9adac063f8337cc4a7a497a"
"""``master`` on 2026-10-06 (committed 2025-05-17)."""
LICENSE = "BSD-3-Clause"
LICENSE_URL = f"https://github.com/{HUB_REPO}/blob/{HUB_COMMIT}/LICENSE"
THIRD_PARTY_MEAN = (0.4914, 0.4822, 0.4465)
THIRD_PARTY_STD = (0.2023, 0.1994, 0.2010)
NORMALISATION_SOURCE = "https://github.com/chenyaofo/image-classification-codebase/blob/master/conf/cifar10.conf"
RELEASES = f"https://github.com/{HUB_REPO}/releases/download"


@dataclass(frozen=True)
class ThirdPartyModel:
    name: str
    """torch.hub entry point, e.g. ``cifar10_resnet20``."""
    release: str
    file: str
    size_bytes: int
    published_top1: float
    """Percent, from the repository README."""

    @property
    def url(self) -> str:
        return f"{RELEASES}/{self.release}/{self.file}"

    @property
    def hash_prefix(self) -> str:
        return self.file.rsplit("-", 1)[1].split(".")[0]


def _m(arch: str, release: str, digest: str, size: int, top1: float) -> ThirdPartyModel:
    return ThirdPartyModel(f"cifar10_{arch}", release, f"cifar10_{arch}-{digest}.pt", size, top1)


CATALOG: tuple[ThirdPartyModel, ...] = (
    _m("resnet20", "resnet", "4118986f", 1_139_055, 92.60),
    _m("resnet32", "resnet", "ef93fc4d", 1_944_567, 93.53),
    _m("resnet44", "resnet", "2a3cabcb", 2_750_015, 94.01),
    _m("resnet56", "resnet", "187c023a", 3_555_463, 94.37),
    _m("vgg11_bn", "vgg", "eaeebf42", 39_068_509, 92.79),
    _m("vgg13_bn", "vgg", "c01e4a43", 39_814_235, 94.00),
    _m("vgg16_bn", "vgg", "6ee7ea24", 61_080_472, 94.16),
    _m("vgg19_bn", "vgg", "57191229", 82_346_709, 93.91),
    _m("mobilenetv2_x0_5", "mobilenetv2", "ca14ced9", 2_986_233, 92.88),
    _m("mobilenetv2_x0_75", "mobilenetv2", "a53c314e", 5_688_121, 93.72),
    _m("mobilenetv2_x1_0", "mobilenetv2", "fe6a5b48", 9_193_273, 93.79),
    _m("mobilenetv2_x1_4", "mobilenetv2", "3bbbd6e2", 17_635_961, 94.22),
    _m("shufflenetv2_x0_5", "shufflenetv2", "1308b4e9", 1_554_833, 90.13),
    _m("shufflenetv2_x1_0", "shufflenetv2", "98807be3", 5_230_673, 92.98),
    _m("shufflenetv2_x1_5", "shufflenetv2", "296694dd", 10_164_241, 93.55),
    _m("shufflenetv2_x2_0", "shufflenetv2", "ec31611c", 21_707_473, 93.81),
    _m("repvgg_a0", "repvgg", "ef08a50e", 31_582_451, 94.39),
    _m("repvgg_a1", "repvgg", "38d2431b", 51_541_043, 94.89),
    _m("repvgg_a2", "repvgg", "09488915", 107_555_782, 94.98),
)


def catalog_bytes() -> int:
    return sum(m.size_bytes for m in CATALOG)


def check_weights_file(entry: ThirdPartyModel, path: Path) -> str:
    """Size and hash-prefix check of a downloaded weights file. Returns its full SHA-256."""
    path = Path(path)
    size = path.stat().st_size
    if size != entry.size_bytes:
        raise ValueError(f"{path.name}: {size} bytes, the release lists {entry.size_bytes}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(entry.hash_prefix):
        raise ValueError(f"{path.name}: SHA-256 {digest[:8]}... does not start with {entry.hash_prefix}")
    return digest


def load_third_party(entry: ThirdPartyModel, hub_dir: Path) -> tuple[nn.Module, dict]:
    """The pretrained model in eval mode, and a record of where it came from. Downloads on first use."""
    import torch

    hub_dir = Path(hub_dir)
    torch.hub.set_dir(str(hub_dir))
    model = torch.hub.load(f"{HUB_REPO}:{HUB_COMMIT}", entry.name, pretrained=True, trust_repo=True,
                           verbose=False)
    path = hub_dir / "checkpoints" / entry.file
    if not path.is_file():
        raise FileNotFoundError(f"torch.hub did not leave {entry.file} under {hub_dir / 'checkpoints'}")
    sha256 = check_weights_file(entry, path)
    return model.eval(), {
        "name": entry.name, "source_url": entry.url, "hub_repo": HUB_REPO, "hub_commit": HUB_COMMIT,
        "license": LICENSE, "license_url": LICENSE_URL, "file": entry.file, "size_bytes": entry.size_bytes,
        "sha256": sha256, "published_top1_percent": entry.published_top1,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def adapter_for(model: nn.Module, source_mean, source_std) -> InputAdapter:
    """Wrap a third-party model so it takes inputs normalised with (`source_mean`, `source_std`)."""
    return InputAdapter(model, source_mean=source_mean, source_std=source_std,
                        target_mean=THIRD_PARTY_MEAN, target_std=THIRD_PARTY_STD).eval()
