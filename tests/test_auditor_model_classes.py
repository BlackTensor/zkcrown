"""P9.4: input adapters, the third-party catalogue, and the model-class audit script's fixed choices."""

from __future__ import annotations

import hashlib
import sys

import pytest
import torch
from torch import nn

from src.data import CIFAR10_MEAN, CIFAR10_STD, MNIST_MEAN, MNIST_STD
from src.models.adapters import InputAdapter
from src.models.third_party import (
    CATALOG,
    HUB_COMMIT,
    THIRD_PARTY_MEAN,
    THIRD_PARTY_STD,
    ThirdPartyModel,
    adapter_for,
    catalog_bytes,
    check_weights_file,
)
from src.models.zk_model import ZKModel
from src.utils.results import repo_root

sys.path.insert(0, str(repo_root() / "experiments"))


class Capture(nn.Module):
    def __init__(self, out_classes: int = 10):
        super().__init__()
        self.seen, self.out_classes = None, out_classes

    def forward(self, x):
        self.seen = x.detach().clone()
        return torch.zeros(len(x), self.out_classes)


def normalised(pixels, mean, std):
    return (pixels - torch.tensor(mean).view(1, -1, 1, 1)) / torch.tensor(std).view(1, -1, 1, 1)


def test_adapter_renormalises_to_the_target_statistics():
    pixels = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    inner = Capture()
    adapter = adapter_for(inner, CIFAR10_MEAN, CIFAR10_STD)
    adapter(normalised(pixels, CIFAR10_MEAN, CIFAR10_STD))
    assert torch.allclose(inner.seen, normalised(pixels, THIRD_PARTY_MEAN, THIRD_PARTY_STD), atol=1e-5)


def test_adapter_with_equal_statistics_is_the_identity():
    x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    inner = Capture()
    InputAdapter(inner, source_mean=CIFAR10_MEAN, source_std=CIFAR10_STD, target_mean=CIFAR10_MEAN,
                 target_std=CIFAR10_STD)(x)
    assert torch.allclose(inner.seen, x, atol=1e-5)


def test_adapter_grayscale_and_resize_for_zk_model():
    pixels = torch.rand(3, 3, 32, 32, generator=torch.Generator().manual_seed(2))
    inner = Capture()
    adapter = InputAdapter(inner, source_mean=CIFAR10_MEAN, source_std=CIFAR10_STD, target_mean=MNIST_MEAN,
                           target_std=MNIST_STD, grayscale=True, size=28)
    adapter(normalised(pixels, CIFAR10_MEAN, CIFAR10_STD))
    assert inner.seen.shape == (3, 1, 28, 28)
    seen_pixels = inner.seen * MNIST_STD[0] + MNIST_MEAN[0]
    assert float(seen_pixels.min()) >= -1e-5 and float(seen_pixels.max()) <= 1 + 1e-5


def test_adapter_feeds_a_real_zk_model():
    adapter = InputAdapter(ZKModel().eval(), source_mean=CIFAR10_MEAN, source_std=CIFAR10_STD,
                           target_mean=MNIST_MEAN, target_std=MNIST_STD, grayscale=True, size=28).eval()
    assert adapter(torch.zeros(5, 3, 32, 32)).shape == (5, 10)


def test_adapter_has_no_state_of_its_own():
    inner = nn.Linear(2, 2)
    adapter = adapter_for(inner, CIFAR10_MEAN, CIFAR10_STD)
    assert set(adapter.state_dict()) == {f"model.{k}" for k in inner.state_dict()}


def test_catalog_is_the_nineteen_non_vit_cifar10_models():
    names = [m.name for m in CATALOG]
    assert len(names) == 19 and len(set(names)) == 19
    assert all(n.startswith("cifar10_") for n in names)
    assert not any("vit" in n for n in names)
    assert {m.release for m in CATALOG} == {"resnet", "vgg", "mobilenetv2", "shufflenetv2", "repvgg"}
    assert catalog_bytes() == 496_539_109
    assert len(HUB_COMMIT) == 40


def test_catalog_urls_and_hash_prefixes():
    for m in CATALOG:
        assert m.url == f"https://github.com/chenyaofo/pytorch-cifar-models/releases/download/{m.release}/{m.file}"
        assert len(m.hash_prefix) == 8 and int(m.hash_prefix, 16) >= 0
        assert 85.0 < m.published_top1 < 100.0


def test_weights_file_check(tmp_path):
    data = b"not really weights"
    digest = hashlib.sha256(data).hexdigest()
    entry = ThirdPartyModel("cifar10_x", "x", f"cifar10_x-{digest[:8]}.pt", len(data), 90.0)
    path = tmp_path / entry.file
    path.write_bytes(data)
    assert check_weights_file(entry, path) == digest
    wrong_size = ThirdPartyModel("cifar10_x", "x", entry.file, len(data) + 1, 90.0)
    with pytest.raises(ValueError, match="bytes"):
        check_weights_file(wrong_size, path)
    wrong_hash = ThirdPartyModel("cifar10_x", "x", "cifar10_x-00000000.pt", len(data), 90.0)
    with pytest.raises(ValueError, match="does not start"):
        check_weights_file(wrong_hash, path)


def test_script_choices_fixed_before_the_run():
    import p9_4_auditor_model_classes as script

    assert len(script.FRESH_SEEDS) == 20 and len(set(script.FRESH_SEEDS)) == 20
    assert 1337 not in script.FRESH_SEEDS and 20261005 not in script.FRESH_SEEDS
    assert script.ACCURACY_TOLERANCE_PP == 1.0
    assert sum(script.EXPECTED_PHASE4_GRADES.values()) == 85
    assert "1,000 wrong keys" in script.WEIGHT_FPR_BASIS and "fresh" in script.WEIGHT_FPR_BASIS
