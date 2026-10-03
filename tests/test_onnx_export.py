"""P8.1: zk_model ONNX export and the ONNX Runtime vs PyTorch comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from src.models.zk_model import zk_model  # noqa: E402
from src.zk.ezkl.onnx_export import compare, export_onnx, onnx_summary, sha256_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED = REPO_ROOT / "results" / "zk" / "p8.1" / "zk_model.onnx"


def seeded_model() -> torch.nn.Module:
    """Seeded, with distinct non-zero biases (identical zero biases get merged by constant folding)."""
    torch.manual_seed(0)
    m = zk_model().eval()
    with torch.no_grad():
        for mod in m.modules():
            if getattr(mod, "bias", None) is not None:
                mod.bias.normal_(0, 0.1)
    return m


def test_export_is_deterministic_and_simple(tmp_path):
    m = seeded_model()
    a, b = export_onnx(m, tmp_path / "a.onnx"), export_onnx(m, tmp_path / "b.onnx")
    assert sha256_file(a) == sha256_file(b)
    s = onnx_summary(a)
    assert s["ops"] == ["Conv", "Relu", "Conv", "Relu", "Conv", "Relu", "Flatten", "Gemm"]
    assert s["inputs"] == [("input", [1, 1, 28, 28])] and s["outputs"] == [("logits", [1, 10])]
    assert s["initializer_elements"] == 6138


def test_compare_agrees_on_a_seeded_model(tmp_path):
    m = seeded_model()
    path = export_onnx(m, tmp_path / "m.onnx")
    x = np.random.default_rng(1).standard_normal((50, 1, 28, 28)).astype(np.float32)
    r = compare(m, path, x)
    assert r.n == 50 and r.top1_agree == 50 and r.max_abs_diff < 1e-4


def test_compare_detects_a_different_model(tmp_path):
    path = export_onnx(seeded_model(), tmp_path / "m.onnx")
    torch.manual_seed(1)
    other = zk_model().eval()
    with torch.no_grad():
        other.classifier[1].weight.mul_(100)
    x = np.random.default_rng(2).standard_normal((50, 1, 28, 28)).astype(np.float32)
    assert compare(other, path, x).max_abs_diff > 1e-3


def test_compare_counts_accuracy_with_labels(tmp_path):
    m = seeded_model()
    path = export_onnx(m, tmp_path / "m.onnx")
    x = np.random.default_rng(3).standard_normal((20, 1, 28, 28)).astype(np.float32)
    with torch.no_grad():
        preds = m(torch.from_numpy(x)).argmax(1).numpy()
    r = compare(m, path, x, preds)
    assert r.torch_correct == r.onnx_correct == 20


@pytest.mark.skipif(not (COMMITTED.exists() and (REPO_ROOT / "results" / "p0.7_zk_model.pt").exists()),
                    reason="committed ONNX or the P0.7 weights not present")
def test_committed_onnx_matches_a_fresh_export_of_p0_7(tmp_path):
    m = zk_model()
    m.load_state_dict(torch.load(REPO_ROOT / "results" / "p0.7_zk_model.pt", map_location="cpu", weights_only=True))
    assert sha256_file(export_onnx(m, tmp_path / "x.onnx")) == sha256_file(COMMITTED)
