"""Export `zk_model` to ONNX for Track B, and compare ONNX Runtime with PyTorch (P8.1).

The exported graph takes one **normalised** MNIST image, shape (1, 1, 28, 28),
float32, and returns 10 raw logits, shape (1, 10). The MNIST normalisation
(`src/data/mnist.py`) stays outside the graph, as it was during training, so
the circuit (P8.3) sees exactly what the model saw. The batch size is fixed at
1 because EZKL proves one fixed-shape graph.

The legacy TorchScript exporter (`dynamo=False`) is used. It gives a plain
graph of Conv, Relu, Flatten and Gemm with the weights as initializers, which
is the simple op set `zk_model` was designed for (see `src/models/zk_model.py`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
INPUT_SHAPE = (1, 1, 28, 28)
DEFAULT_OPSET = 13


def export_onnx(model: nn.Module, path: Path | str, *, opset: int = DEFAULT_OPSET) -> Path:
    """Write `model` (in eval mode) as an ONNX file with a fixed (1, 1, 28, 28) input."""
    path = Path(path)
    model = model.eval()
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(model, (dummy,), str(path), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                          opset_version=opset, do_constant_folding=True, dynamo=False)
    return path


def onnx_summary(path: Path | str) -> dict:
    """Checker result, opset, op types in graph order, initializer count and element total."""
    import onnx

    m = onnx.load(str(path))
    onnx.checker.check_model(m, full_check=True)
    inits = list(m.graph.initializer)
    return {
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in m.opset_import],
        "ops": [n.op_type for n in m.graph.node],
        "inputs": [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
        "outputs": [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output],
        "initializers": len(inits),
        "initializer_elements": int(sum(int(np.prod(t.dims)) for t in inits)),
    }


@dataclass(frozen=True)
class Agreement:
    """PyTorch vs ONNX Runtime over a set of inputs. Aggregates only."""

    n: int
    top1_agree: int
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    torch_correct: int | None
    onnx_correct: int | None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def compare(model: nn.Module, onnx_path: Path | str, inputs: np.ndarray, labels: np.ndarray | None = None) -> Agreement:
    """Run every input (N, 1, 28, 28) one at a time through both, compare logits.

    Relative difference is |a - b| / max(|a|, 1e-6) per logit.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = model.eval()
    inputs = np.ascontiguousarray(inputs, dtype=np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(inputs)).numpy()
    got = np.concatenate([sess.run([OUTPUT_NAME], {INPUT_NAME: inputs[i:i + 1]})[0] for i in range(len(inputs))])
    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    rel = diff / np.maximum(np.abs(ref.astype(np.float64)), 1e-6)
    tp, op = ref.argmax(1), got.argmax(1)
    return Agreement(
        n=len(inputs),
        top1_agree=int((tp == op).sum()),
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        max_rel_diff=float(rel.max()),
        torch_correct=None if labels is None else int((tp == labels).sum()),
        onnx_correct=None if labels is None else int((op == labels).sum()),
    )


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
