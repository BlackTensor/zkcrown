"""P8.1: export the P0.7 `zk_model` to ONNX and check ONNX Runtime against PyTorch.

    python experiments/p8_1_export_onnx.py

Local CPU, under a minute. Reads no key. EZKL is not installed or used (P8.2).

1. Load `results/p0.7_zk_model.pt`, checked against its recorded file SHA-256,
   strictly into `zk_model()`, and compute its P5.1 fingerprint.
2. Export with `src/zk/ezkl/onnx_export.py`: fixed input (1, 1, 28, 28),
   normalised as in training, 10 logits out, opset 13, legacy exporter. Export
   a second time and require identical bytes.
3. `onnx.checker` (full check), and every state_dict tensor must appear
   bit-identically among the graph's initializers.
4. Run all 10,000 MNIST test images, one at a time, through PyTorch and ONNX
   Runtime (CPU), plus 1,000 seeded standard-normal inputs as off-distribution
   inputs. **Pass criteria, fixed before the run:** every top-1 class agrees,
   and the largest absolute logit difference is at most 1e-4. PyTorch test
   accuracy must also reproduce P0.7's 9,896 / 10,000.

The ONNX file is copied to `results/zk/p8.1/zk_model.onnx` and committed
(26 KB; `zk_model` carries no watermark and no key material). P8.3 uses it.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

import numpy as np
import torch

from src.crypto.fingerprint import fingerprint_state_dict
from src.data.mnist import mnist_datasets
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.zk.ezkl.onnx_export import DEFAULT_OPSET, compare, export_onnx, onnx_summary, sha256_file
from src.models.zk_model import zk_model

WEIGHTS = repo_root() / "results" / "p0.7_zk_model.pt"
WEIGHTS_SHA256 = "6bc298d04edfe5af136349904c53ccc9a82e80dfb03729f616e552a3df9ddec4"
FINGERPRINT = "4d6683b73dc0dabf73abfe954f6792f38508462eae581601c707667c08241955"  # P5.1
P0_7_TEST_CORRECT = 9896
OUT = repo_root() / "results" / "zk" / "p8.1" / "zk_model.onnx"
MAX_ABS_DIFF = 1e-4
N_RANDOM = 1000


def load_model() -> torch.nn.Module:
    if sha256_file(WEIGHTS) != WEIGHTS_SHA256:
        raise SystemExit(f"{WEIGHTS} does not match the P0.7 SHA-256")
    model = zk_model()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True), strict=True)
    if fingerprint_state_dict(model.state_dict()).sha256 != FINGERPRINT:
        raise SystemExit("zk_model fingerprint differs from P5.1")
    return model.eval()


def initializers_match(model: torch.nn.Module, onnx_path: Path) -> bool:
    """Every state_dict tensor equals, bit for bit, some initializer of the same shape."""
    import onnx
    from onnx import numpy_helper

    inits = [numpy_helper.to_array(t) for t in onnx.load(str(onnx_path)).graph.initializer]
    for t in model.state_dict().values():
        a = t.detach().numpy()
        if not any(i.shape == a.shape and i.dtype == a.dtype and np.array_equal(i.view(np.uint8), a.view(np.uint8))
                   for i in inits):
            return False
    return True


def test_arrays() -> tuple[np.ndarray, np.ndarray]:
    ds = mnist_datasets(repo_root() / "data", download=False)["test"]
    xs = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32)
    ys = np.asarray(ds.targets, dtype=np.int64)
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seeds the off-distribution inputs")
    args = parser.parse_args()
    git = git_info()
    start = time.perf_counter()

    import onnx
    import onnxruntime

    model = load_model()
    with tempfile.TemporaryDirectory(prefix="zkcrown_p8_1_") as tmp:
        a = export_onnx(model, Path(tmp) / "a.onnx")
        b = export_onnx(model, Path(tmp) / "b.onnx")
        if sha256_file(a) != sha256_file(b):
            raise SystemExit("two exports gave different bytes")
        summary = onnx_summary(a)
        if not initializers_match(model, a):
            raise SystemExit("an exported initializer differs from the PyTorch weights")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a, OUT)

    xs, ys = test_arrays()
    t0 = time.perf_counter()
    test = compare(model, OUT, xs, ys)
    test_seconds = time.perf_counter() - t0
    rng = np.random.default_rng(args.seed)
    rand = compare(model, OUT, rng.standard_normal((N_RANDOM, 1, 28, 28)).astype(np.float32))

    if test.torch_correct != P0_7_TEST_CORRECT:
        raise SystemExit(f"PyTorch test accuracy {test.torch_correct} != P0.7's {P0_7_TEST_CORRECT}")
    passed = all(r.top1_agree == r.n and r.max_abs_diff <= MAX_ABS_DIFF for r in (test, rand))

    path = write_result(
        name="p8.1_onnx_export",
        seed=args.seed,
        task="P8.1",
        params={
            "model": "zk_model (P0.7), results/p0.7_zk_model.pt",
            "weights_file_sha256": WEIGHTS_SHA256,
            "fingerprint_p5_1": FINGERPRINT,
            "exporter": f"torch.onnx.export, legacy (dynamo=False), opset {DEFAULT_OPSET}, constant folding",
            "input": "normalised MNIST image (mean 0.1307, std 0.3081 applied outside the graph), float32 (1, 1, 28, 28)",
            "runtime": "onnxruntime CPUExecutionProvider, one image per run",
            "pass_criteria": f"all top-1 agree and max |logit diff| <= {MAX_ABS_DIFF}, on test and random inputs; "
                             f"PyTorch test correct == {P0_7_TEST_CORRECT}",
            "random_inputs": f"{N_RANDOM} standard-normal inputs from numpy default_rng(seed)",
            "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__},
        },
        metrics={
            "onnx": {**summary, "file_bytes": OUT.stat().st_size, "sha256": sha256_file(OUT),
                     "export_deterministic": True, "initializers_bit_identical_to_weights": True},
            "test_set": test.to_dict(),
            "random_inputs": rand.to_dict(),
            "onnx_test_seconds": round(test_seconds, 3),
            "passed": passed,
        },
        notes="ONNX export only; EZKL not installed (P8.2).",
        duration_seconds=time.perf_counter() - start,
        git=git,
    )
    print(f"ops {summary['ops']}, {OUT.stat().st_size:,} bytes; test: {test.top1_agree}/{test.n} top-1 agree, "
          f"max |diff| {test.max_abs_diff:.3g}, accuracy torch {test.torch_correct} onnx {test.onnx_correct}; "
          f"random: {rand.top1_agree}/{rand.n}, max |diff| {rand.max_abs_diff:.3g}; passed {passed}")
    print(f"wrote {path}")
    if not passed:
        raise SystemExit("ONNX Runtime does not match PyTorch within the fixed criteria")


if __name__ == "__main__":
    main()
