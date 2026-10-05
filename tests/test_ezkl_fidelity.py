"""P8.6: fidelity summary, fixed proof subset, and the batch witness stage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
try:
    import p8_6_ezkl_fidelity as f  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "experiments"))

COMPILED = REPO_ROOT / "zk/ezkl/zk_model/network.compiled"
P8_4_INPUT = REPO_ROOT / "results/zk/p8.4/input.json"
P8_4_PUBLIC = REPO_ROOT / "results/zk/p8.4/public_instances.json"


def test_proof_subset_is_fixed_by_seed():
    a = f.proof_subset()
    assert a == f.proof_subset() and a == sorted(a)
    assert len(set(a)) == f.N_PROVED and all(0 <= i < f.N_TEST for i in a)


def test_argmax_reports_ties():
    assert f.argmax_with_ties(np.array([0.0, 2.0, 1.0])) == (1, False)
    assert f.argmax_with_ties(np.array([2.0, 2.0, 1.0])) == (0, True)


def test_summary_counts_disagreements_and_accuracy():
    labels = np.array([0, 1, 2, 0])
    torch_l = np.eye(3)[[0, 1, 2, 1]] * 5.0          # PyTorch: right, right, right, wrong
    circ = torch_l.copy()
    circ[2] = [0.0, 5.5, 5.0]                         # circuit flips image 2 to class 1 (wrong)
    circ[3] = [6.0, 5.0, 0.0]                         # and image 3 to class 0 (right)
    s = f.summarise(torch_l, circ, labels)
    assert s["top1_agree"] == 2 and [d["index"] for d in s["disagreements"]] == [2, 3]
    assert s["pytorch_correct"] == 3 and s["circuit_correct"] == 3
    assert s["paired_drop_pytorch_to_circuit"]["reference_only_right"] == 1
    assert s["paired_drop_pytorch_to_circuit"]["other_only_right"] == 1
    assert s["abs_logit_diff"]["max"] == 6.0 and s["abs_logit_diff"]["argmax_image_index"] == 3


@pytest.mark.skipif(not COMPILED.exists(), reason="P8.3 compiled circuit not on disk")
def test_batch_stage_reproduces_the_p8_4_outputs(tmp_path):
    pytest.importorskip("ezkl")
    from src.zk.ezkl import stages

    x = np.array(json.loads(P8_4_INPUT.read_text())["input_data"][0], dtype=np.float32)
    np.save(tmp_path / "inputs.npy", np.stack([x, x]))
    out = stages.witness_batch(str(tmp_path / "inputs.npy"), [0, 0], str(COMPILED), str(tmp_path),
                               str(tmp_path / "r.json"))
    assert out == {"images": 2, "errors": 0}
    rows = json.loads((tmp_path / "r.json").read_text())
    want = [float(v) for v in json.loads(P8_4_PUBLIC.read_text())["pretty_public_inputs"]["rescaled_outputs"][0]]
    assert [r["outputs"] for r in rows] == [want, want]
    assert all(r["seconds"] > 0 and r["error"] is None for r in rows)
