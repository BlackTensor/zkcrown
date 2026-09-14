"""Tests for P3.6: the dual-watermark entry point's helpers.

The real run needs `K`, the trained weights and CIFAR-10. These tests cover
the fixed choices, the behavioral gate, deterministic saving and the P3.5 lookup.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.models import main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p3_6_make_dual_model")


def test_decision_constants(script):
    assert script.ALPHA == 0.1
    assert script.BEHAVIORAL_GATE_ALPHA == "1e-6"
    assert script.OUTPUT.endswith(".pt")  # gitignored


@pytest.mark.parametrize("fired, rejects", [(28, False), (29, True), (100, True), (3, False)])
def test_behavioral_gate_matches_p2_8_threshold(script, fired, rejects):
    gate = script.behavioral_gate(fired, 100)
    assert gate["passes_gate"] is rejects
    assert gate["fired"] == fired and gate["gate_alpha"] == "1e-6"
    assert gate["rejects"]["1e-6"] is rejects  # agrees with the P2.8 summary


def test_save_state_is_atomic_and_deterministic(script, tmp_path):
    set_seed(5)
    state = main_model().state_dict()
    a = script.save_state(state, tmp_path / "a" / "w.pt")
    b = script.save_state(state, tmp_path / "b" / "w.pt")
    assert a == b and len(a) == 64
    assert not (tmp_path / "a" / "w.pt.tmp").exists()
    loaded = torch.load(tmp_path / "b" / "w.pt", weights_only=True)
    assert all(torch.equal(loaded[k], v) for k, v in state.items())


def test_p3_5_row_reads_the_committed_alpha_0_1_row(script):
    row = script.p3_5_row(0.1)
    assert row["alpha"] == 0.1
    assert row["test"]["correct"] == 9085 and row["holdout"]["correct"] == 4558
    assert row["detection"]["bit_matches"] == 126


def test_p3_5_row_refuses_an_alpha_not_in_the_sweep(script):
    with pytest.raises(SystemExit):
        script.p3_5_row(0.12)
