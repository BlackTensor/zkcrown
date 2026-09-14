"""Tests for P3.4: the key-specificity experiment and the project owner id.

The experiment's core loop runs here on a seeded `main_model` with test keys
and a few wrong keys. No trained weights and no real `K` are used.

Every key here is a TEST KEY, public by construction.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.models import main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from src.watermark.carrier import carrier_layout  # noqa: E402
from src.watermark.signature import PROJECT_OWNER_ID, check_owner_id  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_KEY = bytes(range(32))
"""TEST KEY ONLY."""


def _script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p3_4_key_specificity")


def test_project_owner_id_is_fixed_and_valid():
    assert PROJECT_OWNER_ID == "blacktensor-zkcrown-owner"
    assert check_owner_id(PROJECT_OWNER_ID) == PROJECT_OWNER_ID


def test_wrong_keys_are_deterministic_distinct_and_seeded(monkeypatch):
    script = _script(monkeypatch)
    keys = [script.wrong_key(1337, j) for j in range(50)]
    assert keys == [script.wrong_key(1337, j) for j in range(50)]
    assert len(set(keys)) == 50 and all(len(k) == 32 for k in keys)
    assert script.wrong_key(1338, 0) != keys[0]


def test_wrong_key_family_differs_from_p2_8(monkeypatch):
    script = _script(monkeypatch)
    p28 = importlib.import_module("p2_8_detection_test")
    assert {script.wrong_key(1337, j) for j in range(20)}.isdisjoint({p28.null_key(1337, j) for j in range(20)})


def test_alpha_grid_includes_the_zero_control(monkeypatch):
    script = _script(monkeypatch)
    assert script.ALPHAS[0] == 0.0 and list(script.ALPHAS) == sorted(script.ALPHAS)


def test_summarise(monkeypatch):
    s = _script(monkeypatch).summarise([-0.2, 0.1, 0.4])
    assert s["count"] == 3 and s["mean"] == pytest.approx(0.1) and s["max_abs"] == pytest.approx(0.4)
    assert s["sd"] == pytest.approx(0.3) and s["min"] == -0.2 and s["max"] == 0.4


@pytest.fixture(scope="module")
def small_run():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    try:
        script = importlib.import_module("p3_4_key_specificity")
    finally:
        sys.path.remove(str(REPO_ROOT / "experiments"))
    set_seed(1234)
    host = main_model().state_dict()
    layout = carrier_layout(main_model())
    wrong = [script.wrong_key(99, j) for j in range(4)]
    rows = script.run_specificity({"host": host}, layout, TEST_KEY, "test owner", (0.0, 0.2), wrong, log=lambda *_: None)
    return rows


def test_run_reports_every_alpha_with_all_sections(small_run):
    assert set(small_run) == {"host"}
    assert [r["alpha"] for r in small_run["host"].values()] == [0.0, 0.2]
    for row in small_run["host"].values():
        assert set(row) == {"alpha", "embedding", "correct_key", "wrong_keys", "wrong_projection_true_signature", "gap"}
        assert row["wrong_keys"]["correlation"]["count"] == 4
        assert not any(k.startswith("_") for k in row)


def test_strong_embedding_separates_and_zero_control_does_not(small_run):
    zero, strong = small_run["host"]["0.0"], small_run["host"]["0.2"]
    assert strong["correct_key"]["bit_matches"] == 128
    assert strong["gap"]["correct_above_every_wrong_key"]
    assert strong["wrong_keys"]["z_max_abs"] < 6
    assert strong["wrong_projection_true_signature"]["z_max_abs"] < 6
    assert abs(zero["correct_key"]["z"]) < 6
    assert zero["embedding"]["delta_l2"] == 0.0


def test_gap_fields_are_consistent(small_run):
    for row in small_run["host"].values():
        c = row["correct_key"]["correlation"]
        w = row["wrong_keys"]["correlation"]
        g = row["gap"]
        assert g["correct_minus_wrong_mean"] == pytest.approx(c - w["mean"])
        assert g["correct_minus_wrong_max"] == pytest.approx(c - w["max"])
        assert g["in_wrong_key_sd"] == pytest.approx((c - w["mean"]) / w["sd"])
        assert row["correct_key"]["z"] == pytest.approx(c * math.sqrt(128))


def test_result_rows_hold_no_signature_or_fingerprint(small_run):
    keys, sequences = set(), []

    def walk(node):
        if isinstance(node, dict):
            keys.update(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            sequences.append(len(node))

    walk(small_run)
    assert {"projected", "recovered_bits", "signs", "value"}.isdisjoint(keys)
    # Only the 7 per-tensor embedding stats are lists: no per-bit (128) or per-key (4) arrays.
    assert set(sequences) == {7}


def test_run_refuses_the_owner_key_among_wrong_keys(monkeypatch):
    script = _script(monkeypatch)
    layout = carrier_layout(main_model())
    with pytest.raises(ValueError):
        script.run_specificity({}, layout, TEST_KEY, "test owner", (0.0,), [TEST_KEY, bytes(32)])
    with pytest.raises(ValueError):
        script.run_specificity({}, layout, TEST_KEY, "test owner", (0.0,), [bytes(32)])
