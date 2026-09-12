"""Tests for the seeding helper and the standard result record (P0.2)."""

from __future__ import annotations

import json
import random

import pytest

from src.utils.results import (
    SCHEMA_VERSION,
    build_record,
    git_info,
    read_result,
    write_result,
)
from src.utils.seeding import DEFAULT_SEED, set_seed


# --- seeding ----------------------------------------------------------------


def _draw(n: int = 8) -> list[float]:
    return [random.random() for _ in range(n)]


def test_same_seed_gives_same_stream():
    set_seed(7)
    first = _draw()
    set_seed(7)
    assert _draw() == first


def test_different_seed_gives_different_stream():
    set_seed(7)
    first = _draw()
    set_seed(8)
    assert _draw() != first


def test_set_seed_reports_backends():
    seeded = set_seed(DEFAULT_SEED)
    assert seeded["python"] is True
    # numpy/torch/cuda depend on the environment; only the keys are guaranteed.
    assert set(seeded) == {"python", "numpy", "torch", "cuda"}


@pytest.mark.parametrize("bad", [-1, "7", 1.0, True, None])
def test_set_seed_rejects_bad_seeds(bad):
    with pytest.raises((TypeError, ValueError)):
        set_seed(bad)


# --- result records ---------------------------------------------------------


def test_record_has_the_required_fields():
    record = build_record("demo", seed=3, params={"lr": 0.1}, metrics={"acc": 0.5})
    for field in ("name", "timestamp_utc", "git", "seed", "params", "metrics"):
        assert field in record
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["seed"] == 3
    assert record["params"]["lr"] == 0.1
    assert record["metrics"]["acc"] == 0.5


def test_record_carries_a_git_commit_in_this_repo():
    info = git_info()
    assert info["available"] is True, "expected to be running inside the git repo"
    assert len(info["commit"]) == 40


def test_write_and_read_round_trip(tmp_path):
    path = write_result(
        "p0.2_round_trip",
        seed=11,
        params={"n": 2},
        metrics={"wdr": 0.75},
        task="P0.2",
        out_dir=tmp_path,
    )
    assert path.exists()
    assert path.parent == tmp_path
    record = read_result(path)
    assert record["name"] == "p0.2_round_trip"
    assert record["task"] == "P0.2"
    assert record["metrics"]["wdr"] == 0.75


def test_written_file_is_strict_json(tmp_path):
    """Non-finite metrics must not produce bare NaN, which strict parsers reject."""
    path = write_result(
        "nonfinite",
        seed=1,
        metrics={"nan": float("nan"), "inf": float("inf"), "ninf": float("-inf")},
        out_dir=tmp_path,
    )
    text = path.read_text(encoding="utf-8")
    record = json.loads(text, parse_constant=lambda c: pytest.fail(f"bare {c} in JSON"))
    assert record["metrics"] == {"nan": "NaN", "inf": "Infinity", "ninf": "-Infinity"}


def test_unserialisable_values_do_not_lose_the_run(tmp_path):
    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    path = write_result(
        "opaque", seed=1, params={"obj": Opaque()}, metrics={"acc": 1.0}, out_dir=tmp_path
    )
    assert read_result(path)["params"]["obj"] == "<opaque>"


def test_subdir_and_unique_filenames(tmp_path):
    first = write_result("sweep", seed=1, metrics={"a": 1}, out_dir=tmp_path, subdir="attacks")
    second = write_result("sweep", seed=2, metrics={"a": 2}, out_dir=tmp_path, subdir="attacks")
    assert first.parent == tmp_path / "attacks"
    assert first != second
    assert len(list((tmp_path / "attacks").glob("*.json"))) == 2


def test_no_temp_files_left_behind(tmp_path):
    write_result("clean", seed=1, metrics={"a": 1}, out_dir=tmp_path)
    assert list(tmp_path.glob("*.tmp")) == []


def test_empty_name_is_rejected():
    with pytest.raises(ValueError):
        build_record("   ", seed=1)
