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


def test_git_info_has_a_consistent_shape():
    """Must not require a git checkout.

    A handoff zip unpacked on Colab is not a repository (CLAUDE.md 0.5), and
    the notebook runs this suite as a pre-flight check before spending GPU
    time. A test that assumed `.git` was present would abort a perfectly good
    run. So: assert the shape always, and the commit only when git can answer.
    """
    info = git_info()
    assert set(info) == {"commit", "branch", "dirty", "available", "source"}

    if info["available"]:
        assert info["source"] == "git"
        assert len(info["commit"]) == 40
        assert isinstance(info["dirty"], bool)
    else:
        assert info["source"] in {"handoff", None}


def test_a_git_snapshot_is_used_instead_of_reading_git_again(tmp_path, monkeypatch):
    """A run writing several records passes one snapshot, so its own earlier files cannot mark later ones dirty."""
    snapshot = {"commit": "abc", "branch": "b", "dirty": False, "available": True, "source": "git"}

    def must_not_run(*a):
        raise AssertionError("git was read despite a snapshot")

    monkeypatch.setattr("src.utils.results._run_git", must_not_run)
    record = read_result(write_result("snap", seed=1, out_dir=tmp_path, git=snapshot))
    assert record["git"] == snapshot


def test_records_are_writable_without_git(tmp_path, monkeypatch):
    """No git binary must not stop a result from being saved."""
    monkeypatch.setattr("src.utils.results._run_git", lambda *a: None)
    monkeypatch.setattr("src.utils.results.repo_root", lambda: tmp_path / "no-build-info")
    record = read_result(write_result("nogit", seed=1, metrics={"a": 1}, out_dir=tmp_path))
    assert record["git"] == {
        "commit": None,
        "branch": None,
        "dirty": None,
        "available": False,
        "source": None,
    }
    assert record["metrics"]["a"] == 1


def test_unpacked_handoff_zip_still_records_its_commit(tmp_path, monkeypatch):
    """On Colab the code is an unpacked zip, not a checkout. The result must
    still name the commit the zip was built from."""
    (tmp_path / "BUILD_INFO.json").write_text(
        json.dumps({"commit": "a" * 40, "branch": "master", "dirty": False}), encoding="utf-8"
    )
    monkeypatch.setattr("src.utils.results._run_git", lambda *a: None)
    monkeypatch.setattr("src.utils.results.repo_root", lambda: tmp_path)
    assert git_info() == {
        "commit": "a" * 40,
        "branch": "master",
        "dirty": False,
        "available": False,
        "source": "handoff",
    }


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
