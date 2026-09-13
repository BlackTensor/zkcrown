"""The standard result record.

CLAUDE.md 0.3: every experiment writes a JSON result file, and no result lives
only in a notebook cell output. This module defines the one record shape that
every experiment in the project uses, so that section 8's ledger and the
Streamlit dashboard can read any result without special-casing.

A record carries the five required fields -- name, timestamp, git hash, seed,
params, metrics -- plus the environment it ran in and whether the working tree
was dirty, because a number produced from uncommitted code is not reproducible
from its commit hash and the file should say so.

Stdlib only: this runs in environments without torch or numpy installed.

Typical use at the end of an experiment::

    from src.utils.results import write_result
    path = write_result(
        name="p0.6_baseline_clean_accuracy",
        seed=seed,
        params={"epochs": 30, "lr": 0.1, "batch_size": 128},
        metrics={"clean_accuracy": 0.8123, "loss": 0.54},
    )
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
"""Bump this if the record shape changes incompatibly, and say so in CLAUDE.md."""

_GIT_TIMEOUT_S = 15


def repo_root() -> Path:
    """Repository root, derived from this file's location (src/utils/results.py)."""
    return Path(__file__).resolve().parents[2]


def results_dir() -> Path:
    """Where result JSON files go. Committed, per Section 3."""
    return repo_root() / "results"


def _run_git(*args: str) -> str | None:
    """Run a git command in the repo root. Returns None if git can't answer."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        # No git binary, or not a repository. Both are survivable; the record
        # just records the absence.
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


BUILD_INFO_NAME = "BUILD_INFO.json"
"""Written into every handoff zip by `handoff/build_handoff.py`."""


def _build_info() -> dict[str, Any] | None:
    path = repo_root() / BUILD_INFO_NAME
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return info if isinstance(info, dict) else None


def git_info() -> dict[str, Any]:
    """Commit, branch and dirty flag, and where they came from.

    `dirty` True means the code that produced the result does not match the
    recorded commit. Treat such a result as provisional.

    `source` is `"git"` when read from a live checkout, `"handoff"` when read
    from the `BUILD_INFO.json` a handoff zip carries (an unpacked zip on Colab
    is not a repository, and without this the result would name no commit at
    all), or None when neither is available. `available` stays True only for a
    live checkout.
    """
    commit = _run_git("rev-parse", "HEAD")
    if commit is None:
        build = _build_info()
        if build is not None and build.get("commit"):
            return {
                "commit": build["commit"],
                "branch": build.get("branch"),
                "dirty": build.get("dirty"),
                "available": False,
                "source": "handoff",
            }
        return {"commit": None, "branch": None, "dirty": None, "available": False, "source": None}

    porcelain = _run_git("status", "--porcelain")
    dirty = None if porcelain is None else bool(porcelain)
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    return {"commit": commit, "branch": branch, "dirty": dirty, "available": True, "source": "git"}


def _optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return getattr(module, "__version__", "unknown")


def system_ram_gb() -> float | None:
    """Total system RAM in GiB, read from /proc/meminfo. None where that is absent.

    Linux only, which is what Colab runs. Section 8.1 asks for this and
    P0.5's record did not have it.
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 2)  # value is in kB
    except (OSError, ValueError, IndexError):
        return None
    return None


def environment_info() -> dict[str, Any]:
    """What the run actually executed on.

    Section 2.1: Colab does not guarantee which accelerator you get, so the
    device is recorded per run rather than assumed.
    """
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _optional_version("torch"),
        "numpy": _optional_version("numpy"),
        "accelerator": None,
        "cuda": None,
        "system_ram_gb": system_ram_gb(),
    }

    try:
        import torch
    except ImportError:
        return info

    if torch.cuda.is_available():
        info["accelerator"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    else:
        info["accelerator"] = "cpu"
    return info


def _json_safe(value: Any) -> Any:
    """Coerce a value into something `json.dump` accepts without lying about it.

    Handles the types that actually show up in metrics: numpy scalars and
    arrays, torch tensors, Paths, datetimes. Non-finite floats become the
    strings "NaN" / "Infinity" / "-Infinity" so the file stays valid strict
    JSON instead of emitting bare `NaN`, which most parsers reject.

    Anything genuinely unrecognised falls back to `str()`. That is deliberate:
    losing an entire GPU run's numbers to a serialisation error at the last
    line is worse than a stringified oddity that a reader can see.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    # numpy scalars and arrays, torch tensors: all expose .item() for
    # 0-d values and .tolist() otherwise, without importing either library.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:  # noqa: BLE001 - fall through to str()
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:  # noqa: BLE001 - fall through to str()
            pass

    return str(value)


def _slug(name: str) -> str:
    """Filesystem-safe stem. Keeps it readable; does not try to be reversible."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    if not slug:
        raise ValueError(f"name {name!r} has no filesystem-safe characters")
    return slug


def build_record(
    name: str,
    seed: int,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    *,
    task: str | None = None,
    notes: str | None = None,
    duration_seconds: float | None = None,
    seeded_backends: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Assemble a result record without writing it.

    Args:
        name: identifies the experiment. Prefer the task id as a prefix, e.g.
            ``"p2.6_false_positive_rate"``.
        seed: the seed passed to `set_seed`.
        params: the knobs that were set. Everything needed to re-run.
        metrics: the numbers that came out.
        task: the CLAUDE.md task id, e.g. ``"P2.6"``.
        notes: free text. Caveats belong here, not in a README.
        duration_seconds: wall clock time of the run.
        seeded_backends: the return value of `set_seed`.

    Returns:
        A JSON-safe dict.
    """
    if not name or not name.strip():
        raise ValueError("name must be a non-empty string")

    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "task": task,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "seed": seed,
        "seeded_backends": _json_safe(seeded_backends) if seeded_backends else None,
        "params": _json_safe(params or {}),
        "metrics": _json_safe(metrics or {}),
        "environment": environment_info(),
        "duration_seconds": _json_safe(duration_seconds),
        "notes": notes,
    }


def write_result(
    name: str,
    seed: int,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    *,
    task: str | None = None,
    notes: str | None = None,
    duration_seconds: float | None = None,
    seeded_backends: dict[str, bool] | None = None,
    out_dir: Path | str | None = None,
    subdir: str | None = None,
    filename: str | None = None,
) -> Path:
    """Build a record and write it to a JSON file. Returns the path written.

    The default filename is ``<name>__seed<seed>__<timestamp>.json``, which is
    unique per run so sweeps accumulate rather than overwrite.

    Args:
        out_dir: overrides the default `results/` directory. Useful for tests
            and for writing to a mounted Drive path from Colab.
        subdir: groups related runs, e.g. ``subdir="attacks"``.
        filename: overrides the generated filename entirely.

    Warns:
        UserWarning: if the git working tree is dirty, since the result is then
            not reproducible from the recorded commit.

    Note:
        A result committed alongside the code that produced it will always read
        `dirty: true`, because the commit containing the file cannot exist when
        the file is written. Read such a record as "based on <commit>, plus the
        uncommitted change that became the next commit". The flag matters when
        it appears on a result committed *later* than the code it describes.
    """
    record = build_record(
        name,
        seed,
        params,
        metrics,
        task=task,
        notes=notes,
        duration_seconds=duration_seconds,
        seeded_backends=seeded_backends,
    )

    if record["git"]["dirty"]:
        warnings.warn(
            f"result {name!r} was produced from a dirty working tree; commit "
            f"{record['git']['commit'][:8]} does not describe the code that ran",
            UserWarning,
            stacklevel=2,
        )

    directory = Path(out_dir) if out_dir is not None else results_dir()
    if subdir:
        directory = directory / _slug(subdir)
    directory.mkdir(parents=True, exist_ok=True)

    if filename is None:
        stamp = record["timestamp_utc"].replace(":", "").replace("-", "")
        filename = f"{_slug(name)}__seed{seed}__{stamp}.json"
    path = directory / filename

    # Write to a temporary file and replace, so a Colab session dying mid-write
    # cannot leave a truncated JSON file behind.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)

    return path


def read_result(path: Path | str) -> dict[str, Any]:
    """Load a result record. Rejects files that are not this schema."""
    with Path(path).open(encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict) or "schema_version" not in record:
        raise ValueError(f"{path} is not a zk-Crown result record")
    return record
