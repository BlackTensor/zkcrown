"""Build a self-contained Colab handoff zip.

CLAUDE.md 0.5 step 3: everything a `[GPU]` run needs goes into one zip, and it
must run top to bottom on Colab without anyone editing code inside it. The
zips themselves are gitignored build artifacts; this script is what is
committed, so any handoff can be rebuilt from the repo.

    python handoff/build_handoff.py P0.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Directories copied wholesale. Logic lives in src/, so src/ is non-negotiable.
INCLUDE_DIRS = ("src", "experiments", "tests")

INCLUDE_FILES = ("requirements.txt", "conftest.py", "CLAUDE.md")

EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pth", ".pt", ".ptau", ".zkey"}


def _should_include(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    return path.suffix not in EXCLUDE_SUFFIXES


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def build_info() -> dict:
    """The commit the zip is built from.

    Shipped as BUILD_INFO.json so a result written on Colab, where the unpacked
    zip is not a repository, still names the code that produced it.
    """
    porcelain = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if porcelain is None else bool(porcelain),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build(task_id: str, out_dir: Path | None = None) -> Path:
    """Package the handoff for `task_id`. Returns the zip path."""
    notebook = ROOT / "notebooks" / f"{task_id}_colab.ipynb"
    if not notebook.is_file():
        raise FileNotFoundError(
            f"{notebook} is missing. CLAUDE.md 0.5 step 2 requires a notebook per handoff."
        )
    instructions = ROOT / "handoff" / f"{task_id}_INSTRUCTIONS.md"
    if not instructions.is_file():
        raise FileNotFoundError(
            f"{instructions} is missing. CLAUDE.md 0.5 step 4 requires an instructions file."
        )

    out_dir = out_dir or (ROOT / "handoff")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{task_id}_colab.zip"

    members: list[tuple[Path, str]] = []
    for directory in INCLUDE_DIRS:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and _should_include(path):
                members.append((path, str(path.relative_to(ROOT)).replace("\\", "/")))
    for name in INCLUDE_FILES:
        path = ROOT / name
        if path.is_file():
            members.append((path, name))
    members.append((notebook, f"notebooks/{notebook.name}"))
    members.append((instructions, f"handoff/{instructions.name}"))

    info = build_info()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in members:
            zf.write(path, arcname)
        zf.writestr("BUILD_INFO.json", json.dumps(info, indent=2) + "\n")

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    size_kb = zip_path.stat().st_size / 1024

    print(f"{zip_path}  ({size_kb:.1f} KB, {len(members) + 1} files)")
    print(f"sha256: {digest}")
    print(f"commit: {info['commit']}  dirty={info['dirty']}")
    if info["dirty"] is not False:
        print(
            "WARNING: built from a working tree that is dirty or not a git checkout. "
            "Results from this zip will not be reproducible from the recorded commit. "
            "Commit first, then rebuild."
        )
    print("\ncontents:")
    for _, arcname in members:
        print(f"  {arcname}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", help="e.g. P0.5")
    args = parser.parse_args()
    build(args.task_id)


if __name__ == "__main__":
    main()
