"""The dashboard's only way to read project data (P9.5).

Every file the app shows comes from a committed file under ``results/``,
``figures/`` or ``provenance/``, and is read through this module:

- the path must be listed in ``app/manifest.json``, which records each file's
  SHA-256 and size as committed (written by the local manifest tool in ``app/``);
- the path must be relative, canonical (no ``.``, ``..`` or repeated ``/``),
  use ``/``, and start with one of `ALLOWED_ROOTS`;
- the bytes on disk must hash to the recorded SHA-256.

Anything else raises `DataIntegrityError`. There is no fallback: a caller that
cannot read a file shows the error, never a number from somewhere else. The
module never touches any other directory, and it holds no measured values.

It does not import Streamlit, so it can be tested on its own.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
MANIFEST_PATH = APP_DIR / "manifest.json"
MANIFEST_SCHEMA = "zk-crown/dashboard-manifest/v1"
ALLOWED_ROOTS = ("results", "figures", "provenance")


class DataIntegrityError(Exception):
    """A file could not be used: not allowed, not in the manifest, missing, or changed."""

    def __init__(self, path: str, problem: str):
        super().__init__(f"{path}: {problem}")
        self.path, self.problem = path, problem


@dataclass(frozen=True)
class FileCheck:
    path: str
    ok: bool
    sha256: str | None
    problem: str | None = None


def check_path(path: str) -> str:
    """The canonical form of an allowed relative path, or `DataIntegrityError`."""
    if not isinstance(path, str) or not path:
        raise DataIntegrityError(repr(path), "not a path")
    if "\\" in path or path.startswith("/") or ":" in path:
        raise DataIntegrityError(path, "only relative paths with '/' separators are allowed")
    parts = PurePosixPath(path).parts
    if ".." in parts or str(PurePosixPath(path)) != path or path.startswith("."):
        raise DataIntegrityError(path, "only canonical paths are allowed (no '.', '..' or repeated '/')")
    if parts[0] not in ALLOWED_ROOTS:
        raise DataIntegrityError(path, f"only files under {', '.join(r + '/' for r in ALLOWED_ROOTS)} are allowed")
    return path


def _is_sha256_hex(value: object) -> bool:
    try:
        return (isinstance(value, str) and value == value.lower()
                and len(bytes.fromhex(value)) == hashlib.sha256().digest_size)
    except ValueError:
        return False


def load_manifest(manifest_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """The manifest's file table: path to ``{"sha256", "bytes"}``. Every listed path is checked."""
    path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DataIntegrityError("app/manifest.json", f"unreadable ({type(error).__name__})") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise DataIntegrityError("app/manifest.json", f"not a {MANIFEST_SCHEMA} manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise DataIntegrityError("app/manifest.json", "has no file table")
    for name, entry in files.items():
        if check_path(name) != name:
            raise DataIntegrityError(name, "manifest path is not in canonical form")
        if not (isinstance(entry, dict) and _is_sha256_hex(entry.get("sha256"))
                and isinstance(entry.get("bytes"), int)):
            raise DataIntegrityError(name, "malformed manifest entry")
    return files


class DataStore:
    """Hash-checked reads of committed files. One per app session; tests pass their own roots."""

    def __init__(self, repo_root: Path | None = None, manifest_path: Path | None = None):
        self.repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        self.files = load_manifest(manifest_path)

    def read_bytes(self, path: str) -> bytes:
        name = check_path(path)
        entry = self.files.get(name)
        if entry is None:
            raise DataIntegrityError(name, "not listed in app/manifest.json")
        file = self.repo_root / name
        if not file.is_file():
            raise DataIntegrityError(name, "missing")
        data = file.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"]:
            raise DataIntegrityError(name, f"SHA-256 {digest} does not match the recorded {entry['sha256']}")
        return data

    def read_text(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8")

    def read_json(self, path: str) -> Any:
        try:
            return json.loads(self.read_text(path))
        except (UnicodeDecodeError, ValueError) as error:
            raise DataIntegrityError(path, f"verified, but not valid JSON ({type(error).__name__})") from error

    def latest(self, prefix: str) -> str:
        """The last manifest path, in sorted order, that starts with `prefix`.

        Result files are named ``<task>__seed<seed>__<UTC timestamp>.json``, so for one
        task prefix the sorted order is the time order and this is the newest record.
        """
        names = sorted(name for name in self.files if name.startswith(prefix))
        if not names:
            raise DataIntegrityError(prefix, "no file with this prefix is listed in app/manifest.json")
        return names[-1]

    def recorded_sha256(self, path: str) -> str:
        name = check_path(path)
        if name not in self.files:
            raise DataIntegrityError(name, "not listed in app/manifest.json")
        return self.files[name]["sha256"]

    def verify(self, path: str) -> FileCheck:
        try:
            data = self.read_bytes(path)
        except DataIntegrityError as error:
            return FileCheck(error.path, False, None, error.problem)
        return FileCheck(check_path(path), True, hashlib.sha256(data).hexdigest())

    def verify_all(self) -> list[FileCheck]:
        return [self.verify(name) for name in sorted(self.files)]
