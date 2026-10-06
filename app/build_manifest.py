"""Write ``app/manifest.json``: the SHA-256 of every committed file the dashboard may read (P9.5).

    python app/build_manifest.py           # write
    python app/build_manifest.py --check   # compare, write nothing

Local tool, not used by the running app. It lists the files git tracks under
``results/``, ``figures/`` and ``provenance/``, and refuses to write unless the
bytes on disk of each one are exactly the committed bytes (git blob id of the
file equals the blob id in the index, and the file has no staged or unstaged
change). So the manifest records committed content, which is what a fresh
checkout on the host will contain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))

from app.data import ALLOWED_ROOTS, MANIFEST_PATH, MANIFEST_SCHEMA, REPO_ROOT  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout


def committed_files() -> dict[str, dict]:
    changed = _git("status", "--porcelain", "--", *ALLOWED_ROOTS).strip()
    if changed:
        raise SystemExit(f"uncommitted changes under {', '.join(ALLOWED_ROOTS)}; commit first:\n{changed}")
    files = {}
    for line in _git("ls-files", "-s", "--", *ALLOWED_ROOTS).splitlines():
        meta, name = line.split("\t", 1)
        blob = meta.split()[1]
        path = REPO_ROOT / name
        data = path.read_bytes()
        if _git("hash-object", "--no-filters", "--", name).strip() != blob:
            raise SystemExit(f"{name}: bytes on disk differ from the committed blob (line endings?)")
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return dict(sorted(files.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare with the existing manifest; write nothing")
    args = parser.parse_args(argv)
    manifest = {"schema": MANIFEST_SCHEMA, "roots": list(ALLOWED_ROOTS), "files": committed_files()}
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.check:
        same = MANIFEST_PATH.is_file() and MANIFEST_PATH.read_text(encoding="utf-8") == text
        print("manifest is current" if same else "manifest is out of date")
        return 0 if same else 1
    MANIFEST_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}: {len(manifest['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
