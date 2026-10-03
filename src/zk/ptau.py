"""Public powers-of-tau files for Groth16 phase 2 (P7.4).

The files are the Hermez / Polygon prepared-phase-2 powers of tau over bn128,
truncated to each power of two, listed in the README of the pinned snarkjs
0.7.6 (`zk/node_modules/snarkjs/README.md`). That table gives, per power, the
download URL and the BLAKE2b-512 hash of the file. The README reaches us
through npm with the integrity hash recorded in `zk/package-lock.json`, so the
expected hashes are tied to the pinned package rather than to whatever a web
page says today.

Every file is hashed before use and refused on any mismatch. The files are
large binaries and are gitignored (`*.ptau`); they live in `zk/ptau/`.

What the file gives: phase 1 of the Groth16 setup. It is sound as long as at
least one contributor to that ceremony destroyed their secret. It says nothing
about phase 2, which is per circuit (P7.6).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.zk.toolchain import NODE_MODULES, REPO_ROOT

SNARKJS_README = NODE_MODULES / "snarkjs" / "README.md"
PTAU_DIR = REPO_ROOT / "zk" / "ptau"

_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*\[(\S+?\.ptau)\]\((https://\S+?\.ptau)\)\s*\|\s*([0-9a-f]{128})\s*\|")


@dataclass(frozen=True)
class PtauEntry:
    power: int
    max_constraints: str
    filename: str
    url: str
    blake2b_512: str


def read_table(readme: Path = SNARKJS_README) -> dict[int, PtauEntry]:
    """Parse the snarkjs README table of prepared ptau files, keyed by power."""
    table = {}
    for line in readme.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if m:
            power = int(m.group(1))
            table[power] = PtauEntry(power, m.group(2), m.group(3), m.group(4), m.group(5))
    return table


def required_power(constraints: int, public_inputs: int, outputs: int = 0) -> int:
    """Smallest power p with 2**p >= constraints + public inputs + outputs + 1 (snarkjs domain size)."""
    need = constraints + public_inputs + outputs + 1
    p = 1
    while 2**p < need:
        p += 1
    return p


def blake2b_file(path: Path) -> str:
    h = hashlib.blake2b(digest_size=64)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class PtauHashMismatch(RuntimeError):
    pass


def fetch(entry: PtauEntry, dest_dir: Path = PTAU_DIR, *, download: bool = True,
          url: str | None = None) -> tuple[Path, bool]:
    """Return a verified local copy of `entry`. Returns (path, downloaded_now).

    An existing file is re-hashed, never trusted. A new download goes to a
    temporary file first and is moved into place only if its BLAKE2b-512
    matches; a mismatch deletes it and raises.

    `url` overrides the README's URL, for a mirror. The expected hash is still
    the README's, so a mirror is never trusted for content.
    """
    url = url or entry.url
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / entry.filename
    if dest.exists():
        got = blake2b_file(dest)
        if got != entry.blake2b_512:
            raise PtauHashMismatch(f"{dest} has BLAKE2b-512 {got}, expected {entry.blake2b_512}")
        return dest, False
    if not download:
        raise FileNotFoundError(dest)
    with tempfile.NamedTemporaryFile(dir=dest_dir, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                shutil.copyfileobj(resp, tmp, length=1 << 20)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    got = blake2b_file(tmp_path)
    if got != entry.blake2b_512:
        tmp_path.unlink(missing_ok=True)
        raise PtauHashMismatch(f"download of {url} has BLAKE2b-512 {got}, expected {entry.blake2b_512}")
    tmp_path.replace(dest)
    return dest, True
