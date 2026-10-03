"""P7.2: check the installed Track A toolchain against its pins and record it.

    python experiments/p7_2_toolchain_versions.py

Local, offline, under a few seconds. Needs the install described in
`zk/README.md`: the circom binary at `tools/bin/circom.exe` (or `circom` on
non-Windows) and `npm ci` run in `zk/`. Reads no key and no model.

Checks, any failure stops the run before a result is written:

- the circom binary's SHA-256 equals the digest GitHub publishes for the
  v2.2.3 `circom-windows-amd64.exe` release asset, and `circom --version`
  reports 2.2.3;
- `zk/package.json` pins snarkjs and circomlib to exact versions, and the
  installed packages and `zk/package-lock.json` both report those versions;
- the snarkjs CLI starts and prints its version banner;
- the circomlib templates P7.3 and P7.5 need (`poseidon.circom`,
  `bitify.circom`) exist; their SHA-256 are recorded.

Nothing is compiled or proved here. That starts in P7.3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

CIRCOM_VERSION = "2.2.3"
CIRCOM_ASSET = "circom-windows-amd64.exe"
CIRCOM_ASSET_SHA256 = "e43f132ee6f0aa79b705beceb59c2a7e6a54d7bdeab917ca34e9fc1951d185e1"
"""Digest GitHub reports for the v2.2.3 release asset (`gh api .../releases/tags/v2.2.3`)."""
SNARKJS_VERSION = "0.7.6"
CIRCOMLIB_VERSION = "2.0.5"
CIRCOMLIB_TEMPLATES = ("poseidon.circom", "bitify.circom")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path) -> str:
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
    return (out.stdout or "") + (out.stderr or "")


def require(cond: bool, msg: str) -> None:
    if not cond:
        sys.exit(f"P7.2 check failed: {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; nothing is random")
    args = parser.parse_args()
    start = time.perf_counter()
    root = repo_root()
    zk = root / "zk"
    git = git_info()

    # circom
    circom = root / "tools" / "bin" / ("circom.exe" if sys.platform == "win32" else "circom")
    require(circom.is_file(), f"circom binary not found at {circom}")
    circom_sha = sha256_file(circom)
    if sys.platform == "win32":
        require(circom_sha == CIRCOM_ASSET_SHA256, f"circom binary SHA-256 {circom_sha} != release asset digest")
    circom_out = run([str(circom), "--version"], root).strip()
    require(circom_out == f"circom compiler {CIRCOM_VERSION}", f"circom --version gave {circom_out!r}")

    # npm pins, lockfile, installed packages
    pkg = json.loads((zk / "package.json").read_text(encoding="utf-8"))
    pins = {"snarkjs": SNARKJS_VERSION, "circomlib": CIRCOMLIB_VERSION}
    require(pkg["dependencies"] == pins, f"zk/package.json pins {pkg['dependencies']} != {pins}")
    lock = json.loads((zk / "package-lock.json").read_text(encoding="utf-8"))
    lock_entries = {}
    for name, version in pins.items():
        entry = lock["packages"][f"node_modules/{name}"]
        require(entry["version"] == version, f"lockfile has {name} {entry['version']}")
        installed = json.loads((zk / "node_modules" / name / "package.json").read_text(encoding="utf-8"))
        require(installed["version"] == version, f"installed {name} is {installed['version']}")
        lock_entries[name] = {"version": version, "resolved": entry["resolved"], "integrity": entry["integrity"],
                              "license": entry.get("license")}

    # snarkjs CLI
    node = shutil.which("node")
    require(node is not None, "node not on PATH")
    node_version = run([node, "--version"], root).strip()
    snarkjs_cli = zk / "node_modules" / "snarkjs" / "build" / "cli.cjs"
    require(snarkjs_cli.is_file(), f"snarkjs CLI not found at {snarkjs_cli}")
    banner = run([node, str(snarkjs_cli), "--version"], zk)
    m = re.search(r"snarkjs@(\S+)", banner)
    require(m is not None and m.group(1) == SNARKJS_VERSION, f"snarkjs banner gave {banner[:80]!r}")

    # circomlib templates
    templates = {}
    for t in CIRCOMLIB_TEMPLATES:
        p = zk / "node_modules" / "circomlib" / "circuits" / t
        require(p.is_file(), f"circomlib template {t} missing")
        templates[t] = sha256_file(p)

    path = write_result(
        name="p7.2_toolchain",
        seed=args.seed,
        task="P7.2",
        params={
            "install_location": "local (tools/bin/ and zk/node_modules/, both gitignored)",
            "circom_source": f"github.com/iden3/circom release v{CIRCOM_VERSION}, asset {CIRCOM_ASSET}",
            "npm_pins": pins,
        },
        metrics={
            "circom": {"version_output": circom_out, "binary_sha256": circom_sha,
                       "matches_release_asset_digest": circom_sha == CIRCOM_ASSET_SHA256},
            "snarkjs": {**lock_entries["snarkjs"], "cli_banner_version": m.group(1)},
            "circomlib": {**lock_entries["circomlib"], "template_sha256": templates},
            "node": node_version,
            "lockfile_sha256": sha256_file(zk / "package-lock.json"),
            "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        },
        notes="Installed and version-checked only; nothing compiled or proved (P7.3).",
        duration_seconds=time.perf_counter() - start,
        git=git,
    )
    print(f"circom {circom_out.split()[-1]}, snarkjs {SNARKJS_VERSION}, circomlib {CIRCOMLIB_VERSION}, node {node_version}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
