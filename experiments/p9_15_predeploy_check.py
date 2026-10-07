"""P9.15: pre-deploy check for the hosted dashboard. Run it before any push; it writes a result and exits non-zero on failure.

Streamlit Community Cloud deploys the tracked tree of a git repository, so
every check is about what git tracks (the working tree must be clean, so
tracked files on disk equal the commit):

1. ``clean_tree``: no uncommitted or untracked changes.
2. ``secrets_dir``: the only tracked path under ``secrets/`` is the empty
   placeholder ``secrets/.gitkeep``.
3. ``secret_file_names``: no tracked file whose name looks like key material
   (``K.bin``, nonce, signing key, trigger bundle, GPG keyrings, ``.env``,
   ``secrets.toml``).
4. ``model_weights``: the only tracked model-weight file is the approved bundle
   ``results/p9.15_dual_W_star.npz`` (owner decision, 2026-10-07). Files that
   were committed before P9.15 are listed under ``owner_decision_needed``
   rather than allowed silently.
5. ``bundle_identity``: the bundle's P5.1 fingerprint is the one the published
   commitment names, and is not the behavioral-only P2.3 model or clean ``W``.
   Bundling P2.3 next to the dual model would publish the owner's weight
   watermark change as their difference.
6. ``secret_values``: when the owner's secrets are on this machine, no tracked
   file contains ``K``, its halves, ``S``, the commitment nonce or the record
   signing seed (raw, hex, base64, decimal), windows of the real trigger bundle,
   or pieces of the GPG secret key. Without the secrets this check is reported
   as not run, never as passed.
7. ``page_modes``: every dashboard page states what runs live and what is
   replayed, and the page header renders it.
8. ``app_requirements``: ``app/requirements.txt`` pins Streamlit and nothing else.
9. ``manifest``: ``app/manifest.json`` lists exactly the tracked data files with
   their committed hashes.

Only counts and names are printed or recorded, never a secret value.

    python experiments/p9_15_predeploy_check.py
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

APPROVED_BUNDLE = "results/p9.15_dual_W_star.npz"
COMMITTED_BEFORE_P9_15 = {
    "results/zk/p8.1/zk_model.onnx": "zk_model (6,138-parameter MNIST model) as ONNX, committed in P8.1 for the "
                                     "Track B proof; not read by the app",
}
WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".npz", ".npy", ".onnx", ".safetensors", ".h5", ".hdf5", ".pkl",
                   ".pickle", ".joblib", ".bin", ".tflite", ".pb", ".zkey", ".ptau", ".srs")
SECRET_NAME = re.compile(r"(^|/)(K\.bin|commitment_nonce\.bin|provenance_signing_key\.bin|trigger_bundle\.npz|"
                         r"secring\.gpg|private-keys-v1\.d|pubring\.kbx|\.env|secrets\.toml)$|\.gpg$",
                         re.IGNORECASE)
P5_1_PREFIX = "p5.1_model_fingerprint__"
MUST_NOT_BUNDLE = ("behavioral_only_W_star", "clean_W")
GPG_FINGERPRINT = "C7301BA7D92FC2A65257BFC2A759F8EC04BF66E7"


def git(*args: str, root: Path) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout


def tracked(root: Path) -> list[str]:
    return [p for p in git("ls-files", "-z", root=root).decode().split("\0") if p]


# --- the individual rules (pure functions, tested on their own) ----------------------------

def check_secrets_dir(paths: list[str], gitkeep: bytes | None) -> dict:
    under = sorted(p for p in paths if p.startswith("secrets/"))
    placeholder_ok = gitkeep is None or gitkeep.lstrip(b"\xef\xbb\xbf").strip() == b""
    return {"passed": under in ([], ["secrets/.gitkeep"]) and placeholder_ok, "tracked_under_secrets": under,
            "gitkeep_is_empty_placeholder": placeholder_ok}


def check_secret_file_names(paths: list[str]) -> dict:
    hits = sorted(p for p in paths if SECRET_NAME.search(p))
    return {"passed": not hits, "matches": hits}


def check_model_weights(paths: list[str]) -> dict:
    weights = sorted(p for p in paths if p.lower().endswith(WEIGHT_SUFFIXES))
    unexpected = [p for p in weights if p != APPROVED_BUNDLE and p not in COMMITTED_BEFORE_P9_15]
    return {"passed": APPROVED_BUNDLE in weights and not unexpected, "approved_bundle_present": APPROVED_BUNDLE in weights,
            "unexpected": unexpected,
            "owner_decision_needed": {p: COMMITTED_BEFORE_P9_15[p] for p in weights if p in COMMITTED_BEFORE_P9_15}}


def check_bundle_identity(bundle_fingerprint: str, published: str, forbidden: dict[str, str]) -> dict:
    equal_forbidden = sorted(name for name, fp in forbidden.items() if fp == bundle_fingerprint)
    return {"passed": bundle_fingerprint == published and not equal_forbidden,
            "equals_published": bundle_fingerprint == published, "equals_forbidden_model": equal_forbidden}


def needle_forms(label: str, value: bytes) -> dict[str, bytes]:
    return {f"{label} raw": value, f"{label} hex": value.hex().encode(), f"{label} HEX": value.hex().upper().encode(),
            f"{label} base64": base64.b64encode(value).rstrip(b"=")[:-2],
            f"{label} decimal": str(int.from_bytes(value, "big")).encode()}


def scan(files: dict[str, bytes], needles: dict[str, bytes]) -> dict[str, int]:
    """How many files contain each needle. Only counts; the needles themselves are never returned."""
    hits: dict[str, int] = {}
    for data in files.values():
        for name, needle in needles.items():
            if needle and needle in data:
                hits[name] = hits.get(name, 0) + 1
    return hits


def check_page_modes(pages, header_source: str, overview_source: str) -> dict:
    missing = [p.slug for p in pages if not p.mode or not re.search(r"\b(live|replay|replayed)\b", p.mode, re.I)]
    rendered = "mode_line(spec)" in header_source and "mode_line(spec)" in overview_source
    return {"passed": not missing and rendered, "pages": len(pages), "missing_or_unclear": missing,
            "header_renders_it": rendered}


def check_requirements(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    names = {ln.split("==")[0] for ln in lines}
    return {"passed": bool(lines) and all("==" in ln for ln in lines) and names == {"streamlit"}, "lines": lines}


# --- secret needles (read locally, never written anywhere) ---------------------------------

def owner_needles(root: Path) -> dict[str, bytes] | None:
    secrets = root / "secrets"
    files = {name: secrets / name for name in ("K.bin", "commitment_nonce.bin", "provenance_signing_key.bin",
                                                "trigger_bundle.npz")}
    if not all(p.exists() for p in files.values()):
        return None
    import numpy as np

    from src.watermark.signature import PROJECT_OWNER_ID, derive_signature

    key = files["K.bin"].read_bytes()
    needles: dict[str, bytes] = {}
    for label, value in (("K", key), ("K_hi", key[:len(key) // 2]), ("K_lo", key[len(key) // 2:]),
                         ("S", derive_signature(key, PROJECT_OWNER_ID).value),
                         ("nonce", files["commitment_nonce.bin"].read_bytes()),
                         ("signing seed", files["provenance_signing_key.bin"].read_bytes())):
        needles.update(needle_forms(label, value))
    bundle = files["trigger_bundle.npz"].read_bytes()
    step = max(1, len(bundle) // 8)
    for i in range(0, len(bundle) - 64, step):
        needles[f"trigger bundle file window {i}"] = bundle[i:i + 64]
    with np.load(io.BytesIO(bundle), allow_pickle=False) as z:
        for name in z.files:
            raw = z[name].tobytes()
            if len(raw) >= 4096:
                for i in range(0, len(raw) - 96, len(raw) // 8):
                    needles[f"trigger bundle {name} window {i}"] = raw[i:i + 96]
    armor = subprocess.run(["gpg", "--batch", "--export-secret-keys", "--armor", GPG_FINGERPRINT],
                           capture_output=True).stdout
    body = [ln for ln in armor.splitlines() if ln and not ln.startswith(b"-----") and b":" not in ln
            and not ln.startswith(b"=")]
    if body:
        secret = base64.b64decode(b"".join(body))
        public = subprocess.run(["gpg", "--batch", "--export", GPG_FINGERPRINT], capture_output=True).stdout
        for i in range(0, len(secret) - 32, 16):
            window = secret[i:i + 32]
            if window not in public:  # parts shared with the public key (user ID, public point) are not secret
                needles[f"gpg secret window {i}"] = window
    needles["PGP private key marker"] = b"PGP PRIVATE KEY BLOCK"
    return needles


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    git_snapshot = git_info()
    started = time.perf_counter()
    root = repo_root()
    paths = tracked(root)
    checks: dict[str, dict] = {}

    status = git("status", "--porcelain", root=root).decode().strip()
    checks["clean_tree"] = {"passed": not status, "changed_entries": len(status.splitlines()) if status else 0}
    gitkeep = (root / "secrets/.gitkeep").read_bytes() if "secrets/.gitkeep" in paths else None
    checks["secrets_dir"] = check_secrets_dir(paths, gitkeep)
    checks["secret_file_names"] = check_secret_file_names(paths)
    checks["model_weights"] = check_model_weights(paths)

    import numpy as np

    from src.crypto.fingerprint import fingerprint_state_dict

    with np.load(root / APPROVED_BUNDLE, allow_pickle=False) as z:
        bundle_fp = fingerprint_state_dict({name: z[name] for name in z.files}).sha256
    published = json.loads((root / "provenance/commitment.json").read_text(encoding="utf-8"))["model_fingerprint"]
    p5_1 = sorted((root / "results").glob(P5_1_PREFIX + "*.json"))[-1]
    models = json.loads(p5_1.read_text(encoding="utf-8"))["metrics"]["models"]
    checks["bundle_identity"] = check_bundle_identity(
        bundle_fp, published["sha256"], {k: models[k]["fingerprint"]["sha256"] for k in MUST_NOT_BUNDLE})

    needles = owner_needles(root)
    if needles is None:
        checks["secret_values"] = {"passed": None, "run": False, "reason": "owner secrets not on this machine"}
    else:
        files = {p: (root / p).read_bytes() for p in paths if (root / p).is_file()}
        hits = scan(files, needles)
        checks["secret_values"] = {"passed": not hits, "run": True, "files_scanned": len(files),
                                   "needles": len(needles), "hits": hits}

    from app.pages import PAGES

    checks["page_modes"] = check_page_modes(PAGES, (root / "app/ui.py").read_text(encoding="utf-8"),
                                            (root / "app/overview.py").read_text(encoding="utf-8"))
    checks["app_requirements"] = check_requirements((root / "app/requirements.txt").read_text(encoding="utf-8"))
    manifest = subprocess.run([sys.executable, str(root / "app/build_manifest.py"), "--check"], capture_output=True,
                              text=True)
    checks["manifest"] = {"passed": manifest.returncode == 0, "output": manifest.stdout.strip()[-300:]}

    failed = [name for name, c in checks.items() if c["passed"] is False]
    not_run = [name for name, c in checks.items() if c["passed"] is None]
    path = write_result(
        name="p9.15_predeploy_check", seed=args.seed, task="P9.15",
        params={"approved_bundle": APPROVED_BUNDLE, "committed_before_p9_15": COMMITTED_BEFORE_P9_15,
                "weight_suffixes": list(WEIGHT_SUFFIXES), "must_not_bundle": list(MUST_NOT_BUNDLE)},
        metrics={"checks": checks, "failed": failed, "not_run": not_run, "passed": not failed and not not_run,
                 "tracked_files": len(paths), "bundle_fingerprint": bundle_fp},
        duration_seconds=time.perf_counter() - started, out_dir=args.out_dir, git=git_snapshot,
        notes="Checks the tracked tree a Streamlit Community Cloud deploy would serve. Secret values are compared "
              "in memory and never written; only counts are recorded.")
    for name, c in checks.items():
        print(f"{'PASS' if c['passed'] else 'NOT RUN' if c['passed'] is None else 'FAIL':7s} {name}")
    if checks["model_weights"]["owner_decision_needed"]:
        print("owner decision needed:", ", ".join(checks["model_weights"]["owner_decision_needed"]))
    print("wrote", path)
    if failed or not_run:
        raise SystemExit(f"pre-deploy check did not pass: failed {failed}, not run {not_run}")
    return {"path": path, "checks": checks}


if __name__ == "__main__":
    main()
