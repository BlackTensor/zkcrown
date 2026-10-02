"""P5.4: write the commitment publication artifact, ``provenance/commitment.json``.

    python experiments/p5_4_publish_commitment.py            # create it, once
    python experiments/p5_4_publish_commitment.py --check    # re-verify the existing one

CPU only, a few seconds. Needs ``secrets/K.bin`` and the dual `W*` weights
file. No data.

What it does:

1. Loads `K` and derives `S` for ``PROJECT_OWNER_ID`` (P2.1).
2. Creates the commitment nonce, 31 bytes from the OS random source, at
   ``secrets/commitment_nonce.bin`` (gitignored). It is created once and never
   overwritten. **Without it the commitment can never be opened or proved**,
   so it needs the same backup as `K`.
3. Computes `C` (P5.3).
4. Fingerprints the final dual `W*` (P5.1). The weights file is checked
   against its recorded file SHA-256, and the fingerprint must equal the one
   P5.1 committed.
5. Writes the artifact with the owner id and the current UTC time, then reads
   it back from disk and checks that it is canonical, well formed, and that
   its `C` is what `K`, `S` and the nonce give.

The artifact is written once. If it already exists the script refuses, since a
published commitment must not be replaced silently. ``--check`` repeats steps
1, 3, 4 and the read-back against the existing file and nonce, and writes no
artifact.

Nothing secret is printed or written outside ``secrets/``. The result record
holds public values only: `C`, the fingerprint, the owner id, the artifact's
hash. ``created_utc`` is this machine's clock; an independent timestamp is
P5.5.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
from make_master_key import DEFAULT_KEY_PATH, load_key
from p2_4_measure_wdr import sha256_file

from src.crypto.commitment import NONCE_BYTES, commit, new_nonce
from src.crypto.fingerprint import fingerprint_file
from src.crypto.publication import (
    ARTIFACT_PATH,
    build_publication,
    read_publication,
    utc_now,
    write_publication,
)
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.watermark.signature import PROJECT_OWNER_ID, derive_signature

DEFAULT_NONCE_PATH = repo_root() / "secrets" / "commitment_nonce.bin"
W_DUAL = "results/p3.6_dual_wm_W_star.pt"
W_DUAL_SHA256 = "7a9a9f141b55c7894b208c1f338f4c7385362f969b82b319ce5893fb4bb434c4"
"""Dual `W*` weights file, from results/p3.6_dual_wm__seed1337__20260914T085541+0000.json."""
P5_1_RESULT = "results/p5.1_model_fingerprint__seed1337__20261002T173049+0000.json"
MODEL_LABEL = "zk-crown main_model, dual-watermarked W* (P3.6), CIFAR-10"


def create_nonce(path: Path) -> Path:
    """Write a fresh nonce to `path`. Raises `FileExistsError` if it exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(new_nonce())
    if path.stat().st_size != NONCE_BYTES:
        raise RuntimeError(f"{path} is not {NONCE_BYTES} bytes after writing")
    return path


def load_nonce(path: Path) -> bytes:
    """Read the nonce from `path`, checking its length."""
    data = Path(path).read_bytes()
    if len(data) != NONCE_BYTES:
        raise ValueError(f"{path} holds {len(data)} bytes, expected {NONCE_BYTES}")
    return data


def model_fingerprint(weights: Path, expected_file_sha256: str, p5_1_result: Path):
    """Fingerprint the dual `W*`, refusing a wrong file or a fingerprint P5.1 did not record."""
    actual = sha256_file(weights)
    if actual != expected_file_sha256:
        raise SystemExit(f"{weights} has SHA-256 {actual}, expected {expected_file_sha256}. Refusing to publish it.")
    fingerprint = fingerprint_file(weights)
    recorded = json.loads(p5_1_result.read_text(encoding="utf-8"))["metrics"]["models"]["dual_W_star"]["fingerprint"]
    if fingerprint.to_dict() != recorded:
        raise SystemExit(f"fingerprint {fingerprint.sha256} differs from P5.1's {recorded['sha256']}")
    return fingerprint


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; the nonce is OS randomness")
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--nonce", type=Path, default=DEFAULT_NONCE_PATH)
    parser.add_argument("--weights", type=Path, default=repo_root() / W_DUAL)
    parser.add_argument("--artifact", type=Path, default=repo_root() / ARTIFACT_PATH)
    parser.add_argument("--check", action="store_true", help="verify the existing artifact; write no artifact")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    git = git_info()
    started = time.perf_counter()

    if args.check:
        if not args.artifact.exists():
            raise SystemExit(f"{args.artifact} does not exist; nothing to check")
    elif args.artifact.exists():
        raise SystemExit(f"{args.artifact} already exists. Refusing to replace a published commitment. Use --check.")

    key = load_key(args.key)
    signature = derive_signature(key, PROJECT_OWNER_ID)
    fingerprint = model_fingerprint(args.weights, W_DUAL_SHA256, repo_root() / P5_1_RESULT)

    nonce_created = False
    if not args.check and not args.nonce.exists():
        create_nonce(args.nonce)
        nonce_created = True
    commitment = commit(key, signature, load_nonce(args.nonce))

    if not args.check:
        record = build_publication(commitment, fingerprint, PROJECT_OWNER_ID, MODEL_LABEL, utc_now())
        write_publication(record, args.artifact)

    # Everything below is about the file on disk.
    published, artifact_hash = read_publication(args.artifact)
    if published["commitment"] != commitment.to_dict():
        raise SystemExit("the artifact's commitment is not what K, S and the nonce give")
    if published["model_fingerprint"] != fingerprint.to_dict():
        raise SystemExit("the artifact's model fingerprint is not the dual W*'s")
    if published["owner_id"] != PROJECT_OWNER_ID:
        raise SystemExit("the artifact names another owner id")
    serialized = args.artifact.read_bytes()
    secrets_absent = all(
        blob.hex().encode() not in serialized.lower() and blob not in serialized
        for blob in (key, signature.value, load_nonce(args.nonce))
    )
    if not secrets_absent:
        raise SystemExit("a secret appears in the artifact")

    metrics = {
        "mode": "check" if args.check else "create",
        "artifact_path": ARTIFACT_PATH,
        "artifact_sha256": artifact_hash,
        "artifact_bytes": len(serialized),
        "commitment": published["commitment"],
        "model_fingerprint_sha256": published["model_fingerprint"]["sha256"],
        "owner_id": published["owner_id"],
        "created_utc": published["created_utc"],
        "nonce_created_this_run": nonce_created,
        "reopened_from_secrets": True,
        "fingerprint_matches_p5_1": True,
        "canonical_on_disk": True,
        "secrets_absent_from_artifact": True,
    }
    path = write_result(
        name="p5.4_commitment_publication",
        seed=args.seed,
        task="P5.4",
        params={
            "schema": published["schema"],
            "commitment_version": published["commitment"]["version"],
            "fingerprint_version": published["model_fingerprint"]["version"],
            "model": {"task": "P3.6", "file": W_DUAL, "file_sha256": W_DUAL_SHA256},
            "p5_1_reference": P5_1_RESULT,
            "nonce": f"{NONCE_BYTES} bytes from the OS random source, kept in secrets/ (gitignored)",
            "timestamp": "created_utc is this machine's clock, self-asserted; independent timestamp is P5.5",
        },
        metrics=metrics,
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git,
        notes=(
            "Public values only. The artifact is unsigned (P6.2) and its timestamp is self-asserted (P5.5). The "
            "commitment was re-computed from K, S and the nonce and equals the published one; that is the owner "
            "checking their own file, not an opening shown to anyone."
        ),
    )

    print(f"{'checked' if args.check else 'wrote'} {args.artifact}")
    print(f"C            {published['commitment']['decimal']}")
    print(f"fingerprint  {published['model_fingerprint']['sha256']}")
    print(f"owner id     {published['owner_id']}")
    print(f"created_utc  {published['created_utc']}  (self-asserted)")
    print(f"artifact SHA-256 {artifact_hash}")
    if nonce_created:
        print(f"created the commitment nonce at {args.nonce}. Back it up with K, outside this repo. "
              "Without it the commitment cannot be opened.")
    print("wrote", path)
    return {"path": path, "metrics": metrics}


if __name__ == "__main__":
    main()
