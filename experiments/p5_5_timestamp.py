"""P5.5: timestamp the commitment publication artifact, and record what the timestamps show.

    python experiments/p5_5_timestamp.py stamp      # once: submit the artifact's hash to OpenTimestamps calendars
    python experiments/p5_5_timestamp.py upgrade    # hours later: fetch the Bitcoin attestation into the proof
    python experiments/p5_5_timestamp.py record     # verify the signed tag and the proof offline, write a result

``stamp`` and ``upgrade`` contact public calendar servers and send them a
hash. ``record`` contacts nobody.

The signed tag is made by hand, once, after the commit that holds the artifact,
its ``.ots`` proof and the owner's public key:

    git tag -s -u <OWNER_KEY_FINGERPRINT> provenance-commitment-v1 <commit> -m "..."

What ``record`` checks, and stops on:

- the artifact on disk is canonical and has the SHA-256 that P5.4 recorded;
- the proof file is for exactly that SHA-256;
- the tag has a good signature from ``OWNER_KEY_FINGERPRINT``;
- the artifact, the proof and the public key inside the tagged commit are
  byte-identical to the files on disk;
- the committed public key file is the key that signed, when ``gpg`` is on
  the PATH to read it.

What it reports without judging: whether the proof is still pending or carries
a Bitcoin attestation. A Bitcoin attestation is parsed, not checked against
the blockchain here.

See `src/crypto/timestamping.py` for what each mechanism is evidence of.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.crypto.publication import ARTIFACT_PATH, read_publication
from src.crypto.timestamping import DEFAULT_CALENDARS, describe_proof, parse_verify_tag, stamp_file, upgrade_proof
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

TAG = "provenance-commitment-v1"
OWNER_KEY_FINGERPRINT = "C7301BA7D92FC2A65257BFC2A759F8EC04BF66E7"
"""The owner's Ed25519 signing key, created in P5.5. Public; the private key is in the owner's GPG keyring."""
PUBLIC_KEY_PATH = "provenance/owner_signing_key.asc"
OTS_PATH = ARTIFACT_PATH + ".ots"
P5_4_RESULT = "results/p5.4_commitment_publication__seed1337__20261002T180506+0000.json"


def git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd or repo_root(), capture_output=True)


def tagged_blob_sha256(tag: str, path: str, cwd: Path | None = None) -> str | None:
    """SHA-256 of `path` as stored in the commit `tag` points at. None if absent."""
    done = git("cat-file", "blob", f"{tag}:{path}", cwd=cwd)
    return hashlib.sha256(done.stdout).hexdigest() if done.returncode == 0 else None


def verify_tag(tag: str, cwd: Path | None = None) -> dict:
    """Signature status of an annotated tag, plus the commit it points at."""
    done = git("verify-tag", "--raw", tag, cwd=cwd)
    status = parse_verify_tag(done.stderr.decode("utf-8", "replace"))
    commit = git("rev-parse", f"{tag}^{{commit}}", cwd=cwd)
    kind = git("cat-file", "-t", tag, cwd=cwd)
    return {
        **status,
        "tag": tag,
        "verify_exit_code": done.returncode,
        "object_type": kind.stdout.decode().strip() or None,
        "commit": commit.stdout.decode().strip() or None,
    }


def public_key_fingerprint(path: Path) -> str | None:
    """Fingerprint of the primary key in an armored key file. None if gpg is not on the PATH."""
    gpg = shutil.which("gpg")
    if gpg is None:
        return None
    done = subprocess.run([gpg, "--show-keys", "--with-colons", str(path)], capture_output=True, text=True)
    for line in done.stdout.splitlines():
        if line.startswith("fpr:"):
            return line.split(":")[9]
    return None


def record(args) -> dict:
    git_snapshot = git_info()
    started = time.perf_counter()
    root = repo_root()
    artifact, ots, key = root / ARTIFACT_PATH, root / OTS_PATH, root / PUBLIC_KEY_PATH

    published, artifact_hash = read_publication(artifact)
    recorded = json.loads((root / P5_4_RESULT).read_text(encoding="utf-8"))["metrics"]["artifact_sha256"]
    if artifact_hash != recorded:
        raise SystemExit(f"artifact SHA-256 {artifact_hash} differs from P5.4's {recorded}")

    proof = describe_proof(ots, artifact)
    if not proof["matches_file"] or proof["file_digest"] != artifact_hash:
        raise SystemExit("the OpenTimestamps proof is not for this artifact")

    tag = verify_tag(args.tag)
    if not tag["good"] or tag["verify_exit_code"] != 0:
        raise SystemExit(f"tag {args.tag} has no good signature")
    if tag["fingerprint"] != OWNER_KEY_FINGERPRINT:
        raise SystemExit(f"tag {args.tag} is signed by {tag['fingerprint']}, expected {OWNER_KEY_FINGERPRINT}")
    if tag["object_type"] != "tag":
        raise SystemExit(f"{args.tag} is not an annotated tag")
    on_disk = {ARTIFACT_PATH: artifact_hash, OTS_PATH: None, PUBLIC_KEY_PATH: hashlib.sha256(key.read_bytes()).hexdigest()}
    tagged = {path: tagged_blob_sha256(args.tag, path) for path in on_disk}
    if tagged[ARTIFACT_PATH] != artifact_hash or tagged[PUBLIC_KEY_PATH] != on_disk[PUBLIC_KEY_PATH]:
        raise SystemExit("the tagged commit does not hold these exact artifact and public key files")
    if tagged[OTS_PATH] is None:
        raise SystemExit("the tagged commit holds no OpenTimestamps proof")
    key_fingerprint = public_key_fingerprint(key)
    if key_fingerprint is not None and key_fingerprint != OWNER_KEY_FINGERPRINT:
        raise SystemExit(f"{PUBLIC_KEY_PATH} holds key {key_fingerprint}, not the signing key")

    metrics = {
        "artifact_sha256": artifact_hash,
        "artifact_created_utc_self_asserted": published["created_utc"],
        "opentimestamps": {
            **proof,
            "proof_path": OTS_PATH,
            "independent_time_evidence": bool(proof["bitcoin_attestations"]),
            "proof_in_tag_is_current": tagged[OTS_PATH] == proof["ots_sha256"],
        },
        "gpg_tag": {
            **tag,
            "expected_fingerprint": OWNER_KEY_FINGERPRINT,
            "tagged_commit_holds_artifact": True,
            "tagged_commit_holds_public_key": True,
            "public_key_path": PUBLIC_KEY_PATH,
            "public_key_file_fingerprint": key_fingerprint,
            "signature_time_is_self_asserted": True,
            "pushed_to_a_third_party": False,
        },
    }
    path = write_result(
        name="p5.5_timestamp",
        seed=args.seed,
        task="P5.5",
        params={
            "artifact": ARTIFACT_PATH,
            "p5_4_reference": P5_4_RESULT,
            "tag": args.tag,
            "calendars_submitted_to": list(DEFAULT_CALENDARS),
            "ots_client": "opentimestamps library (the ots CLI does not start on this machine)",
            "bitcoin_attestation_checked_against_chain": False,
        },
        metrics=metrics,
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git_snapshot,
        notes=(
            "The GPG tag shows who vouched for the commit; its date is the signer's clock. The OpenTimestamps proof "
            "is time evidence only once it carries a Bitcoin attestation that has been checked against the chain. "
            "A pending proof holds calendar promises only."
        ),
    )
    print(f"artifact SHA-256      {artifact_hash}")
    print(f"OpenTimestamps        {proof['status']}; pending calendars {len(proof['pending_calendars'])}; "
          f"Bitcoin attestations {[a['height'] for a in proof['bitcoin_attestations']]}")
    print(f"tag {args.tag}  good signature by {tag['fingerprint']} on commit {tag['commit'][:7]}, "
          f"signed {tag['signed_date']} (signer's clock)")
    print("wrote", path)
    return {"path": path, "metrics": metrics}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("stamp", "upgrade", "record"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only")
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "stamp":
        read_publication(repo_root() / ARTIFACT_PATH)  # only a canonical, valid artifact is stamped
        outcome = stamp_file(repo_root() / ARTIFACT_PATH)
        print(json.dumps(outcome, indent=1))
        return outcome
    if args.command == "upgrade":
        outcome = upgrade_proof(repo_root() / OTS_PATH)
        print(json.dumps(outcome, indent=1))
        return outcome
    return record(args)


if __name__ == "__main__":
    main()
