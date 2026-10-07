"""P5.5: timestamp the commitment publication artifact, and record what the timestamps show.

    python experiments/p5_5_timestamp.py stamp      # once: submit the artifact's hash to OpenTimestamps calendars
    python experiments/p5_5_timestamp.py upgrade    # hours later: fetch the Bitcoin attestation into the proof
    python experiments/p5_5_timestamp.py record     # verify the signed tag and the proof offline, write a result
    python experiments/p5_5_timestamp.py status     # describe both proofs offline, write a result
    python experiments/p5_5_timestamp.py record --target record   # chain-check the provenance record's own proof

``stamp`` and ``upgrade`` act on ``provenance/commitment.json`` by default.
With ``--target record`` they act on the signed provenance record
``provenance/record.json`` instead (the owner's requirement under P6.2), whose
proof is ``provenance/record.json.ots``. The P5.5 timestamp does not cover the
record, so the record gets its own proof. ``status`` reports both proofs.

``stamp`` and ``upgrade`` contact public calendar servers and send them a
hash. ``record`` contacts two public block explorers and sends them only block
heights and hashes. ``status`` contacts nobody.

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

- every Bitcoin attestation in the proof checks against the chain: the
  block's header, fetched from two explorers and hashed here, meets its
  proof-of-work target and carries the Merkle root the proof requires
  (`check_bitcoin_attestations`). A pending proof with no attestation is
  reported as such, not stopped on.

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

from src.crypto.provenance import RECORD_PATH, read_record
from src.crypto.publication import ARTIFACT_PATH, read_publication
from src.crypto.timestamping import (
    DEFAULT_CALENDARS,
    DEFAULT_EXPLORERS,
    check_bitcoin_attestations,
    describe_proof,
    parse_verify_tag,
    stamp_file,
    upgrade_proof,
)
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

TAG = "provenance-commitment-v1"
OWNER_KEY_FINGERPRINT = "C7301BA7D92FC2A65257BFC2A759F8EC04BF66E7"
"""The owner's Ed25519 signing key, created in P5.5. Public; the private key is in the owner's GPG keyring."""
PUBLIC_KEY_PATH = "provenance/owner_signing_key.asc"
OTS_PATH = ARTIFACT_PATH + ".ots"
P5_4_RESULT = "results/p5.4_commitment_publication__seed1337__20261002T180506+0000.json"
RECORD_OTS_PATH = RECORD_PATH + ".ots"
P6_2_RECORD_SHA256 = "28a3ad667ebb6690bae46b7efcfacb0d3ac16cba1c29e340e25a34e4576169c1"
"""The signed provenance record written by P6.2."""
TARGETS = {"commitment": (ARTIFACT_PATH, OTS_PATH), "record": (RECORD_PATH, RECORD_OTS_PATH)}


def read_target(target: str) -> str:
    """Read and validate the target file; return its SHA-256. Only a canonical, valid file is stamped."""
    path = repo_root() / TARGETS[target][0]
    if target == "commitment":
        return read_publication(path)[1]
    _, digest = read_record(path)
    if digest != P6_2_RECORD_SHA256:
        raise SystemExit(f"{path} has SHA-256 {digest}, not the P6.2 record {P6_2_RECORD_SHA256}")
    return digest


def status(args) -> dict:
    """Describe both proofs offline. Checks each proof is for the file on disk; judges nothing else."""
    git_snapshot = git_info()
    started = time.perf_counter()
    proofs = {}
    for target, (file_path, ots_path) in TARGETS.items():
        digest = read_target(target)
        ots = repo_root() / ots_path
        if not ots.exists():
            proofs[target] = {"file": file_path, "file_sha256": digest, "proof_path": ots_path, "status": "no proof"}
            continue
        proof = describe_proof(ots, repo_root() / file_path)
        if not proof["matches_file"] or proof["file_digest"] != digest:
            raise SystemExit(f"{ots_path} is not a proof for {file_path}")
        proofs[target] = {"file": file_path, "file_sha256": digest, "proof_path": ots_path, **proof,
                          "independent_time_evidence": False}
    path = write_result(
        name="p5.5_timestamp_status",
        seed=args.seed,
        task="P5.5",
        params={
            "targets": {t: f for t, (f, _) in TARGETS.items()},
            "calendars_submitted_to": list(DEFAULT_CALENDARS),
            "ots_client": "opentimestamps library (the ots CLI does not start on this machine)",
            "bitcoin_attestation_checked_against_chain": False,
        },
        metrics={"proofs": proofs},
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git_snapshot,
        notes=(
            "Offline description of the OpenTimestamps proofs. independent_time_evidence stays false until a Bitcoin "
            "attestation has been checked against the chain, which this command does not do."
        ),
    )
    for target, proof in proofs.items():
        heights = [a["height"] for a in proof.get("bitcoin_attestations", [])]
        print(f"{target:10s} {proof['file']}  {proof['status']}; pending calendars "
              f"{len(proof.get('pending_calendars', []))}; Bitcoin attestations {heights}")
    print("wrote", path)
    return {"path": path, "proofs": proofs}


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

    chain = None
    if proof["bitcoin_attestations"]:
        chain = check_bitcoin_attestations(proof["bitcoin_attestations"], explorers=args.explorers)
        if not chain["all_verified"]:
            raise SystemExit(f"a Bitcoin attestation did not check against the chain: {json.dumps(chain, indent=1)}")

    metrics = {
        "artifact_sha256": artifact_hash,
        "artifact_created_utc_self_asserted": published["created_utc"],
        "opentimestamps": {
            **proof,
            "proof_path": OTS_PATH,
            "chain_check": chain,
            "independent_time_evidence": chain is not None and chain["all_verified"],
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
            "bitcoin_attestation_checked_against_chain": chain is not None,
            "explorers": list(args.explorers),
        },
        metrics=metrics,
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git_snapshot,
        notes=(
            "The GPG tag shows who vouched for the commit; its date is the signer's clock. The OpenTimestamps proof "
            "is time evidence only once it carries a Bitcoin attestation that has been checked against the chain. "
            "A pending proof holds calendar promises only. The chain check trusts two explorers' view of which block "
            "is at each height; it hashes the headers and checks their work, but it is not a full node."
        ),
    )
    print(f"artifact SHA-256      {artifact_hash}")
    print(f"OpenTimestamps        {proof['status']}; pending calendars {len(proof['pending_calendars'])}; "
          f"Bitcoin attestations {[a['height'] for a in proof['bitcoin_attestations']]}")
    if chain is not None:
        for block in chain["blocks"]:
            print(f"block {block['height']}  {block['block_hash']}  header time {block['header_time_utc']}  "
                  f"Merkle root matches on {len(block['explorers'])} explorers")
    print(f"tag {args.tag}  good signature by {tag['fingerprint']} on commit {tag['commit'][:7]}, "
          f"signed {tag['signed_date']} (signer's clock)")
    print("wrote", path)
    return {"path": path, "metrics": metrics}


def record_proof(args) -> dict:
    """Check the provenance record's own proof against the chain and write a result, pass or fail.

    The record's proof (``provenance/record.json.ots``, P6.2) gets the same chain
    check as the commitment publication: each attested block's header is fetched
    from the explorers, hashed here, checked for work, and its Merkle root
    compared with the proof's. Unlike ``record`` for the commitment, a failed
    check does not stop the run: the result says which blocks were and were not
    verified, so the dashboard can state it. There is no tag for the record.
    """
    git_snapshot = git_info()
    started = time.perf_counter()
    root = repo_root()
    digest = read_target("record")
    proof = describe_proof(root / RECORD_OTS_PATH, root / RECORD_PATH)
    if not proof["matches_file"] or proof["file_digest"] != digest:
        raise SystemExit(f"{RECORD_OTS_PATH} is not a proof for {RECORD_PATH}")
    chain = None
    if proof["bitcoin_attestations"]:
        chain = check_bitcoin_attestations(proof["bitcoin_attestations"], explorers=args.explorers)
    verified = chain is not None and chain["all_verified"]
    metrics = {
        "record_path": RECORD_PATH,
        "record_sha256": digest,
        "opentimestamps": {
            **proof,
            "proof_path": RECORD_OTS_PATH,
            "chain_check": chain,
            "independent_time_evidence": verified,
        },
    }
    path = write_result(
        name="p6.2_record_timestamp",
        seed=args.seed,
        task="P6.2",
        params={
            "record": RECORD_PATH,
            "proof": RECORD_OTS_PATH,
            "calendars_submitted_to": list(DEFAULT_CALENDARS),
            "ots_client": "opentimestamps library (the ots CLI does not start on this machine)",
            "bitcoin_attestation_checked_against_chain": chain is not None,
            "explorers": list(args.explorers),
            "tag": "none: the record has no signed tag, and none was made",
        },
        metrics=metrics,
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git_snapshot,
        notes=(
            "Same chain check as the commitment publication (P5.5): two explorers, header hashed locally, work "
            "checked, Merkle root compared. Not a full node. Recorded whether or not every block verified; "
            "independent_time_evidence is true only if all did."
        ),
    )
    print(f"record SHA-256        {digest}")
    print(f"OpenTimestamps        {proof['status']}; Bitcoin attestations {[a['height'] for a in proof['bitcoin_attestations']]}")
    for block in (chain or {}).get("blocks", []):
        print(f"block {block['height']}  verified {block['verified']}  {block.get('block_hash')}  "
              f"header time {block.get('header_time_utc')}")
    print(f"independent time evidence: {verified}")
    print("wrote", path)
    return {"path": path, "metrics": metrics}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("stamp", "upgrade", "record", "status"))
    parser.add_argument("--target", choices=tuple(TARGETS), default="commitment", help="file for stamp and upgrade")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only")
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--explorers", nargs="+", default=list(DEFAULT_EXPLORERS), help="for record's chain check")
    args = parser.parse_args(argv)

    file_path, ots_path = TARGETS[args.target]
    if args.command == "stamp":
        read_target(args.target)
        outcome = stamp_file(repo_root() / file_path)
        print(json.dumps(outcome, indent=1))
        return outcome
    if args.command == "upgrade":
        outcome = upgrade_proof(repo_root() / ots_path)
        print(json.dumps(outcome, indent=1))
        return outcome
    if args.command == "status":
        return status(args)
    if args.target == "record":
        return record_proof(args)
    return record(args)


if __name__ == "__main__":
    main()
