"""P6.2: sign the provenance record, ``provenance/record.json``.

    python experiments/p6_2_sign_provenance_record.py            # create it, once
    python experiments/p6_2_sign_provenance_record.py --check    # re-verify the existing one

CPU only, a few seconds. Needs ``provenance/commitment.json`` (P5.4) and
``secrets/trigger_bundle.npz`` (P2.3). It does not read `K`, the commitment
nonce or any model.

What it does:

1. Reads the commitment publication, which must be canonical on disk.
2. Loads the owner's trigger bundle and checks its digest against the one
   P2.3 trained on. Only the digest and the trigger count go into the record.
3. Creates the Ed25519 signing key at ``secrets/provenance_signing_key.bin``
   (gitignored, no passphrase). It is created once and never overwritten.
   **It needs a backup**: without it no further record can be signed under
   the same public key.
4. Builds the record (P6.1 schema) with the current UTC time, signs it, and
   writes it. The record is written once; if it exists the script refuses.
5. Reads the record back from disk and checks that it is canonical, that its
   signature verifies under the public key it names, that this key is the one
   in ``secrets/``, that it matches the publication and the bundle, and that
   the private key does not appear in it.
6. Tampers with copies of the record in memory and counts how many still
   verify. A signature never seen to fail has not been tested.

``--check`` repeats steps 1, 2, 5 and 6 against the existing record and key
and writes no record and no key.

The result record holds public values only. ``created_utc`` is this machine's
clock. The record is not independently timestamped here.
"""

from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime, timedelta
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.crypto.commitment import Commitment
from src.crypto.poseidon import BN254_SCALAR_FIELD
from src.crypto.provenance import (
    RECORD_PATH,
    attach_signature,
    build_unsigned_record,
    read_record,
    signing_payload,
    without_signature,
    write_record,
)
from src.crypto.publication import ARTIFACT_PATH, read_publication, utc_now
from src.crypto.signing import (
    create_signing_key,
    load_signing_key,
    public_key_bytes,
    sign_record,
    signature_is_valid,
)
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.watermark.bundle import load_bundle

DEFAULT_SIGNING_KEY_PATH = repo_root() / "secrets" / "provenance_signing_key.bin"
DEFAULT_BUNDLE_PATH = repo_root() / "secrets" / "trigger_bundle.npz"
P2_3_BUNDLE_SHA256 = "fbd65ec730a3555c5921f13d1a5d4840b6310261ddcaf31d3a724195122baec8"
"""The trigger bundle `W*` was trained on, from results/p2.3_behavioral_wm__seed1337__20260913T112054+0000.json."""


def _flip_hex(text: str) -> str:
    """Change the last hex digit, staying lowercase hex."""
    return text[:-1] + ("0" if text[-1] != "0" else "1")


def field_tampers(record: dict) -> dict[str, dict]:
    """Well-formed copies of `record`, each with one signed field changed and the signature left alone."""

    def changed(path: tuple, value) -> dict:
        out = copy.deepcopy(record)
        target = out
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        return out

    c = Commitment.from_decimal(record["watermark_commitment"]["commitment"]["decimal"])
    later = (datetime.fromisoformat(record["timestamp"]["created_utc"]) + timedelta(seconds=1)).isoformat(timespec="seconds")
    fingerprint = record["model"]["fingerprint"]
    triggers = record["trigger_set_commitment"]
    return {
        "owner_id": changed(("owner", "owner_id"), record["owner"]["owner_id"] + "-x"),
        "public_key": changed(("owner", "public_key", "hex"), _flip_hex(record["owner"]["public_key"]["hex"])),
        "model_label": changed(("model", "label"), record["model"]["label"] + " x"),
        "fingerprint_sha256": changed(("model", "fingerprint", "sha256"), _flip_hex(fingerprint["sha256"])),
        "fingerprint_elements": changed(("model", "fingerprint", "elements"), fingerprint["elements"] + 1),
        "commitment": changed(("watermark_commitment", "commitment"), Commitment((c.value + 1) % BN254_SCALAR_FIELD).to_dict()),
        "trigger_digest": changed(("trigger_set_commitment", "sha256"), _flip_hex(triggers["sha256"])),
        "trigger_count": changed(("trigger_set_commitment", "triggers"), triggers["triggers"] + 1),
        "publication_sha256": changed(("commitment_publication", "sha256"),
                                      _flip_hex(record["commitment_publication"]["sha256"])),
        "publication_path": changed(("commitment_publication", "path"), record["commitment_publication"]["path"] + ".x"),
        "created_utc": changed(("timestamp", "created_utc"), later),
    }


def _safely_valid(candidate: dict) -> bool:
    """`signature_is_valid`, counting a public key the library refuses as not valid."""
    try:
        return signature_is_valid(candidate)
    except ValueError:
        return False


def tamper_checks(record: dict) -> dict:
    """Count tampered copies of `record` that still verify. Every count of accepted copies should be 0 but the last."""
    fields = field_tampers(record)
    accepted_fields = sorted(name for name, candidate in fields.items() if _safely_valid(candidate))

    signature = bytes.fromhex(record["signature"]["hex"])
    unsigned = without_signature(record)
    bit_flips_accepted = 0
    for bit in range(8 * len(signature)):
        flipped = bytearray(signature)
        flipped[bit // 8] ^= 1 << (bit % 8)
        bit_flips_accepted += _safely_valid(attach_signature(unsigned, bytes(flipped)))

    # Another key signs the same bytes. Under the record's own public key that must fail.
    other = Ed25519PrivateKey.generate()
    other_signature_accepted = _safely_valid(attach_signature(unsigned, other.sign(signing_payload(unsigned))))

    # The same statement re-issued under the other key, naming that key. This verifies, and it should:
    # the signature binds the record to a key, not the key to a person.
    reissued = copy.deepcopy(unsigned)
    reissued["owner"]["public_key"]["hex"] = public_key_bytes(other).hex()
    reissued_verifies = signature_is_valid(sign_record(reissued, other))

    return {
        "field_changes_tried": len(fields),
        "field_changes_accepted": len(accepted_fields),
        "field_changes_accepted_names": accepted_fields,
        "signature_bit_flips_tried": 8 * len(signature),
        "signature_bit_flips_accepted": bit_flips_accepted,
        "other_key_signature_accepted_under_owner_key": bool(other_signature_accepted),
        "reissued_under_another_key_verifies_against_that_key": bool(reissued_verifies),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; the key is OS randomness")
    parser.add_argument("--signing-key", type=Path, default=DEFAULT_SIGNING_KEY_PATH)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--publication", type=Path, default=repo_root() / ARTIFACT_PATH)
    parser.add_argument("--record", type=Path, default=repo_root() / RECORD_PATH)
    parser.add_argument("--check", action="store_true", help="verify the existing record; write no record and no key")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    git = git_info()
    started = time.perf_counter()

    if args.check:
        if not args.record.exists():
            raise SystemExit(f"{args.record} does not exist; nothing to check")
        if not args.signing_key.exists():
            raise SystemExit(f"{args.signing_key} does not exist; a check never creates a key")
    elif args.record.exists():
        raise SystemExit(f"{args.record} already exists. Refusing to replace a signed provenance record. Use --check.")

    publication, publication_hash = read_publication(args.publication)
    try:
        bundle = load_bundle(args.bundle, expected_digest=P2_3_BUNDLE_SHA256)
    except ValueError as error:
        raise SystemExit(f"Refusing to sign: {error}") from error
    if bundle.key_kind != "owner":
        raise SystemExit(f"Refusing to sign: {args.bundle} is a {bundle.key_kind!r} bundle, not the owner's")

    key_created = False
    if not args.check and not args.signing_key.exists():
        create_signing_key(args.signing_key)
        key_created = True
    private_key = load_signing_key(args.signing_key)
    public = public_key_bytes(private_key)

    if not args.check:
        unsigned = build_unsigned_record(publication, bundle.digest(), len(bundle), public, utc_now())
        write_record(sign_record(unsigned, private_key), args.record)

    # Everything below is about the file on disk.
    record, record_hash = read_record(args.record)
    if not signature_is_valid(record):
        raise SystemExit("the record's signature does not verify under the public key it names")
    if record["owner"]["public_key"]["hex"] != public.hex():
        raise SystemExit("the record was signed by another key than the one in secrets/")
    expected = build_unsigned_record(publication, bundle.digest(), len(bundle), public, record["timestamp"]["created_utc"])
    if without_signature(record) != expected:
        raise SystemExit("the record does not match the commitment publication and the trigger bundle")
    if record["commitment_publication"]["sha256"] != publication_hash:
        raise SystemExit("the record names another commitment publication")
    resigned = sign_record(expected, private_key)
    if resigned["signature"] != record["signature"]:
        raise SystemExit("signing the same record again gave a different signature")
    serialized = args.record.read_bytes()
    seed = args.signing_key.read_bytes()
    if seed in serialized or seed.hex().encode() in serialized.lower():
        raise SystemExit("the private signing key appears in the record")

    tampering = tamper_checks(record)
    if (tampering["field_changes_accepted"] or tampering["signature_bit_flips_accepted"]
            or tampering["other_key_signature_accepted_under_owner_key"]):
        raise SystemExit(f"a tampered record verified: {tampering}")

    metrics = {
        "mode": "check" if args.check else "create",
        "record_path": RECORD_PATH,
        "record_sha256": record_hash,
        "record_bytes": len(serialized),
        "signature_algorithm": record["signature"]["algorithm"],
        "public_key_hex": record["owner"]["public_key"]["hex"],
        "signature_hex": record["signature"]["hex"],
        "owner_id": record["owner"]["owner_id"],
        "created_utc": record["timestamp"]["created_utc"],
        "commitment_decimal": record["watermark_commitment"]["commitment"]["decimal"],
        "model_fingerprint_sha256": record["model"]["fingerprint"]["sha256"],
        "trigger_set_commitment_sha256": record["trigger_set_commitment"]["sha256"],
        "trigger_count": record["trigger_set_commitment"]["triggers"],
        "commitment_publication_sha256": record["commitment_publication"]["sha256"],
        "signing_key_created_this_run": key_created,
        "signature_verifies": True,
        "signed_by_the_key_in_secrets": True,
        "matches_publication_and_bundle": True,
        "signature_is_deterministic": True,
        "canonical_on_disk": True,
        "private_key_absent_from_record": True,
        "tampering": tampering,
    }
    path = write_result(
        name="p6.2_signed_provenance_record",
        seed=args.seed,
        task="P6.2",
        params={
            "schema": record["schema"],
            "publication": {"path": ARTIFACT_PATH, "sha256": publication_hash},
            "trigger_bundle": {"digest": P2_3_BUNDLE_SHA256, "source": "secrets/trigger_bundle.npz, digest checked"},
            "signing_key": "Ed25519, 32-byte seed from the OS random source, kept in secrets/ (gitignored), no passphrase",
            "library": "cryptography",
            "timestamp": "created_utc is this machine's clock, self-asserted; the record is not independently timestamped",
        },
        metrics=metrics,
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git,
        notes=(
            "Public values only. A valid signature shows the record is unchanged since the named key signed it. It "
            "does not show who holds that key: the same statement re-issued under another key verifies against that "
            "key. The tamper counts are checks on the listed cases, not a proof and not a rate."
        ),
    )

    print(f"{'checked' if args.check else 'wrote'} {args.record}")
    print(f"public key   {record['owner']['public_key']['hex']}  (ed25519)")
    print(f"owner id     {record['owner']['owner_id']}")
    print(f"created_utc  {record['timestamp']['created_utc']}  (self-asserted)")
    print(f"record SHA-256 {record_hash}")
    print(f"tampering    {tampering['field_changes_accepted']} of {tampering['field_changes_tried']} field changes and "
          f"{tampering['signature_bit_flips_accepted']} of {tampering['signature_bit_flips_tried']} signature bit flips verified")
    if key_created:
        print(f"created the signing key at {args.signing_key}. Back it up outside this repo. It has no passphrase.")
    print("wrote", path)
    return {"path": path, "metrics": metrics}


if __name__ == "__main__":
    main()
