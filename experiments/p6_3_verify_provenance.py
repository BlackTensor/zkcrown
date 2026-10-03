"""P6.3: verify the signed provenance record, and check that the verifier fails when it should.

    python experiments/p6_3_verify_provenance.py

CPU only, under a minute. Reads ``provenance/record.json``,
``provenance/commitment.json`` and three weights files, each checked against
its recorded file SHA-256. Reads nothing under ``secrets/``: the verifier needs
public values only, and so does every case below.

Cases (each must give the stated verdict, or the script stops):

1. **Genuine.** The record, its publication, the dual `W*` as suspect and the
   owner's public key as trusted key: every check passes. Also without a
   suspect and key, where the two optional checks are reported as not run.
2. **Other suspects.** Clean `W`, the behavioral-only model and the dual `W*`
   with one bit flipped: the record stays valid, the fingerprint check fails.
3. **Tampered records.** The 11 single-field changes of P6.2 and all 512
   signature bit flips: the signature check fails on every one.
4. **Malformed commitments.** `C` = p, decimal and hex disagreeing, a leading
   zero, a wrong layout version: the commitment check fails on every one.
5. **Tampered publications.** `C` changed, owner id changed, time changed,
   with the genuine record: the publication check fails on every one.
6. **Forgeries under another key**, made here with a fresh key:
   - the owner's statement re-signed by that key;
   - a consistent publication and record of the forger's own, for clean `W`;
   - the genuine record with its fingerprint changed, re-signed.
   The last fails the publication check. The first two pass every check
   except the trusted key. That is the limit of the signature, and the reason
   the trust check exists.

The trusted key here is the public key recorded by P6.2's result file, so this
is a check of the mechanism, not an independent source of trust. A real
verifier would take the key from somewhere the owner cannot rewrite after the
fact.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from p2_4_measure_wdr import W_CLEAN_SHA256, W_STAR_SHA256, sha256_file
from p6_2_sign_provenance_record import field_tampers
from src.crypto.commitment import Commitment
from src.crypto.fingerprint import fingerprint_state_dict
from src.crypto.poseidon import BN254_SCALAR_FIELD
from src.crypto.provenance import RECORD_PATH, attach_signature, build_unsigned_record, read_record, without_signature
from src.crypto.provenance_verifier import verify_provenance
from src.crypto.publication import ARTIFACT_PATH, build_publication, read_publication
from src.crypto.signing import public_key_bytes, sign_record
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

W_DUAL_SHA256 = "7a9a9f141b55c7894b208c1f338f4c7385362f969b82b319ce5893fb4bb434c4"
OWNER_PUBLIC_KEY = "94b0224bc4c2c7bdd80773cf142a3ec9c5c84ab6d55d8e808547a04f28b02d52"
"""From results/p6.2_signed_provenance_record__seed1337__20261002T190547+0000.json."""
RECORD_SHA256 = "28a3ad667ebb6690bae46b7efcfacb0d3ac16cba1c29e340e25a34e4576169c1"
PUBLICATION_SHA256 = "cbdd82d96dd80dade0ab673ba74c37f53c404df358714324d307d0bafb9f231c"
MODELS = {
    "dual_W_star": ("results/p3.6_dual_wm_W_star.pt", W_DUAL_SHA256),
    "clean_W": ("results/p0.5_clean_baseline_W.pt", W_CLEAN_SHA256),
    "behavioral_only_W_star": ("results/p2.3_behavioral_wm_W_star.pt", W_STAR_SHA256),
}
TIMING_REPEATS = 200


def load_state(name: str) -> dict:
    relative, expected = MODELS[name]
    path = repo_root() / relative
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"{path} has SHA-256 {actual}, expected {expected}. Refusing to use it.")
    return torch.load(path, map_location="cpu", weights_only=True)


def flip_one_bit(state: dict) -> dict:
    out = {name: tensor.clone() for name, tensor in state.items()}
    first = out["features.0.weight"].view(-1)
    first[0:1] = (first[0:1].view(torch.int32) ^ 1).view(torch.float32)
    return out


def summary(verdict) -> dict:
    """The verdict without the long hex values that are the same in every case."""
    d = verdict.to_dict()
    return {k: d[k] for k in ("valid", "record_valid", "record_well_formed", "signature_valid", "commitment_well_formed",
                              "matches_publication", "fingerprint_matches", "public_key_trusted", "checks_not_run",
                              "problems")}


def expect(name: str, verdict, **wanted) -> dict:
    got = verdict.to_dict()
    wrong = {k: (got[k], v) for k, v in wanted.items() if got[k] != v}
    if wrong:
        raise SystemExit(f"case {name!r} gave the wrong verdict (got, wanted): {wrong}\nproblems: {got['problems']}")
    return summary(verdict)


def malformed_commitments(record: dict) -> dict[str, dict]:
    def with_commitment(change) -> dict:
        out = copy.deepcopy(record)
        change(out["watermark_commitment"]["commitment"])
        return out

    c = record["watermark_commitment"]["commitment"]
    return {
        "c_equals_p": with_commitment(lambda x: x.update(decimal=str(BN254_SCALAR_FIELD),
                                                          hex=BN254_SCALAR_FIELD.to_bytes(32, "big").hex())),
        "hex_disagrees": with_commitment(lambda x: x.update(hex=Commitment(1).hex())),
        "leading_zero": with_commitment(lambda x: x.update(decimal="0" + c["decimal"])),
        "wrong_version": with_commitment(lambda x: x.update(version="zk-crown/commitment/v0")),
    }


def tampered_publications(publication: dict) -> dict[str, dict]:
    c = Commitment.from_decimal(publication["commitment"]["decimal"])
    out = {}
    changed = copy.deepcopy(publication)
    changed["commitment"] = Commitment((c.value + 1) % BN254_SCALAR_FIELD).to_dict()
    out["commitment_plus_one"] = changed
    changed = copy.deepcopy(publication)
    changed["owner_id"] = publication["owner_id"] + "-x"
    out["owner_id"] = changed
    changed = copy.deepcopy(publication)
    changed["created_utc"] = "2026-01-01T00:00:00+00:00"
    out["created_utc"] = changed
    return out


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; nothing here is random but "
                                                                         "the forger's key")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    git = git_info()
    started = time.perf_counter()

    record, record_hash = read_record(repo_root() / RECORD_PATH)
    publication, publication_hash = read_publication(repo_root() / ARTIFACT_PATH)
    if record_hash != RECORD_SHA256 or publication_hash != PUBLICATION_SHA256:
        raise SystemExit(f"record or publication is not the P6.2 / P5.4 file: {record_hash}, {publication_hash}")

    states = {name: load_state(name) for name in MODELS}
    fingerprints = {name: fingerprint_state_dict(state) for name, state in states.items()}
    flipped = fingerprint_state_dict(flip_one_bit(states["dual_W_star"]))
    cases: dict[str, dict] = {}

    # 1. Genuine.
    genuine = verify_provenance(record, publication, suspect=states["dual_W_star"], trusted_public_key=OWNER_PUBLIC_KEY)
    cases["genuine"] = expect("genuine", genuine, valid=True, record_valid=True, fingerprint_matches=True,
                              public_key_trusted=True, checks_not_run=[], problems=[])
    if genuine.record_sha256 != RECORD_SHA256:
        raise SystemExit("the verdict names another record")
    bare = verify_provenance(record, publication)
    cases["genuine_no_suspect_no_key"] = expect(
        "genuine_no_suspect_no_key", bare, valid=True, record_valid=True, fingerprint_matches=None,
        public_key_trusted=None, checks_not_run=["fingerprint_matches", "public_key_trusted"])

    # 2. Other suspects.
    for name, fp in (("clean_W", fingerprints["clean_W"]),
                     ("behavioral_only_W_star", fingerprints["behavioral_only_W_star"]),
                     ("dual_W_star_one_bit_flipped", flipped)):
        cases[f"suspect_{name}"] = expect(f"suspect_{name}", verify_provenance(record, publication, fp, OWNER_PUBLIC_KEY),
                                          valid=False, record_valid=True, fingerprint_matches=False,
                                          public_key_trusted=True)

    # 3. Tampered records.
    field_rejected = {name: not verify_provenance(candidate, publication).signature_valid
                      for name, candidate in field_tampers(record).items()}
    signature = bytes.fromhex(record["signature"]["hex"])
    unsigned = without_signature(record)
    bit_flips_accepted = 0
    for bit in range(8 * len(signature)):
        flip = bytearray(signature)
        flip[bit // 8] ^= 0x80 >> (bit % 8)
        bit_flips_accepted += verify_provenance(attach_signature(unsigned, bytes(flip)), publication).signature_valid
    if not all(field_rejected.values()) or bit_flips_accepted:
        raise SystemExit(f"a tampered record verified: fields {field_rejected}, bit flips {bit_flips_accepted}")

    # 4. Malformed commitments.
    commitment_rejected = {name: not verify_provenance(candidate, publication).commitment_well_formed
                           for name, candidate in malformed_commitments(record).items()}
    if not all(commitment_rejected.values()):
        raise SystemExit(f"a malformed commitment passed: {commitment_rejected}")

    # 5. Tampered publications.
    publication_rejected = {}
    for name, candidate in tampered_publications(publication).items():
        verdict = verify_provenance(record, candidate, states["dual_W_star"], OWNER_PUBLIC_KEY)
        publication_rejected[name] = not verdict.matches_publication and verdict.signature_valid
    if not all(publication_rejected.values()):
        raise SystemExit(f"a tampered publication was accepted: {publication_rejected}")

    # 6. Forgeries under a fresh key.
    forger = Ed25519PrivateKey.generate()
    forger_public = public_key_bytes(forger)
    reissued = copy.deepcopy(unsigned)
    reissued["owner"]["public_key"]["hex"] = forger_public.hex()
    reissued = sign_record(reissued, forger)
    for key, label, wanted in ((None, "forged_reissue", dict(valid=True, public_key_trusted=None)),
                               (OWNER_PUBLIC_KEY, "forged_reissue_trusted_key", dict(valid=False, public_key_trusted=False))):
        cases[label] = expect(label, verify_provenance(reissued, publication, states["dual_W_star"], key),
                              record_valid=True, fingerprint_matches=True, **wanted)

    own_publication = build_publication(Commitment(12345), fingerprints["clean_W"], publication["owner_id"],
                                        "forger's claim on clean W", publication["created_utc"])
    own_record = sign_record(build_unsigned_record(own_publication, record["trigger_set_commitment"]["sha256"],
                                                   record["trigger_set_commitment"]["triggers"], forger_public,
                                                   record["timestamp"]["created_utc"]), forger)
    for key, label, wanted in ((None, "forged_own_claim", dict(valid=True, public_key_trusted=None)),
                               (OWNER_PUBLIC_KEY, "forged_own_claim_trusted_key", dict(valid=False, public_key_trusted=False))):
        cases[label] = expect(label, verify_provenance(own_record, own_publication, states["clean_W"], key),
                              record_valid=True, fingerprint_matches=True, **wanted)

    edited = copy.deepcopy(reissued)
    del edited["signature"]
    edited["model"]["fingerprint"] = fingerprints["clean_W"].to_dict()
    edited = sign_record(edited, forger)
    cases["forged_fingerprint_swap"] = expect(
        "forged_fingerprint_swap", verify_provenance(edited, publication, states["clean_W"]),
        valid=False, record_valid=False, signature_valid=True, matches_publication=False, fingerprint_matches=True)

    # Timing: the full check with a precomputed suspect fingerprint, and with the fingerprint computed each time.
    start = time.perf_counter()
    for _ in range(TIMING_REPEATS):
        verify_provenance(record, publication, fingerprints["dual_W_star"], OWNER_PUBLIC_KEY)
    record_ms = 1000 * (time.perf_counter() - start) / TIMING_REPEATS
    start = time.perf_counter()
    for _ in range(20):
        verify_provenance(record, publication, states["dual_W_star"], OWNER_PUBLIC_KEY)
    with_fingerprint_ms = 1000 * (time.perf_counter() - start) / 20

    metrics = {
        "record_path": RECORD_PATH,
        "record_sha256": record_hash,
        "publication_path": ARTIFACT_PATH,
        "publication_sha256": publication_hash,
        "trusted_public_key_hex": OWNER_PUBLIC_KEY,
        "trusted_public_key_source": "the P6.2 result record (same repository; a mechanism check, not independent trust)",
        "suspect_fingerprints": {name: fp.sha256 for name, fp in fingerprints.items()},
        "genuine_verdict": genuine.to_dict(),
        "cases": cases,
        "record_field_tampers_tried": len(field_rejected),
        "record_field_tampers_rejected": sum(field_rejected.values()),
        "signature_bit_flips_tried": 8 * len(signature),
        "signature_bit_flips_accepted": int(bit_flips_accepted),
        "malformed_commitments_tried": len(commitment_rejected),
        "malformed_commitments_rejected": sum(commitment_rejected.values()),
        "tampered_publications_tried": len(publication_rejected),
        "tampered_publications_rejected": sum(publication_rejected.values()),
        "verify_ms_precomputed_fingerprint": round(record_ms, 3),
        "verify_ms_including_fingerprint_of_dual_W_star": round(with_fingerprint_ms, 3),
        "timing_repeats": {"precomputed": TIMING_REPEATS, "including_fingerprint": 20},
    }
    path = write_result(
        name="p6.3_provenance_verifier",
        seed=args.seed,
        task="P6.3",
        params={
            "verifier": "src/crypto/provenance_verifier.py",
            "models": {name: {"file": rel, "file_sha256": sha} for name, (rel, sha) in MODELS.items()},
            "forger_key": "fresh Ed25519 key from the OS random source, discarded after the run",
            "secrets_read": "none",
        },
        metrics=metrics,
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git,
        notes=(
            "Public values only. A valid verdict shows the record is intact, agrees with the publication it names, and "
            "(with a suspect) that the suspect is bit for bit the named model. It does not show who holds the signing "
            "key: two forgeries under a fresh key pass every check except the trusted key. It does not check the P5.5 "
            "timestamp, open C, or run any watermark test. Tamper counts are checks on the listed cases, not a proof."
        ),
    )

    print(f"genuine: valid={genuine.valid}  record {record_hash[:16]}...  suspect dual W* fingerprint matches")
    for name, case in cases.items():
        print(f"  {name:32s} valid={case['valid']!s:5}  record_valid={case['record_valid']!s:5}  "
              f"fingerprint={case['fingerprint_matches']!s:5}  trusted_key={case['public_key_trusted']}")
    print(f"tampered records rejected: {sum(field_rejected.values())}/{len(field_rejected)} fields, "
          f"{8 * len(signature) - bit_flips_accepted}/{8 * len(signature)} signature bit flips")
    print(f"malformed commitments rejected {sum(commitment_rejected.values())}/{len(commitment_rejected)}, "
          f"tampered publications rejected {sum(publication_rejected.values())}/{len(publication_rejected)}")
    print(f"verify {record_ms:.2f} ms (fingerprint precomputed), {with_fingerprint_ms:.1f} ms with fingerprinting")
    print("wrote", path)
    return {"path": path, "metrics": metrics}


if __name__ == "__main__":
    main()
