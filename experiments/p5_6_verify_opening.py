"""P5.6: verify a revealed secret against the published commitment (the non-ZK baseline).

    python experiments/p5_6_verify_opening.py                       # owner self-check, with secrets/
    python experiments/p5_6_verify_opening.py --opening file.json   # check an opening someone handed over

CPU only, a few seconds.

Default mode is the owner checking that the real secrets open the real
artifact, and that nothing else does. The opening is built **in memory** from
``secrets/K.bin``, the signature for ``PROJECT_OWNER_ID`` and
``secrets/commitment_nonce.bin``. It is never written, printed or put in the
result. Steps:

1. Read ``provenance/commitment.json`` (canonical, and with P5.4's SHA-256).
2. The true opening must be accepted.
3. Wrong openings must all be rejected:

   - each of the 632 single-bit flips of `K`, `S` and the nonce;
   - `K`'s two halves swapped; `S` and the nonce each replaced by zeros;
   - 1,000 public wrong keys ``SHA-256("zk-crown/p5.6/wrong-key/v1" || u64 seed
     || u64 j)``, each tried twice: with the true `S` and nonce, and with its
     own correctly derived `S` and the true nonce.

4. A tampered publication must be rejected with the true opening: `C` changed
   by one, and the owner id changed (which leaves `C` matching but breaks the
   signature derivation).
5. Time one verification.

Any acceptance of a wrong opening, or rejection of the true one, stops the run
before a result is written.

``--opening`` mode is what a third party runs: it needs only the artifact and
an opening file, reads nothing under ``secrets/``, and prints the verdict.

This is a self-check, not a disclosure: the owner has shown the opening to
nobody. Really opening the commitment hands the verifier `K` and burns the
watermark key. See `src/crypto/opening.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.crypto.opening import REVEALED_BYTES, Opening, verify_opening
from src.crypto.poseidon import BN254_SCALAR_FIELD
from src.crypto.publication import ARTIFACT_PATH, read_publication
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.watermark.signature import PROJECT_OWNER_ID, derive_signature

P5_4_RESULT = "results/p5.4_commitment_publication__seed1337__20261002T180506+0000.json"
WRONG_KEY_DOMAIN = b"zk-crown/p5.6/wrong-key/v1"
N_WRONG_KEYS = 1000
TIMING_REPEATS = 50


def wrong_key(seed: int, j: int) -> bytes:
    """Public wrong key `j`. Not an owner key."""
    return hashlib.sha256(WRONG_KEY_DOMAIN + struct.pack(">QQ", seed, j)).digest()


def bit_flips(blob: bytes):
    for bit in range(len(blob) * 8):
        flipped = bytearray(blob)
        flipped[bit // 8] ^= 0x80 >> (bit % 8)
        yield bytes(flipped)


def wrong_openings(true: Opening, seed: int, n_keys: int = N_WRONG_KEYS):
    """Yield ``(kind, opening)`` for every wrong opening the self-check tries."""
    for key in bit_flips(true.key):
        yield "key_bit_flip", Opening(key, true.signature, true.nonce)
    for signature in bit_flips(true.signature):
        yield "signature_bit_flip", Opening(true.key, signature, true.nonce)
    for nonce in bit_flips(true.nonce):
        yield "nonce_bit_flip", Opening(true.key, true.signature, nonce)
    yield "key_halves_swapped", Opening(true.key[16:] + true.key[:16], true.signature, true.nonce)
    yield "signature_zero", Opening(true.key, bytes(len(true.signature)), true.nonce)
    yield "nonce_zero", Opening(true.key, true.signature, bytes(len(true.nonce)))
    for j in range(n_keys):
        key = wrong_key(seed, j)
        yield "wrong_key_true_signature", Opening(key, true.signature, true.nonce)
        yield "wrong_key_own_signature", Opening(key, derive_signature(key, PROJECT_OWNER_ID).value, true.nonce)


def self_check(publication: dict, true: Opening, seed: int, n_keys: int = N_WRONG_KEYS) -> dict:
    """Run steps 2 to 5. Raises SystemExit on any wrong outcome. Returns aggregates only."""
    verdict = verify_opening(publication, true)
    if not verdict.valid:
        raise SystemExit(f"the true opening was rejected: {verdict.problem}")

    rejected: dict[str, int] = {}
    for kind, opening in wrong_openings(true, seed, n_keys):
        if verify_opening(publication, opening).valid:
            raise SystemExit(f"a wrong opening was accepted ({kind})")
        rejected[kind] = rejected.get(kind, 0) + 1

    c = int(publication["commitment"]["decimal"])
    other_c = (c + 1) % BN254_SCALAR_FIELD
    tampered_c = json.loads(json.dumps(publication))
    tampered_c["commitment"]["decimal"] = str(other_c)
    tampered_c["commitment"]["hex"] = other_c.to_bytes(32, "big").hex()
    tampered_owner = {**publication, "owner_id": publication["owner_id"] + "-impostor"}
    c_verdict, owner_verdict = verify_opening(tampered_c, true), verify_opening(tampered_owner, true)
    if c_verdict.valid or c_verdict.commitment_matches:
        raise SystemExit("a publication with another C was accepted")
    if owner_verdict.valid or not owner_verdict.commitment_matches or owner_verdict.signature_derives_from_key:
        raise SystemExit("a publication with another owner id was not caught by the signature derivation")

    started = time.perf_counter()
    for _ in range(TIMING_REPEATS):
        verify_opening(publication, true)
    seconds = (time.perf_counter() - started) / TIMING_REPEATS

    return {
        "true_opening": verdict.to_dict(),
        "wrong_openings_tried": sum(rejected.values()),
        "wrong_openings_accepted": 0,
        "wrong_openings_rejected_by_kind": rejected,
        "tampered_publications": {
            "commitment_changed": {"valid": False, "commitment_matches": False},
            "owner_id_changed": {"valid": False, "commitment_matches": True, "signature_derives_from_key": False},
        },
        "seconds_per_verification": seconds,
        "secret_bytes_revealed_by_an_opening": REVEALED_BYTES,
        "opening_disclosed_to_anyone": False,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="selects the public wrong keys")
    parser.add_argument("--artifact", type=Path, default=repo_root() / ARTIFACT_PATH)
    parser.add_argument("--opening", type=Path, default=None, help="verify this opening file; reads no secrets")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    publication, artifact_hash = read_publication(args.artifact)

    if args.opening is not None:
        opening = Opening.from_dict(json.loads(args.opening.read_text(encoding="utf-8")))
        verdict = verify_opening(publication, opening).to_dict()
        print(json.dumps(verdict, indent=1))
        if not verdict["valid"]:
            raise SystemExit(1)
        return verdict

    from make_master_key import DEFAULT_KEY_PATH, load_key
    from p5_4_publish_commitment import DEFAULT_NONCE_PATH, load_nonce

    git = git_info()
    started = time.perf_counter()
    recorded = json.loads((repo_root() / P5_4_RESULT).read_text(encoding="utf-8"))["metrics"]["artifact_sha256"]
    if artifact_hash != recorded:
        raise SystemExit(f"artifact SHA-256 {artifact_hash} differs from P5.4's {recorded}")
    key = load_key(DEFAULT_KEY_PATH)
    true = Opening(key, derive_signature(key, PROJECT_OWNER_ID).value, load_nonce(DEFAULT_NONCE_PATH))
    metrics = {"artifact_sha256": artifact_hash, **self_check(publication, true, args.seed)}

    path = write_result(
        name="p5.6_opening_verifier",
        seed=args.seed,
        task="P5.6",
        params={
            "artifact": ARTIFACT_PATH,
            "p5_4_reference": P5_4_RESULT,
            "checks": ["publication_well_formed", "commitment_matches", "signature_derives_from_key"],
            "wrong_key_family": "SHA-256('zk-crown/p5.6/wrong-key/v1' || u64 seed || u64 j), public",
            "n_wrong_keys": N_WRONG_KEYS,
            "timing_repeats": TIMING_REPEATS,
            "device": "cpu",
        },
        metrics=metrics,
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git,
        notes=(
            "Owner self-check with the opening held in memory only; nothing was disclosed. A real non-ZK opening "
            "hands the verifier K. Rejection of wrong openings is checked on the listed cases, not proved here; "
            "binding rests on Poseidon's collision resistance."
        ),
    )
    print(f"true opening accepted: C matches, S derives from K for {metrics['true_opening']['owner_id']!r}")
    print(f"wrong openings: {metrics['wrong_openings_tried']} tried, 0 accepted  {metrics['wrong_openings_rejected_by_kind']}")
    print("tampered publications (C changed, owner id changed): both rejected")
    print(f"one verification: {metrics['seconds_per_verification'] * 1e3:.2f} ms; "
          f"a real opening reveals {REVEALED_BYTES} secret bytes, including all of K")
    print("wrote", path)
    return {"path": path, "metrics": metrics}


if __name__ == "__main__":
    main()
