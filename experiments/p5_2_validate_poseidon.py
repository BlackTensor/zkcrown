"""P5.2: validate the Python Poseidon against known test vectors before anything trusts it.

    python experiments/p5_2_validate_poseidon.py

CPU only, a few seconds. No key, no model, no data.

`src/crypto/poseidon.py` has to equal circomlib's Poseidon, because the P7
circuit recomputes the P5.3 commitment with circomlib. Three independent
references are checked, and any mismatch stops the run before a result is
written:

1. **Published vectors**, typed in below, not produced by any code in this
   repo. They were entered from recall of their sources and the sources were
   not re-fetched. The three circomlibjs values are confirmed by check 2,
   where they reappear as the "counting" vectors for 2, 4 and 16 inputs. The
   permutation vector has no second confirmation here beyond matching:

   - the Poseidon reference implementation's test vector for the x^5, 254-bit,
     t = 3 permutation on the state ``[0, 1, 2]``
     (``poseidonperm_x5_254_3`` in the authors' reference repository);
   - the values circomlibjs's own test suite asserts for ``poseidon([1, 2])``,
     ``poseidon([1, 2, 3, 4])`` and ``poseidon([1 … 16])``.

2. **circomlibjs vectors**, ``tests/data/poseidon_circomlibjs_vectors.json``,
   produced by ``experiments/p5_2_make_poseidon_vectors.mjs`` with Node and
   the npm package alone: 8 vectors for each input count 1 to 16 (counting,
   all zeros, all ``p - 1``, and 5 pseudo-random), each hashed by both of
   circomlibjs's implementations. The pseudo-random inputs are re-derived
   here from their stated rule, so the file's inputs are checked too.

3. **circomlibjs's constants.** For every width 2 to 17, the SHA-256 of the
   round constants and of the MDS matrix this module generates with the Grain
   LFSR must equal the digests of circomlibjs's shipped constant tables.

What this does not check: the circom *circuit* itself. circomlibjs is the
JavaScript side of circomlib, and snarkjs witnesses agree with it, but the
first run of the actual ``Poseidon`` template against this module is P7.3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.crypto.poseidon import (
    BN254_SCALAR_FIELD,
    FULL_ROUNDS,
    MAX_INPUTS,
    POSEIDON_INSTANCE,
    grain_parameters,
    partial_rounds,
    poseidon,
    poseidon_permutation,
)
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

VECTORS = "tests/data/poseidon_circomlibjs_vectors.json"

REFERENCE_PERMUTATION_T3 = {
    "source": "Poseidon reference implementation, test vector poseidonperm_x5_254_3",
    "input": [0, 1, 2],
    "output": [
        0x115CC0F5E7D690413DF64C6B9662E9CF2A3617F2743245519E19607A4417189A,
        0x0FCA49B798923AB0239DE1C9E7A4A9A2210312B6A2F616D18B5A87F9B628AE29,
        0x0E7AE82E40091E63CBD4F16A6D16310B3729D4B6E138FCF54110E2867045A30C,
    ],
}
PUBLISHED_HASHES = [
    {"source": "circomlibjs test suite", "inputs": [1, 2],
     "output": 7853200120776062878684798364095072458815029376092732009249414926327459813530},
    {"source": "circomlibjs test suite", "inputs": [1, 2, 3, 4],
     "output": 18821383157269793795438455681495246036402687001665670618754263018637548127333},
    {"source": "circomlibjs test suite", "inputs": list(range(1, 17)),
     "output": 9989051620750914585850546081941653841776809718687451684622678807385399211877},
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def constant_digests(t: int) -> dict:
    """Digests of this module's constants for width `t`, in the vector file's encoding."""
    constants, mds = grain_parameters(t)
    return {
        "count": len(constants),
        "C_sha256": sha256_text(",".join(str(c) for c in constants)),
        "M_sha256": sha256_text(";".join(",".join(str(m) for m in row) for row in mds)),
    }


def expected_inputs(record: dict, n: int, kind: str) -> list[int]:
    """The inputs a vector of this kind must have, by the generator's stated rule."""
    p = BN254_SCALAR_FIELD
    if kind == "counting":
        return list(range(1, n + 1))
    if kind == "zeros":
        return [0] * n
    if kind == "max":
        return [p - 1] * n
    if kind.startswith("random"):
        j = int(kind[len("random"):])
        return [int(sha256_text(f"{record['vector_domain']}|{n}|{j}|{i}"), 16) % p for i in range(n)]
    raise SystemExit(f"unknown vector kind {kind!r}")


def validate(record: dict) -> dict:
    """Run all three checks. Raises SystemExit on the first mismatch."""
    if int(record["field"]) != BN254_SCALAR_FIELD:
        raise SystemExit("the vector file is for another field")

    if poseidon_permutation(REFERENCE_PERMUTATION_T3["input"]) != REFERENCE_PERMUTATION_T3["output"]:
        raise SystemExit("the reference permutation vector for t = 3 does not match")
    for vector in PUBLISHED_HASHES:
        if poseidon(vector["inputs"]) != vector["output"]:
            raise SystemExit(f"published vector for {len(vector['inputs'])} inputs does not match")

    per_arity = {n: 0 for n in range(1, MAX_INPUTS + 1)}
    kinds: dict[str, int] = {}
    for vector in record["vectors"]:
        n, kind = vector["n"], vector["kind"]
        inputs = [int(x) for x in vector["inputs"]]
        if inputs != expected_inputs(record, n, kind):
            raise SystemExit(f"vector n={n} {kind}: inputs do not follow the generator's rule")
        if poseidon(inputs) != int(vector["output"]):
            raise SystemExit(f"vector n={n} {kind}: output differs from circomlibjs")
        per_arity[n] += 1
        kinds[kind.rstrip("0123456789")] = kinds.get(kind.rstrip("0123456789"), 0) + 1
    if min(per_arity.values()) < 1:
        raise SystemExit("some input count has no vector")

    total_constants = 0
    for t in range(2, MAX_INPUTS + 2):
        mine, theirs = constant_digests(t), record["constant_digests"][str(t)]
        if mine != theirs:
            raise SystemExit(f"constants for width {t} differ from circomlibjs")
        if mine["count"] != (FULL_ROUNDS + partial_rounds(t)) * t:
            raise SystemExit(f"width {t}: unexpected number of round constants")
        total_constants += mine["count"]

    return {
        "reference_permutation_t3_matches": True,
        "published_hash_vectors_matched": len(PUBLISHED_HASHES),
        "circomlibjs_vectors_matched": len(record["vectors"]),
        "circomlibjs_vectors_total": len(record["vectors"]),
        "vectors_per_input_count": per_arity,
        "vectors_by_kind": kinds,
        "widths_with_matching_constants": MAX_INPUTS,
        "round_constants_compared": total_constants,
        "mds_matrices_compared": MAX_INPUTS,
        "mismatches": 0,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; nothing here is random")
    parser.add_argument("--vectors", type=Path, default=repo_root() / VECTORS)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    git = git_info()
    started = time.perf_counter()
    raw = args.vectors.read_bytes()
    record = json.loads(raw)
    metrics = validate(record)

    # Host-side cost of one 3-input hash (the P5.3 commitment's shape), constants already generated.
    repeats = 200
    t0 = time.perf_counter()
    for i in range(repeats):
        poseidon([i, 2, 3])
    metrics["seconds_per_3_input_hash"] = (time.perf_counter() - t0) / repeats

    path = write_result(
        name="p5.2_poseidon_validation",
        seed=args.seed,
        task="P5.2",
        params={
            "instance": POSEIDON_INSTANCE,
            "field": str(BN254_SCALAR_FIELD),
            "full_rounds": FULL_ROUNDS,
            "partial_rounds": {t: partial_rounds(t) for t in range(2, MAX_INPUTS + 2)},
            "constants": "generated by the Grain LFSR in src/crypto/poseidon.py; no table copied in",
            "vectors_file": VECTORS,
            "vectors_file_sha256": hashlib.sha256(raw).hexdigest(),
            "circomlibjs_version": record["circomlibjs_version"],
            "node_version": record["node_version"],
            "published_sources": sorted({REFERENCE_PERMUTATION_T3["source"], *(v["source"] for v in PUBLISHED_HASHES)}),
        },
        metrics=metrics,
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git,
        notes=(
            "Validation against circomlibjs and published vectors only. The circom Poseidon template itself is first "
            "run against this implementation in P7.3. Not constant time."
        ),
    )
    print(f"reference permutation (t = 3): match; published hashes: {metrics['published_hash_vectors_matched']} match")
    print(f"circomlibjs {record['circomlibjs_version']} vectors: {metrics['circomlibjs_vectors_matched']} of "
          f"{metrics['circomlibjs_vectors_total']} match")
    print(f"constants: {metrics['round_constants_compared']} round constants and {metrics['mds_matrices_compared']} "
          f"MDS matrices equal circomlibjs's, widths 2 to {MAX_INPUTS + 1}")
    print(f"one 3-input hash: {metrics['seconds_per_3_input_hash'] * 1e3:.2f} ms")
    print("wrote", path)
    return {"path": path, "metrics": metrics}


if __name__ == "__main__":
    main()
