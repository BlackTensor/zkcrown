"""P7.3: toy Groth16 loop, knowledge of a Poseidon preimage, setup through verification.

    python experiments/p7_3_toy_poseidon_proof.py [--work-dir DIR]

Local CPU, seconds. Needs the P7.2 toolchain (`zk/README.md`). Reads no key:
the preimage is two public demo values derived from a fixed label.

Steps, each a recorded circom or snarkjs command:

1. compile `src/zk/circuits/toy_poseidon_preimage.circom` (circomlib
   `Poseidon(2)`, private `preimage[2]`, public `hash`) and read its R1CS
   counts;
2. make a **local, single-contributor** powers of tau just large enough for
   the toy. This is for the toy only: whoever runs it could have kept the
   toxic waste. P7.4 picks the public Hermez file for the real circuit;
3. Groth16 phase 2: setup, one contribution, `zkey verify`, export the
   verification key;
4. compute `hash` with the host implementation (`src/crypto/poseidon.py`),
   compute the witness with circom's generated code, and `wtns check` it.
   The circuit constrains `hash === Poseidon(preimage)`, so the witness can
   only be computed if circom's template and the host code agree. This is the
   first run of the circom template against the P5.2 code;
5. prove and verify.

Two sanity checks that the loop can fail, not the P7.8 negative tests: a
witness with `hash + 1` must not compute, and the honest proof must not verify
against `hash + 1` as its public input.

The proof, public signals and verification key are copied to
`results/zk/p7.3_toy/`. The ptau, zkey, r1cs and witness stay in the work
directory (a temporary one by default), as they are gitignored binaries.
Sizes and times are recorded for this toy only; the real-circuit figures are
P7.6 and P7.7.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.crypto.poseidon import BN254_SCALAR_FIELD, poseidon
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.zk.toolchain import CIRCUITS_DIR, Toolchain

CIRCUIT = CIRCUITS_DIR / "toy_poseidon_preimage.circom"
NAME = "toy_poseidon_preimage"
PREIMAGE_LABEL = b"zk-crown/p7.3/toy-preimage/v1\x00"


def demo_preimage() -> list[int]:
    """Two public 248-bit demo field elements: SHA-256(label || i), first 31 bytes, big-endian."""
    return [int.from_bytes(hashlib.sha256(PREIMAGE_LABEL + bytes([i])).digest()[:31], "big") for i in range(2)]


def ptau_power(info: dict[str, int]) -> int:
    """Smallest power of two covering the constraints plus public inputs and outputs, plus one."""
    need = info["constraints"] + info.get("public_inputs", 0) + info.get("outputs", 0) + 1
    return max(1, math.ceil(math.log2(need)))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(work_dir: Path) -> dict:
    tc = Toolchain(work_dir)
    out = tc.compile(CIRCUIT)
    info = tc.r1cs_info(out["r1cs"])
    power = ptau_power(info)
    ptau = tc.local_powers_of_tau(power, name="toy_pot")
    keys = tc.groth16_setup(out["r1cs"], ptau, NAME)

    preimage = demo_preimage()
    assert all(0 <= x < BN254_SCALAR_FIELD for x in preimage)
    h = poseidon(preimage)

    wtns, _ = tc.witness(out["wasm"], {"preimage": preimage, "hash": h}, "honest")
    tc.check_witness(out["r1cs"], wtns)
    proof = tc.prove(keys["zkey"], wtns, "honest")
    public = json.loads(proof["public"].read_text(encoding="utf-8"))
    if public != [str(h)]:
        raise SystemExit(f"public signals {public} != host Poseidon {h}")
    verified = tc.verify(keys["vkey"], proof["public"], proof["proof"], "honest")
    if not verified:
        raise SystemExit("honest proof did not verify")

    # Sanity: the loop must be able to fail.
    _, bad_wtns = tc.witness(out["wasm"], {"preimage": preimage, "hash": (h + 1) % BN254_SCALAR_FIELD},
                             "wrong_hash", check=False)
    wrong_public = work_dir / "wrong_public.json"
    wrong_public.write_text(json.dumps([str((h + 1) % BN254_SCALAR_FIELD)]), encoding="utf-8")
    wrong_verified = tc.verify(keys["vkey"], wrong_public, proof["proof"], "wrong_public")
    if bad_wtns.returncode == 0:
        raise SystemExit("witness for hash + 1 computed; the circuit does not constrain the hash")
    if "Assert Failed" not in bad_wtns.output:
        raise SystemExit(f"witness for hash + 1 failed for another reason:\n{bad_wtns.output[-800:]}")
    if wrong_verified:
        raise SystemExit("honest proof verified against hash + 1")

    files = {"r1cs": out["r1cs"], "wasm": out["wasm"], "ptau": ptau, "zkey": keys["zkey"], "vkey": keys["vkey"],
             "witness": wtns, "proof": proof["proof"], "public": proof["public"]}
    return {
        "tc": tc, "info": info, "power": power, "preimage": preimage, "hash": h, "files": files,
        "verified": verified, "wrong_hash_witness_exit": bad_wtns.returncode,
        "wrong_hash_failed_on_constraint": "Assert Failed" in bad_wtns.output, "wrong_public_verified": wrong_verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; ceremony entropy is OS random")
    parser.add_argument("--work-dir", type=Path, default=None, help="keep build files here (default: a temp dir)")
    args = parser.parse_args()
    git = git_info()
    start = time.perf_counter()

    tmp = None
    work_dir = args.work_dir
    if work_dir is None:
        tmp = tempfile.mkdtemp(prefix="zkcrown_p7_3_")
        work_dir = Path(tmp)
    try:
        r = run(work_dir)
        files = r["files"]
        dest = repo_root() / "results" / "zk" / "p7.3_toy"
        dest.mkdir(parents=True, exist_ok=True)
        for key, fname in [("proof", "proof.json"), ("public", "public.json"), ("vkey", "verification_key.json")]:
            shutil.copyfile(files[key], dest / fname)
        steps = [{"name": s.name, "seconds": round(s.seconds, 3), "returncode": s.returncode} for s in r["tc"].steps]
        path = write_result(
            name="p7.3_toy_poseidon_proof",
            seed=args.seed,
            task="P7.3",
            params={
                "circuit": str(CIRCUIT.relative_to(repo_root())).replace("\\", "/"),
                "circuit_sha256": sha256_file(CIRCUIT),
                "statement": "know preimage[0..1] with Poseidon(preimage) == hash; hash public",
                "poseidon_template": "circomlib Poseidon(2)",
                "proof_system": "groth16 over bn128 (snarkjs)",
                "circom_flags": "--r1cs --wasm --sym, default optimisation (-O1)",
                "powers_of_tau": f"local single-contributor, power {r['power']} (toy only; not trusted)",
                "phase2": "one local contribution, OS-random entropy, not stored",
                "preimage_label": PREIMAGE_LABEL.decode("ascii").rstrip("\x00") + "\\0",
            },
            metrics={
                "r1cs": r["info"],
                "preimage": [str(x) for x in r["preimage"]],
                "hash_host_poseidon": str(r["hash"]),
                "witness_computed_and_checked": True,
                "public_signals_equal_host_hash": True,
                "honest_proof_verified": r["verified"],
                "sanity_wrong_hash_witness_exit_code": r["wrong_hash_witness_exit"],
                "sanity_wrong_hash_failed_on_constraint": r["wrong_hash_failed_on_constraint"],
                "sanity_proof_verified_against_hash_plus_1": r["wrong_public_verified"],
                "file_bytes": {k: v.stat().st_size for k, v in files.items()},
                "sha256": {k: sha256_file(files[k]) for k in ("vkey", "proof", "public")},
                "steps": steps,
                "committed_copies": "results/zk/p7.3_toy/{proof,public,verification_key}.json",
            },
            notes="Toy circuit and toy, untrusted setup. Sizes and times describe this toy only, not the P7.5 circuit.",
            duration_seconds=time.perf_counter() - start,
            git=git,
        )
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"constraints {r['info'].get('constraints')}, ptau power {r['power']}, verified {r['verified']}, "
          f"wrong-hash witness exit {r['wrong_hash_witness_exit']}, wrong public verified {r['wrong_public_verified']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
