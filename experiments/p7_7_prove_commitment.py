"""P7.7: prove knowledge of the real opening of the published commitment, and verify it.

    python experiments/p7_7_prove_commitment.py [--verify-repeats 5]

Local CPU. Reads `secrets/K.bin` and `secrets/commitment_nonce.bin`.

Statement (the P7.5 circuit): I know `K_hi, K_lo, S < 2^128` and
`nonce < 2^248` with `Poseidon(DOMAIN, K_hi, K_lo, S, nonce) == C`, where `C`
is the only public signal, taken from `provenance/commitment.json`.

Steps:

1. Check the inputs are the ones P7.5 and P7.6 fixed: the circuit's SHA-256
   equals P7.5's, the proving key `zk/keys/commitment_opening_final.zkey` and
   the committed verification key `results/zk/p7.6/verification_key.json`
   equal P7.6's SHA-256s, and the R1CS compiled here equals P7.6's.
2. Build the real private inputs from `K`, `S` (derived for the project owner
   id) and the nonce, and check the host `commit()` equals the published `C`.
3. In `secrets/p7.7_witness/` (gitignored) only: write the input JSON,
   compute the witness, `wtns check` it, and `groth16 prove`. The directory
   is deleted afterwards, whatever happens.
4. Check the public signals are exactly `[C]`.
5. Verify the proof with the **committed** verification key, `--verify-repeats`
   times, timing each call. A baseline of the same number of `snarkjs --help`
   calls measures Node start-up, which is included in every snarkjs time.
6. Copy `proof.json` and `public.json` to `results/zk/p7.7/`, write the
   result record, then scan every one of those files for any private value
   (each field element in decimal and hex, and `K`, `S`, nonce as hex). Any
   hit deletes the files and stops.

Times and memory are measured per snarkjs process on this machine
(`GetProcessMemoryInfo`, Windows), not on Colab. Phase 2 of the setup had a
single contributor (P7.6), so a proof under this key is not sound against the
key's creator.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.crypto.commitment import commit, commitment_inputs
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.zk.toolchain import CIRCUITS_DIR, Toolchain

CIRCUIT = CIRCUITS_DIR / "commitment_opening.circom"
ZKEY = repo_root() / "zk" / "keys" / "commitment_opening_final.zkey"
VKEY = repo_root() / "results" / "zk" / "p7.6" / "verification_key.json"
OUT_DIR = repo_root() / "results" / "zk" / "p7.7"
SECRET_WORK_DIR = repo_root() / "secrets" / "p7.7_witness"
PRIVATE = ("K_hi", "K_lo", "S", "nonce")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_record(prefix: str) -> dict:
    files = sorted(glob.glob(str(repo_root() / "results" / f"{prefix}__*.json")))
    if not files:
        raise SystemExit(f"no {prefix} result found")
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def secret_needles() -> list[str]:
    """Every private value as it could appear in a file: decimal and hex (lower case)."""
    from make_master_key import DEFAULT_KEY_PATH, load_key
    from p5_4_publish_commitment import DEFAULT_NONCE_PATH, load_nonce

    from src.watermark.signature import PROJECT_OWNER_ID, derive_signature

    key = load_key(DEFAULT_KEY_PATH)
    nonce = load_nonce(DEFAULT_NONCE_PATH)
    sig = derive_signature(key, PROJECT_OWNER_ID).value
    elements = commitment_inputs(key, sig, nonce)[1:]
    needles = [str(e) for e in elements] + [format(e, "x") for e in elements]
    needles += [key.hex(), key[:16].hex(), key[16:].hex(), sig.hex(), nonce.hex()]
    return needles


def find_secrets(paths: list[Path], needles: list[str]) -> list[str]:
    """Names of the files in which any needle appears (case-insensitive)."""
    return [p.name for p in paths if any(n in p.read_text(encoding="utf-8").lower() for n in needles)]


def check_public_signals(public: list, c: int) -> None:
    if public != [str(c)]:
        raise SystemExit(f"public signals are not exactly [C]: {len(public)} signal(s)")


def run(work_dir: Path, verify_repeats: int) -> dict:
    if (OUT_DIR / "proof.json").exists():
        raise SystemExit(f"{OUT_DIR / 'proof.json'} exists; refusing to replace the committed proof")
    from make_master_key import DEFAULT_KEY_PATH, load_key
    from p5_4_publish_commitment import DEFAULT_NONCE_PATH, load_nonce

    from src.crypto.publication import ARTIFACT_PATH, read_publication
    from src.watermark.signature import PROJECT_OWNER_ID, derive_signature


    p75, p76 = latest_record("p7.5_commitment_circuit"), latest_record("p7.6_groth16_setup")
    if sha256_file(CIRCUIT) != p75["params"]["circuit_sha256"]:
        raise SystemExit("circuit differs from P7.5")
    if sha256_file(ZKEY) != p76["metrics"]["sha256"]["zkey_final"]:
        raise SystemExit("proving key differs from P7.6")
    if sha256_file(VKEY) != p76["metrics"]["sha256"]["verification_key_json"]:
        raise SystemExit("committed verification key differs from P7.6")

    tc = Toolchain(work_dir)
    out = tc.compile(CIRCUIT)
    if sha256_file(out["r1cs"]) != p76["metrics"]["sha256"]["r1cs"]:
        raise SystemExit("R1CS compiled here differs from P7.6's")

    publication, _ = read_publication(ARTIFACT_PATH)
    c = int(publication["commitment"]["decimal"])
    key = load_key(DEFAULT_KEY_PATH)
    nonce = load_nonce(DEFAULT_NONCE_PATH)
    sig = derive_signature(key, PROJECT_OWNER_ID).value
    if commit(key, sig, nonce).value != c:
        raise SystemExit("host commit() of the secrets does not equal the published C")
    private = dict(zip(PRIVATE, commitment_inputs(key, sig, nonce)[1:]))
    del key, nonce, sig

    if SECRET_WORK_DIR.exists():
        shutil.rmtree(SECRET_WORK_DIR)
    try:
        stc = Toolchain(SECRET_WORK_DIR)
        wtns, _ = stc.witness(out["wasm"], {**private, "C": c}, "real")
        stc.check_witness(out["r1cs"], wtns)
        files = stc.prove(ZKEY, wtns, "real")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(files["proof"], OUT_DIR / "proof.json")
        shutil.copyfile(files["public"], OUT_DIR / "public.json")
    finally:
        del private
        shutil.rmtree(SECRET_WORK_DIR, ignore_errors=True)
    if SECRET_WORK_DIR.exists():
        raise SystemExit(f"could not remove {SECRET_WORK_DIR}")

    public = json.loads((OUT_DIR / "public.json").read_text(encoding="utf-8"))
    check_public_signals(public, c)
    proof = json.loads((OUT_DIR / "proof.json").read_text(encoding="utf-8"))

    verify_steps = []
    for i in range(verify_repeats):
        ok = tc.verify(VKEY, OUT_DIR / "public.json", OUT_DIR / "proof.json", f"committed_{i}")
        if not ok:
            raise SystemExit("proof did not verify with the committed verification key")
        verify_steps.append(tc.steps[-1])
    baseline = [tc.snarkjs(f"node_baseline_{i}", "--help", check=False) for i in range(verify_repeats)]

    return {"tc": tc, "stc": stc, "c": c, "proof": proof, "public": public,
            "verify_steps": verify_steps, "baseline": baseline}


def step_dict(s) -> dict:
    return {"name": s.name, "seconds": round(s.seconds, 3), "returncode": s.returncode,
            "peak_working_set_bytes": s.peak_working_set_bytes, "peak_private_bytes": s.peak_private_bytes}


def summary(xs: list[float]) -> dict:
    return {"n": len(xs), "median": round(statistics.median(xs), 3), "min": round(min(xs), 3),
            "max": round(max(xs), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; proving randomness is snarkjs's")
    parser.add_argument("--verify-repeats", type=int, default=5)
    args = parser.parse_args()
    git = git_info()
    start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="zkcrown_p7_7_") as tmp:
        r = run(Path(tmp), args.verify_repeats)
    secret_steps = {s.name: s for s in r["stc"].steps}
    prove, wit = secret_steps["groth16_prove_real"], secret_steps["wtns_calculate_real"]
    verify_s = [s.seconds for s in r["verify_steps"]]
    base_s = [s.seconds for s in r["baseline"]]
    pi_b = r["proof"]["pi_b"]
    proof_path, public_path = OUT_DIR / "proof.json", OUT_DIR / "public.json"

    path = write_result(
        name="p7.7_groth16_proof",
        seed=args.seed,
        task="P7.7",
        params={
            "circuit": "src/zk/circuits/commitment_opening.circom",
            "statement": "know K_hi, K_lo, S < 2^128 and nonce < 2^248 with Poseidon(DOMAIN, K_hi, K_lo, S, nonce) == C; C public",
            "public_C_source": "provenance/commitment.json",
            "proving_key": "zk/keys/commitment_opening_final.zkey (P7.6, SHA-256 checked, not committed)",
            "verification_key": "results/zk/p7.6/verification_key.json (committed, SHA-256 checked)",
            "witness_location": "secrets/p7.7_witness/ (gitignored), deleted after proving",
            "setup_note": "phase 2 had a single contributor (the owner, P7.6): not sound against the key's creator",
            "timing_note": "each time is one snarkjs process including Node start-up; node_baseline is `snarkjs --help`",
            "memory_measurement": "per snarkjs process, Windows GetProcessMemoryInfo, local machine (not Colab)",
        },
        metrics={
            "public_signals": r["public"],
            "public_signals_exactly_C": True,
            "C_published": str(r["c"]),
            "witness_checked": True,
            "proof_verified_with_committed_vkey": True,
            "proof_protocol": r["proof"].get("protocol"),
            "proof_curve": r["proof"].get("curve"),
            "proof_group_elements": {"pi_a": "G1", "pi_b": "G2", "pi_c": "G1"},
            "proof_json_bytes": proof_path.stat().st_size,
            "public_json_bytes": public_path.stat().st_size,
            "proof_sha256": sha256_file(proof_path),
            "pi_b_coordinates": sum(len(x) for x in pi_b),
            "witness_seconds": round(wit.seconds, 3),
            "witness_peak_working_set_bytes": wit.peak_working_set_bytes,
            "prove_seconds": round(prove.seconds, 3),
            "prove_peak_working_set_bytes": prove.peak_working_set_bytes,
            "prove_peak_private_bytes": prove.peak_private_bytes,
            "verify_seconds": summary(verify_s),
            "verify_peak_working_set_bytes_max": max(s.peak_working_set_bytes or 0 for s in r["verify_steps"]) or None,
            "node_baseline_seconds": summary(base_s),
            "steps": [step_dict(s) for s in r["tc"].steps]
                     + [step_dict(s) for s in r["stc"].steps if s.name != "wtns_calculate_real"]
                     + [{**step_dict(wit), "name": "wtns_calculate_real"}],
        },
        notes="Real opening proved in zero knowledge against the published C; witness written under secrets/ only "
              "and deleted. Single-contributor setup (P7.6). Negative tests are P7.8.",
        duration_seconds=time.perf_counter() - start,
        git=git,
    )

    hits = find_secrets([proof_path, public_path, Path(path)], secret_needles())
    if hits:
        for p in (proof_path, public_path, Path(path)):
            p.unlink(missing_ok=True)
        raise SystemExit(f"a private value appeared in {hits}; files deleted")
    mib = (lambda b: f"{b / 2**20:.1f} MiB" if b else "n/a")
    print(f"public signals == [C]; proof {proof_path.stat().st_size:,} bytes JSON; witness {wit.seconds:.2f} s; "
          f"prove {prove.seconds:.2f} s ({mib(prove.peak_working_set_bytes)}); verify median "
          f"{statistics.median(verify_s):.2f} s over {len(verify_s)}; node baseline median "
          f"{statistics.median(base_s):.2f} s; no secret in proof, public or record")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
