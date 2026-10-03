"""P7.6: Groth16 proving and verification keys for the commitment-opening circuit.

    python experiments/p7_6_groth16_setup.py

Local CPU. Reads no key, no nonce and no witness: the setup needs only the
circuit and the powers of tau.

Steps, each a recorded snarkjs command with its time and peak memory:

1. check that `src/zk/circuits/commitment_opening.circom` is byte-identical
   to the circuit P7.5 checked (its SHA-256 in the P7.5 result), compile it,
   and check it still has 1,471 constraints and one public input;
2. re-hash the P7.4 file `zk/ptau/powersOfTau28_hez_final_15.ptau` against
   the BLAKE2b-512 in the snarkjs 0.7.6 README (no download);
3. `groth16 setup` (the circuit-specific phase 2 start), **one** `zkey
   contribute` with fresh OS entropy, `zkey verify` against the R1CS and the
   ptau, and `zkey export verificationkey`.

**Single contributor.** Phase 2 has exactly one contribution, made by this
script on the owner's machine. Its entropy is generated in memory, passed to
snarkjs on the command line, redacted from every record, and never written to
a file. Groth16 is sound as long as one phase 2 contributor discarded their
secret; with one contributor, that is the owner alone. So this proving key
gives **no soundness against its own creator**: whoever ran this could, in
principle, have kept the toxic waste and could forge proofs. Third parties
must trust the owner for that, or the setup must be redone with independent
contributors (not done).

Outputs:
- `zk/keys/commitment_opening_final.zkey`: the proving key, gitignored, kept
  for P7.7. Written once; the script refuses to replace it, since a new setup
  gives a different key and invalidates the committed verification key.
- `results/zk/p7.6/verification_key.json`: committed.
The initial (pre-contribution) zkey and the build files are deleted.

Peak memory is read per command from Windows (`GetProcessMemoryInfo` on the
snarkjs node process: peak working set and peak private bytes). It is this
machine, not Colab. On other platforms it is recorded as not measured.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.zk.ptau import PTAU_DIR, fetch, read_table
from src.zk.toolchain import CIRCUITS_DIR, SETUP_TIMEOUT_SECONDS, Toolchain

CIRCUIT = CIRCUITS_DIR / "commitment_opening.circom"
NAME = "commitment_opening"
PTAU_POWER = 15
EXPECTED_CONSTRAINTS = 1471
KEYS_DIR = repo_root() / "zk" / "keys"
ZKEY_PATH = KEYS_DIR / f"{NAME}_final.zkey"
VKEY_DEST = repo_root() / "results" / "zk" / "p7.6" / "verification_key.json"
CONTRIBUTOR = "blacktensor-zkcrown-owner, single phase 2 contributor (P7.6)"
SETUP_STEPS = ("groth16_setup", "zkey_contribute", "zkey_verify", "zkey_export_vkey")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p7_5_circuit_sha256() -> str:
    files = sorted(glob.glob(str(repo_root() / "results" / "p7.5_commitment_circuit__*.json")))
    if not files:
        raise SystemExit("no P7.5 result found")
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))["params"]["circuit_sha256"]


def check_vkey(vkey: dict) -> None:
    if vkey.get("protocol") != "groth16" or vkey.get("curve") != "bn128" or vkey.get("nPublic") != 1:
        raise SystemExit(f"unexpected verification key header: protocol={vkey.get('protocol')}, "
                         f"curve={vkey.get('curve')}, nPublic={vkey.get('nPublic')}")
    if len(vkey.get("IC", [])) != 2:
        raise SystemExit("verification key should have 2 IC points (1 public input)")


def contribution_hash(output: str) -> str | None:
    """The contribution hash snarkjs prints after `zkey contribute` (public; it identifies the contribution)."""
    m = re.search(r"Contribution Hash:\s*((?:[0-9a-f]{8}\s*)+)", output, flags=re.IGNORECASE)
    return "".join(m.group(1).split()) if m else None


def run(work_dir: Path) -> dict:
    if ZKEY_PATH.exists():
        raise SystemExit(f"{ZKEY_PATH} exists; refusing to replace the proving key")
    if VKEY_DEST.exists():
        raise SystemExit(f"{VKEY_DEST} exists; refusing to replace the verification key")

    circuit_sha = sha256_file(CIRCUIT)
    if circuit_sha != p7_5_circuit_sha256():
        raise SystemExit("circuit differs from the one P7.5 checked")

    entry = read_table()[PTAU_POWER]
    ptau, downloaded = fetch(entry, PTAU_DIR, download=False)
    assert not downloaded

    tc = Toolchain(work_dir)
    out = tc.compile(CIRCUIT)
    info = tc.r1cs_info(out["r1cs"])
    if info["constraints"] != EXPECTED_CONSTRAINTS or info["public_inputs"] != 1:
        raise SystemExit(f"R1CS differs from P7.5: {info}")

    keys = tc.groth16_setup(out["r1cs"], ptau, NAME, contributor=CONTRIBUTOR)
    steps = {s.name: s for s in tc.steps}
    for s in tc.steps:
        if any(a.startswith("-e=") and a != "-e=<redacted>" for a in s.argv):
            raise SystemExit("unredacted entropy in a recorded command")

    vkey = json.loads(keys["vkey"].read_text(encoding="utf-8"))
    check_vkey(vkey)
    verify_out = steps["zkey_verify"].output
    contribs = re.findall(r"contribution #(\d+)", verify_out, flags=re.IGNORECASE)

    sizes = {"r1cs": out["r1cs"].stat().st_size, "zkey_initial": keys["zkey0"].stat().st_size,
             "zkey_final": keys["zkey"].stat().st_size, "verification_key_json": keys["vkey"].stat().st_size,
             "ptau": ptau.stat().st_size}
    hashes = {"r1cs": sha256_file(out["r1cs"]), "zkey_final": sha256_file(keys["zkey"]),
              "verification_key_json": sha256_file(keys["vkey"])}

    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    VKEY_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(keys["zkey"], ZKEY_PATH)
    shutil.copyfile(keys["vkey"], VKEY_DEST)
    if sha256_file(ZKEY_PATH) != hashes["zkey_final"] or sha256_file(VKEY_DEST) != hashes["verification_key_json"]:
        raise SystemExit("copied key files differ from the originals")

    return {"tc": tc, "info": info, "circuit_sha": circuit_sha, "ptau": ptau, "ptau_blake2b": entry.blake2b_512,
            "sizes": sizes, "hashes": hashes, "contribution_hash": contribution_hash(steps["zkey_contribute"].output),
            "contributions_listed_by_verify": sorted(set(contribs)), "verify_ok": "ZKey Ok!" in verify_out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; ceremony entropy is OS random")
    args = parser.parse_args()
    git = git_info()
    start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="zkcrown_p7_6_") as tmp:
        r = run(Path(tmp))
    tc = r["tc"]
    steps = [{"name": s.name, "seconds": round(s.seconds, 3), "returncode": s.returncode,
              "peak_working_set_bytes": s.peak_working_set_bytes, "peak_private_bytes": s.peak_private_bytes}
             for s in tc.steps]
    setup = [s for s in steps if s["name"] in SETUP_STEPS]
    measured = all(s["peak_working_set_bytes"] is not None for s in setup)
    peak_ws = max(s["peak_working_set_bytes"] for s in setup) if measured else None
    peak_private = max(s["peak_private_bytes"] for s in setup) if measured else None

    path = write_result(
        name="p7.6_groth16_setup",
        seed=args.seed,
        task="P7.6",
        params={
            "circuit": "src/zk/circuits/commitment_opening.circom",
            "circuit_sha256": r["circuit_sha"],
            "circuit_equals_p7_5": True,
            "ptau": f"zk/ptau/{r['ptau'].name} (Hermez, power {PTAU_POWER}, P7.4; BLAKE2b-512 re-checked, "
                    "powersoftau verify not completed in P7.4)",
            "ptau_blake2b_512": r["ptau_blake2b"],
            "proof_system": "groth16 over bn128 (snarkjs 0.7.6)",
            "phase2": "single contribution by the owner on this machine; fresh OS entropy passed on the command "
                      "line only, never written to a file, redacted from the record",
            "phase2_contributor_name": CONTRIBUTOR,
            "soundness_note": "single phase 2 contributor: this proving key gives no soundness against its own "
                              "creator, who could have kept the toxic waste",
            "setup_step_timeout_seconds": SETUP_TIMEOUT_SECONDS,
            "memory_measurement": "per snarkjs process, Windows GetProcessMemoryInfo, local machine (not Colab)",
        },
        metrics={
            "r1cs": r["info"],
            "zkey_verify": "ZKey Ok!" if r["verify_ok"] else "failed",
            "contributions_listed_by_zkey_verify": r["contributions_listed_by_verify"],
            "contribution_hash": r["contribution_hash"],
            "file_bytes": r["sizes"],
            "sha256": r["hashes"],
            "setup_peak_working_set_bytes": peak_ws,
            "setup_peak_private_bytes": peak_private,
            "setup_seconds_total": round(sum(s["seconds"] for s in setup), 3),
            "steps": steps,
            "proving_key_path": "zk/keys/commitment_opening_final.zkey (gitignored, not committed)",
            "verification_key_committed": "results/zk/p7.6/verification_key.json",
        },
        notes="Phase 2 had a single contributor (the owner), so the proving key gives no soundness against its "
              "creator. No proof generated (P7.7).",
        duration_seconds=time.perf_counter() - start,
        git=git,
    )
    mib = (lambda b: f"{b / 2**20:.1f} MiB" if b is not None else "n/a")
    print(f"zkey verify: {'ZKey Ok!' if r['verify_ok'] else 'FAILED'}; zkey {r['sizes']['zkey_final']:,} bytes, "
          f"vkey {r['sizes']['verification_key_json']:,} bytes; setup peak working set {mib(peak_ws)}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
