"""P7.5: compile the commitment-opening circuit and check it against P5.3.

    python experiments/p7_5_commitment_circuit.py [--no-real]

Local CPU, under a minute. No setup, no proof (P7.6, P7.7).

Checks, each a witness computation with circom's generated code followed by
`snarkjs wtns check` against the R1CS. A witness exists only if every
constraint holds, so a computed and checked witness means the circuit accepts
that input. Any check that disagrees stops the run before a result is written.

1. **Layout.** The `DOMAIN` literal in the circuit equals
   `src.crypto.commitment.DOMAIN_ELEMENT`.
2. **Size.** Constraint count from `snarkjs r1cs info`; whether the P7.4 file
   (power 15) covers it, and the smallest power that would.
3. **Fixed vector.** The public demo values in
   `tests/data/commitment_circomlibjs_vector.json` (computed by circomlibjs from
   the P5.3 specification) are accepted with their `C`, and the host's
   `commit()` gives the same `C`.
4. **Range checks.** For each private input, with `C` recomputed by the host
   Poseidon for the exact values used, so only the range check can fail:
   the largest in-range value (2^128 - 1, nonce 2^248 - 1) is accepted, and
   the smallest out-of-range value (2^128, nonce 2^248) and p - 1 are refused,
   with the refusal coming from a Num2Bits assertion. These show the circuit
   accepts exactly the values the host's byte encoding can produce. Systematic
   negative tests of proofs are P7.8.
5. **Wrong C.** The demo opening with `C + 1` is refused on the `C === h.out`
   line.
6. **Real opening** (skipped with `--no-real`). `K` from `secrets/K.bin`, `S`
   derived for the project owner id, the nonce from
   `secrets/commitment_nonce.bin`, and `C` from `provenance/commitment.json`.
   The input file and the witness, which contain the secrets, are written only
   under `secrets/p7.5_witness/` (gitignored) and deleted afterwards. Nothing
   about them is printed or recorded except whether the check passed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.crypto.commitment import DOMAIN_ELEMENT, commit, commitment_inputs
from src.crypto.poseidon import BN254_SCALAR_FIELD, poseidon
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.zk.ptau import required_power
from src.zk.toolchain import CIRCUITS_DIR, Toolchain, ToolError

CIRCUIT = CIRCUITS_DIR / "commitment_opening.circom"
VECTOR = repo_root() / "tests" / "data" / "commitment_circomlibjs_vector.json"
PTAU_POWER = 15  # the P7.4 file
REAL_WORK_DIR = repo_root() / "secrets" / "p7.5_witness"
PRIVATE = ("K_hi", "K_lo", "S", "nonce")
BITS = {"K_hi": 128, "K_lo": 128, "S": 128, "nonce": 248}


def circuit_domain(source: str) -> int:
    m = re.search(r"var DOMAIN = (\d+);", source)
    if m is None:
        raise SystemExit("no DOMAIN literal in the circuit")
    return int(m.group(1))


def code_without_comments(source: str) -> str:
    """The circuit source with // and /* */ comments removed."""
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", source, flags=re.S))


def demo_vector() -> dict:
    v = json.loads(VECTOR.read_text(encoding="utf-8"))
    names, inputs = v["input_names"], [int(x) for x in v["inputs"]]
    if names != ["DOMAIN", *PRIVATE]:
        raise SystemExit(f"unexpected vector input names {names}")
    if inputs[0] != DOMAIN_ELEMENT:
        raise SystemExit("vector DOMAIN differs from the host DOMAIN_ELEMENT")
    return {"v": v, "private": dict(zip(PRIVATE, inputs[1:])), "C": int(v["commitment_decimal"])}


def host_c(private: dict) -> int:
    return poseidon([DOMAIN_ELEMENT, *(private[k] for k in PRIVATE)])


def witness_ok(tc: Toolchain, wasm: Path, r1cs: Path, inputs: dict, name: str) -> tuple[bool, str]:
    """(accepted, failure output). Accepted means computed and `wtns check` passed."""
    wtns, step = tc.witness(wasm, inputs, name, check=False)
    if step.returncode != 0:
        return False, step.output
    tc.check_witness(r1cs, wtns)
    return True, ""


def range_cases(base: dict) -> list[tuple[str, dict, bool]]:
    cases = []
    for k in PRIVATE:
        n = BITS[k]
        for label, value, accept in [("max_in_range", 2**n - 1, True), ("min_out_of_range", 2**n, False),
                                     ("p_minus_1", BN254_SCALAR_FIELD - 1, False)]:
            cases.append((f"{k}_{label}", {**base, k: value}, accept))
    return cases


def run(work_dir: Path, real: bool) -> dict:
    source = CIRCUIT.read_text(encoding="utf-8")
    if circuit_domain(source) != DOMAIN_ELEMENT:
        raise SystemExit("circuit DOMAIN literal differs from src.crypto.commitment.DOMAIN_ELEMENT")
    if "<--" in code_without_comments(source):
        raise SystemExit("`<--` found in the circuit")

    tc = Toolchain(work_dir)
    out = tc.compile(CIRCUIT)
    info = tc.r1cs_info(out["r1cs"])
    need = required_power(info["constraints"], info["public_inputs"], info.get("outputs", 0))

    demo = demo_vector()
    d = demo["v"]
    host = commit(bytes.fromhex(d["key_hex"]), bytes.fromhex(d["signature_hex"]), bytes.fromhex(d["nonce_hex"]))
    if host.value != demo["C"] or host_c(demo["private"]) != demo["C"]:
        raise SystemExit("host commit() disagrees with the circomlibjs vector")
    ok, err = witness_ok(tc, out["wasm"], out["r1cs"], {**demo["private"], "C": demo["C"]}, "demo")
    if not ok:
        raise SystemExit(f"circuit refused the circomlibjs demo vector:\n{err[-1500:]}")

    ranges = {}
    for name, private, accept in range_cases(demo["private"]):
        ok, err = witness_ok(tc, out["wasm"], out["r1cs"], {**private, "C": host_c(private)}, name)
        on_range = "Num2Bits" in err and "Assert Failed" in err
        if ok != accept or (not accept and not on_range):
            raise SystemExit(f"range case {name}: accepted={ok}, expected {accept}\n{err[-1500:]}")
        ranges[name] = {"expected": "accept" if accept else "refuse", "accepted": ok,
                        "refused_by_num2bits": (not ok) and on_range}

    ok, err = witness_ok(tc, out["wasm"], out["r1cs"],
                         {**demo["private"], "C": (demo["C"] + 1) % BN254_SCALAR_FIELD}, "wrong_c")
    wrong_c_on_c_line = "CommitmentOpening" in err and "Assert Failed" in err and "Num2Bits" not in err
    if ok or not wrong_c_on_c_line:
        raise SystemExit(f"wrong C: accepted={ok}\n{err[-1500:]}")

    real_result = None
    if real:
        real_result = check_real_opening(out)

    return {"tc": tc, "info": info, "need": need, "ranges": ranges, "wrong_c_refused_on_c_line": wrong_c_on_c_line,
            "demo_C": demo["C"], "real": real_result, "source": source}


def check_real_opening(out: dict) -> dict:
    """Accept the real opening against the published C, leaving no secret on disk."""
    from make_master_key import DEFAULT_KEY_PATH, load_key
    from p5_4_publish_commitment import DEFAULT_NONCE_PATH, load_nonce

    from src.crypto.publication import ARTIFACT_PATH, read_publication
    from src.watermark.signature import PROJECT_OWNER_ID, derive_signature

    publication, _ = read_publication(ARTIFACT_PATH)
    published_c = int(publication["commitment"]["decimal"])
    key = load_key(DEFAULT_KEY_PATH)
    nonce = load_nonce(DEFAULT_NONCE_PATH)
    signature = derive_signature(key, PROJECT_OWNER_ID).value
    elements = commitment_inputs(key, signature, nonce)
    private = dict(zip(PRIVATE, elements[1:]))
    host_matches = commit(key, signature, nonce).value == published_c
    del key, nonce, signature, elements

    if REAL_WORK_DIR.exists():
        shutil.rmtree(REAL_WORK_DIR)
    try:
        tc = Toolchain(REAL_WORK_DIR)
        wtns, step = tc.witness(out["wasm"], {**private, "C": published_c}, "real", check=False)
        accepted = step.returncode == 0
        if accepted:
            tc.check_witness(out["r1cs"], wtns)
    finally:
        del private
        shutil.rmtree(REAL_WORK_DIR, ignore_errors=True)
    if not host_matches or not accepted:
        raise SystemExit(f"real opening: host matches published C={host_matches}, circuit accepted={accepted}")
    return {"published_C_source": "provenance/commitment.json", "host_commit_equals_published_C": True,
            "circuit_accepted_real_opening": True, "secret_files_removed": not REAL_WORK_DIR.exists()}


def assert_no_secret_in(path: Path) -> None:
    """Delete the record and stop if any real private input appears in it, in decimal or hex."""
    from make_master_key import DEFAULT_KEY_PATH, load_key
    from p5_4_publish_commitment import DEFAULT_NONCE_PATH, load_nonce

    from src.watermark.signature import PROJECT_OWNER_ID, derive_signature

    key = load_key(DEFAULT_KEY_PATH)
    nonce = load_nonce(DEFAULT_NONCE_PATH)
    elements = commitment_inputs(key, derive_signature(key, PROJECT_OWNER_ID).value, nonce)[1:]
    needles = [str(e) for e in elements] + [format(e, "x") for e in elements] + [key.hex(), nonce.hex()]
    text = path.read_text(encoding="utf-8").lower()
    if any(n in text for n in needles):
        path.unlink()
        raise SystemExit("a real secret value appeared in the result record; record deleted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; nothing is random")
    parser.add_argument("--no-real", action="store_true", help="skip the check with the owner's real opening")
    args = parser.parse_args()
    git = git_info()
    start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="zkcrown_p7_5_") as tmp:
        r = run(Path(tmp), real=not args.no_real)
        steps = [{"name": s.name, "seconds": round(s.seconds, 3), "returncode": s.returncode}
                 for s in r["tc"].steps if not s.name.startswith("wtns_calculate_real")]
    info, need = r["info"], r["need"]
    path = write_result(
        name="p7.5_commitment_circuit",
        seed=args.seed,
        task="P7.5",
        params={
            "circuit": "src/zk/circuits/commitment_opening.circom",
            "circuit_sha256": __import__("hashlib").sha256(CIRCUIT.read_bytes()).hexdigest(),
            "statement": "know K_hi, K_lo, S < 2^128 and nonce < 2^248 with Poseidon(DOMAIN, K_hi, K_lo, S, nonce) == C; C public",
            "layout": "zk-crown/commitment/v1 (P5.3)",
            "domain_element": str(DOMAIN_ELEMENT),
            "circom_flags": "--r1cs --wasm --sym, default optimisation (-O1)",
            "uses_left_arrow": "none in this circuit; circomlib Num2Bits uses <-- per bit, each immediately constrained",
            "vector": "tests/data/commitment_circomlibjs_vector.json (public demo values)",
            "ptau_power_available": PTAU_POWER,
        },
        metrics={
            "r1cs": info,
            "smallest_ptau_power_needed": need,
            "fits_ptau_power_15": need <= PTAU_POWER,
            "p7_4_estimate_constraints": 1471,
            "demo_vector_accepted": True,
            "demo_C": str(r["demo_C"]),
            "range_cases": r["ranges"],
            "wrong_C_refused_on_C_line": r["wrong_c_refused_on_c_line"],
            "real_opening": r["real"] if r["real"] is not None else "not run (--no-real)",
            "steps": steps,
        },
        notes="Circuit compiled and checked by witness generation only; no setup or proof (P7.6, P7.7).",
        duration_seconds=time.perf_counter() - start,
        git=git,
    )
    if r["real"] is not None:
        assert_no_secret_in(path)
    print(f"constraints {info['constraints']} (public {info['public_inputs']}, private {info['private_inputs']}), "
          f"needs power {need}, fits 2^15: {need <= PTAU_POWER}; demo vector accepted; "
          f"{len(r['ranges'])} range cases as expected; wrong C refused; real opening: "
          f"{'accepted' if r['real'] else 'not run'}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
