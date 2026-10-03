"""P7.4: fetch the public Hermez powers-of-tau file for the P7.5 circuit and verify it.

    python experiments/p7_4_fetch_ptau.py

Local CPU. Downloads one file (about 2.4 MB at power 11) unless a verified
copy is already in `zk/ptau/`. Reads no key.

Choosing the size before the real circuit exists:

1. compile a sizing probe of each circomlib component the P7.5 circuit will
   use, with the pinned circom: `Poseidon(5)`, `Num2Bits(128)` (for `K_hi`,
   `K_lo`, `S`) and `Num2Bits(248)` (for the nonce). Their constraint counts
   are summed, one `Num2Bits(128)` counted three times. The probes are not the
   P7.5 circuit and their sum is an estimate, recorded as such;
2. take the smallest power `p` with `2**p >= estimate + public inputs + 1`.
   P7.5 must check that its real count fits; if it does not, it takes the next
   power from the same table.

Where the file comes from: the prepared-phase-2 Hermez ceremony files listed
in the README of the pinned snarkjs 0.7.6, which gives each file's URL and
BLAKE2b-512. The script checks that this script's own copy of the chosen
hash equals the README's, hashes the download before moving it into place,
then runs `snarkjs powersoftau verify` on it. Any failure stops the run before
a result is written. The file is gitignored and never committed.

Owner decision (2026-10-03): the README's URLs (Google Cloud Storage) and the
older Hermez S3 bucket return HTTP 403, and the only mirror found carries
2^15 and 2^20 but not 2^11. So the file used is **2^15**, larger than the
sizing needs, downloaded from a third-party GitHub release mirror. The mirror
is not trusted for content: the file must match the README's BLAKE2b-512 for
power 15. Same ceremony, same security; the cost is a bigger file and a
slower phase 2.
"""

from __future__ import annotations

import argparse
import re
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.zk.ptau import PTAU_DIR, SNARKJS_README, fetch, read_table, required_power, sha256_file
from src.zk.toolchain import Toolchain

SIZED_POWER = 11
"""Smallest power the sizing probe needs. The script stops if the probe gives another value."""
CHOSEN_POWER = 15
CHOSEN_BLAKE2B_512 = ("982372c867d229c236091f767e703253249a9b432c1710b4f326306bfa2428a1"
                      "7b06240359606cfe4d580b10a5a1f63fbed499527069c18ae17060472969ae6e")
"""Copied from the snarkjs 0.7.6 README table, power 15. Checked against the README at run time."""
MIRROR_URL = ("https://github.com/hilawe/dash-mno-verify/releases/download/ptau-hermez-v1/"
              "powersOfTau28_hez_final_15.ptau")
"""Third-party mirror (owner decision); content is checked against CHOSEN_BLAKE2B_512, not trusted."""

PUBLIC_INPUTS = 1  # C

PROBES = {
    "poseidon5": ('include "circomlib/circuits/poseidon.circom";\n'
                  "template P() { signal input x[5]; signal input c; component h = Poseidon(5);\n"
                  "  for (var i = 0; i < 5; i++) { h.inputs[i] <== x[i]; } c === h.out; }\n"
                  "component main {public [c]} = P();\n"),
    "num2bits128": ('include "circomlib/circuits/bitify.circom";\n'
                    "template B() { signal input x; component b = Num2Bits(128); b.in <== x; }\n"
                    "component main = B();\n"),
    "num2bits248": ('include "circomlib/circuits/bitify.circom";\n'
                    "template B() { signal input x; component b = Num2Bits(248); b.in <== x; }\n"
                    "component main = B();\n"),
}
PROBE_MULTIPLICITY = {"poseidon5": 1, "num2bits128": 3, "num2bits248": 1}


def probe_counts(work_dir: Path) -> dict[str, int]:
    tc = Toolchain(work_dir)
    counts = {}
    for name, body in PROBES.items():
        src = work_dir / f"probe_{name}.circom"
        src.write_text("pragma circom 2.1.0;\n" + body, encoding="utf-8")
        out = tc.compile(src)
        counts[name] = tc.r1cs_info(out["r1cs"])["constraints"]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; nothing is random")
    args = parser.parse_args()
    git = git_info()
    start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="zkcrown_p7_4_") as tmp:
        counts = probe_counts(Path(tmp))
    estimate = sum(counts[k] * PROBE_MULTIPLICITY[k] for k in counts)
    sized = required_power(estimate, PUBLIC_INPUTS)
    if sized != SIZED_POWER:
        raise SystemExit(f"sizing gives power {sized}, script expects {SIZED_POWER}; update deliberately")
    power = CHOSEN_POWER
    if power < sized:
        raise SystemExit(f"chosen power {power} is below the sized power {sized}")

    table = read_table()
    entry = table.get(power)
    if entry is None or entry.blake2b_512 != CHOSEN_BLAKE2B_512:
        raise SystemExit(f"snarkjs README hash for power {power} does not equal the pinned hash")
    if not MIRROR_URL.endswith("/" + entry.filename):
        raise SystemExit(f"mirror URL {MIRROR_URL} does not name {entry.filename}")

    t0 = time.perf_counter()
    path, downloaded = fetch(entry, url=MIRROR_URL)
    fetch_seconds = time.perf_counter() - t0

    with tempfile.TemporaryDirectory(prefix="zkcrown_p7_4v_") as tmp:
        tc = Toolchain(Path(tmp))
        step = tc.snarkjs("ptau_verify", "powersoftau", "verify", str(path), check=False)
    out = step.output
    if step.returncode != 0 or "Powers of Tau Ok!" not in out:
        raise SystemExit(f"snarkjs powersoftau verify failed:\n{out[-2000:]}")
    contributions = len(re.findall(r"contribution #\d+", out, flags=re.IGNORECASE))
    beacon = "beacon" in out.lower()

    record = write_result(
        name="p7.4_ptau",
        seed=args.seed,
        task="P7.4",
        params={
            "sizing": "sum of circomlib component probes compiled with the pinned circom (default -O1); estimate, not the P7.5 circuit",
            "probe_multiplicity": PROBE_MULTIPLICITY,
            "public_inputs": PUBLIC_INPUTS,
            "rule": "smallest p with 2**p >= constraints + public inputs + 1",
            "source": "snarkjs 0.7.6 README table of prepared-phase-2 Hermez ptau files (bn128); expected hash from it",
            "readme_url_status": "HTTP 403 AccessDenied (storage.googleapis.com and the older Hermez S3 bucket), 2026-10-03",
            "download_url": MIRROR_URL,
            "mirror_trust": "none: content accepted only if BLAKE2b-512 equals the README's",
            "sized_power": SIZED_POWER,
            "chosen_power_reason": "owner decision: only 2^15 and 2^20 found on a reachable mirror; 2^15 is the smallest available that fits",
            "snarkjs_readme_sha256": sha256_file(SNARKJS_README),
        },
        metrics={
            "probe_constraints": counts,
            "estimated_constraints": estimate,
            "sized_power": sized,
            "power": power,
            "max_domain_size": 2**power,
            "headroom_constraints": 2**power - (estimate + PUBLIC_INPUTS + 1),
            "file": entry.filename,
            "readme_url": entry.url,
            "bytes": path.stat().st_size,
            "blake2b_512": entry.blake2b_512,
            "blake2b_512_matches_readme": True,
            "sha256": sha256_file(path),
            "downloaded_this_run": downloaded,
            "fetch_and_hash_seconds": round(fetch_seconds, 3),
            "snarkjs_powersoftau_verify": "Powers of Tau Ok!",
            "snarkjs_verify_seconds": round(step.seconds, 3),
            "contribution_lines_reported": contributions,
            "beacon_reported": beacon,
            "local_path": str(path.relative_to(repo_root())).replace("\\", "/") + " (gitignored)",
        },
        notes="Phase 1 only. Phase 2 for the P7.5 circuit is P7.6. The file itself is not committed.",
        duration_seconds=time.perf_counter() - start,
        git=git,
    )
    print(f"probes {counts}, estimate {estimate}, sized power {sized}, using power {power}, {entry.filename} {path.stat().st_size} bytes, "
          f"BLAKE2b ok, snarkjs verify ok ({contributions} contribution lines, beacon {beacon}), downloaded {downloaded}")
    print(f"wrote {record}")


if __name__ == "__main__":
    main()
