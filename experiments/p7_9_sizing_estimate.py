"""P7.9 sizing: constraint estimate for deriving the v1 triggers inside the circuit.

    python experiments/p7_9_sizing_estimate.py

Local CPU, about a minute. Reads no key. No P7.9 circuit is written: this is
a sizing probe, in the style of P7.4. Each circomlib component the circuit
would need is compiled on its own with the pinned circom (default -O1) and its
constraint count read from `snarkjs r1cs info`. Those measured counts are
multiplied by how often the v1 derivation uses each component. The total is
an estimate, not a measured circuit.

What "bind the trigger derivation" would have to prove, using the v1
construction unchanged (CLAUDE.md P1.1, P1.2, P2.2, P2.3):

1. **Keystream.** Block i of a stream is
   `HMAC-SHA256(K, "zk-crown/keystream/v1\\0" || u16 len || label || u64 i)`.
   Lower bound used here: the K^ipad and K^opad compression states are
   computed once in the circuit and shared (2 compressions), then each block
   costs its inner message compressions plus 1 outer compression.
   - signs (`triggers/v1/perturbation-sign`): N * 3072 bits, 256 bits per block;
   - base indices (`triggers/v1/base-index`): N 64-bit draws, 4 per block,
     no rejection assumed (probability below 2.5e-15 per draw);
   - targets (`responses/v1/target-class`): N 64-bit draws, 4 per block.
2. **Trigger commitment.** The published trigger set commitment is the P2.3
   bundle digest: one SHA-256 over the domain, header, all N*3072 pixels, and
   the indices, labels and targets. Its compression count follows from the
   exact byte length.
3. **Pixel selection and perturbation.** Each trigger's base image is a
   training image chosen by index. Keeping it private needs a committed
   dataset (a Poseidon Merkle tree over the 45,000 images, which does not
   exist yet): per trigger, a leaf hash of the 3,072 bytes packed into
   31-byte elements and a depth-16 path. Then per pixel: an 8-bit range check
   on the base value and two 10-bit comparisons for the clip. The partial
   Fisher-Yates and the modulo reductions are smaller still and are not
   counted, so the total is a lower bound for this design.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

import numpy as np

from src.utils.results import git_info, write_result
from src.utils.seeding import DEFAULT_SEED
from src.watermark.bundle import DOMAIN as BUNDLE_DOMAIN
from src.watermark.bundle import TriggerBundle
from src.watermark.keygen import DOMAIN as KEYSTREAM_DOMAIN
from src.watermark.responses import TARGET_CLASS_LABEL
from src.watermark.triggers import BASE_INDEX_LABEL, DEFAULT_AMPLITUDE, PERTURBATION_SIGN_LABEL
from src.zk.ptau import required_power
from src.zk.toolchain import Toolchain

N = 100
PIXELS = 32 * 32 * 3
POOL = 45_000
P7_5_CONSTRAINTS = 1471
PTAU_POWER = 15
HERMEZ_MAX_POWER = 28

PROBES = {
    "sha256_compression": ('include "circomlib/circuits/sha256/sha256compression.circom";\n'
                           "component main = Sha256compression();"),
    "poseidon_2": 'include "circomlib/circuits/poseidon.circom";\ncomponent main = Poseidon(2);',
    "poseidon_16": 'include "circomlib/circuits/poseidon.circom";\ncomponent main = Poseidon(16);',
    "num2bits_8": 'include "circomlib/circuits/bitify.circom";\ncomponent main = Num2Bits(8);',
    "lessthan_10": 'include "circomlib/circuits/comparators.circom";\ncomponent main = LessThan(10);',
}


def compressions(message_bytes: int) -> int:
    """SHA-256 compressions for a message of this length (padding: 0x80 and a 64-bit length)."""
    return math.ceil((message_bytes + 9) / 64)


def hmac_block_compressions(label: str) -> int:
    """Per keystream block, with the K^ipad / K^opad states shared: inner message blocks plus 1 outer."""
    inner_message = len(KEYSTREAM_DOMAIN) + 2 + len(label.encode()) + 8
    return math.ceil((inner_message + 9) / 64) + 1


def bundle_digest_bytes() -> int:
    b = TriggerBundle(images=np.zeros((N, 32, 32, 3), np.uint8), base_indices=np.arange(N, dtype=np.int64),
                      base_labels=np.zeros(N, np.int64), targets=np.ones(N, np.int64),
                      amplitude=DEFAULT_AMPLITUDE, num_classes=10, key_kind="owner",
                      trigger_version="triggers/v1", response_version="responses/v1")
    import json
    header = json.dumps(b.header(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(BUNDLE_DOMAIN) + 4 + len(header) + N * PIXELS + 3 * N * 8


def measure(work: Path) -> dict[str, int]:
    tc = Toolchain(work)
    counts = {}
    for name, body in PROBES.items():
        src = work / f"{name}.circom"
        src.write_text("pragma circom 2.1.0;\n" + body + "\n", encoding="utf-8")
        out = tc.compile(src)
        counts[name] = tc.r1cs_info(out["r1cs"])["constraints"]
    return counts


def estimate(c: dict[str, int]) -> dict:
    sign_blocks = N * PIXELS // 256
    draw_blocks = math.ceil(N * 8 / 32)
    keystream = {
        "shared_ipad_opad": 2,
        "perturbation_sign": sign_blocks * hmac_block_compressions(PERTURBATION_SIGN_LABEL),
        "base_index": draw_blocks * hmac_block_compressions(BASE_INDEX_LABEL),
        "target_class": draw_blocks * hmac_block_compressions(TARGET_CLASS_LABEL),
    }
    digest_bytes = bundle_digest_bytes()
    bundle = compressions(digest_bytes)
    sha_total = sum(keystream.values()) + bundle

    leaf_elements = math.ceil(PIXELS / 31)
    leaf_calls = math.ceil((leaf_elements - 16) / 15) + 1  # sponge: 16 in the first call, then 15 + chaining
    depth = math.ceil(math.log2(POOL))
    per_trigger_dataset = leaf_calls * c["poseidon_16"] + depth * c["poseidon_2"]
    per_pixel = c["num2bits_8"] + 2 * c["lessthan_10"]

    parts = {
        "keystream_sha256": sum(keystream.values()) * c["sha256_compression"],
        "bundle_digest_sha256": bundle * c["sha256_compression"],
        "dataset_membership": N * per_trigger_dataset,
        "pixel_perturbation": N * PIXELS * per_pixel,
        "commitment_opening_p7_5": P7_5_CONSTRAINTS,
    }
    total = sum(parts.values())
    one_hmac_block = (2 + hmac_block_compressions(PERTURBATION_SIGN_LABEL)) * c["sha256_compression"]
    return {
        "compressions": {**keystream, "bundle_digest": bundle, "total": sha_total},
        "keystream_blocks": {"perturbation_sign": sign_blocks, "base_index": draw_blocks, "target_class": draw_blocks},
        "hmac_block_compressions": {"perturbation_sign": hmac_block_compressions(PERTURBATION_SIGN_LABEL),
                                    "base_index": hmac_block_compressions(BASE_INDEX_LABEL),
                                    "target_class": hmac_block_compressions(TARGET_CLASS_LABEL)},
        "bundle_digest_bytes": digest_bytes,
        "per_trigger_dataset_constraints": per_trigger_dataset,
        "per_pixel_constraints": per_pixel,
        "parts": parts,
        "total_estimate": total,
        "sha256_share": round((parts["keystream_sha256"] + parts["bundle_digest_sha256"]) / total, 4),
        "power_needed": required_power(total, 1),
        "ptau_15_capacity": 2**PTAU_POWER,
        "fits_2_15": total + 2 <= 2**PTAU_POWER,
        "fits_largest_hermez_2_28": total + 2 <= 2**HERMEZ_MAX_POWER,
        "single_keystream_block_with_shared_states": one_hmac_block,
        "single_keystream_block_plus_p7_5_fits_2_15": one_hmac_block + P7_5_CONSTRAINTS + 2 <= 2**PTAU_POWER,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; nothing is random")
    args = parser.parse_args()
    git = git_info()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="zkcrown_p7_9_") as tmp:
        counts = measure(Path(tmp))
    est = estimate(counts)
    path = write_result(
        name="p7.9_sizing_estimate",
        seed=args.seed,
        task="P7.9",
        params={
            "method": "circomlib components compiled alone with the pinned circom (-O1), counts multiplied by the "
                      "v1 derivation's usage; an estimate and a lower bound for this design, not a circuit",
            "keystream": "v1 unchanged: HMAC-SHA256(K, DOMAIN || u16 len || label || u64 i)",
            "assumptions": "K^ipad/K^opad states shared; no rejection in randbelow; Fisher-Yates and modulo not "
                           "counted; dataset Merkle tree (Poseidon, depth 16) assumed, does not exist",
            "N": N, "pixels_per_trigger": PIXELS, "pool": POOL,
        },
        metrics={"component_constraints": counts, **est},
        notes="Sizing only; no P7.9 circuit written.",
        duration_seconds=time.perf_counter() - start,
        git=git,
    )
    print(f"component counts {counts}")
    print(f"SHA-256 compressions {est['compressions']['total']:,}; parts {est['parts']}")
    print(f"total >= {est['total_estimate']:,} constraints, needs power {est['power_needed']}; fits 2^15: "
          f"{est['fits_2_15']}; fits 2^28: {est['fits_largest_hermez_2_28']}; one keystream block alone: "
          f"{est['single_keystream_block_with_shared_states']:,}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
