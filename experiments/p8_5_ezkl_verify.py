"""P8.5: verify the P8.4 EZKL proof off chain, then try tampered variants.

    python experiments/p8_5_ezkl_verify.py

Local CPU. Verifies `results/zk/p8.4/proof.json` with the committed
`results/zk/p8.3/settings.json` and `results/zk/p8.3/vk.key`, and the P8.3
SRS in `zk/ezkl/srs/` (gitignored). Each is checked against the SHA-256 in
the P8.3 or P8.4 record first. No proving key is read.

1. **Timing:** the honest proof is verified 5 times, each in its own Python
   process, which measures its own peak memory (P8.3 method). Verify time is
   the median of the 5 `ezkl.verify` calls.
2. **Negative checks**, all in one process (`verify_batch`), with the honest
   proof as a control first and last, and once after a JSON round trip with
   no change. Each case's outcome is classified as `accepted` (True),
   `rejected` (a clean False) or `error` (an exception). Cases:
   - one output score changed in the public instances (field +1, +1.0 at the
     circuit scale, negated, two scores swapped so the predicted class changes);
   - one input pixel changed the same ways;
   - single bit flips at a spread of byte positions in the proof bytes
     (changed in both `proof` and `hex_proof`, which carry the same bytes);
   - the instances of a different image: MNIST test image 1, the next image
     by position, fixed before the run; its instances come from a witness
     generated with the P8.3 compiled circuit;
   - an instance dropped, an instance appended.
   Only `instances`, `proof` and `hex_proof` are edited. ezkl's informational
   `pretty_public_inputs` is left as it was.

Zero accepted negatives is evidence about the cases tried, not a soundness
proof. Tampered files stay in `zk/ezkl/zk_model/p8.5/` (gitignored); only the
result record is committed.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.utils.results import git_info, read_result, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

import p8_1_export_onnx as p81
import p8_3_ezkl_setup as p83

P8_3_RECORD = Path("results/p8.3_ezkl_setup__seed1337__20261003T185318+0000.json")
P8_4_RECORD = Path("results/p8.4_ezkl_prove__seed1337__20261003T185838+0000.json")
PROOF = Path("results/zk/p8.4/proof.json")
SETTINGS = Path("results/zk/p8.3/settings.json")
VK = Path("results/zk/p8.3/vk.key")
WORK = Path("zk/ezkl/zk_model/p8.5")
TIMING_RUNS = 5
OTHER_IMAGE_INDEX = 1
"""Fixed before the run: the next MNIST test image by position after P8.4's index 0."""
N_INPUTS, N_OUTPUTS = 784, 10
BN254_R = 21888242871839275222246405745257275088548364400416034343698204186575808495617
OUTPUT_IDX = (0, 3, 7, 9)
PIXEL_IDX = (0, 150, 400, 783)
FLIP_BYTES = (0, 1, 63, 64, 255, 512, 1024, 1536, 2048, 2560, 3000, 3071)
FLIP_BITS = (0, 7)


def felt(h: str) -> int:
    return int.from_bytes(bytes.fromhex(h), "little")


def unfelt(v: int) -> str:
    return (v % BN254_R).to_bytes(32, "little").hex()


def with_instance(pf: dict, i: int, value: int) -> dict:
    out = copy.deepcopy(pf)
    out["instances"][0][i] = unfelt(value)
    return out


def with_bit_flip(pf: dict, byte: int, bit: int) -> dict:
    out = copy.deepcopy(pf)
    out["proof"][byte] ^= 1 << bit
    out["hex_proof"] = "0x" + bytes(out["proof"]).hex()
    return out


def negative_cases(pf: dict, other_instances: list[str], scale: int) -> list[tuple[str, str, dict]]:
    inst = pf["instances"][0]
    one = 1 << scale  # 1.0 at the circuit scale
    cases = []
    for k in OUTPUT_IDX:
        i = N_INPUTS + k
        cases.append(("output", f"output {k} +1 field unit", with_instance(pf, i, felt(inst[i]) + 1)))
    k = 7
    i = N_INPUTS + k
    cases.append(("output", f"output {k} +1.0 (+{one})", with_instance(pf, i, felt(inst[i]) + one)))
    cases.append(("output", f"output {k} negated", with_instance(pf, i, -felt(inst[i]))))
    swapped = copy.deepcopy(pf)
    a, b = N_INPUTS + 3, N_INPUTS + 7
    swapped["instances"][0][a], swapped["instances"][0][b] = inst[b], inst[a]
    cases.append(("output", "outputs 3 and 7 swapped (predicted class becomes 3)", swapped))
    for px in PIXEL_IDX:
        cases.append(("input", f"pixel {px} +1 field unit", with_instance(pf, px, felt(inst[px]) + 1)))
    cases.append(("input", f"pixel 400 +1.0 (+{one})", with_instance(pf, 400, felt(inst[400]) + one)))
    cases.append(("input", "pixel 0 set to 0", with_instance(pf, 0, 0)))
    for byte in FLIP_BYTES:
        for bit in FLIP_BITS:
            cases.append(("proof_bits", f"proof byte {byte} bit {bit} flipped", with_bit_flip(pf, byte, bit)))
    full = copy.deepcopy(pf)
    full["instances"][0] = list(other_instances)
    cases.append(("other_image", f"all instances of test image {OTHER_IMAGE_INDEX}", full))
    mixed_in = copy.deepcopy(pf)
    mixed_in["instances"][0][:N_INPUTS] = other_instances[:N_INPUTS]
    cases.append(("other_image", f"inputs of image {OTHER_IMAGE_INDEX}, outputs of image 0", mixed_in))
    mixed_out = copy.deepcopy(pf)
    mixed_out["instances"][0][N_INPUTS:] = other_instances[N_INPUTS:]
    cases.append(("other_image", f"inputs of image 0, outputs of image {OTHER_IMAGE_INDEX}", mixed_out))
    dropped = copy.deepcopy(pf)
    dropped["instances"][0] = inst[:-1]
    cases.append(("structure", "last instance dropped (793)", dropped))
    appended = copy.deepcopy(pf)
    appended["instances"][0] = inst + [unfelt(0)]
    cases.append(("structure", "zero instance appended (795)", appended))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    git = git_info()
    root = repo_root()
    t_start = time.perf_counter()

    import ezkl
    if ezkl.__version__ != p83.EZKL_VERSION:
        raise SystemExit(f"ezkl {ezkl.__version__} installed, expected {p83.EZKL_VERSION}")
    r3, r4 = read_result(root / P8_3_RECORD), read_result(root / P8_4_RECORD)
    srs = root / r3["params"]["srs"]["path"]
    compiled = root / "zk/ezkl/zk_model/network.compiled"
    checks = {PROOF: r4["metrics"]["proof_sha256"], SETTINGS: r3["metrics"]["sha256"]["settings"],
              VK: r3["metrics"]["sha256"]["vk"], srs.relative_to(root): r3["metrics"]["sha256"]["srs"],
              compiled.relative_to(root): r3["metrics"]["sha256"]["compiled_circuit"]}
    for rel_path, digest in checks.items():
        if p83.sha256_file(root / rel_path) != digest:
            raise SystemExit(f"{rel_path} does not match its recorded SHA-256")

    work = root / WORK
    work.mkdir(parents=True, exist_ok=True)
    rel = lambda p: str(Path(p).relative_to(root)).replace("\\", "/")  # noqa: E731
    common = {"settings": rel(root / SETTINGS), "vk": rel(root / VK), "srs": rel(srs)}
    scale = json.loads((root / SETTINGS).read_text())["run_args"]["input_scale"]

    # 1. Timing: honest proof, one process per run.
    timing = [p83.run_stage("verify", log_dir=work / "logs", proof=rel(root / PROOF), **common)
              for _ in range(TIMING_RUNS)]
    outcomes = [t["value"]["outcome"] for t in timing]
    if outcomes != ["accepted"] * TIMING_RUNS:
        raise SystemExit(f"honest proof not accepted in every timing run: {outcomes}")
    verify_secs = [t["value"]["seconds"] for t in timing]

    # Instances of a different image (test index 1), from a witness of the P8.3 circuit.
    xs, ys = p81.test_arrays()
    other_data, other_wit = work / "other_input.json", work / "other_witness.json"
    other_data.write_text(json.dumps({"input_data": [xs[OTHER_IMAGE_INDEX].reshape(-1).astype(float).tolist()]}))
    p83.run_stage("gen_witness", log_dir=work / "logs", data=rel(other_data), compiled=rel(compiled),
                  witness=rel(other_wit))
    w = json.loads(other_wit.read_text())
    other_instances = [v for g in w["inputs"] for v in g] + [v for g in w["outputs"] for v in g]
    assert len(other_instances) == N_INPUTS + N_OUTPUTS

    # 2. Negative checks with controls, one process.
    pf = json.loads((root / PROOF).read_text())
    assert len(pf["proof"]) == 3072 and bytes(pf["proof"]).hex() == pf["hex_proof"][2:]
    if pf["instances"][0] == other_instances:
        raise SystemExit("image 1 has the same instances as image 0")
    cases = [("control", "honest proof (first)", None),
             ("control", "honest proof after JSON round trip, unchanged", copy.deepcopy(pf))]
    cases += negative_cases(pf, other_instances, scale)
    cases.append(("control", "honest proof (last)", None))
    paths, meta = [], []
    for n, (family, label, obj) in enumerate(cases):
        if obj is None:
            path = root / PROOF
        else:
            path = work / f"case_{n:03d}.json"
            path.write_text(json.dumps(obj))
        paths.append(rel(path))
        meta.append({"family": family, "case": label})
    batch = p83.run_stage("verify_batch", log_dir=work / "logs", proofs=paths, **common)
    results = []
    for m, r in zip(meta, batch["value"]["results"]):
        results.append({**m, "outcome": r["outcome"], "error": r["error"], "seconds": round(r["seconds"], 4)})

    controls = [r for r in results if r["family"] == "control"]
    negatives = [r for r in results if r["family"] != "control"]
    by_family = {}
    for r in negatives:
        f = by_family.setdefault(r["family"], {"tried": 0, "accepted": 0, "rejected_false": 0, "error": 0,
                                               "error_messages": []})
        f["tried"] += 1
        f["accepted"] += r["outcome"] == "accepted"
        f["rejected_false"] += r["outcome"] == "rejected"
        if r["outcome"] == "error":
            f["error"] += 1
            if r["error"] not in f["error_messages"]:
                f["error_messages"].append(r["error"])
    summary = {"negatives_tried": len(negatives),
               "negatives_accepted": sum(r["outcome"] == "accepted" for r in negatives),
               "negatives_rejected_false": sum(r["outcome"] == "rejected" for r in negatives),
               "negatives_error": sum(r["outcome"] == "error" for r in negatives),
               "controls_tried": len(controls),
               "controls_accepted": sum(r["outcome"] == "accepted" for r in controls)}

    path = write_result(
        name="p8.5_ezkl_verify", seed=args.seed, task="P8.5", git=git,
        duration_seconds=time.perf_counter() - t_start,
        params={
            "ezkl_version": ezkl.__version__,
            "proof": {"path": str(PROOF).replace("\\", "/"), "sha256": checks[PROOF]},
            "settings": str(SETTINGS).replace("\\", "/"), "vk": str(VK).replace("\\", "/"),
            "srs": rel(srs), "sha256_checked": {str(k).replace("\\", "/"): v for k, v in checks.items()},
            "proving_key_read": False,
            "timing_runs": TIMING_RUNS,
            "other_image_index": OTHER_IMAGE_INDEX,
            "edited_fields": ["instances", "proof", "hex_proof"],
            "measurement": "each stage in its own Python process, which reads its own peak memory (P8.3)",
        },
        metrics={
            "verify_seconds": [round(v, 4) for v in verify_secs],
            "verify_seconds_median": round(statistics.median(verify_secs), 4),
            "verify_peak_working_set_bytes": [t["peak_working_set_bytes"] for t in timing],
            "verify_peak_private_bytes": [t["peak_private_bytes"] for t in timing],
            "verify_process_seconds": [t["process_seconds"] for t in timing],
            "timing_stages": [{k: v for k, v in t.items() if k != "value"} for t in timing],
            "summary": summary,
            "by_family": by_family,
            "cases": results,
            "batch_peak_working_set_bytes": batch["peak_working_set_bytes"],
        },
        notes=("Local Windows CPU (x64 Python emulated on ARM), not Colab. Zero accepted negatives is "
               "evidence about the cases tried, not a soundness proof."),
    )
    print(f"verify median {statistics.median(verify_secs):.3f} s; {summary}")
    for fam, f in by_family.items():
        print(f"  {fam}: tried {f['tried']}, accepted {f['accepted']}, false {f['rejected_false']}, "
              f"error {f['error']} {f['error_messages'][:2]}")
    print(f"record {path}")
    return 0 if summary["negatives_accepted"] == 0 and summary["controls_accepted"] == len(controls) else 1


if __name__ == "__main__":
    sys.exit(main())
