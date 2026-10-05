"""P8.6: PyTorch against EZKL circuit agreement for `zk_model` on MNIST test images.

    python experiments/p8_6_ezkl_fidelity.py

Local CPU. Uses the P8.3 compiled circuit, settings, proving key, verification
key and SRS from `zk/ezkl/`, each checked against the SHA-256 the committed
P8.3 record holds. One model (the P0.7 `zk_model`, P8.1 ONNX) and one set of
scale settings (input and param scale 13, logrows 18, from P8.3's `accuracy`
calibration). Nothing here says how other scales or models behave.

**Image set, fixed before the run:** all 10,000 images of the official MNIST
test set, in index order, normalised as in training. Not a sample. A scratch
timing of 20 in-process `gen_witness` calls before this script was written
gave about 0.28 s per image, so the whole test set is about 47 minutes in
witness-only mode, cheap enough that sampling would only add sampling error.

**Bulk (witness only):** `ezkl.gen_witness` runs the circuit's quantised
forward pass and writes the values the prover would use, without a proof.
The test set is split into 10 chunks of 1,000, each in its own Python process
(`gen_witness_batch` in `src/zk/ezkl/stages.py`) under the P8.3 watchdog
(8 GiB, one hour per process), run one after another so the per-image times
are not inflated by contention. The circuit's class is the argmax of ezkl's
dequantised outputs (`rescaled_outputs`); ties are counted.

**Proved subset, fixed before the run:** 10 test indices drawn without
replacement with `numpy.random.default_rng(1337)`, sorted. Not chosen from
results. Each is proved with the P8.3 proving key in its own process and
verified with the committed verification key. The check is that each proof
verifies and that its public outputs equal, value for value, the outputs the
bulk witness pass gave for that image, so the witness-only figures are the
figures a proof would carry.

**Reported:** top-1 agreement with PyTorch, every disagreement with its index
and label, the largest and mean absolute logit difference, circuit accuracy
against the labels next to PyTorch's (which must reproduce P0.7's 9,896 /
10,000), the paired McNemar test between them, and time per image.

Nothing is committed but the result record; witnesses and proofs stay in
`zk/ezkl/zk_model/p8.6/` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

import numpy as np
import torch

from src.utils.results import git_info, read_result, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.utils.stats import paired_accuracy_difference

import p8_1_export_onnx as p81
import p8_3_ezkl_setup as p83

N_TEST = 10_000
CHUNK = 1_000
N_PROVED = 10
PROOF_SUBSET_SEED = 1337
P0_7_CORRECT = 9_896
P8_3_RECORD = Path("results/p8.3_ezkl_setup__seed1337__20261003T185318+0000.json")
P8_4_PUBLIC = Path("results/zk/p8.4/public_instances.json")
WORK = Path("zk/ezkl/zk_model")
RUN_DIR = WORK / "p8.6"


def proof_subset() -> list[int]:
    """The 10 test indices proved in full. Fixed by seed, not by results."""
    rng = np.random.default_rng(PROOF_SUBSET_SEED)
    return sorted(int(i) for i in rng.choice(N_TEST, size=N_PROVED, replace=False))


def argmax_with_ties(row: np.ndarray) -> tuple[int, bool]:
    top = np.flatnonzero(row == row.max())
    return int(top[0]), len(top) > 1


def summarise(torch_logits: np.ndarray, circuit_logits: np.ndarray, labels: np.ndarray) -> dict:
    """Agreement, disagreements, logit differences and paired accuracy."""
    t_cls = torch_logits.argmax(axis=1)
    c = [argmax_with_ties(r) for r in circuit_logits]
    c_cls = np.array([k for k, _ in c])
    ties = [i for i, (_, tie) in enumerate(c) if tie]
    diff = np.abs(circuit_logits - torch_logits)
    per_image_max = diff.max(axis=1)
    disagree = np.flatnonzero(c_cls != t_cls)
    worst = int(per_image_max.argmax())
    paired = paired_accuracy_difference((t_cls == labels).tolist(), (c_cls == labels).tolist())
    return {
        "n": int(len(labels)),
        "top1_agree": int((c_cls == t_cls).sum()),
        "top1_agreement_rate": float((c_cls == t_cls).mean()),
        "disagreements": [{"index": int(i), "label": int(labels[i]), "pytorch_class": int(t_cls[i]),
                           "circuit_class": int(c_cls[i]),
                           "pytorch_logits": torch_logits[i].tolist(),
                           "circuit_logits": circuit_logits[i].tolist()} for i in disagree],
        "circuit_argmax_ties": ties,
        "abs_logit_diff": {"max": float(diff.max()), "mean": float(diff.mean()),
                           "median": float(np.median(diff)), "p99": float(np.quantile(diff, 0.99)),
                           "per_image_max_mean": float(per_image_max.mean()),
                           "argmax_image_index": worst, "argmax_image_label": int(labels[worst])},
        "pytorch_correct": int((t_cls == labels).sum()),
        "circuit_correct": int((c_cls == labels).sum()),
        "pytorch_accuracy": float((t_cls == labels).mean()),
        "circuit_accuracy": float((c_cls == labels).mean()),
        "paired_drop_pytorch_to_circuit": paired,
    }


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
    p83_rec = read_result(root / P8_3_RECORD)
    expected = p83_rec["metrics"]["sha256"]
    work = root / WORK
    files = {"pk": work / "pk.key", "compiled_circuit": work / "network.compiled",
             "settings": work / "settings.json", "vk": work / "vk.key",
             "srs": root / p83_rec["params"]["srs"]["path"]}
    for name, path in files.items():
        if p83.sha256_file(path) != expected[name]:
            raise SystemExit(f"{path} does not match the P8.3 SHA-256 for {name}")
    run_args = json.loads(files["settings"].read_text())["run_args"]

    run_dir = root / RUN_DIR
    if (run_dir / "bulk_outputs.json").exists():
        raise SystemExit(f"{run_dir} already holds a P8.6 run; move it aside first")
    run_dir.mkdir(parents=True, exist_ok=True)
    rel = lambda p: str(p.relative_to(root)).replace("\\", "/")  # noqa: E731

    xs, ys = p81.test_arrays()
    assert xs.shape == (N_TEST, 1, 28, 28)
    labels = ys.astype(int)
    model = p81.load_model()
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(xs)).numpy().astype(float)
    if int((torch_logits.argmax(axis=1) == labels).sum()) != P0_7_CORRECT:
        raise SystemExit("PyTorch does not reproduce P0.7's 9,896 / 10,000")
    flat = xs.reshape(N_TEST, -1).astype(np.float32)

    # Bulk: witness-only circuit evaluation of all 10,000 images, one process per chunk.
    chunks, rows = [], []
    for start in range(0, N_TEST, CHUNK):
        cdir = run_dir / f"chunk_{start:05d}"
        cdir.mkdir(parents=True, exist_ok=True)
        np.save(cdir / "inputs.npy", flat[start:start + CHUNK])
        st = p83.run_stage("gen_witness_batch", log_dir=cdir / "logs", inputs=rel(cdir / "inputs.npy"),
                           indices=list(range(start, start + CHUNK)),
                           compiled=rel(files["compiled_circuit"]), work_dir=rel(cdir),
                           results=rel(cdir / "results.json"))
        part = json.loads((cdir / "results.json").read_text())
        rows += part
        chunks.append({"first_index": start, **{k: st.get(k) for k in
                       ("process_seconds", "call_seconds", "peak_working_set_bytes", "peak_private_bytes",
                        "watchdog_peak_tree_working_set_bytes", "value")}})
        print(f"chunk {start:5d}: {st['process_seconds']:.0f} s, errors {st['value']['errors']}", flush=True)
    (run_dir / "bulk_outputs.json").write_text(json.dumps(rows))
    errors = [r for r in rows if r["error"] is not None]
    if errors or [r["index"] for r in rows] != list(range(N_TEST)):
        raise SystemExit(f"bulk pass incomplete: {len(errors)} errors, first {errors[:3]}")
    circuit_logits = np.array([r["outputs"] for r in rows], dtype=float)
    times = [r["seconds"] for r in rows]

    # Consistency with P8.4: image 0's witness outputs equal the outputs in the committed P8.4 proof.
    p84 = json.loads((root / P8_4_PUBLIC).read_text())["pretty_public_inputs"]["rescaled_outputs"][0]
    p84_matches = [float(v) for v in p84] == circuit_logits[0].tolist()

    # Proved subset: full proof and verification, outputs compared with the bulk pass.
    proved = []
    for idx in proof_subset():
        pdir = run_dir / f"proof_{idx:05d}"
        pdir.mkdir(parents=True, exist_ok=True)
        data, wit, prf = pdir / "input.json", pdir / "witness.json", pdir / "proof.json"
        data.write_text(json.dumps({"input_data": [flat[idx].astype(float).tolist()]}))
        p83.run_stage("gen_witness", log_dir=pdir / "logs", data=rel(data),
                      compiled=rel(files["compiled_circuit"]), witness=rel(wit))
        st = p83.run_stage("prove", log_dir=pdir / "logs", witness=rel(wit),
                           compiled=rel(files["compiled_circuit"]), pk=rel(files["pk"]), proof=rel(prf),
                           srs=rel(files["srs"]))
        vf = p83.run_stage("verify", log_dir=pdir / "logs", proof=rel(prf), settings=rel(files["settings"]),
                           vk=rel(files["vk"]), srs=rel(files["srs"]))
        pf = json.loads(prf.read_text())
        proof_out = [float(v) for v in pf["pretty_public_inputs"]["rescaled_outputs"][0]]
        proved.append({"index": idx, "label": int(labels[idx]),
                       "verify_outcome": vf["value"]["outcome"],
                       "proof_outputs_equal_bulk_witness": proof_out == circuit_logits[idx].tolist(),
                       "circuit_class": int(np.argmax(proof_out)),
                       "pytorch_class": int(torch_logits[idx].argmax()),
                       "prove_seconds": st.get("call_seconds"), "prove_peak_working_set_bytes":
                       st.get("peak_working_set_bytes")})
        print(f"proved {idx}: verify {vf['value']['outcome']}, "
              f"equal to bulk {proved[-1]['proof_outputs_equal_bulk_witness']}", flush=True)
    subset_ok = all(p["verify_outcome"] == "accepted" and p["proof_outputs_equal_bulk_witness"] for p in proved)

    summary = summarise(torch_logits, circuit_logits, labels)
    path = write_result(
        name="p8.6_ezkl_fidelity", seed=args.seed, task="P8.6", git=git,
        duration_seconds=time.perf_counter() - t_start,
        params={
            "ezkl_version": ezkl.__version__,
            "model": "P0.7 zk_model via the P8.1 ONNX file; one model",
            "scale_settings": {k: run_args[k] for k in ("input_scale", "param_scale", "scale_rebase_multiplier",
                                                        "logrows", "check_mode")},
            "one_set_of_scale_settings": True,
            "image_set": {"split": "MNIST test (official 10,000)", "indices": "all, 0..9999 in order",
                          "chosen": "fixed before the run: the whole test set, not a sample",
                          "why_all": ("a scratch timing of 20 in-process gen_witness calls gave ~0.28 s per "
                                      "image, so all 10,000 take ~47 min"),
                          "normalisation": "as in training, (x - 0.1307) / 0.3081"},
            "bulk_mode": "witness only (ezkl.gen_witness), 10 chunks of 1,000, one process each, sequential",
            "circuit_class": "argmax of ezkl's dequantised outputs (pretty_elements.rescaled_outputs)",
            "pytorch_logits": "float32, eval mode, batched over the test set",
            "proved_subset": {"indices": proof_subset(), "chosen": (f"numpy default_rng({PROOF_SUBSET_SEED})"
                              f".choice({N_TEST}, {N_PROVED}, replace=False), sorted; fixed before the run")},
            "p8_3_artifacts_sha256_checked": expected,
            "limits": {"stage_timeout_seconds": p83.STAGE_TIMEOUT_SECONDS,
                       "stage_memory_limit_bytes": p83.STAGE_MEMORY_LIMIT_BYTES},
        },
        metrics={
            **summary,
            "time_per_image_seconds": {"what": "the ezkl.gen_witness call only, one image, in-process",
                                       "median": statistics.median(times), "mean": statistics.fmean(times),
                                       "min": min(times), "max": max(times),
                                       "p99": float(np.quantile(times, 0.99))},
            "bulk_wall_seconds": sum(c["process_seconds"] for c in chunks),
            "chunks": chunks,
            "image0_matches_p8_4_proof_outputs": p84_matches,
            "proved_subset": proved,
            "proved_subset_all_verify_and_equal_bulk": subset_ok,
        },
        notes=("One model (P0.7 zk_model) and one set of scale settings (input/param scale 13, logrows 18, "
               "P8.3 accuracy calibration). Local Windows CPU (x64 Python emulated on ARM), not Colab. "
               "Bulk figures are from witness generation; 10 fixed images were also proved and verified."),
    )
    s = summary
    print(f"top-1 agree {s['top1_agree']}/{s['n']}; disagreements {[d['index'] for d in s['disagreements']]}; "
          f"ties {len(s['circuit_argmax_ties'])}")
    print(f"|logit diff| max {s['abs_logit_diff']['max']:.4g}, mean {s['abs_logit_diff']['mean']:.4g}")
    print(f"accuracy PyTorch {s['pytorch_correct']}, circuit {s['circuit_correct']}; "
          f"median {statistics.median(times):.3f} s per image; subset ok {subset_ok}; P8.4 match {p84_matches}")
    print(f"record {path}")
    return 0 if subset_ok and p84_matches else 1


if __name__ == "__main__":
    sys.exit(main())
