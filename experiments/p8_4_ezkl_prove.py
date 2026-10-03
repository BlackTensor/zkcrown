"""P8.4: EZKL witness and proof of `zk_model` inference on one public MNIST test image.

    python experiments/p8_4_ezkl_prove.py

Local CPU. Uses the P8.3 compiled circuit, proving key and SRS from
`zk/ezkl/` (gitignored), each checked against the SHA-256 the committed P8.3
record holds. The SRS is already on disk from P8.3, so `get_srs` is not
called; if it were needed it would have to be awaited (P8.2).

The image is **index 0 of the official 10,000-image MNIST test set**, fixed
before any run by position alone, not chosen by looking at a result. It is
normalised as in training, the same input the P8.1 graph takes. The input
and output are public (P8.3 settings), so both are in the proof's public
instances.

Each of `gen_witness` and `prove` runs in its own Python process
(`src/zk/ezkl/stages.py`), which measures its own peak working set and peak
private bytes; the P8.3 watchdog stops a stage past 8 GiB or one hour.

No verification beyond a sanity check, which never calls `ezkl.verify` (that
is P8.5): the proof file parses, carries proof bytes, and its public
instances encode exactly the witness's input and output values.

Committed in `results/zk/p8.4/`: the proof, the public instances, and the
input file. The witness stays in `zk/ezkl/zk_model/p8.4/` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

import numpy as np
import torch

from src.utils.results import git_info, read_result, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

import p8_1_export_onnx as p81
import p8_3_ezkl_setup as p83

TEST_INDEX = 0
"""Fixed before the run: the first image of the official MNIST test set."""
P8_3_RECORD = Path("results/p8.3_ezkl_setup__seed1337__20261003T185318+0000.json")
WORK = Path("zk/ezkl/zk_model")
OUT = Path("results/zk/p8.4")


def rescaled(section: dict | None, key: str):
    if not section or key not in section:
        return None
    return section[key]


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
    settings = json.loads(files["settings"].read_text())

    out, run_dir = root / OUT, work / "p8.4"
    if (out / "proof.json").exists():
        raise SystemExit(f"{out / 'proof.json'} exists; refusing to replace it")
    run_dir.mkdir(parents=True, exist_ok=True)
    data, witness, proof = run_dir / "input.json", run_dir / "witness.json", run_dir / "proof.json"

    xs, ys = p81.test_arrays()
    x, label = xs[TEST_INDEX], int(ys[TEST_INDEX])
    assert x.shape == (1, 28, 28)
    data.write_text(json.dumps({"input_data": [x.reshape(-1).astype(float).tolist()]}))
    model = p81.load_model()
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(x[None])).numpy()[0].astype(float)
    torch_class = int(np.argmax(torch_logits))

    rel = lambda p: str(p.relative_to(root)).replace("\\", "/")  # noqa: E731
    stages = [p83.run_stage("baseline", log_dir=run_dir / "logs")]
    stages.append(p83.run_stage("gen_witness", log_dir=run_dir / "logs", data=rel(data),
                                compiled=rel(files["compiled_circuit"]), witness=rel(witness)))
    stages.append(p83.run_stage("prove", log_dir=run_dir / "logs", witness=rel(witness),
                                compiled=rel(files["compiled_circuit"]), pk=rel(files["pk"]),
                                proof=rel(proof), srs=rel(files["srs"])))

    w = json.loads(witness.read_text())
    pf = json.loads(proof.read_text())
    w_pretty, pf_pretty = w.get("pretty_elements") or {}, pf.get("pretty_public_inputs") or {}
    circuit_out = rescaled(pf_pretty, "rescaled_outputs") or rescaled(w_pretty, "rescaled_outputs")
    circuit_logits = np.asarray(circuit_out[0], dtype=float)
    circuit_in = np.asarray((rescaled(pf_pretty, "rescaled_inputs") or rescaled(w_pretty, "rescaled_inputs"))[0],
                            dtype=float)
    circuit_class = int(np.argmax(circuit_logits))

    # Sanity check only (no ezkl.verify): the proof's public instances encode the witness's input and output.
    instances = pf.get("instances")
    flat_instances = [v for group in instances for v in group] if instances else []
    witness_public = [v for group in w.get("inputs", []) for v in group] + \
                     [v for group in w.get("outputs", []) for v in group]
    proof_field = pf.get("proof")
    proof_bytes = (len(bytes.fromhex(proof_field[2:] if proof_field.startswith("0x") else proof_field))
                   if isinstance(proof_field, str) else len(proof_field or []))
    sanity = {
        "proof_file_parses": True,
        "proof_bytes_nonzero": proof_bytes > 0,
        "public_instance_count": len(flat_instances),
        "public_instance_count_expected": 784 + 10,
        "instances_equal_witness_input_and_output": flat_instances == witness_public,
        "ezkl_verify_called": False,
    }
    if not (sanity["proof_bytes_nonzero"] and sanity["instances_equal_witness_input_and_output"]
            and len(flat_instances) == 794):
        raise SystemExit(f"sanity check failed: {sanity}")

    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(proof, out / "proof.json")
    shutil.copyfile(data, out / "input.json")
    public = {"note": ("Public instances of the P8.4 proof: the 784 input values then the 10 output "
                       "logits, as field elements, plus ezkl's rescaled (dequantised) view."),
              "instances": instances, "pretty_public_inputs": pf_pretty or None}
    (out / "public_instances.json").write_text(json.dumps(public, indent=1) + "\n")

    input_scale = settings["run_args"]["input_scale"]
    in_err = float(np.max(np.abs(circuit_in - x.reshape(-1))))
    path = write_result(
        name="p8.4_ezkl_prove", seed=args.seed, task="P8.4", git=git,
        duration_seconds=time.perf_counter() - t_start,
        params={
            "ezkl_version": ezkl.__version__,
            "test_image": {"split": "MNIST test (official 10,000)", "index": TEST_INDEX,
                           "chosen": "fixed before the run, by position (first image); not selected by results",
                           "normalisation": "as in training, (x - 0.1307) / 0.3081"},
            "visibility": {k: settings["run_args"][k] for k in ("input_visibility", "output_visibility",
                                                                 "param_visibility")},
            "scales": {"input_scale": input_scale, "param_scale": settings["run_args"]["param_scale"]},
            "logrows": settings["run_args"]["logrows"],
            "p8_3_artifacts_sha256_checked": expected,
            "srs_fetched": False,
            "limits": {"stage_timeout_seconds": p83.STAGE_TIMEOUT_SECONDS,
                       "stage_memory_limit_bytes": p83.STAGE_MEMORY_LIMIT_BYTES},
            "measurement": "each stage in its own Python process, which reads its own peak memory (P8.3)",
        },
        metrics={
            "label": label,
            "pytorch": {"logits": torch_logits.tolist(), "class": torch_class},
            "circuit": {"rescaled_output_logits": circuit_logits.tolist(), "class": circuit_class},
            "class_matches_pytorch": circuit_class == torch_class,
            "max_abs_logit_diff_vs_pytorch": float(np.max(np.abs(circuit_logits - torch_logits))),
            "max_abs_input_quantisation_error": in_err,
            "stages": stages,
            "proof_file_bytes": (out / "proof.json").stat().st_size,
            "proof_bytes": proof_bytes,
            "public_instances_file_bytes": (out / "public_instances.json").stat().st_size,
            "witness_file_bytes": witness.stat().st_size,
            "key_bytes": {"pk": files["pk"].stat().st_size, "vk": files["vk"].stat().st_size,
                          "srs": files["srs"].stat().st_size},
            "sanity_check": sanity,
            "proof_sha256": p83.sha256_file(out / "proof.json"),
        },
        notes=("One proof of one image on local Windows CPU (x64 Python emulated on ARM), not Colab. "
               "Not verified here (P8.5). One image says nothing general about circuit fidelity (P8.6)."),
    )
    print(f"label {label}; PyTorch class {torch_class}; circuit class {circuit_class}; "
          f"max |logit diff| {np.max(np.abs(circuit_logits - torch_logits)):.4g}")
    print(f"proof {proof_bytes:,} bytes ({(out / 'proof.json').stat().st_size:,} as JSON); record {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
