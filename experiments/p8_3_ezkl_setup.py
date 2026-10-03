"""P8.3: EZKL gen-settings, calibrate-settings, compile and setup for `zk_model`.

    python experiments/p8_3_ezkl_setup.py [--allow-logrows-above-17]

Local CPU. Input: the committed P8.1 ONNX file `results/zk/p8.1/zk_model.onnx`,
checked by SHA-256. Each stage runs in its own child Python process
(`src/zk/ezkl/stages.py`). That process reports its own peak working set and
peak private bytes (Windows); the parent also records the launcher's figure,
which is not the stage's memory. A
`baseline` child that only imports ezkl gives the fixed start-up cost.

Settings chosen here, fixed before the run:

- Visibility: input public, output public, parameters `fixed`. The statement
  for Phase 8 (owner decision (c3)) is inference on a public input with a
  public output. With `fixed` parameters the weights are built into the
  circuit, so the verification key is tied to them.
- Scales: not set by hand. `gen_settings` starts from ezkl's defaults, and
  `calibrate_settings` picks the input and parameter scales.
- Calibration target `accuracy`, not `resources`: P8.6 measures PyTorch vs
  circuit agreement, so precision is preferred over a smaller circuit.
- Calibration data: 200 MNIST **training** images (never test), drawn without
  replacement from the 60,000 with numpy seed 1337, normalised exactly as in
  training and as the P8.1 graph expects.

The SRS is fetched by `ezkl.get_srs`, awaited, into `zk/ezkl/srs/`. If the
calibrated logrows is above 17 the run stops before fetching it and reports
the expected file size, unless `--allow-logrows-above-17` is given (owner
instruction).

Kept out of git (`/zk/ezkl/`): the proving key, the compiled circuit, the
calibration data and the SRS. Committed under `results/zk/p8.3/`: the
calibrated settings and the verification key. The script refuses to replace
an existing proving key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

import numpy as np

from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.zk.ezkl.stages import RESULT_PREFIX
from src.zk.toolchain import _peak_memory

EZKL_VERSION = "23.0.5"
P8_1_ONNX = Path("results/zk/p8.1/zk_model.onnx")
P8_1_ONNX_SHA256 = "6416735f6d7eef04d30906003a40b85c210b61253a9e53935426f3b56f7f5ca9"
WORK = Path("zk/ezkl/zk_model")
SRS_DIR = Path("zk/ezkl/srs")
OUT = Path("results/zk/p8.3")
VISIBILITY = {"input_visibility": "public", "output_visibility": "public", "param_visibility": "fixed"}
CALIBRATION_TARGET = "accuracy"
CALIBRATION_IMAGES = 200
LOGROWS_ASK_ABOVE = 17
STAGE_TIMEOUT_SECONDS = 7200


def srs_bytes(logrows: int) -> int:
    """Size of ezkl's KZG SRS file for 2^logrows rows.

    2^k G1 points and 2^k Lagrange G1 points at 64 bytes each, plus 260 bytes
    of header and G2 elements. Inferred from `kzg15.srs` (4,194,564 bytes,
    P8.2) and checked against the downloaded file in this run.
    """
    return (1 << logrows) * 128 + 260


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_stage(stage: str, **kw) -> dict:
    argv = [sys.executable, "-m", "src.zk.ezkl.stages", stage, json.dumps(kw)]
    start = time.perf_counter()
    proc = subprocess.Popen(argv, cwd=repo_root(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    try:
        out, err = proc.communicate(timeout=STAGE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise SystemExit(f"stage {stage} exceeded {STAGE_TIMEOUT_SECONDS} s; stopped")
    wall = time.perf_counter() - start
    launcher_ws, _ = _peak_memory(proc)
    lines = [ln for ln in out.splitlines() if ln.startswith(RESULT_PREFIX)]
    log = (out + err)
    rec = {"stage": stage, "returncode": proc.returncode, "process_seconds": round(wall, 3),
           "launcher_peak_working_set_bytes": launcher_ws, "log_tail": log[-1500:]}
    if proc.returncode != 0 or not lines:
        print(log[-4000:], file=sys.stderr)
        raise SystemExit(f"stage {stage} failed (exit {proc.returncode})")
    res = json.loads(lines[-1][len(RESULT_PREFIX):])
    rec["value"] = res["value"]
    rec["call_seconds"] = round(res["seconds"], 3)
    rec["peak_working_set_bytes"] = res["peak_working_set_bytes"]
    rec["peak_private_bytes"] = res["peak_private_bytes"]
    mib = (res["peak_working_set_bytes"] or 0) / 2**20
    print(f"{stage:13s} {rec['call_seconds']:8.2f} s call, {wall:8.2f} s process, peak {mib:8.1f} MiB  -> {res['value']}",
          flush=True)
    return rec


def settings_view(path: Path) -> dict:
    s = json.loads(path.read_text())
    ra = s["run_args"]
    keep = ("logrows", "input_scale", "param_scale", "scale_rebase_multiplier", "lookup_range",
            "input_visibility", "output_visibility", "param_visibility", "num_inner_cols",
            "decomp_base", "decomp_legs", "check_mode", "bounded_log_lookup")
    return {"run_args": {k: ra[k] for k in keep if k in ra},
            **{k: s.get(k) for k in ("num_rows", "total_assignments", "total_const_size",
                                     "required_lookups", "required_range_checks", "version")}}


def calibration_data(path: Path, seed: int) -> dict:
    from src.data.mnist import mnist_datasets

    train = mnist_datasets(repo_root() / "data", download=False)["train"]
    idx = np.random.default_rng(seed).choice(len(train), size=CALIBRATION_IMAGES, replace=False)
    x = np.stack([train[int(i)][0].numpy() for i in idx]).astype(np.float32)
    assert x.shape == (CALIBRATION_IMAGES, 1, 28, 28)
    path.write_text(json.dumps({"input_data": [x.reshape(-1).tolist()]}))
    return {"split": "MNIST train (60,000)", "n": CALIBRATION_IMAGES, "numpy_seed": seed,
            "indices_sha256": hashlib.sha256(np.sort(idx).astype("<i8").tobytes()).hexdigest(),
            "normalisation": "ToTensor then Normalize((0.1307,), (0.3081,)), as in training",
            "value_range": [float(x.min()), float(x.max())]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--allow-logrows-above-17", action="store_true",
                    help="fetch the SRS even if logrows > 17 (only after the owner has agreed)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    git = git_info()
    root = repo_root()
    t_start = time.perf_counter()

    import ezkl
    if ezkl.__version__ != EZKL_VERSION:
        raise SystemExit(f"ezkl {ezkl.__version__} installed, expected {EZKL_VERSION}")
    onnx_path = root / P8_1_ONNX
    if sha256_file(onnx_path) != P8_1_ONNX_SHA256:
        raise SystemExit("P8.1 ONNX file does not match its recorded SHA-256")

    work, srs_dir, out = root / WORK, root / SRS_DIR, root / OUT
    pk, vk = work / "pk.key", work / "vk.key"
    if pk.exists() or (out / "vk.key").exists():
        raise SystemExit(f"{pk} or committed vk already exists; refusing to replace keys")
    work.mkdir(parents=True, exist_ok=True)
    srs_dir.mkdir(parents=True, exist_ok=True)
    settings, compiled, cal = work / "settings.json", work / "network.compiled", work / "calibration.json"
    rel = lambda p: str(p.relative_to(root)).replace("\\", "/")  # noqa: E731

    cal_info = calibration_data(cal, args.seed)
    stages = [run_stage("baseline")]
    stages.append(run_stage("gen_settings", model=rel(onnx_path), settings=rel(settings), **VISIBILITY))
    settings_initial = settings_view(settings)
    stages.append(run_stage("calibrate", data=rel(cal), model=rel(onnx_path), settings=rel(settings),
                            target=CALIBRATION_TARGET))
    settings_calibrated = settings_view(settings)
    stages.append(run_stage("compile", model=rel(onnx_path), compiled=rel(compiled), settings=rel(settings)))

    logrows = settings_calibrated["run_args"]["logrows"]
    srs_path = srs_dir / f"kzg{logrows}.srs"
    print(f"calibrated logrows {logrows}; SRS kzg{logrows}.srs expected {srs_bytes(logrows):,} bytes", flush=True)
    if logrows > LOGROWS_ASK_ABOVE and not args.allow_logrows_above_17 and not srs_path.exists():
        print(f"STOPPED before get_srs: logrows {logrows} > {LOGROWS_ASK_ABOVE}. Expected SRS size "
              f"{srs_bytes(logrows):,} bytes ({srs_bytes(logrows) / 2**20:.1f} MiB). Ask the owner, then re-run "
              "with --allow-logrows-above-17. No record written.", flush=True)
        print(f"calibrated run_args: {json.dumps(settings_calibrated['run_args'])}; "
              f"rows {settings_calibrated['num_rows']}", flush=True)
        shutil.rmtree(work)
        return 2
    srs_cached_before = srs_path.exists()
    stages.append(run_stage("get_srs", settings=rel(settings), srs=rel(srs_path)))
    stages.append(run_stage("setup", compiled=rel(compiled), vk=rel(vk), pk=rel(pk), srs=rel(srs_path)))

    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(settings, out / "settings.json")
    shutil.copyfile(vk, out / "vk.key")
    sizes = {name: p.stat().st_size for name, p in
             (("pk", pk), ("vk", vk), ("compiled_circuit", compiled), ("settings", settings),
              ("srs", srs_path), ("calibration_data", cal))}
    if sizes["srs"] != srs_bytes(logrows):
        print(f"note: SRS file is {sizes['srs']:,} bytes, formula gave {srs_bytes(logrows):,}", flush=True)

    path = write_result(
        name="p8.3_ezkl_setup", seed=args.seed, task="P8.3", git=git,
        duration_seconds=time.perf_counter() - t_start,
        params={
            "ezkl_version": ezkl.__version__,
            "onnx": {"path": rel(onnx_path), "sha256": P8_1_ONNX_SHA256},
            "visibility": VISIBILITY,
            "scales": "not set by hand: gen_settings defaults, then chosen by calibrate_settings",
            "calibration_target": CALIBRATION_TARGET,
            "calibration_other_args": "ezkl defaults (lookup_safety_margin, scales, scale_rebase_multiplier, max_logrows)",
            "calibration_data": cal_info,
            "srs": {"path": rel(srs_path), "fetched_by": "ezkl.get_srs, awaited", "cached_before_run": srs_cached_before},
            "kept_out_of_git": ["pk.key", "network.compiled", "calibration.json", "SRS"],
            "measurement": ("each stage in its own Python process, which reads its own peak working set and "
                            "peak private bytes with GetProcessMemoryInfo after the ezkl call; includes the "
                            "Python + ezkl import cost measured by the baseline stage. The parent's reading is "
                            "of the venv launcher stub and is recorded only as launcher_peak_working_set_bytes"),
        },
        metrics={
            "stages": stages,
            "settings_initial": settings_initial,
            "settings_calibrated": settings_calibrated,
            "logrows": logrows,
            "file_bytes": sizes,
            "srs_bytes_expected": srs_bytes(logrows),
            "sha256": {"vk": sha256_file(vk), "pk": sha256_file(pk), "settings": sha256_file(settings),
                       "compiled_circuit": sha256_file(compiled), "srs": sha256_file(srs_path)},
        },
        notes=("Local Windows CPU, x64 Python under emulation on ARM; not Colab. Single run. The proving "
               "key, compiled circuit and SRS stay in zk/ezkl/ (gitignored); settings and vk are committed "
               "in results/zk/p8.3/."),
    )
    print(f"record {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
