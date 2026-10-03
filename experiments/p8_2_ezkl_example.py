"""P8.2: run EZKL's official example notebook, with one documented change.

    python experiments/p8_2_ezkl_example.py [--work-dir DIR] [--unchanged]

Downloads `examples/notebooks/simple_demo_all_public.ipynb` from the
`zkonduit/ezkl` repository at tag `v23.0.5` (the pinned ezkl version), checks
its SHA-256 against the digest pinned below, and executes it cell by cell with
nbclient in the venv's own kernel, in a scratch directory outside the repo.

One documented change (owner decision, 2026-10-03): `dynamo=False` is added
to the notebook's `torch.onnx.export` call, and nothing else is edited, added
or removed. Run unchanged (`--unchanged`, the first run, commit `dee2479`),
the notebook fails at `gen_settings`. The cause is torch >= 2.9's default
dynamo exporter, which ignores `opset_version=10` and writes opset 18, IR
version 10 and an external `.data` file, and EZKL's tract loader rejects that.
`dynamo=False` selects torch's legacy exporter, which honours opset 10.

After the notebook, the committed P8.1 `zk_model` ONNX file (hash-checked) is
passed to `ezkl.gen_settings` with default run args, to confirm EZKL loads it.
Settings only; calibration, compile and setup are P8.3. The notebook makes a tiny random-weight CNN
(no training, no project model), exports it to ONNX, and runs gen_settings,
calibrate_settings, compile_circuit, get_srs, gen_witness, setup, prove and
verify.

Network: GitHub (the notebook) and, inside the notebook, `ezkl.get_srs`, which
downloads EZKL's public KZG SRS into `~/.ezkl/srs`. Reads no key, no model.

The result record is written whether the notebook passes or fails, so a
failure is recorded rather than lost: the failing cell, its error, and the
cells that did run. The exit code is non-zero on failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

EZKL_VERSION = "23.0.5"
NOTEBOOK_TAG = "v23.0.5"
NOTEBOOK_NAME = "simple_demo_all_public.ipynb"
NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/zkonduit/ezkl/"
    f"{NOTEBOOK_TAG}/examples/notebooks/{NOTEBOOK_NAME}"
)
NOTEBOOK_SHA256 = "f921003074b5505b5bf4fa0dc1d5d06bb2292704f14b6787e7f00184e2cd2bd4"
"""Digest of the notebook as fetched from the tag on 2026-10-03 (11,826 bytes)."""
DEVIATION_OLD = "do_constant_folding=True,"
DEVIATION_NEW = "do_constant_folding=True, dynamo=False,"
P8_1_ONNX = Path("results/zk/p8.1/zk_model.onnx")
P8_1_ONNX_SHA256 = "6416735f6d7eef04d30906003a40b85c210b61253a9e53935426f3b56f7f5ca9"
FIRST_RUN_RECORD = "results/p8.2_ezkl_example__seed1337__20261003T182134+0000.json"
CELL_TIMEOUT_SECONDS = 1800
ARTIFACTS = ("network.onnx", "settings.json", "network.compiled", "witness.json",
             "test.pk", "test.vk", "test.pf", "input.json", "calibration.json")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cell_text_outputs(cell) -> tuple[str, list[dict]]:
    """Concatenated stream/text output, plus any error outputs, truncated."""
    texts, errors = [], []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            texts.append(out.get("text", ""))
        elif out.get("output_type") in ("execute_result", "display_data"):
            texts.append(out.get("data", {}).get("text/plain", ""))
        elif out.get("output_type") == "error":
            errors.append({"ename": out.get("ename"), "evalue": out.get("evalue")})
    text = "".join(texts)
    return (text if len(text) <= 4000 else text[:2000] + "\n...[truncated]...\n" + text[-2000:]), errors


def onnx_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    import onnx
    m = onnx.load(str(path))
    return {
        "opset_imports": {imp.domain or "ai.onnx": imp.version for imp in m.opset_import},
        "producer": f"{m.producer_name} {m.producer_version}".strip(),
        "ops": sorted({n.op_type for n in m.graph.node}),
    }


def settings_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    s = json.loads(path.read_text())
    ra = s.get("run_args", {})
    keep = ("logrows", "input_scale", "param_scale", "scale_rebase_multiplier",
            "lookup_range", "input_visibility", "output_visibility",
            "param_visibility", "commitment", "num_inner_cols", "decomp_base", "decomp_legs")
    return {
        "run_args": {k: ra[k] for k in keep if k in ra},
        "num_rows": s.get("num_rows"),
        "total_assignments": s.get("total_assignments"),
        "total_const_size": s.get("total_const_size"),
        "version": s.get("version"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="where the notebook runs and writes its files (default: a new temp dir)")
    ap.add_argument("--unchanged", action="store_true",
                    help="run the notebook with no change at all (fails under torch >= 2.9)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="recorded only; the notebook uses torch.rand unseeded, as written upstream")
    args = ap.parse_args()

    git = git_info()
    t_start = time.perf_counter()

    import ezkl
    import nbformat
    import onnx
    import onnxscript
    import torch
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    if ezkl.__version__ != EZKL_VERSION:
        raise SystemExit(f"ezkl {ezkl.__version__} installed, expected {EZKL_VERSION}")

    raw = urllib.request.urlopen(NOTEBOOK_URL, timeout=60).read()
    nb_sha = sha256_bytes(raw)
    if nb_sha != NOTEBOOK_SHA256:
        raise SystemExit(f"notebook SHA-256 {nb_sha} != pinned {NOTEBOOK_SHA256}")

    work = args.work_dir or Path(tempfile.mkdtemp(prefix="p8.2_ezkl_"))
    work = work.resolve()
    if repo_root().resolve() in (work, *work.parents):
        raise SystemExit("work dir must be outside the repo (the notebook writes keys and proofs)")
    work.mkdir(parents=True, exist_ok=True)
    (work / NOTEBOOK_NAME).write_bytes(raw)

    nb = nbformat.reads(raw.decode("utf-8"), as_version=4)
    deviation = None
    if not args.unchanged:
        hits = [i for i, c in enumerate(nb.cells)
                if c.cell_type == "code" and DEVIATION_OLD in c.source]
        if len(hits) != 1 or nb.cells[hits[0]].source.count(DEVIATION_OLD) != 1:
            raise SystemExit(f"expected exactly one {DEVIATION_OLD!r} in the notebook, found {hits}")
        before = nb.cells[hits[0]].source
        nb.cells[hits[0]].source = before.replace(DEVIATION_OLD, DEVIATION_NEW)
        changed = [(a, b) for a, b in zip(before.splitlines(), nb.cells[hits[0]].source.splitlines()) if a != b]
        assert len(changed) == 1
        deviation = {"cell_index": hits[0], "line_before": changed[0][0].strip(),
                     "line_after": changed[0][1].strip(),
                     "reason": ("torch >= 2.9 defaults to the dynamo ONNX exporter, which ignores "
                                "opset_version=10 and writes opset 18 / IR 10 with external data; "
                                "EZKL 23.0.5's tract loader rejects it at gen_settings. dynamo=False "
                                "selects the legacy exporter. Owner decision 2026-10-03 (option 1)."),
                     "unchanged_run_record": FIRST_RUN_RECORD,
                     "unchanged_run_error": ("RuntimeError: Failed to generate settings: [graph] "
                                             "[tract] Translating proto model to model")}
        (work / ("modified_" + NOTEBOOK_NAME)).write_text(nbformat.writes(nb), encoding="utf-8")
    client = NotebookClient(nb, timeout=CELL_TIMEOUT_SECONDS, kernel_name="python3",
                            resources={"metadata": {"path": str(work)}}, allow_errors=False)

    cells: list[dict] = []
    failure = None
    kernel_peak_wset = None
    with client.setup_kernel():
        kernel_pid = None
        try:
            kernel_pid = client.km.provisioner.process.pid
        except Exception:
            pass
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            first_line = next((ln for ln in cell.source.splitlines() if ln.strip()), "")
            t0 = time.perf_counter()
            ok = True
            try:
                client.execute_cell(cell, index)
            except CellExecutionError as exc:
                ok = False
                failure = {"cell_index": index, "first_line": first_line,
                           "error": str(exc)[-6000:]}
            text, errors = cell_text_outputs(cell)
            cells.append({"cell_index": index, "first_line": first_line, "ok": ok,
                          "seconds": round(time.perf_counter() - t0, 3),
                          "output_text": text, "errors": errors})
            print(f"cell [{index}] {'ok' if ok else 'FAILED'} "
                  f"{cells[-1]['seconds']:.1f}s  {first_line[:60]}", flush=True)
            if not ok:
                break
        if kernel_pid is not None:
            try:
                import psutil
                mi = psutil.Process(kernel_pid).memory_info()
                kernel_peak_wset = getattr(mi, "peak_wset", None)
            except Exception:
                pass

    p81 = None
    if failure is None:
        onnx_path = repo_root() / P8_1_ONNX
        p81_sha = sha256_bytes(onnx_path.read_bytes())
        if p81_sha != P8_1_ONNX_SHA256:
            raise SystemExit(f"P8.1 ONNX SHA-256 {p81_sha} != {P8_1_ONNX_SHA256}")
        settings_out = work / "p8.1_zk_model_settings.json"
        t0 = time.perf_counter()
        try:
            ok = ezkl.gen_settings(str(onnx_path), str(settings_out))
            err = None
        except Exception as exc:  # recorded, not hidden
            ok, err = False, f"{type(exc).__name__}: {exc}"
        p81 = {"path": str(P8_1_ONNX).replace("\\", "/"), "sha256": p81_sha,
               "run_args": "ezkl defaults (PyRunArgs())", "gen_settings_ok": ok is True,
               "error": err, "seconds": round(time.perf_counter() - t0, 3),
               "onnx_model": onnx_summary(onnx_path), "settings": settings_summary(settings_out)}
        print(f"P8.1 zk_model.onnx gen_settings: {'ok' if ok is True else 'FAILED ' + str(err)}")

    artifacts = {name: (work / name).stat().st_size for name in ARTIFACTS if (work / name).is_file()}
    srs_dir = Path.home() / ".ezkl" / "srs"
    srs_files = ({p.name: p.stat().st_size for p in sorted(srs_dir.iterdir())}
                 if srs_dir.is_dir() else {})
    passed = failure is None and (p81 is not None and p81["gen_settings_ok"])
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")

    path = write_result(
        name="p8.2_ezkl_example",
        seed=args.seed,
        task="P8.2",
        git=git,
        duration_seconds=time.perf_counter() - t_start,
        params={
            "ezkl_version": ezkl.__version__,
            "notebook_url": NOTEBOOK_URL,
            "notebook_sha256": nb_sha,
            "notebook_unchanged": args.unchanged,
            "deviation": deviation,
            "work_dir_outside_repo": True,
            "kernel": "python3 (the venv's own interpreter)",
            "cell_timeout_seconds": CELL_TIMEOUT_SECONDS,
            "python": sys.version.split()[0],
            "python_machine": platform.machine(),
            "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                         "onnxscript": onnxscript.__version__},
        },
        metrics={
            "passed": passed,
            "code_cells_total": n_code,
            "code_cells_run": len(cells),
            "notebook_passed": failure is None,
            "failure": failure,
            "p8_1_zk_model_gen_settings": p81,
            "cells": cells,
            "onnx_model": onnx_summary(work / "network.onnx"),
            "settings": settings_summary(work / "settings.json"),
            "artifact_bytes": artifacts,
            "srs_files_bytes": srs_files,
            "kernel_peak_working_set_bytes": kernel_peak_wset,
        },
        notes=(("EZKL's official example run unchanged" if args.unchanged else
                "EZKL's official example run with one documented change (dynamo=False in the "
                "ONNX export call; see params.deviation)") + " on this machine (x64 Python under "
               "emulation on an ARM CPU, Windows). Random untrained weights, unseeded "
               "inputs as written upstream: the circuit sizes and times are a toolchain "
               "check, not P8 measurements of zk_model."),
    )
    print(f"{'PASSED' if passed else 'FAILED'}: {len(cells)}/{n_code} code cells; record {path}")
    if passed:
        if args.work_dir is None:
            shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"work dir kept for inspection: {work}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
