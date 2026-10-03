"""EZKL pipeline stages for Track B, each run in its own child process (P8.3).

    python -m src.zk.ezkl.stages <stage> '<json kwargs>'

One stage per process, so the process's peak memory belongs to that stage
alone, plus the fixed cost of starting Python and importing ezkl (the
`baseline` stage measures that). The child measures its own peak working set
and peak private bytes at the end (`own_peak_memory`), because on Windows the
venv `python.exe` the parent starts is only a launcher for the real
interpreter, and the parent would measure the launcher (P8.3's first run read
about 23.5 MiB for every stage that way). The child prints one JSON line,
prefixed with `RESULT `, with the stage's return value, its wall time around
the ezkl call, and its peak memory.

Stages:

- `baseline`: import ezkl and return. No work.
- `gen_settings`: `ezkl.gen_settings` with the given visibilities.
- `calibrate`: `ezkl.calibrate_settings` on a calibration data file.
- `compile`: `ezkl.compile_circuit`.
- `get_srs`: `ezkl.get_srs`, awaited inside an event loop. In ezkl 23.0.5 it
  is async; unawaited it returns a pending Future (P8.2).
- `setup`: `ezkl.setup`, writing the verification and proving keys.
- `gen_witness`: `ezkl.gen_witness` for one input file (P8.4).
- `prove`: `ezkl.prove` from a witness, compiled circuit and proving key (P8.4).
- `verify`: `ezkl.verify` on one proof file (P8.5). Reports the outcome, not
  just the return value: `accepted` (True), `rejected` (a clean False), or
  `error` (an exception, with its message).
- `verify_batch`: the same for a list of proof files in one process, each
  with its own time (P8.5 negative checks).

A stage that returns a dict (the witness, the proof) reports only its type;
the content is in the file it wrote.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time

RESULT_PREFIX = "RESULT "
STAGES = ("baseline", "gen_settings", "calibrate", "compile", "get_srs", "setup", "gen_witness", "prove",
          "verify", "verify_batch")


async def _await_srs(**kwargs):
    import ezkl

    res = ezkl.get_srs(**kwargs)
    return (await res) if inspect.isawaitable(res) else res


def verify_outcome(proof: str, settings: str, vk: str, srs: str) -> dict:
    """Run `ezkl.verify` once and classify what happened."""
    import ezkl

    t0 = time.perf_counter()
    try:
        ok = ezkl.verify(proof, settings, vk, srs_path=srs)
        outcome, error = ("accepted" if ok is True else "rejected"), None
        if ok not in (True, False):
            outcome, error = "error", f"non-boolean return {ok!r}"
    except Exception as exc:  # recorded, not hidden
        outcome, error = "error", f"{type(exc).__name__}: {exc}"[:500]
    return {"outcome": outcome, "error": error, "seconds": time.perf_counter() - t0}


def run_stage(stage: str, kw: dict):
    import ezkl

    if stage == "baseline":
        return ezkl.__version__
    if stage == "gen_settings":
        args = ezkl.PyRunArgs()
        args.input_visibility = kw["input_visibility"]
        args.output_visibility = kw["output_visibility"]
        args.param_visibility = kw["param_visibility"]
        return ezkl.gen_settings(kw["model"], kw["settings"], py_run_args=args)
    if stage == "calibrate":
        return ezkl.calibrate_settings(kw["data"], kw["model"], kw["settings"], kw["target"])
    if stage == "compile":
        return ezkl.compile_circuit(kw["model"], kw["compiled"], kw["settings"])
    if stage == "get_srs":
        return asyncio.run(_await_srs(settings_path=kw["settings"], srs_path=kw["srs"]))
    if stage == "setup":
        return ezkl.setup(kw["compiled"], kw["vk"], kw["pk"], srs_path=kw["srs"])
    if stage == "gen_witness":
        return ezkl.gen_witness(kw["data"], kw["compiled"], kw["witness"])
    if stage == "prove":
        return ezkl.prove(kw["witness"], kw["compiled"], kw["pk"], kw["proof"], srs_path=kw["srs"])
    if stage == "verify":
        return verify_outcome(kw["proof"], kw["settings"], kw["vk"], kw["srs"])
    if stage == "verify_batch":
        return {"results": [{"case": c, **verify_outcome(c, kw["settings"], kw["vk"], kw["srs"])}
                            for c in kw["proofs"]]}
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")


def main(argv: list[str]) -> int:
    stage, kw = argv[0], json.loads(argv[1]) if len(argv) > 1 else {}
    t0 = time.perf_counter()
    value = run_stage(stage, kw)
    seconds = time.perf_counter() - t0
    from src.zk.toolchain import own_peak_memory

    peak_ws, peak_private = own_peak_memory()
    print(RESULT_PREFIX + json.dumps({"stage": stage, "value": (value if isinstance(value, (bool, str, int, float)) or stage in ("verify", "verify_batch")
                                                else type(value).__name__),
                                      "seconds": seconds, "peak_working_set_bytes": peak_ws,
                                      "peak_private_bytes": peak_private}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
