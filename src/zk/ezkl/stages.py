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
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time

RESULT_PREFIX = "RESULT "
STAGES = ("baseline", "gen_settings", "calibrate", "compile", "get_srs", "setup")


async def _await_srs(**kwargs):
    import ezkl

    res = ezkl.get_srs(**kwargs)
    return (await res) if inspect.isawaitable(res) else res


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
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")


def main(argv: list[str]) -> int:
    stage, kw = argv[0], json.loads(argv[1]) if len(argv) > 1 else {}
    t0 = time.perf_counter()
    value = run_stage(stage, kw)
    seconds = time.perf_counter() - t0
    from src.zk.toolchain import own_peak_memory

    peak_ws, peak_private = own_peak_memory()
    print(RESULT_PREFIX + json.dumps({"stage": stage, "value": value if isinstance(value, (bool, str, int, float)) else repr(value),
                                      "seconds": seconds, "peak_working_set_bytes": peak_ws,
                                      "peak_private_bytes": peak_private}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
