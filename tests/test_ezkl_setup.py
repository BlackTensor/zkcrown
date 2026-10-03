"""P8.3: EZKL stage runner and setup-script helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
try:
    import p8_3_ezkl_setup as setup  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "experiments"))

from src.zk.ezkl import stages  # noqa: E402

ezkl = pytest.importorskip("ezkl")


def test_srs_size_formula_matches_the_p8_2_file():
    assert setup.srs_bytes(15) == 4_194_564  # kzg15.srs as downloaded in P8.2
    assert setup.srs_bytes(17) == 4 * setup.srs_bytes(15) - 3 * 260


def test_settings_are_fixed_before_the_run():
    assert setup.VISIBILITY == {"input_visibility": "public", "output_visibility": "public",
                                "param_visibility": "fixed"}
    assert setup.CALIBRATION_TARGET == "accuracy"
    assert setup.LOGROWS_ASK_ABOVE == 17


def test_unknown_stage_is_refused():
    with pytest.raises(ValueError):
        stages.run_stage("no_such_stage", {})


def test_baseline_stage_in_child_process_reports_version_and_peak_memory(tmp_path):
    rec = setup.run_stage("baseline", log_dir=tmp_path)
    assert rec["returncode"] == 0 and rec["value"] == ezkl.__version__
    if sys.platform == "win32":
        assert rec["peak_working_set_bytes"] > 10 * 2**20


@pytest.mark.skipif(sys.platform != "win32", reason="Windows memory counters")
def test_stage_peak_memory_is_the_working_process_not_the_launcher():
    import subprocess
    code = ("from src.zk.toolchain import own_peak_memory; b = bytearray(300 * 2**20); "
            "b[::4096] = bytes(len(b[::4096])); print(own_peak_memory()[0])")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert int(out.stdout.strip()) > 300 * 2**20


def test_gen_settings_stage_on_the_p8_1_onnx(tmp_path):
    out = tmp_path / "settings.json"
    rec = setup.run_stage("gen_settings", log_dir=tmp_path, model=str(REPO_ROOT / setup.P8_1_ONNX), settings=str(out),
                          **setup.VISIBILITY)
    assert rec["value"] is True
    ra = json.loads(out.read_text())["run_args"]
    assert (ra["input_visibility"], ra["output_visibility"], ra["param_visibility"]) == ("Public", "Public", "Fixed")


def test_get_srs_is_awaited_not_left_pending():
    import inspect
    assert inspect.iscoroutinefunction(stages._await_srs)


def test_watchdog_stops_a_stage_past_the_memory_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "STAGE_MEMORY_LIMIT_BYTES", 1)  # any running child exceeds it
    monkeypatch.setattr(setup, "WATCH_INTERVAL_SECONDS", 0.05)
    with pytest.raises(SystemExit, match="STOPPED"):
        setup.run_stage("gen_settings", log_dir=tmp_path, model=str(REPO_ROOT / setup.P8_1_ONNX),
                        settings=str(tmp_path / "s.json"), **setup.VISIBILITY)


def test_limits_are_the_owners():
    assert setup.STAGE_TIMEOUT_SECONDS == 3600
    assert setup.STAGE_MEMORY_LIMIT_BYTES == 8 * 2**30
