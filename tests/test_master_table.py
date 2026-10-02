"""Tests for P4.9: the master robustness table.

The script only rearranges committed attack rows, so the tests cover the
grouping, the refusal of rows that do not belong, the outcome labels, and that
the table and figure render. One test runs over the committed rows themselves.
"""

from __future__ import annotations

import copy
import importlib
import json
import warnings
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
ROWS_DIR = REPO_ROOT / "results" / "attacks"


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p4_9_master_table")


def make_record(script, config, task, *, fired=100, z=10.0, accuracy=0.9, applicable=True, dirty=False):
    behavioral_detected, weight_detected = fired >= 29, applicable and z >= 5.257
    weight = ({"applicable": True, "correlation": z / script.Z_CAP, "z": z, "bit_matches": 100, "p_value_bound": 1e-9,
               "rejects": {"1e-6": weight_detected}, "detected": weight_detected}
              if applicable else {"applicable": False, "reason": "layout gone", "detected": False})
    return {
        "schema_version": 1,
        "task": task,
        "git": {"commit": "abc", "dirty": dirty},
        "metrics": {"row": {
            "row_version": script.ROW_VERSION,
            "table": {"detection_alpha": "1e-6"},
            "config": config,
            "source": {"name": "dual", "weights_sha256": script.W_DUAL_SHA256},
            "clean_accuracy": {"accuracy": accuracy, "correct": round(accuracy * 10000), "drop_vs_source_pp": 1.0,
                               "drop_ci95_pp": [0.5, 1.5], "mcnemar_exact_p": 0.01},
            "behavioral": {"fired": fired, "n": 100, "p_value": 1e-9, "rejects": {"1e-6": behavioral_detected},
                           "detected": behavioral_detected},
            "weight": weight,
            "owner": {"owner_id": "o", "trigger_bundle_sha256": "b", "carrier_digest": "c"},
        }},
    }


CONFIGS = {
    "control": ({"attack": "none", "strength": 0.0, "params": {}}, "P4.1"),
    "prune_layerwise": ({"attack": "magnitude_prune_layerwise", "strength": 0.5, "params": {}}, "P4.2"),
    "prune_global": ({"attack": "magnitude_prune_global", "strength": 0.5, "params": {}}, "P4.2"),
    "prune_channel": ({"attack": "channel_prune_l1", "strength": 0.05, "params": {}}, "P4.3"),
    "quantize": ({"attack": "ptq_int8_static", "strength": 8.0, "params": {}}, "P4.4"),
    "finetune": ({"attack": "finetune_holdout", "strength": 20.0, "params": {"lr": 0.01}}, "P4.5"),
    "prune_finetune_global": ({"attack": "prune_finetune", "strength": 0.5,
                               "params": {"prune": "magnitude_prune_global", "lr": 0.01, "epochs": 20}}, "P4.6"),
    "prune_finetune_channel": ({"attack": "prune_finetune", "strength": 0.5,
                                "params": {"prune": "channel_prune_l1", "lr": 0.1, "epochs": 60}}, "P4.6"),
    "distill": ({"attack": "distill", "strength": 16.0, "params": {"transfer": "train50k", "lr": 0.1}}, "P4.7"),
    "overwrite_weight": ({"attack": "overwrite_weight", "strength": 2.0, "params": {}}, "P4.8"),
    "overwrite_behavioral": ({"attack": "overwrite_behavioral", "strength": 0.01, "params": {"epochs": 20}}, "P4.8"),
    "overwrite_both": ({"attack": "overwrite_both", "strength": 0.01, "params": {"epochs": 20, "weight_alpha": 0.1}}, "P4.8"),
}


def test_every_attack_maps_to_exactly_one_family(script):
    assert {f["id"] for f in script.FAMILIES} == set(CONFIGS)
    for family_id, (config, task) in CONFIGS.items():
        family = script.family_of(config)
        assert family["id"] == family_id and family["task"] == task
        assert family["label"](config)
    with pytest.raises(ValueError):
        script.family_of({"attack": "unknown", "strength": 1.0, "params": {}})


def test_setting_labels(script):
    label = lambda fid: script.family_of(CONFIGS[fid][0])["label"](CONFIGS[fid][0])  # noqa: E731
    assert label("prune_channel") == "channels 5%"
    assert label("finetune") == "LR 0.01, 20 ep"
    assert label("prune_finetune_channel") == "channels 50%, LR 0.1, 60 ep"
    assert label("distill") == "50,000 images, LR 0.1, width 16"
    assert label("overwrite_both") == "LR 0.01, 20 ep, alpha' 0.1"


def test_outcome_labels(script):
    assert script.outcome(True, True) == "both detected"
    assert script.outcome(False, True) == "behavioral lost, weight detected"
    assert script.outcome(True, False) == "weight lost, behavioral detected"
    assert script.outcome(False, False) == "neither detected"


def test_entry_copies_the_row_and_marks_not_applicable(script):
    config, task = CONFIGS["distill"]
    entry = script.entry_from_record(make_record(script, config, task, fired=5, applicable=False, accuracy=0.9067), "f.json")
    assert entry["fired"] == 5 and entry["test_accuracy"] == 0.9067
    assert entry["weight_applicable"] is False and entry["weight_z"] is None
    assert entry["outcome"] == "neither detected"
    cells = script.cells(entry)
    assert cells[2] == {"fill": None, "text": "n/a", "lost": True}
    assert cells[1]["lost"] is True and cells[0]["lost"] is False


@pytest.mark.parametrize("mutate", [
    lambda r: r["git"].update(dirty=True),
    lambda r: r["git"].update(dirty=None),
    lambda r: r["metrics"]["row"].update(row_version="other/v1"),
    lambda r: r["metrics"]["row"]["source"].update(name="clean"),
    lambda r: r["metrics"]["row"]["source"].update(weights_sha256="00"),
    lambda r: r["metrics"]["row"]["table"].update(detection_alpha="0.05"),
    lambda r: r.update(task="P4.2"),
])
def test_rows_that_do_not_belong_are_refused(script, mutate):
    config, task = CONFIGS["finetune"]
    record = make_record(script, config, task)
    script.entry_from_record(copy.deepcopy(record), "ok.json")
    mutate(record)
    with pytest.raises(ValueError):
        script.entry_from_record(record, "bad.json")


def test_cells_clip_to_the_scale(script):
    config, task = CONFIGS["finetune"]
    low = script.entry_from_record(make_record(script, config, task, fired=0, z=-1.1, accuracy=0.0997), "a.json")
    assert [c["fill"] for c in script.cells(low)] == [0.0, 0.0, 0.0]
    high = script.entry_from_record(make_record(script, config, task, fired=100, z=script.Z_CAP, accuracy=1.0), "b.json")
    assert [c["fill"] for c in script.cells(high)] == [1.0, 1.0, 1.0]


def test_summary_counts(script):
    config, task = CONFIGS["finetune"]
    rows = [script.entry_from_record(make_record(script, config, task, fired=f, z=z, accuracy=a), f"{i}.json")
            for i, (f, z, a) in enumerate([(100, 10.0, 0.90), (8, 9.0, 0.86), (5, 2.0, 0.80), (6, 3.0, 0.79)])]
    s = script.summarise(rows)
    assert (s["rows"], s["behavioral_detected"], s["weight_detected"]) == (4, 1, 2)
    assert s["outcomes"] == {"both detected": 1, "behavioral lost, weight detected": 1,
                             "weight lost, behavioral detected": 0, "neither detected": 2}
    assert s["highest_accuracy_behavioral_not_detected"]["test_accuracy"] == 0.86
    assert s["highest_accuracy_neither_detected"]["test_accuracy"] == 0.80
    assert (s["fired_min"], s["fired_max"], s["weight_z_min"], s["weight_z_max"]) == (5, 100, 2.0, 10.0)


def write_rows(script, directory: Path, records: dict[str, dict]) -> None:
    for name, record in records.items():
        (directory / name).write_text(json.dumps(record), encoding="utf-8")


def small_table(script, monkeypatch, directory: Path) -> tuple:
    """Shrink the plan to the control plus one attack row, and write those two rows."""
    by_id = {f["id"]: f for f in script.FAMILIES}
    families = (by_id["control"], dict(by_id["overwrite_weight"], expected=1))
    monkeypatch.setattr(script, "FAMILIES", families)
    write_rows(script, directory, {"a.json": make_record(script, *CONFIGS["control"]),
                                   "w.json": make_record(script, *CONFIGS["overwrite_weight"])})
    return families


def test_collect_refuses_a_partial_or_duplicated_table(script, tmp_path, monkeypatch):
    config, task = CONFIGS["control"]
    families = small_table(script, monkeypatch, tmp_path)
    (tmp_path / "p4.x_apply").mkdir()  # apply-record subfolders are ignored
    (tmp_path / "p4.x_apply" / "x.json").write_text("{}", encoding="utf-8")
    entries, owner = script.collect(tmp_path)
    assert [e["family"] for e in entries] == ["control", "overwrite_weight"] and owner["owner_id"] == "o" and "sort_key" not in entries[0] and "_owner" not in entries[0]

    write_rows(script, tmp_path, {"b.json": make_record(script, config, task)})
    with pytest.raises(ValueError, match="same attack setting"):
        script.collect(tmp_path)
    (tmp_path / "b.json").unlink()

    (tmp_path / "a.json").unlink()
    with pytest.raises(ValueError, match="expected 1"):
        script.collect(tmp_path)

    other = make_record(script, config, task)
    other["metrics"]["row"]["owner"]["trigger_bundle_sha256"] = "different"
    write_rows(script, tmp_path, {"a.json": make_record(script, config, task)})
    two = dict(families[0], expected=2, label=lambda c: str(id(c)))
    monkeypatch.setattr(script, "FAMILIES", (two, families[1]))
    write_rows(script, tmp_path, {"c.json": other})
    with pytest.raises(ValueError, match="different owner materials"):
        script.collect(tmp_path)


@pytest.mark.skipif(not any(ROWS_DIR.glob("p4.8_overwrite_both*.json")), reason="committed attack rows not present")
def test_committed_rows_make_a_complete_table(script, tmp_path):
    entries, _ = script.collect(ROWS_DIR)
    assert len(entries) == sum(f["expected"] for f in script.FAMILIES) == 85
    summary = script.summaries(entries)
    assert summary["all_attacks"]["rows"] == 84
    assert summary["control"]["outcomes"]["both detected"] == 1
    assert summary["quantize"]["behavioral_detected"] == summary["quantize"]["weight_detected"] == 3
    assert summary["distill"]["outcomes"]["neither detected"] == 6 and summary["distill"]["weight_not_applicable"] == 3
    for e in entries:  # the detected flags are the rows' own 1e-6 decisions
        assert e["behavioral_detected"] == (e["fired"] >= 29)
        if e["weight_applicable"]:
            assert e["weight_detected"] == (e["weight_z"] >= 5.257)

    text = script.markdown(entries, summary, "p4.9_x.json")
    assert text.count("\n| ") >= 85 + len(summary)
    lowered = text.lower()
    for banned in ("unremovable", "court proof", "legally proves", "immune to pruning"):
        assert banned not in lowered

    figure = tmp_path / "heatmap.png"
    script.plot(entries, figure)
    assert figure.stat().st_size > 10_000
    again = tmp_path / "again.png"
    script.plot(entries, again)
    assert figure.read_bytes() == again.read_bytes()


def test_main_writes_record_table_and_figure(script, tmp_path, monkeypatch):
    rows = tmp_path / "rows"
    rows.mkdir()
    small_table(script, monkeypatch, rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the dirty-tree warning, irrelevant here
        out = script.main(["--rows-dir", str(rows), "--out-dir", str(tmp_path / "out"),
                           "--table", str(tmp_path / "t.md"), "--figure", str(tmp_path / "f.png")])
    record = json.loads(out["path"].read_text(encoding="utf-8"))
    assert record["task"] == "P4.9" and record["metrics"]["row_count"] == 2
    assert record["metrics"]["rows"][0]["outcome"] == "both detected"
    assert out["path"].name in (tmp_path / "t.md").read_text(encoding="utf-8")
    assert (tmp_path / "f.png").is_file()
