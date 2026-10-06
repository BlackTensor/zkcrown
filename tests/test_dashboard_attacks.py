"""P9.10: the attack-lab explorer. P4.9 rows, live tiers equal to the committed grades, honest filters."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from app import attacks_logic as lab_logic
from app.data import REPO_ROOT, DataStore

APP = REPO_ROOT / "app"


@pytest.fixture(scope="module")
def store():
    return DataStore()


@pytest.fixture(scope="module")
def lab(store):
    return lab_logic.load(store)


def test_rows_are_the_committed_p4_9_rows(store, lab):
    record = store.read_json(lab.p4_9_path)["metrics"]
    assert len(lab.rows) == record["row_count"] == len(record["rows"])
    for row, committed in zip(lab.rows, record["rows"]):
        assert (row.result_file, row.setting, row.test_accuracy, row.drop_pp, row.fired, row.weight_z,
                row.outcome) == (committed["result_file"], committed["setting"], committed["test_accuracy"],
                                 committed["drop_pp"], committed["fired"], committed["weight_z"],
                                 committed["outcome"])
    assert lab.outcomes == record["summary"]["all_attacks"]["outcomes"]
    assert lab.settings_run == record["summary"]["all_attacks"]["rows"]


def test_live_tiers_equal_the_committed_grades(store, lab):
    assert lab.grade_mismatches == []
    p9_3 = store.read_json(store.latest("results/p9.3_graded_verdicts__"))["metrics"]["phase4_grade_counts"]
    assert dict(Counter(r.tier for r in lab.rows)) == p9_3


def test_outcome_counts_over_the_settings_run(lab):
    counts = lab_logic.outcome_counts(lab.rows, list(lab.outcomes))
    assert counts == lab.outcomes
    assert sum(counts.values()) == lab.settings_run


def test_not_applicable_is_said_not_left_blank(lab):
    missing = [r for r in lab.rows if not r.weight_applicable]
    assert missing and all(r.weight_status == "not applicable" for r in missing)
    assert all(r.weight_status in ("detected", "not detected") for r in lab.rows if r.weight_applicable)


def test_channel_rows_carry_the_alignment_note(store, lab):
    summary = store.read_json(lab.p4_9_path)["metrics"]["summary"]
    channel = {key for key, f in summary.items() if f.get("note") and "re-alignment" in f["note"]}
    assert channel == {"prune_channel", "prune_finetune_channel"}
    assert all(r.channel_pruned == (r.family in channel) for r in lab.rows)
    assert "never built" in lab_logic.NEVER_BUILT


def test_neither_filter_finds_the_honest_rows(lab):
    neither = lab_logic.filter_rows(lab.rows, None, "neither detected", None)
    assert len(neither) == lab.outcomes["neither detected"]
    assert all(not r.behavioral_detected and not r.weight_detected for r in neither)
    channel = lab_logic.filter_rows(lab.rows, ["prune_channel"], None, None)
    assert {r.family for r in channel} == {"prune_channel"}
    one = f"{channel[0].family_title} · {channel[0].setting}"
    assert lab_logic.filter_rows(lab.rows, None, None, [one]) == [channel[0]]


@pytest.fixture
def app_test():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60)


def test_page_renders_everything(app_test, lab):
    at = app_test.run()
    at.switch_page("views/attacks.py").run()
    assert not at.exception and not at.error
    body = " ".join(h.proto.body for h in at.get("html"))
    for count in lab.outcomes.values():
        assert f"{count} of {lab.settings_run}" in body
    assert any("all 85 equal the committed grades" in s.value for s in at.success)
    assert len(at.get("vega_lite_chart")) == 2
    assert any(lab_logic.NEVER_BUILT in w.value for w in at.warning)
    table = at.dataframe[0].value
    assert len(table) == len(lab.rows)
    assert set(table["Weight test"]) == {"detected", "not detected", "not applicable"}
    captions = " ".join(c.value for c in at.caption)
    assert "not rates" in captions and "each setting was run once" in captions
    assert "deliberately chosen grid" in captions


def test_one_click_neither_filter(app_test, lab):
    at = app_test
    at.session_state["zk_lab_outcome"] = "neither detected"
    at.run()
    at.switch_page("views/attacks.py").run()
    assert not at.exception
    table = at.dataframe[0].value
    assert len(table) == lab.outcomes["neither detected"]
    assert set(table["Behavioral test"]) == {"not detected"}
    assert "stolen" not in " ".join(h.proto.body for h in at.get("html")).lower()


def test_channel_caveat_sits_next_to_the_rows(app_test, lab):
    at = app_test.run()
    at.switch_page("views/attacks.py").run()
    assert not at.exception
    table = at.dataframe[0].value
    channel = table["Family"].isin({r.family_title for r in lab.rows if r.channel_pruned})
    assert set(table.loc[channel, "Weight caveat"]) == {lab_logic.CAVEAT_SHORT}
    assert set(table.loc[~channel, "Weight caveat"]) == {""}
    specs = [json.dumps(json.loads(c.proto.spec), ensure_ascii=False) for c in at.get("vega_lite_chart")]
    scatter = next(s for s in specs if '"shape"' in s)
    assert lab_logic.CAVEAT_SHORT in scatter
    from app.attacks_page import heatmap_cells

    flagged = [c for c in heatmap_cells(lab.rows) if c["metric"] == "Weight z" and c["text"].endswith("⚠")]
    assert len(flagged) == sum(r.channel_pruned for r in lab.rows) > 0
    assert any("Triangles are channel-pruning settings" in c.value for c in at.caption)
