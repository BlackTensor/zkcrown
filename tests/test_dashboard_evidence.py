"""P9.9: the evidence-strength page. Committed nulls, the repo's own statistics, and honest labels."""

from __future__ import annotations

import math
import subprocess
import sys
from fractions import Fraction

import pytest

from app import audit_logic as al
from app import evidence_logic as ev
from app import repo_code
from app.data import REPO_ROOT, DataStore

APP = REPO_ROOT / "app"


@pytest.fixture(scope="module")
def store():
    return DataStore()


@pytest.fixture(scope="module")
def behavioral(store):
    return ev.behavioral(store)


@pytest.fixture(scope="module")
def weight(store):
    return ev.weight(store)


def test_wrong_key_counts_are_the_committed_ones(store, behavioral):
    record = store.read_json(behavioral.path)["metrics"]["null_check"]
    for model, label in ev.NULL_MODELS.items():
        drawn = {h["k"]: h["count"] for h in behavioral.histogram if h["model"] == label}
        committed = record[model]["fired_histogram"]
        assert drawn == {k: c for k, c in enumerate(committed) if c}
        assert sum(drawn.values()) == record[model]["keys"] == behavioral.keys[label]
        assert max(drawn) == record[model]["fired_max"] == behavioral.max_fired[label]


def test_thresholds_match_the_record_and_the_repo_code(store, behavioral):
    sig = repo_code.significance()
    recorded = store.read_json(behavioral.path)["metrics"]["thresholds"]
    for t in behavioral.thresholds:
        assert t["k"] == recorded[t["alpha"]]["threshold"]
        k_star, bound = sig.detection_threshold(behavioral.n, t["alpha"], behavioral.num_classes)
        assert k_star == t["k"] and bound <= Fraction(t["alpha"])
        assert behavioral.tail[t["k"]]["p"] == pytest.approx(recorded[t["alpha"]]["exact_bound"], rel=1e-12)


def test_null_curves_come_from_the_repo_code(behavioral):
    sig = repo_code.significance()
    for row in behavioral.tail:
        assert row["p"] == float(sig.detection_p_value(row["k"], behavioral.n, behavioral.num_classes))
    keys = max(behavioral.keys.values())
    assert sum(r["expected"] for r in behavioral.bound_expected) == pytest.approx(keys)


def test_weight_bound_and_thresholds(store, weight):
    ws = repo_code.weight_significance()
    recorded = store.read_json(weight.path)["metrics"]
    assert weight.cap == math.sqrt(weight.rows)
    assert weight.floor == pytest.approx(recorded["p_value_floor"], rel=1e-12)
    for t in weight.thresholds:
        assert t["z"] == recorded["thresholds"][t["alpha"]]["z"] == pytest.approx(ws.z_threshold(t["alpha"]))
    curve = ev.bound_curve(weight, -1.5)
    assert curve[0]["z"] == -1.5 and curve[-1]["z"] == pytest.approx(weight.cap)
    assert all(row["bound"] == ws.p_value_bound(row["z"]) for row in curve)


def test_suspect_statistics_are_the_recorded_ones(store):
    sources = al.load_sources(store)
    for s in al.SUSPECTS:
        name = sources.names[s.prefix]
        stats = ev.suspect_statistics(sources, name)
        checks = {c["slot"]: c for c in sources.verdicts[name]["checks"]}
        assert stats.fired == checks["behavioral"]["statistic"]["fired"]
        assert stats.behavioral_p == checks["behavioral"]["p_value"]
        assert stats.z == checks["weight"]["statistic"]["z"]
        assert stats.weight_p == checks["weight"]["p_value"]


def test_repo_code_is_the_only_door_into_src_and_never_loads_torch():
    for path in APP.rglob("*.py"):
        if path.name in ("repo_code.py", "build_manifest.py"):
            continue
        text = path.read_text(encoding="utf-8")
        assert "src." not in text.replace("`src.", "") and "importlib" not in text, path
    code = ("import sys; sys.path.insert(0, '.'); from app import repo_code as r; "
            "r.grading(); r.significance(); r.weight_significance(); "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'torchvision')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


@pytest.fixture
def app_test():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60)


@pytest.mark.parametrize("suspect", [s.prefix for s in al.SUSPECTS])
def test_page_renders_each_suspect_with_honest_labels(app_test, suspect):
    at = app_test
    at.session_state["zk_suspect_evidence"] = suspect
    at.run()
    at.switch_page("views/evidence.py").run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert len(at.get("vega_lite_chart")) == 4
    null = ev.measured_null(DataStore())
    warning = " ".join(w.value for w in at.warning)
    assert "**proven bound**" in warning and f"**measured, {null.count:,} wrong keys**" in warning
    assert "mathematical upper limit, not data" in warning and "binned form" in warning
    captions = [c.value for c in at.caption]
    assert sum("**Conclude:**" in c for c in captions) == 2
    assert sum("**Not shown:**" in c for c in captions) == 4
    assert any("only the levels near 0.05 and 0.01 are checked by measurement" in c
               and "rests on the proof alone" in c for c in captions)


def test_picker_is_shared_with_live_audit(app_test):
    from app.suspect_picker import SHARED_KEY

    at = app_test
    at.session_state[SHARED_KEY] = "clean_W"
    at.run()
    at.switch_page("views/evidence.py").run()
    assert not at.exception
    fired = next(m for m in at.metric if m.label == "Triggers fired")
    store = DataStore()
    sources = al.load_sources(store)
    assert fired.value == f"{ev.suspect_statistics(sources, sources.names['clean_W']).fired} / " \
                          f"{ev.suspect_statistics(sources, sources.names['clean_W']).n}"


def test_measured_null_is_the_committed_binned_data(store):
    null = ev.measured_null(store)
    record = store.read_json(null.path)["metrics"]["null"][ev.MEASURED_MODEL[0]]
    edges, counts = record["histogram"]["edges"], record["histogram"]["counts"]
    assert null.histogram == [{"z_low": lo, "z_high": hi, "count": c}
                              for lo, hi, c in zip(edges, edges[1:], counts) if c]
    assert sum(b["count"] for b in null.histogram) == record["count"] == null.count
    survival = record["survival"]
    assert null.exceedance == [{"z": t, "rate": f} for t, f in zip(survival["t"], survival["fraction_at_or_above"])
                               if f > 0]
    for e in null.exceedances:
        committed = record["exceedances_at_proven_thresholds"][e["alpha"]]
        assert (e["observed"], e["bound"], e["z"]) == (committed["exceedances"], committed["bound"],
                                                       committed["z_threshold"])
        assert e["observed"] <= e["bound"] or e["bound"] < 1


def test_only_loose_levels_are_checked_by_measurement(store):
    null = ev.measured_null(store)
    checked = [e["alpha"] for e in null.exceedances if e["bound"] > 1]
    assert checked == ["0.05", "0.01"]
    assert ev.measured_below_bound(null)
