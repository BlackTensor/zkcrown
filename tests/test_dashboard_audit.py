"""P9.7: the live audit page. Replayed checks, the live commitment re-check, and the live grade.

The central test: the grade the page computes live, with the auditor's own
grading code, from the replayed P9.2 check results equals the committed P9.3
grade for every suspect, and the committed P9.4 grade for every suspect P9.4
also audited (matched by model fingerprint, not by name).
"""

from __future__ import annotations

import copy
import types

import pytest

from app import audit_logic as al
from app.data import DataStore

NO_EVIDENCE = "No evidence is not exoneration"


@pytest.fixture(scope="module")
def sources():
    return al.load_sources(DataStore())


def test_seven_suspects_each_resolve_to_one_recorded_verdict(sources):
    names = [sources.names[s.prefix] for s in al.SUSPECTS]
    assert len(names) == len(set(names)) == len(al.SUSPECTS)
    distill = sources.verdicts[sources.names["distill"]]
    weight = next(c for c in distill["checks"] if c["slot"] == "weight")
    assert weight["status"] != "not_applicable"


def test_grading_code_is_the_auditors_own():
    from src.auditor import grading

    module = al.grading_module()
    assert module.grade.__code__.co_code == grading.grade.__code__.co_code
    assert module.NOT_LEGAL == grading.NOT_LEGAL


def test_live_grade_equals_the_committed_p9_3_grade(sources):
    for s in al.SUSPECTS:
        name = sources.names[s.prefix]
        grade = al.live_grade(sources, name)
        assert grade == sources.committed_grades[name], name
        assert al.matches_committed(sources, name, grade)


def test_live_grade_equals_the_committed_p9_4_grade_where_audited(sources):
    store = DataStore()
    p9_4 = store.read_json(store.latest("results/p9.4_auditor_model_classes__"))["metrics"]["verdicts"]
    by_fingerprint = {v["inputs"]["suspect"]["fingerprint"]["sha256"]: v for v in p9_4.values()}
    matched = []
    for s in al.SUSPECTS:
        name = sources.names[s.prefix]
        fingerprint = sources.verdicts[name]["inputs"]["suspect"]["fingerprint"]["sha256"]
        if fingerprint in by_fingerprint:
            assert al.live_grade(sources, name) == by_fingerprint[fingerprint]["grade"], name
            matched.append(s.prefix)
    assert set(matched) == {s.prefix for s in al.SUSPECTS} - {"untrained"}


def test_none_grades_carry_the_no_exoneration_caveat(sources):
    seen = 0
    for s in al.SUSPECTS:
        grade = al.live_grade(sources, sources.names[s.prefix])["suspect"]
        if grade["technical_evidence_strength"] == "none":
            seen += 1
            assert any(c.startswith(NO_EVIDENCE) for c in grade["caveats"])
    assert seen >= 1


def test_owner_evidence_never_changes_the_grade(sources):
    name = sources.names["verbatim"]
    checks = al.checks_for_grading(sources, name)
    status = sources.verdicts[name]["record_status"]
    module = al.grading_module()
    base = module.grade(checks, status)["suspect"]
    tampered = dict(checks, commitment=dict(checks["commitment"], status="failed"),
                    zk_proof=dict(checks["zk_proof"], status="failed"))
    assert module.grade(tampered, status)["suspect"] == base


def test_a_changed_p_value_changes_the_live_grade(sources):
    name = sources.names["clean_W"]
    altered = copy.deepcopy(sources)
    for check in altered.verdicts[name]["checks"]:
        if check["slot"] == "behavioral":
            check["p_value"] = 1e-12
    grade = al.live_grade(altered, name)
    assert grade["suspect"]["technical_evidence_strength"] != "none"
    assert not al.matches_committed(altered, name, grade)


def test_step_labels(sources):
    for slot in al.REPLAYED_SLOTS:
        assert al.label_for(slot, sources) == f"replayed from {sources.p9_2_path}"
        assert al.why_replayed(slot)
    assert al.label_for("commitment", sources).startswith("live")
    assert al.why_replayed("commitment") is None
    assert set(al.REPLAYED_SLOTS) | {"commitment"} == set(al.SLOTS)


def test_live_commitment_passes_and_catches_a_mismatch(sources):
    assert al.live_commitment(sources) == {"status": "passed", "facts": {
        "record_c_equals_published_c": True, "decimal_and_hex_agree": True, "publication_hash_matches_record": True}}
    changed = copy.deepcopy(sources)
    changed.publication["commitment"]["decimal"] = "1"
    assert al.live_commitment(changed)["status"] == "failed"
    changed = copy.deepcopy(sources)
    changed.publication_sha256 = "0" * len(changed.publication_sha256)
    assert al.live_commitment(changed)["facts"]["publication_hash_matches_record"] is False


@pytest.fixture
def app_test(monkeypatch):
    import app.audit_page as page
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(page, "time", types.SimpleNamespace(sleep=lambda seconds: None))
    from app.data import REPO_ROOT

    return AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=60)


@pytest.mark.parametrize("suspect", [s.prefix for s in al.SUSPECTS])
def test_page_runs_every_suspect(app_test, suspect, sources):
    at = app_test
    at.session_state["zk_suspect"] = suspect
    at.run()
    at.switch_page("views/audit.py").run()
    assert not at.exception
    at.button(key=f"zk_run_{suspect}").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    labels = [x.label for x in at.main if type(x).__name__ == "Status"]
    assert len(labels) == len(al.SLOTS) + 1
    short = al.short_name(sources.p9_2_path)
    for slot, label in zip(al.SLOTS, labels):
        assert label.endswith("· live" if slot == "commitment" else f"· replayed from {short}"), label
    assert labels[-1].endswith("· live")
    body = " ".join(h.proto.body for h in at.get("html"))
    grade = sources.committed_grades[sources.names[suspect]]["suspect"]
    assert "zk-grade" in body and "Owner evidence" in body
    assert (NO_EVIDENCE in body) == (grade["technical_evidence_strength"] == "none")
    assert any("identical to the committed grade" in s.value for s in at.success)
    assert "stolen" not in body.lower()
