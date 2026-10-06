"""P9.8: the "Private data revealed" section of the live audit page.

Built from ``secrets_used`` in the recorded verdicts and from the committed P5.6
and P7.7 records, never from typed values. It must state what the checks used,
that the hosted app read no secrets, the guard's known limit, and the contrast
with an opening and with the Groth16 proof, and it must not claim that nothing
was revealed.
"""

from __future__ import annotations

import copy
import types

import pytest

from app import audit_logic as al
from app.data import ALLOWED_ROOTS, DataStore

FORBIDDEN = ("0 bytes revealed", "zero bytes revealed", "nothing was revealed", "nothing revealed",
             "no data revealed", "reveals nothing", "revealed nothing")


@pytest.fixture(scope="module")
def store():
    return DataStore()


@pytest.fixture(scope="module")
def sources(store):
    return al.load_sources(store)


def test_facts_come_from_the_committed_records(store, sources):
    for s in al.SUSPECTS:
        name = sources.names[s.prefix]
        facts = al.privacy_facts(store, sources, name)
        assert facts.secrets_used == tuple(sources.verdicts[name]["secrets_used"])
    opening = store.read_json(store.latest(al.P5_6))["metrics"]
    proof = store.read_json(store.latest(al.P7_7))["metrics"]
    facts = al.privacy_facts(store, sources, sources.names["verbatim"])
    assert facts.opening_bytes == opening["secret_bytes_revealed_by_an_opening"]
    assert facts.opening_disclosed == opening["opening_disclosed_to_anyone"]
    assert facts.proof_public_signals == len(proof["public_signals"])
    assert facts.proof_public_signals_exactly_c == proof["public_signals_exactly_C"]
    assert facts.proof_bytes == proof["proof_json_bytes"]


def test_secrets_used_follows_the_verdict(store, sources):
    name = sources.names["clean_W"]
    altered = copy.deepcopy(sources)
    altered.verdicts[name]["secrets_used"] = []
    assert al.privacy_facts(store, altered, name).secrets_used == ()


def test_files_read_are_committed_manifest_files_only(store, sources):
    facts = al.privacy_facts(store, sources, sources.names["verbatim"])
    for path in facts.files_read:
        assert path in store.files and path.split("/")[0] in ALLOWED_ROOTS
        assert store.verify(path).ok


def test_guard_limit_names_the_known_gaps():
    text = al.GUARD_LIMIT
    assert "one target class" in text and "one bit" in text and "encodings it does not search" in text


def test_no_claim_that_nothing_was_revealed():
    text = " ".join((al.GUARD_LIMIT, *al.VERDICT_EXCLUDES)).lower()
    assert not [f for f in FORBIDDEN if f in text]


@pytest.fixture
def app_test(monkeypatch):
    import app.audit_page as page
    from streamlit.testing.v1 import AppTest

    from app.data import REPO_ROOT

    monkeypatch.setattr(page, "time", types.SimpleNamespace(sleep=lambda seconds: None))
    return AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=60)


@pytest.mark.parametrize("suspect", ["verbatim", "distill", "untrained"])
def test_page_shows_the_section(app_test, suspect, store, sources):
    at = app_test
    at.session_state["zk_suspect"] = suspect
    at.run()
    at.switch_page("views/audit.py").run()
    at.button(key=f"zk_run_{suspect}").click().run()
    assert not at.exception and not at.error
    facts = al.privacy_facts(store, sources, sources.names[suspect])
    markdown = [m for m in at.markdown]
    used_line = next(m for m in markdown if m.value.startswith("**Owner secrets the checks used:"))
    assert f"used: {', '.join(facts.secrets_used)}." in used_line.value
    assert used_line.proto.help == al.GUARD_LIMIT
    assert any(m.value.startswith("**This hosted app read no secrets.**") for m in markdown)
    body = " ".join(h.proto.body for h in at.get("html"))
    assert "Private data revealed" in body
    assert f"{facts.opening_bytes} secret bytes" in body
    assert f"{facts.proof_public_signals} public signal" in body
    for line in al.VERDICT_EXCLUDES:
        assert line.replace("'", "&#x27;") in body or line in body
    text = (body + " ".join(m.value for m in markdown)).lower()
    assert not [f for f in FORBIDDEN if f in text]
    files = [e for e in at.expander if e.label == "Files this page read"]
    assert len(files) == 1
