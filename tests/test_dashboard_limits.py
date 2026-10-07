"""P9.14: the honest-limits page states each limit with its measured fact, source file and evidence page."""

from __future__ import annotations

import html

import pytest

from app import limits_logic as logic
from app import provenance_logic, ui, zk_logic
from app.data import REPO_ROOT, DataIntegrityError, DataStore
from app.pages import PAGES

APP = REPO_ROOT / "app"
BANNED = ("stolen", "proves ownership", "prove ownership", "proof of ownership", "unremovable", "court", "legally",
          "immune to")
REQUIRED = {"distillation", "channel_finetune", "layout", "single_runs", "single_contributor", "hermez",
            "trigger_circuit", "record_pending", "stale_tag", "trusted_key", "not_zero_knowledge", "replayed",
            "not_legal"}


@pytest.fixture(scope="module")
def store():
    return DataStore()


@pytest.fixture(scope="module")
def limits(store):
    return {limit.key: limit for _, build in logic.GROUPS for limit in build(store)}


def test_every_required_limit_is_present(limits):
    assert set(limits) == REQUIRED
    assert {l.category for l in limits.values()} <= set(logic.CATEGORIES)


def test_each_limit_has_verified_sources_and_an_existing_page(store, limits):
    slugs = {p.slug for p in PAGES}
    for limit in limits.values():
        assert limit.sources and limit.page in slugs and limit.page != "limits"
        for path in limit.sources:
            assert store.verify(path).ok, path


def test_attack_facts_come_from_the_master_table(store, limits):
    summary = store.read_json(store.latest(logic.P4_9))["metrics"]["summary"]
    student = summary["distill"]["highest_accuracy_neither_detected"]
    assert f"{student['test_accuracy']:.2%}" in limits["distillation"].fact
    assert "not distinguishable from zero" in limits["distillation"].fact
    channel = summary["prune_finetune_channel"]
    assert f"{channel['outcomes']['neither detected']} of {channel['rows']}" in limits["channel_finetune"].fact
    assert "never built" in limits["layout"].fact
    assert "seed 1337" in limits["single_runs"].fact


def test_crypto_and_provenance_facts_reuse_their_panels(store, limits):
    z = zk_logic.load(store)
    assert f"{zk_logic.blocked(z).total:,}" in limits["trigger_circuit"].fact
    assert zk_logic.track_a(z).ptau_verify in limits["hermez"].fact
    p = provenance_logic.open_points(provenance_logic.load(store))
    assert p.record_proof_status in limits["record_pending"].fact
    assert p.trusted_key_source in limits["trusted_key"].fact
    assert "earlier, pending" in limits["stale_tag"].fact


def test_audit_facts_come_from_the_verdicts(store, limits):
    opening = store.read_json(store.latest(logic.P5_6))["metrics"]
    assert f"{opening['secret_bytes_revealed_by_an_opening']} secret bytes" in limits["not_zero_knowledge"].fact
    assert limits["not_legal"].fact == ui.NOT_LEGAL


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()


def test_page_renders_every_limit_with_its_link(limits):
    at = _app()
    at.switch_page("views/limits.py").run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    body = " ".join(h.proto.body for h in at.get("html"))
    for limit in limits.values():
        assert html.escape(limit.title) in body and html.escape(limit.fact) in body
    links = [e for e in at.get("page_link")]
    assert len(links) == len(limits)
    text = (body + " ".join(c.value for c in at.caption)).lower()
    assert not [w for w in BANNED if w in text]
    assert ui.NOT_LEGAL in body


def test_a_failing_source_hides_only_its_own_group(monkeypatch):
    class Broken(DataStore):
        def read_json(self, path):
            if path.startswith(logic.P4_9):
                raise DataIntegrityError(path, "SHA-256 does not match the recorded value (test)")
            return super().read_json(path)

    monkeypatch.setattr(ui, "_cached_store", lambda: Broken())
    at = _app()
    at.switch_page("views/limits.py").run()
    errors = " ".join(e.value for e in at.error)
    assert "Attack results" in errors and "p4.9_master_table" in errors
    body = " ".join(h.proto.body for h in at.get("html"))
    assert "Distillation removes both" not in body and "single contributor" in body


def test_page_source_reads_nothing_but_the_data_layer():
    for name in ("limits_page.py", "limits_logic.py"):
        source = (APP / name).read_text(encoding="utf-8")
        assert "open(" not in source and "read_text" not in source and "Path(" not in source
