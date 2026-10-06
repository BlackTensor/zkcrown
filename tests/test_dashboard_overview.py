"""P9.6: the landing page's headline figures, their sources, the pipeline links and the wording rules."""

from __future__ import annotations

import json
import re
import shutil

import pytest

from app.data import DataStore
from app.headlines import ORDER, SOURCE_PREFIXES, build_headlines
from app.overview import STAGES
from app.pages import PAGES
from src.utils.results import repo_root

APP = repo_root() / "app"
BANNED = ("proves ownership", "prove ownership", "proof of ownership", "stolen", "unremovable", "court",
          "legally", "immune to")
"""Section 7 wording, and the P9.6 rule against "stolen" as a verdict."""


def records(store: DataStore) -> dict:
    return {name: (store.latest(prefix), store.read_json(store.latest(prefix))) for name, prefix in
            SOURCE_PREFIXES.items()}


@pytest.fixture(scope="module")
def committed():
    store = DataStore()
    recs = records(store)
    return store, recs, {h.key: h for h in build_headlines(recs)}


def test_every_headline_is_built_and_names_a_manifest_file(committed):
    store, _, heads = committed
    assert list(heads) == list(ORDER)
    for h in heads.values():
        assert h.sources and all(s in store.files for s in h.sources), h.key
        assert h.tone in ("hold", "limit")


def test_headlines_equal_the_committed_values(committed):
    _, recs, heads = committed
    summary = recs["robustness"][1]["metrics"]["summary"]
    attacks, distill = summary["all_attacks"], summary["distill"]
    total = attacks["rows"]
    assert heads["both_detected"].value == f"{attacks['outcomes']['both detected']} of {total}"
    assert heads["neither_detected"].value == f"{attacks['outcomes']['neither detected']} of {total}"
    student = distill["highest_accuracy_neither_detected"]
    assert heads["distillation"].value == f"{student['test_accuracy']:.2%}"
    assert distill["note"] in heads["distillation"].detail
    table = recs["auditor"][1]["metrics"]["table"]
    unrelated = [r for r in table if r["class"].startswith("unrelated")]
    flagged = sum(r["technical_evidence_strength"] != "none" for r in unrelated)
    assert heads["unrelated"].value == f"{flagged} of {len(unrelated)}"
    blocks = [b for b in recs["timestamp"][1]["metrics"]["opentimestamps"]["chain_check"]["blocks"] if b["verified"]]
    assert heads["timestamp"].value == f"block {min(b['height'] for b in blocks):,}"
    control = next(r for r in recs["robustness"][1]["metrics"]["rows"] if r["family"] == "control")
    assert heads["owner_model"].value == f"{control['fired']} / {control['n_triggers']}"


def test_honest_figures_are_present_and_marked_as_limits(committed):
    _, _, heads = committed
    assert heads["neither_detected"].tone == "limit" and heads["distillation"].tone == "limit"
    assert "Distillation" in heads["neither_detected"].detail


def test_a_missing_record_drops_only_its_figures(committed):
    _, recs, _ = committed
    partial = dict(recs, robustness=None)
    keys = [h.key for h in build_headlines(partial)]
    assert keys == ["unrelated", "timestamp"]


def test_pipeline_stages_link_to_existing_pages():
    slugs = {p.slug for p in PAGES}
    assert [name for name, _, _ in STAGES] == ["Watermark", "Attack", "Commit", "Prove", "Audit"]
    assert all(slug in slugs for _, slug, _ in STAGES)


def test_no_banned_wording_in_the_app():
    hits = []
    for path in APP.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".css", ".toml", ".txt") and "__pycache__" not in path.parts:
            text = path.read_text(encoding="utf-8").lower()
            hits += [f"{path.name}: {w}" for w in BANNED if w in text]
    assert hits == []


def test_no_banned_wording_in_the_rendered_figures(committed):
    _, _, heads = committed
    text = " ".join(f"{h.label} {h.value} {h.detail}" for h in heads.values()).lower()
    assert not [w for w in BANNED if w in text]


def _app_test():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60)


def test_overview_renders_tiles_pipeline_and_sources(committed):
    _, recs, heads = committed
    at = _app_test().run()
    assert not at.exception and not at.error
    body = " ".join(h.proto.body for h in at.get("html"))
    assert body.count('class="zk-tile ') == len(heads)
    for _, slug, _ in STAGES:
        assert f'href="{slug}"' in body
    for h in heads.values():
        assert h.value in body
    assert [e.label for e in at.expander] == ["Where each figure comes from"]
    shown = at.dataframe[0].value
    assert set(shown["Committed file"]) == {path for path, _ in recs.values()}


def test_a_tampered_source_is_shown_and_its_figures_left_out(monkeypatch, tmp_path, committed):
    import app.ui as ui

    store, recs, _ = committed
    files = {}
    for path, _ in recs.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root() / path, target)
        files[path] = store.files[path]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "zk-crown/dashboard-manifest/v1", "files": files}), encoding="utf-8")
    tampered = recs["robustness"][0]
    (tmp_path / tampered).write_bytes((tmp_path / tampered).read_bytes() + b" ")
    monkeypatch.setattr(ui, "_cached_store", lambda: DataStore(repo_root=tmp_path, manifest_path=manifest))
    at = _app_test().run()
    errors = [e.value for e in at.error]
    assert any(tampered in e and "does not match the recorded" in e for e in errors), errors
    body = " ".join(h.proto.body for h in at.get("html"))
    assert body.count('class="zk-tile ') == 2
    assert not re.search(r"\bof 84\b", body)
