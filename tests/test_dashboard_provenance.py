"""P9.12: the provenance and theft-timeline panel shows committed values only, and states its open points."""

from __future__ import annotations

import copy
import dataclasses

import pytest

from app import provenance_logic as logic
from app.data import REPO_ROOT, DataStore

APP = REPO_ROOT / "app"
BANNED = ("stolen", "proves ownership", "prove ownership", "unremovable", "court", "legally")


@pytest.fixture(scope="module")
def store():
    return DataStore()


@pytest.fixture(scope="module")
def sources(store):
    return logic.load(store)


def test_every_source_is_a_verified_manifest_file(store, sources):
    for path in sources.paths.values():
        assert path in store.files and store.verify(path).ok
    assert sources.publication_sha256 == store.recorded_sha256(logic.PUBLICATION)


def test_live_consistency_checks_pass_on_the_committed_files(sources):
    checks = logic.consistency_checks(sources)
    assert checks and all(c.ok for c in checks), [c.label for c in checks if not c.ok]


@pytest.mark.parametrize("change", ["c", "fingerprint", "publication_hash"])
def test_consistency_checks_catch_a_changed_record(sources, change):
    record = copy.deepcopy(sources.record)
    if change == "c":
        record["watermark_commitment"]["commitment"]["decimal"] = str(int(record["watermark_commitment"]["commitment"]["decimal"]) + 1)
    elif change == "fingerprint":
        record["model"]["fingerprint"]["sha256"] = "0" * 64
    else:
        record["commitment_publication"]["sha256"] = "0" * 64
    assert not all(c.ok for c in logic.consistency_checks(dataclasses.replace(sources, record=record)))


def test_open_points_are_read_from_the_records(sources):
    p = logic.open_points(sources)
    status = sources.timestamp_status["metrics"]["proofs"]["record"]
    assert p.record_proof_status == status["status"] == "pending"
    assert p.record_bitcoin_attestations == 0 and p.record_proof_is_the_described_one
    assert p.tag_holds_current_proof is False and p.tag_pushed is False
    assert "same repository" in p.trusted_key_source and p.all_runs_used_that_key
    other = dataclasses.replace(sources, record_proof_sha256="0" * 64)
    assert not logic.open_points(other).record_proof_is_the_described_one


def test_bitcoin_blocks_come_from_the_chain_check(sources):
    blocks = logic.blocks(sources)
    earliest = sources.timestamp["metrics"]["opentimestamps"]["chain_check"]["earliest_verified_block"]
    assert blocks[0].height == earliest["height"] == 969627
    assert blocks[0].block_hash == earliest["block_hash"] and all(b.verified for b in blocks)
    assert sum(b.calendars for b in blocks) == len(sources.timestamp["metrics"]["opentimestamps"]["bitcoin_attestations"])


def test_timeline_is_in_clock_order_with_the_backdated_claim_first(sources):
    events = logic.timeline(sources)
    assert len(events) == len(sources.theft["metrics"]["timeline"])
    assert [logic.utc(e.utc) for e in events] == sorted(logic.utc(e.utc) for e in events)
    assert events[0].party == "thief" and events[0].kind == "self"
    assert [e.kind for e in events].count("bitcoin") == 1


def test_claim_rows_mark_what_is_not_symmetric(sources):
    rows = logic.claim_rows(sources)
    asym = {r.aspect for r in rows if not r.symmetric}
    assert asym == {"Independent time", "Weight watermark on the model the owner published before the hand-over",
                    "Accepted under the owner's trusted key"}
    cc = sources.theft["metrics"]["counter_claim"]
    text = " ".join(f"{r.owner} {r.thief}" for r in rows)
    assert f"{cc['thief_weight_z_on_owner_published_model']:.2f}" in text
    assert f"{cc['thief_weight_z_on_shipped_model']:.2f}" in text


def test_audit_rows_equal_the_recorded_suspects(sources):
    rows = logic.audit_rows(sources)
    suspects = list(sources.theft["metrics"]["suspects"].values())
    assert [r.fired for r in rows] == [s["triggers_fired"] for s in suspects]
    assert [r.test_accuracy for r in rows] == [s["test_accuracy"] for s in suspects]
    assert sum(r.matches_record for r in rows) == 1
    assert not [r.label for r in rows if any(w in r.label.lower() for w in BANNED)]


def test_page_source_reads_nothing_but_the_data_layer():
    for name in ("provenance_page.py", "provenance_logic.py"):
        source = (APP / name).read_text(encoding="utf-8")
        assert "open(" not in source and "read_text" not in source and "Path(" not in source


def test_page_renders_with_open_points_and_committed_values(sources):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    at.switch_page("views/provenance.py").run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    warning = " ".join(w.value for w in at.warning)
    assert "still pending" in warning and "holds the older, pending proof" in warning
    assert "came from this repository" in warning and "does not show the key belongs to the owner" in warning
    body = " ".join(h.proto.body for h in at.get("html"))
    for value in (sources.publication["commitment"]["decimal"], sources.publication["model_fingerprint"]["sha256"],
                  sources.timestamp["metrics"]["opentimestamps"]["chain_check"]["earliest_verified_block"]["block_hash"],
                  sources.record["signature"]["hex"], sources.timestamp["metrics"]["gpg_tag"]["fingerprint"]):
        assert value in body
    assert "backdated claim" in body and "differs" in body
    text = (body + warning + " ".join(c.value for c in at.caption)).lower()
    assert not [w for w in BANNED if w in text]
