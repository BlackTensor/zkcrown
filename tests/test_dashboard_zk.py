"""P9.13: the zero-knowledge panel shows committed figures only, labels timings, and states what is not proved."""

from __future__ import annotations

import dataclasses

import pytest

from app import zk_logic as logic
from app.data import REPO_ROOT, DataStore

APP = REPO_ROOT / "app"
BANNED = ("stolen", "proves ownership", "prove ownership", "proof of ownership", "unremovable", "court", "legally")


@pytest.fixture(scope="module")
def store():
    return DataStore()


@pytest.fixture(scope="module")
def sources(store):
    return logic.load(store)


def test_every_source_is_a_verified_manifest_file(store, sources):
    for path in sources.paths.values():
        assert path in store.files and store.verify(path).ok


def test_track_a_measured_from_committed_files_equals_the_records(sources):
    a = logic.track_a(sources)
    proof = sources.records["proof"]["metrics"]
    assert a.proof_bytes == a.proof_bytes_recorded == proof["proof_json_bytes"]
    assert a.public_signals == proof["public_signals"] and len(a.public_signals) == 1
    assert a.public_signal_is_published_c
    assert a.contributors == 1 and "not completed" in a.ptau_verify
    assert (a.negatives_tried, a.negatives_accepted) == (206, 0) and a.crashes == 1


def test_a_public_signal_other_than_c_is_flagged(sources):
    other = dataclasses.replace(sources, groth16_public=[str(int(sources.groth16_public[0]) + 1)])
    assert not logic.track_a(other).public_signal_is_published_c


def test_public_signals_compare_by_value():
    assert logic.field_element("0042") == logic.field_element("0x2a") == 42


def test_track_b_measured_from_committed_files_equals_the_records(sources):
    b = logic.track_b(sources)
    prove = sources.records["ezkl_prove"]["metrics"]
    assert b.proof_bytes == b.proof_bytes_recorded == prove["proof_bytes"]
    assert b.public_instances == prove["sanity_check"]["public_instance_count"]
    assert (b.negatives_tried, b.negatives_accepted, b.negatives_error) == (42, 0, 42)
    assert b.fidelity_agree == b.fidelity_n == 10000
    assert b.verify_median == sources.records["ezkl_verify"]["metrics"]["verify_seconds_median"]


def test_family_tables_add_up(sources):
    a, b = logic.track_a(sources), logic.track_b(sources)
    for families, tried in ((logic.track_a_families(sources), a.negatives_tried),
                            (logic.track_b_families(sources), b.negatives_tried)):
        assert sum(f.tried for f in families) == tried and not any(f.accepted for f in families)


def test_blocked_estimate_is_the_sum_of_its_parts(sources):
    x = logic.blocked(sources)
    assert x.total == sources.records["sizing"]["metrics"]["total_estimate"] == sum(c for _, c in x.parts)
    assert x.times_over > 1 and x.single_block > x.ptau_capacity and not x.fits_largest_hermez


def test_page_renders_with_limits_labels_and_committed_values(sources):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    at.switch_page("views/zk.py").run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    warning = " ".join(w.value for w in at.warning)
    for phrase in ("nothing about any model", "says nothing about watermarks", "single contributor",
                   "powers-of-tau check did not finish", "blocked"):
        assert phrase in warning
    assert "A verifier error counts as a rejection" in " ".join(i.value for i in at.info)
    body = " ".join(h.proto.body for h in at.get("html"))
    a, b, x = logic.track_a(sources), logic.track_b(sources), logic.blocked(sources)
    for value in (f"{a.proof_bytes:,} bytes", f"{a.negatives_accepted} of {a.negatives_tried:,}",
                  f"{b.proof_bytes:,} bytes", f"{b.negatives_accepted} of {b.negatives_tried}",
                  f"{b.fidelity_agree:,} / {b.fidelity_n:,}", f"{x.total:,}"):
        assert value in body
    assert body.count(logic.MACHINE) >= 2
    costs = [d for d in at.dataframe if any(logic.MACHINE in str(c) for c in d.value.columns)]
    assert costs, "the cost table must label its columns as measured on this machine"
    text = (body + warning).lower()
    assert not [w for w in BANNED if w in text]


def test_page_source_reads_nothing_but_the_data_layer():
    for name in ("zk_page.py", "zk_logic.py"):
        source = (APP / name).read_text(encoding="utf-8")
        assert "open(" not in source and "read_text" not in source and "Path(" not in source
        assert "subprocess" not in source and "import ezkl" not in source
