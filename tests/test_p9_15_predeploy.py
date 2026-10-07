"""P9.15: the pre-deploy rules catch what they are meant to catch."""

from __future__ import annotations

import sys
from dataclasses import replace

import pytest

from app.pages import PAGES
from src.utils.results import repo_root

ROOT = repo_root()
sys.path.insert(0, str(ROOT / "experiments"))
import p9_15_predeploy_check as pre  # noqa: E402


def test_only_the_gitkeep_placeholder_may_be_tracked_under_secrets():
    assert pre.check_secrets_dir(["secrets/.gitkeep", "app/x.py"], b"\xef\xbb\xbf\n")["passed"]
    assert not pre.check_secrets_dir(["secrets/.gitkeep", "secrets/K.bin"], b"")["passed"]
    assert not pre.check_secrets_dir(["secrets/.gitkeep"], b"not empty")["passed"]


@pytest.mark.parametrize("path", ["secrets/K.bin", "x/commitment_nonce.bin", "provenance_signing_key.bin",
                                  "data/trigger_bundle.npz", "home/secring.gpg", ".env", ".streamlit/secrets.toml"])
def test_secret_looking_file_names_are_caught(path):
    assert not pre.check_secret_file_names([path])["passed"]


def test_public_key_and_results_are_not_secret_names():
    assert pre.check_secret_file_names(["provenance/owner_signing_key.asc", "results/a.json"])["passed"]


@pytest.mark.parametrize("extra", ["results/p2.3_behavioral_wm_W_star.pt", "results/p0.5_clean_baseline_W.npz",
                                   "models/student.safetensors", "results/other.onnx"])
def test_any_model_weights_besides_the_bundle_fail(extra):
    result = pre.check_model_weights([pre.APPROVED_BUNDLE, extra])
    assert not result["passed"] and result["unexpected"] == [extra]


def test_the_bundle_alone_passes_and_older_files_need_a_decision():
    assert pre.check_model_weights([pre.APPROVED_BUNDLE, "results/a.json"])["passed"]
    older = next(iter(pre.COMMITTED_BEFORE_P9_15))
    result = pre.check_model_weights([pre.APPROVED_BUNDLE, older])
    assert result["passed"] and older in result["owner_decision_needed"]
    assert not pre.check_model_weights(["results/a.json"])["passed"]


def test_bundle_identity_refuses_the_behavioral_only_model():
    forbidden = {"behavioral_only_W_star": "p23", "clean_W": "w"}
    assert pre.check_bundle_identity("dual", "dual", forbidden)["passed"]
    assert not pre.check_bundle_identity("p23", "dual", forbidden)["passed"]
    assert not pre.check_bundle_identity("dual", "dual", {"behavioral_only_W_star": "dual"})["passed"]


def test_needle_scan_counts_planted_values_and_never_returns_them():
    key = bytes(range(32))
    needles = pre.needle_forms("K", key)
    files = {"a.md": b"text " + key.hex().encode(), "b.json": b'{"x": "' + str(int.from_bytes(key, "big")).encode(),
             "c.txt": b"nothing here"}
    hits = pre.scan(files, needles)
    assert hits == {"K hex": 1, "K decimal": 1}
    assert key.hex() not in repr(hits)


def test_page_modes_require_a_live_or_replayed_statement():
    header = "def header(spec): mode_line(spec)"
    assert pre.check_page_modes(PAGES, header, header)["passed"]
    blank = [replace(PAGES[0], mode="")] + list(PAGES[1:])
    assert pre.check_page_modes(blank, header, header)["missing_or_unclear"] == [PAGES[0].slug]
    assert not pre.check_page_modes(PAGES, "", header)["passed"]


def test_requirements_must_pin_streamlit_only():
    assert pre.check_requirements("# c\nstreamlit==1.65.0\n")["passed"]
    assert not pre.check_requirements("streamlit==1.65.0\ntorch==2.0\n")["passed"]
    assert not pre.check_requirements("streamlit\n")["passed"]
