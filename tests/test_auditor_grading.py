"""Tests for P9.3: graded verdicts. Pure functions on check results; no keys involved."""

from __future__ import annotations

import json
import math
import random

import pytest

from src.auditor.grading import (
    NO_EVIDENCE_CAVEAT,
    NOT_ASSESSED,
    NOT_LEGAL,
    borderline_levels,
    grade,
    tier_of,
)

VALID = {"record_valid": True, "public_key_trusted": True}
BANNED = ("stolen", "steal", "guilty", "proves ownership", "proof of ownership", "infring", "pirat")


def wm(status="detected", p=None, kind="exact", **stat):
    return {"status": status, "p_value": p, "p_value_kind": kind, "statistic": stat, "reason": "r"}


def checks(beh=None, wgt=None, fp="failed", commitment="passed", zk="passed"):
    return {"fingerprint": {"status": fp}, "behavioral": beh or wm("not_run"), "weight": wgt or wm("not_run"),
            "commitment": {"status": commitment}, "zk_proof": {"status": zk}}


def suspect(g):
    return g["suspect"]


@pytest.mark.parametrize("p, tier", [(0.0, "very strong"), (1e-9, "very strong"), (1.0000001e-9, "strong"),
                                     (1e-6, "strong"), (1.0000001e-6, "moderate"), (1e-3, "moderate"),
                                     (0.01, "weak"), (0.05, "marginal"), (0.0500001, "none"), (1.0, "none"),
                                     (None, NOT_ASSESSED)])
def test_tier_boundaries_are_exact(p, tier):
    assert tier_of(p) == tier


def test_tiers_are_monotone_in_p():
    order = ["very strong", "strong", "moderate", "weak", "marginal", "none"]
    ps = sorted(10 ** random.Random(0).uniform(-12, 0) for _ in range(2000))
    ranks = [order.index(tier_of(p)) for p in ps]
    assert ranks == sorted(ranks)


def test_bonferroni_over_two_tests():
    g = suspect(grade(checks(wm("detected", 1e-8), wm("detected", 1e-4, "upper_bound")), VALID))
    assert g["combined_p_value"] == pytest.approx(2e-8) and g["technical_evidence_strength"] == "strong"
    assert g["watermarks"]["behavioral"]["tier"] == "strong" and g["watermarks"]["weight"]["tier"] == "moderate"
    assert "Bonferroni" in g["combination"] and "not pooled" in g["combination"]


def test_two_marginal_tests_do_not_add_up_to_evidence():
    g = suspect(grade(checks(wm("not_detected", 0.03), wm("not_detected", 0.03, "upper_bound")), VALID))
    assert g["watermarks"]["behavioral"]["tier"] == "marginal"
    assert g["combined_p_value"] == pytest.approx(0.06) and g["technical_evidence_strength"] == "none"


def test_no_evidence_is_not_exoneration():
    g = suspect(grade(checks(wm("not_detected", 0.99), wm("not_detected", 0.98, "upper_bound")), VALID))
    assert g["technical_evidence_strength"] == "none" and "no evidence" in g["statement"]
    assert NO_EVIDENCE_CAVEAT in g["caveats"]
    assert "distillation" in NO_EVIDENCE_CAVEAT.lower() and "channel pruning" in NO_EVIDENCE_CAVEAT
    assert "does not mean the model is independent" in NO_EVIDENCE_CAVEAT


def test_weight_not_applicable_lowers_confidence_and_is_not_a_negative():
    na = {"status": "not_applicable", "p_value": None, "reason": "no carrier layout"}
    g = suspect(grade(checks(wm("not_detected", 0.94), na), VALID))
    assert g["watermarks"]["weight"]["tier"] == NOT_ASSESSED and g["tests_assessed"] == 1
    assert any("Lower confidence" in c and "not a negative" in c for c in g["caveats"])
    g2 = suspect(grade(checks(wm("detected", 1e-30), na), VALID))
    assert g2["technical_evidence_strength"] == "very strong"
    assert any("Lower confidence" in c for c in g2["caveats"])


def test_nothing_assessed():
    g = suspect(grade(checks(), VALID))
    assert g["technical_evidence_strength"] == NOT_ASSESSED and g["combined_p_value"] is None


def test_owner_evidence_never_changes_the_suspect_grade():
    base = checks(wm("not_detected", 0.5), wm("not_detected", 0.7, "upper_bound"))
    reference = grade(base, VALID)["suspect"]
    for commitment in ("passed", "failed", "not_run", "error"):
        for zk in ("passed", "failed", "not_run", "error"):
            for record in (VALID, {"record_valid": False}):
                g = grade({**base, "commitment": {"status": commitment}, "zk_proof": {"status": zk}}, record)
                assert g["suspect"]["technical_evidence_strength"] == reference["technical_evidence_strength"]
                assert g["suspect"]["combined_p_value"] == reference["combined_p_value"]
                assert g["owner_evidence"]["commitment"] == commitment and g["owner_evidence"]["zk_proof"] == zk


def test_fingerprint_only_says_exact_copy_or_not():
    p = (wm("detected", 1e-20), wm("detected", 1e-20, "upper_bound"))
    failed = suspect(grade(checks(*p, fp="failed"), VALID))
    passed = suspect(grade(checks(*p, fp="passed"), VALID))
    assert failed["exact_copy"] is False and "Not an exact copy" in failed["fingerprint_statement"]
    assert passed["exact_copy"] is True
    assert failed["technical_evidence_strength"] == passed["technical_evidence_strength"]
    weak = suspect(grade(checks(wm("not_detected", 0.9), wm("not_detected", 0.9, "upper_bound"), fp="passed"), VALID))
    assert weak["technical_evidence_strength"] == "none" and weak["exact_copy"] is True


def test_an_invalid_record_adds_a_caveat():
    g = suspect(grade(checks(wm("detected", 1e-20)), {"record_valid": False}))
    assert any("did not verify" in c for c in g["caveats"])


def test_borderline_cases_are_flagged_but_not_rounded():
    # P4.5 LR 0.01, 60 epochs: 30 fired, p = 2.5e-7, one trigger above k* = 29.
    g = suspect(grade(checks(wm("detected", 2.5e-7, fired=30, k_star_at_alpha=29),
                             wm("detected", 9.9e-23, "upper_bound")), VALID))
    beh = g["watermarks"]["behavioral"]
    assert beh["tier"] == "strong" and beh["borderline"] and "30 fired, within 1 trigger of the 29" in beh["borderline"][0]
    # P4.5 LR 0.1, 60 epochs: z 6.49 vs z*(1e-9) = 6.438.
    w = suspect(grade(checks(wm("not_detected", 0.9), wm("detected", 7.4e-10, "upper_bound", z=6.49)), VALID))
    assert w["watermarks"]["weight"]["tier"] == "very strong" and w["watermarks"]["weight"]["borderline"]
    assert w["combined_p_value"] == pytest.approx(1.48e-9) and w["technical_evidence_strength"] == "strong"
    assert borderline_levels(1e-3 * 2) == [1e-3] and borderline_levels(0.2) == [] and borderline_levels(0.0) == []
    assert any(math.isclose(level, 1e-9) for level in borderline_levels(2.9e-9))


def test_wording_is_neutral_and_carries_the_not_legal_line():
    cases = [checks(wm("detected", 1e-50), wm("detected", 1e-20, "upper_bound"), fp="passed"),
             checks(wm("not_detected", 0.9), {"status": "not_applicable", "reason": "x"}),
             checks(wm("not_detected", 0.9), wm("not_detected", 0.9, "upper_bound")), checks()]
    for c in cases:
        for record in (VALID, {"record_valid": False}):
            g = grade(c, record)
            text = json.dumps(g).lower()
            assert not any(word in text for word in BANNED), text
            assert g["not_legal_evidence"] == NOT_LEGAL and "technical evidence strength" in text
