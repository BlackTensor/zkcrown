"""Graded verdicts (P9.3): technical evidence strength from the measured test statistics.

The grade is a pure function of the five check results and the record status.
Nothing in it is decided by hand: change a p-value and the grade follows.

Per watermark
-------------
Each watermark is graded from its own p-value: the exact P2.8 p-value for the
behavioral test, the P3.7 upper bound for the weight test. The tier is the
strictest of the standard levels the p-value rejects at (``p <= level``):

    p <= 1e-9  very strong      p <= 1e-3  moderate      p <= 0.05  marginal
    p <= 1e-6  strong           p <= 0.01  weak          otherwise  none

The comparison is exact and never rounded. These levels are the ones P2.8
(k* = 17 / 20 / 23 / 29 / 35 of 100 triggers) and P3.7 (z* = 2.448 / 3.035 /
3.717 / 5.257 / 6.438) turned into thresholds; 1e-6 is the level Phase 4
called "detected". A test with status ``not_run``, ``not_applicable`` or
``error`` is not assessed and contributes nothing.

Combining the two
-----------------
The suspect's grade uses **Bonferroni over the two pre-declared watermark
tests**: ``p_combined = min(1, 2 * min(p_behavioral, p_weight))``, using only
the tests that were assessed but always multiplying by 2, so the procedure is
the same for every suspect. This is a valid level-alpha test of "the suspect
is independent of `K`" under any dependence between the two tests, and the
weight bound being conservative keeps it valid. The two p-values are not
pooled (no Fisher or Stouffer combination): that would need the tests to be
independent under the null, which rests on the trigger and projection
streams being independent PRF outputs, an argument not checked here. The cost
is some power when both tests are weak at once.

Borderline cases
----------------
A p-value within a factor of sqrt(10) (half a decade) of one of the levels is
flagged as borderline, with the level named. Because the behavioral count is
discrete, a fired count within one trigger of k* at 1e-6 (29 of 100) is
flagged too, whatever its p-value. The grade still uses the exact comparison
and is never rounded either way.

What the grade is not
---------------------
- The commitment and ZK checks describe the owner's published evidence, not
  the suspect. They are reported in ``owner_evidence`` and never change the
  suspect's grade.
- A failed fingerprint check means only "not an exact copy". A passed one is
  reported as an exact copy, separately from the statistical grade.
- "No evidence" is not exoneration: distillation and channel pruning followed
  by fine-tuning removed both watermarks in Phase 4 and leave none.
- A weight test that is not applicable lowers confidence; it is not a
  negative result.
- This is a technical ownership verification demonstration, not legal
  evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

GRADE_SCHEMA = "zk-crown/evidence-grade/v1"
TIERS = ((1e-9, "very strong"), (1e-6, "strong"), (1e-3, "moderate"), (0.01, "weak"), (0.05, "marginal"))
NONE, NOT_ASSESSED = "none", "not assessed"
WATERMARK_SLOTS = ("behavioral", "weight")
BORDERLINE_DECADES = 0.5
BORDERLINE_TRIGGERS = 1
NOT_LEGAL = "This is a technical ownership verification demonstration, not legal evidence."
NO_EVIDENCE_CAVEAT = (
    "No evidence is not exoneration, and it does not mean the model is independent of the owner's. Distillation "
    "into a fresh student and channel pruning followed by fine-tuning both removed both watermarks in Phase 4 "
    "(P4.6, P4.7) and leave no evidence for these checks.")
COMBINATION = ("Bonferroni over the two pre-declared watermark tests: p_combined = min(1, 2 * min(p_behavioral, "
               "p_weight)) over the tests assessed. Valid under any dependence between them; the p-values are not "
               "pooled.")


def tier_of(p_value: float | None) -> str:
    """The strictest tier whose level `p_value` is at or below; ``none`` above 0.05."""
    if p_value is None:
        return NOT_ASSESSED
    for level, name in TIERS:
        if p_value <= level:
            return name
    return NONE


def strongest_level(p_value: float | None) -> float | None:
    if p_value is None:
        return None
    rejected = [level for level, _ in TIERS if p_value <= level]
    return min(rejected) if rejected else None


def borderline_levels(p_value: float | None) -> list[float]:
    """Levels within half a decade of `p_value` (p = 0 is never borderline)."""
    if p_value is None or p_value <= 0:
        return []
    return [level for level, _ in TIERS if abs(math.log10(p_value) - math.log10(level)) < BORDERLINE_DECADES]


def _watermark(slot: str, check: Mapping[str, Any]) -> dict[str, Any]:
    status = check.get("status")
    has_p = isinstance(check.get("p_value"), (int, float))
    out: dict[str, Any] = {"status": status, "assessed": status in ("detected", "not_detected") and has_p,
                           "p_value": None, "p_value_kind": None, "strongest_level_rejected": None,
                           "tier": NOT_ASSESSED, "borderline": []}
    if not out["assessed"]:
        if status in ("detected", "not_detected"):
            out["note"] = f"not assessed: the check reported {status} without a p-value"
        else:
            out["note"] = f"not assessed ({status}): {check.get('reason') or 'no reason given'}"
        return out
    p = check.get("p_value")
    out.update({"p_value": p, "p_value_kind": check.get("p_value_kind"), "strongest_level_rejected": strongest_level(p),
                "tier": tier_of(p)})
    stat = check.get("statistic") or {}
    for level in borderline_levels(p):
        text = f"p = {p:.3g} is within a factor of 3.2 of the {level:g} level; the grade uses the exact comparison"
        if slot == "weight" and "z" in stat:
            text += f" (z = {stat['z']:.3f})"
        out["borderline"].append(text)
    if slot == "behavioral" and isinstance(stat.get("fired"), int) and isinstance(stat.get("k_star_at_alpha"), int):
        fired, k_star = stat["fired"], stat["k_star_at_alpha"]
        if abs(fired - k_star) <= BORDERLINE_TRIGGERS:
            out["borderline"].append(f"{fired} fired, within {BORDERLINE_TRIGGERS} trigger of the {k_star} the 1e-6 "
                                     "level needs; one trigger more or fewer changes the tier. The grade uses the "
                                     "exact count")
    if out["p_value_kind"] == "upper_bound":
        out["note"] = "the p-value is an upper bound, so this tier is conservative"
    return out


def grade(checks: Mapping[str, Mapping[str, Any]], record_status: Mapping[str, Any]) -> dict[str, Any]:
    """Grade one verdict, given its checks by slot (dict form) and its record status."""
    marks = {slot: _watermark(slot, checks.get(slot, {})) for slot in WATERMARK_SLOTS}
    assessed = [slot for slot in WATERMARK_SLOTS if marks[slot]["assessed"]]
    combined = min(1.0, 2.0 * min(marks[s]["p_value"] for s in assessed)) if assessed else None
    tier = tier_of(combined) if assessed else NOT_ASSESSED
    caveats: list[str] = []

    fingerprint = checks.get("fingerprint", {}).get("status")
    exact_copy = {"passed": True, "failed": False}.get(fingerprint)
    fingerprint_statement = {
        True: "Exact copy: the suspect is bit for bit the model the owner's record names.",
        False: "Not an exact copy of the model the record names. That is all a failed fingerprint means.",
    }.get(exact_copy, f"Fingerprint not assessed ({fingerprint}).")

    detected = [s for s in assessed if marks[s]["tier"] != NONE]
    if tier == NOT_ASSESSED:
        statement = ("Technical evidence strength: not assessed. Neither watermark test could be run on this "
                     "suspect, so this verdict carries no watermark evidence either way.")
    elif tier == NONE:
        statement = (f"Technical evidence strength: no evidence. Neither assessed watermark test rejects "
                     f"independence from the owner's key at 0.05 (combined p = {combined:.3g}).")
        caveats.append(NO_EVIDENCE_CAVEAT)
    else:
        which = " and ".join(detected)
        statement = (f"Technical evidence strength: {tier}. The suspect responds to the owner's keyed {which} "
                     f"test{'s' if len(detected) > 1 else ''} beyond what a model independent of the owner's key "
                     f"would, combined p = {combined:.3g} (Bonferroni over two tests).")
        missing = [s for s in assessed if marks[s]["tier"] == NONE]
        if missing:
            statement += f" The {missing[0]} test finds no evidence; the grade rests on one test."
    if not marks["weight"]["assessed"] and marks["weight"]["status"] == "not_applicable":
        caveats.append("Lower confidence: the weight test could not be run because the suspect lacks the owner's "
                       "carrier layout (a different architecture or width, or physically removed channels). That is "
                       "not a negative result; only the behavioral test was assessed.")
    elif len(assessed) < 2:
        caveats.append("Lower confidence: only one of the two watermark tests was assessed.")
    if not record_status.get("record_valid"):
        caveats.append("The provenance record did not verify, so it does not support the assumption behind the "
                       "p-values that the key and triggers were fixed before this suspect was seen.")
    borderline = [f"{s}: {b}" for s in WATERMARK_SLOTS for b in marks[s]["borderline"]]
    for level in borderline_levels(combined):
        borderline.append(f"combined: p = {combined:.3g} is within a factor of 3.2 of the {level:g} level; "
                          "the grade uses the exact comparison")

    owner_checks = {s: checks.get(s, {}).get("status") for s in ("commitment", "zk_proof")}
    return {
        "scheme": GRADE_SCHEMA,
        "suspect": {
            "technical_evidence_strength": tier,
            "combined_p_value": combined,
            "combined_strongest_level_rejected": strongest_level(combined),
            "combination": COMBINATION,
            "tests_assessed": len(assessed),
            "watermarks": marks,
            "exact_copy": exact_copy,
            "fingerprint_statement": fingerprint_statement,
            "borderline": borderline,
            "statement": statement,
            "caveats": caveats,
        },
        "owner_evidence": {
            "record_valid": bool(record_status.get("record_valid")),
            "public_key_trusted": record_status.get("public_key_trusted"),
            "commitment": owner_checks["commitment"],
            "zk_proof": owner_checks["zk_proof"],
            "statement": ("These describe the owner's published evidence (the signed record, the commitment and its "
                          "timestamp, the committed proofs), not the suspect. They never change the suspect's grade."),
        },
        "not_legal_evidence": NOT_LEGAL,
    }


def grade_verdict_dict(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Grade a verdict in its ``to_dict`` form, for example one read back from a result file."""
    return grade({c["slot"]: c for c in verdict["checks"]}, verdict["record_status"])
