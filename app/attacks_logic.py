"""Attack-lab data (P9.10): the P4.9 rows with their P9.3 evidence tier, from committed records only.

- ``results/p4.9_master_table__*``: the 85 Phase 4 rows (the unattacked control
  plus the attack settings), the family titles and notes, and the outcome
  counts over the settings run.
- ``results/p2.8_detection_test__*``: k* at the detection level, used only for
  the grade's borderline flag.
- ``results/p9.4_auditor_model_classes__*``: the auditor's committed grade of
  every Phase 4 row. The tier shown is recomputed live with the repo's grading
  code from the row's p-values and checked against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app import repo_code
from app.data import DataStore

P4_9 = "results/p4.9_master_table__"
P2_8 = "results/p2.8_detection_test__"
P9_4 = "results/p9.4_auditor_model_classes__"
ALL = "all_attacks"
CONTROL = "control"
NEVER_BUILT = ("The weight figure assumes the removed channels are re-aligned to the owner's layout first. That "
               "alignment step was never built, so against a model with the channels physically deleted the "
               "weight test could not be run at all.")


@dataclass(frozen=True)
class Row:
    family: str
    family_title: str
    task: str
    setting: str
    test_accuracy: float
    drop_pp: float
    fired: int
    n_triggers: int
    behavioral_p: float
    behavioral_detected: bool
    weight_applicable: bool
    weight_z: float | None
    weight_p: float | None
    weight_detected: bool
    outcome: str
    tier: str
    combined_p: float | None
    note: str | None
    result_file: str

    @property
    def weight_status(self) -> str:
        if not self.weight_applicable:
            return "not applicable"
        return "detected" if self.weight_detected else "not detected"

    @property
    def behavioral_status(self) -> str:
        return "detected" if self.behavioral_detected else "not detected"

    @property
    def channel_pruned(self) -> bool:
        return self.note is not None and "re-alignment" in self.note


@dataclass(frozen=True)
class AttackLab:
    p4_9_path: str
    p9_4_path: str
    rows: list[Row]
    families: list[tuple[str, str]]
    """(id, title) in table order, control first."""
    outcomes: dict[str, int]
    """P4.9's outcome counts over the attack settings run (control excluded)."""
    settings_run: int
    grade_mismatches: list[str]
    """Result files whose live tier or combined p differs from the committed P9.4 grade (should be empty)."""


def _checks(row: dict, k_star: int) -> dict:
    behavioral = {"status": "detected" if row["behavioral_detected"] else "not_detected",
                  "p_value": row["behavioral_p"], "p_value_kind": "exact",
                  "statistic": {"fired": row["fired"], "k_star_at_alpha": k_star}}
    if row["weight_applicable"]:
        weight = {"status": "detected" if row["weight_detected"] else "not_detected",
                  "p_value": row["weight_p_bound"], "p_value_kind": "upper_bound",
                  "statistic": {"z": row["weight_z"]}}
    else:
        weight = {"status": "not_applicable", "p_value": None,
                  "reason": "the owner's carrier layout is not present"}
    return {"behavioral": behavioral, "weight": weight}


def load(store: DataStore) -> AttackLab:
    p4_9_path, p9_4_path = store.latest(P4_9), store.latest(P9_4)
    p4_9 = store.read_json(p4_9_path)["metrics"]
    p9_4 = store.read_json(p9_4_path)
    alpha = p9_4["params"]["detection_alpha"]
    k_star = store.read_json(store.latest(P2_8))["metrics"]["thresholds"][alpha]["threshold"]
    committed = {r["phase4_result_file"]: r for r in p9_4["metrics"]["table"] if r.get("phase4_result_file")}
    summary = p4_9["summary"]
    grading = repo_code.grading()
    rows, mismatches = [], []
    for r in p4_9["rows"]:
        grade = grading.grade(_checks(r, k_star), {"record_valid": True})["suspect"]
        tier, combined = grade["technical_evidence_strength"], grade["combined_p_value"]
        audited = committed.get(r["result_file"])
        if audited is None or audited["technical_evidence_strength"] != tier or not _close(
                audited["combined_p_value"], combined):
            mismatches.append(r["result_file"])
        family = summary[r["family"]]
        rows.append(Row(
            family=r["family"], family_title=family["title"], task=r["task"], setting=r["setting"],
            test_accuracy=r["test_accuracy"], drop_pp=r["drop_pp"], fired=r["fired"], n_triggers=r["n_triggers"],
            behavioral_p=r["behavioral_p"], behavioral_detected=r["behavioral_detected"],
            weight_applicable=r["weight_applicable"], weight_z=r["weight_z"], weight_p=r["weight_p_bound"],
            weight_detected=r["weight_detected"], outcome=r["outcome"], tier=tier, combined_p=combined,
            note=family.get("note"), result_file=r["result_file"]))
    families = [(key, f["title"]) for key, f in summary.items() if key != ALL]
    return AttackLab(p4_9_path=p4_9_path, p9_4_path=p9_4_path, rows=rows, families=families,
                     outcomes=dict(summary[ALL]["outcomes"]), settings_run=summary[ALL]["rows"],
                     grade_mismatches=mismatches)


def _close(a: float | None, b: float | None) -> bool:
    """Equal up to float rounding (``math.isclose`` with its default relative tolerance)."""
    if a is None or b is None:
        return a is b
    return math.isclose(a, b)


def filter_rows(rows: list[Row], families: list[str] | None, outcome: str | None,
                settings: list[str] | None) -> list[Row]:
    return [r for r in rows
            if (not families or r.family in families)
            and (outcome is None or r.outcome == outcome)
            and (not settings or f"{r.family_title} · {r.setting}" in settings)]


def outcome_counts(rows: list[Row], outcomes: list[str]) -> dict[str, int]:
    return {o: sum(r.outcome == o for r in rows if r.family != CONTROL) for o in outcomes}
