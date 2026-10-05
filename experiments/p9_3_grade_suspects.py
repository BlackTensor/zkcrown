"""P9.3: graded verdicts for the P9.2 suspects, and a borderline survey over Phase 4.

    python experiments/p9_3_grade_suspects.py

CPU, under a second. Reads no secrets and no model: the grade is a pure
function of a verdict's check results, so it is computed here from the
committed P9.2 verdicts (the result file is checked by SHA-256). An audit run
now attaches the same grade, because `AuditVerdict.grade` calls the same
function.

1. **P9.2 suspects.** Each verdict is graded. The script stops unless clean
   `W` and the untrained model get "none" (no evidence), and the verbatim copy
   is the only exact copy.
2. **Borderline survey.** The same grading applied to the 85 committed Phase 4
   rows (from the P4.9 record), using their P2.8 p-values, fired counts and
   P3.7 bounds. It lists every row flagged borderline and how it was graded.
   Phase 4 rows carry no commitment or ZK checks, which never affect the
   suspect grade anyway.
"""

from __future__ import annotations

import glob
import hashlib
import json
import time
from collections import Counter

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.auditor.grading import grade, grade_verdict_dict
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

P9_2_RESULT = "results/p9.2_audit_suspects__seed1337__20261005T124654+0000.json"
K_STAR_1E6 = 29
"""P2.8 threshold at 1e-6 for N = 100."""


def phase4_checks(row: dict) -> dict:
    behavioral = {"status": "detected" if row["behavioral_detected"] else "not_detected", "p_value": row["behavioral_p"],
                  "p_value_kind": "exact", "statistic": {"fired": row["fired"], "k_star_at_alpha": K_STAR_1E6}}
    if row["weight_applicable"]:
        weight = {"status": "detected" if row["weight_detected"] else "not_detected", "p_value": row["weight_p_bound"],
                  "p_value_kind": "upper_bound", "statistic": {"z": row["weight_z"]}}
    else:
        weight = {"status": "not_applicable", "p_value": None, "reason": "owner carrier layout not present"}
    return {"behavioral": behavioral, "weight": weight}


def main() -> dict:
    started, git = time.perf_counter(), git_info()
    path = repo_root() / P9_2_RESULT
    p9_2_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    p9_2 = json.loads(path.read_text(encoding="utf-8"))

    suspects, problems = [], []
    for row in p9_2["metrics"]["table"]:
        name = row["suspect"]
        g = grade_verdict_dict(p9_2["metrics"]["verdicts"][name])
        s = g["suspect"]
        suspects.append({
            "suspect": name, "kind": row["kind"],
            "technical_evidence_strength": s["technical_evidence_strength"],
            "combined_p_value": s["combined_p_value"],
            "behavioral_tier": s["watermarks"]["behavioral"]["tier"], "behavioral_p": s["watermarks"]["behavioral"]["p_value"],
            "weight_tier": s["watermarks"]["weight"]["tier"], "weight_p": s["watermarks"]["weight"]["p_value"],
            "tests_assessed": s["tests_assessed"], "exact_copy": s["exact_copy"],
            "borderline": s["borderline"], "statement": s["statement"], "caveats": s["caveats"],
            "owner_evidence": g["owner_evidence"], "grade": g,
        })
    by = {s["suspect"]: s for s in suspects}
    for name in ("clean_W", "untrained_unrelated"):
        if by[name]["technical_evidence_strength"] != "none":
            problems.append(f"{name} graded {by[name]['technical_evidence_strength']}, expected none")
    if [s["suspect"] for s in suspects if s["exact_copy"]] != ["verbatim"]:
        problems.append("the verbatim copy is not the only exact copy")

    rows = json.loads(open(sorted(glob.glob(str(repo_root() / "results" / "p4.9_master_table__*.json")))[-1],
                           encoding="utf-8").read())["metrics"]["rows"]
    survey, borderline = Counter(), []
    for row in rows:
        g = grade(phase4_checks(row), {"record_valid": True})["suspect"]
        survey[g["technical_evidence_strength"]] += 1
        if g["borderline"]:
            borderline.append({"task": row["task"], "family": row["family"], "setting": row["setting"],
                               "fired": row["fired"], "weight_z": row["weight_z"],
                               "behavioral_tier": g["watermarks"]["behavioral"]["tier"],
                               "weight_tier": g["watermarks"]["weight"]["tier"],
                               "combined_p_value": g["combined_p_value"],
                               "technical_evidence_strength": g["technical_evidence_strength"],
                               "flags": g["borderline"]})
    if len(rows) != 85:
        problems.append(f"expected 85 Phase 4 rows, got {len(rows)}")
    if problems:
        raise SystemExit("P9.3 checks failed:\n  " + "\n  ".join(problems))

    for s in suspects:
        cp = "n/a" if s["combined_p_value"] is None else f"{s['combined_p_value']:.2g}"
        print(f"{s['suspect']:30s} {s['technical_evidence_strength']:12s} combined p {cp:>8}  "
              f"beh {s['behavioral_tier']:12s} wgt {s['weight_tier']:12s} borderline {len(s['borderline'])}")
    print("Phase 4 rows by grade:", dict(survey))
    for b in borderline:
        print(f"  borderline: {b['task']} {b['family']} {b['setting']}: {b['technical_evidence_strength']}; "
              + " | ".join(b["flags"]))

    out = write_result(
        "p9.3_graded_verdicts", DEFAULT_SEED, task="P9.3", git=git, duration_seconds=time.perf_counter() - started,
        params={"p9_2_result": P9_2_RESULT, "p9_2_sha256": p9_2_sha, "phase4_rows": "P4.9 master table record",
                "tiers": "p <= 1e-9 very strong, 1e-6 strong, 1e-3 moderate, 0.01 weak, 0.05 marginal, else none",
                "combination": "Bonferroni over two tests: min(1, 2 * min(p_behavioral, p_weight))",
                "borderline": "within half a decade of a level, or fired within 1 trigger of k* = 29 at 1e-6"},
        metrics={"suspects": suspects, "phase4_grade_counts": dict(survey), "phase4_borderline": borderline},
        notes="Grades computed from the committed P9.2 verdicts and the Phase 4 rows; no model or secret read. "
              "Technical evidence strength only; not legal evidence.")
    print(f"wrote {out}")
    return {"suspects": suspects, "survey": dict(survey), "borderline": borderline}


if __name__ == "__main__":
    main()
