"""Evidence-strength data (P9.9): each suspect's statistic against its null, from committed records only.

Behavioral (P2.8):

- the measured null: the fire counts of 1,000 public wrong keys on two models
  that carry no keyed response to those keys, committed by P2.8 as one count
  per possible value (so the histogram is the full set of 1,000 counts);
- the proven bound: ``Binomial(N, 1/(C-1))``, the worst case for any model
  independent of `K`, computed with the repo's own `significance` code;
- the exact p-value curve ``P(fired >= k)`` and the rejection thresholds k*
  that P2.8 recorded.

Weight (P3.7): the proven bound ``P(z >= t) <= exp(-t^2/2)``, computed with the
repo's own `weight_significance` code, with the thresholds P3.7 recorded. It is
an upper limit proven for every model independent of `K`, not a histogram of
measured values.

The suspect's statistic comes from its recorded P9.2 verdict (`audit_logic`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app import repo_code
from app.audit_logic import AuditSources
from app.data import DataStore

P2_8 = "results/p2.8_detection_test__"
P3_7 = "results/p3.7_weight_null__"
NULL_MODELS = {"W_star": "behavioral-watermarked model", "W_clean": "clean W"}
"""P2.8's two null-check models, both scored on keys that are not the owner's."""


@dataclass(frozen=True)
class BehavioralEvidence:
    path: str
    n: int
    num_classes: int
    keys: dict[str, int]
    histogram: list[dict]
    """{k, count, model} for every k with a non-zero count."""
    max_fired: dict[str, int]
    bound_expected: list[dict]
    """{k, expected}: wrong keys per count under the binomial bound, for the same number of keys."""
    tail: list[dict]
    """{k, p}: the exact P2.8 p-value of firing on at least k triggers."""
    thresholds: list[dict]
    """{alpha, k}: P2.8's rejection thresholds."""


@dataclass(frozen=True)
class WeightEvidence:
    path: str
    rows: int
    cap: float
    floor: float
    thresholds: list[dict]
    """{alpha, z}: P3.7's proven thresholds."""


def behavioral(store: DataStore) -> BehavioralEvidence:
    path = store.latest(P2_8)
    record = store.read_json(path)
    params, metrics = record["params"], record["metrics"]
    n, classes = params["n"], params["num_classes"]
    sig = repo_code.significance()
    tail = [sig.detection_p_value(k, n, classes) for k in range(n + 1)]
    pmf = [tail[k] - (tail[k + 1] if k < n else 0) for k in range(n + 1)]
    histogram, keys, max_fired = [], {}, {}
    for model, label in NULL_MODELS.items():
        check = metrics["null_check"][model]
        keys[label], max_fired[label] = check["keys"], check["fired_max"]
        histogram += [{"k": k, "count": c, "model": label} for k, c in enumerate(check["fired_histogram"]) if c]
    per_model = max(keys.values())
    return BehavioralEvidence(
        path=path, n=n, num_classes=classes, keys=keys, histogram=histogram, max_fired=max_fired,
        bound_expected=[{"k": k, "expected": float(p * per_model)} for k, p in enumerate(pmf)],
        tail=[{"k": k, "p": float(p)} for k, p in enumerate(tail)],
        thresholds=[{"alpha": alpha, "k": t["threshold"]} for alpha, t in metrics["thresholds"].items()],
    )


def weight(store: DataStore) -> WeightEvidence:
    path = store.latest(P3_7)
    metrics = store.read_json(path)["metrics"]
    rows = metrics["owner_key"]["W_dual"]["rows"]
    ws = repo_code.weight_significance()
    return WeightEvidence(
        path=path, rows=rows, cap=math.sqrt(rows), floor=ws.p_value_floor(rows),
        thresholds=[{"alpha": alpha, "z": t["z"]} for alpha, t in metrics["thresholds"].items()],
    )


def bound_curve(evidence: WeightEvidence, suspect_z: float | None) -> list[dict]:
    """{z, bound}: the proven bound on a grid from min(0, suspect z) to the cap sqrt(rows)."""
    ws = repo_code.weight_significance()
    low = min(0.0, suspect_z) if suspect_z is not None else 0.0
    steps = evidence.rows
    return [{"z": z, "bound": ws.p_value_bound(z)}
            for z in (low + (evidence.cap - low) * i / steps for i in range(steps + 1))]


@dataclass(frozen=True)
class SuspectStatistics:
    fired: int | None
    n: int | None
    behavioral_p: float | None
    behavioral_status: str
    z: float | None
    rows: int | None
    weight_p: float | None
    weight_status: str
    weight_reason: str | None


def suspect_statistics(sources: AuditSources, name: str) -> SuspectStatistics:
    checks = {c["slot"]: c for c in sources.verdicts[name]["checks"]}
    b, w = checks["behavioral"], checks["weight"]
    bs, wst = b.get("statistic") or {}, w.get("statistic") or {}
    return SuspectStatistics(
        fired=bs.get("fired"), n=bs.get("n"), behavioral_p=b.get("p_value"), behavioral_status=b["status"],
        z=wst.get("z"), rows=wst.get("rows"), weight_p=w.get("p_value"), weight_status=w["status"],
        weight_reason=w.get("reason"))


def null_summary(store: DataStore) -> dict:
    """P3.7's measured wrong-key check on the dual model, as aggregates (the per-key values were not committed)."""
    null = store.read_json(store.latest(P3_7))["metrics"]["null"]["W_dual"]
    return {"count": null["count"], "sd": null["z_sd"], "max": null["z_max"]}
