"""Headline figures for the landing page (P9.6), computed from committed result records.

Nothing here is a typed number: each figure is read from a verified file, and
each `Headline` names the files it came from. The set is deliberately mixed:
what the watermarks survive, what removes them, and how the auditor behaves
on models that carry no watermark.

Sources (resolved by name prefix through the data layer):

- ``results/p4.9_master_table__*``: the P4.9 aggregation of the 85 Phase 4
  attack rows (control plus the attack settings).
- ``results/p9.4_auditor_model_classes__*``: the P9.4 auditor run on the owner's
  model, the attacked variants and the unrelated models.
- ``results/p5.5_timestamp__*``: the P5.5 record whose OpenTimestamps proof was
  checked against the Bitcoin chain on two explorers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

SOURCE_PREFIXES = {
    "robustness": "results/p4.9_master_table__",
    "auditor": "results/p9.4_auditor_model_classes__",
    "timestamp": "results/p5.5_timestamp__",
}


@dataclass(frozen=True)
class Headline:
    key: str
    label: str
    value: str
    detail: str
    tone: str
    """``hold`` (what survives or works) or ``limit`` (what fails)."""
    sources: tuple[str, ...]


def _robustness(record: dict, path: str) -> list[Headline]:
    summary = record["metrics"]["summary"]
    control_row = next(r for r in record["metrics"]["rows"] if r["family"] == "control")
    control, attacks, distill = summary["control"], summary["all_attacks"], summary["distill"]
    total, outcomes = attacks["rows"], attacks["outcomes"]
    student = distill["highest_accuracy_neither_detected"]
    families = sorted(((f["outcomes"]["neither detected"], f["title"]) for key, f in summary.items()
                       if key not in ("all_attacks", "control") and f["outcomes"]["neither detected"]), reverse=True)
    neither_by_family = "; ".join(f"{title}: {n}" for n, title in families)
    only_trigger = outcomes["weight lost, behavioral detected"]
    return [
        Headline(
            "owner_model", "Owner's model, unattacked",
            f"{control_row['fired']} / {control_row['n_triggers']}",
            f"owner triggers give their keyed response; weight watermark z = {control['weight_z_max']:.2f}; "
            f"test accuracy {control['test_accuracy_max']:.2%}.",
            "hold", (path,)),
        Headline(
            "both_detected", "Both watermarks survive",
            f"{outcomes['both detected']} of {total}",
            f"attack settings where both tests still detect the watermark. In another "
            f"{outcomes['behavioral lost, weight detected']} only the weight watermark survives; only the trigger "
            f"watermark survives in {only_trigger or 'none'}.",
            "hold", (path,)),
        Headline(
            "neither_detected", "Neither watermark detected",
            f"{outcomes['neither detected']} of {total}",
            f"attack settings that leave no evidence for either test: {neither_by_family}.",
            "limit", (path,)),
        Headline(
            "distillation", "Distillation removes both",
            f"{student['test_accuracy']:.2%}",
            f"test accuracy kept by a student trained only on the model's outputs ({student['setting']}; "
            f"drop {student['drop_pp']:+.2f} pp), with neither watermark detected. Caveat: {distill['note']}.",
            "limit", (path,)),
    ]


def _auditor(record: dict, path: str) -> list[Headline]:
    rows = [r for r in record["metrics"]["table"] if r["class"].startswith("unrelated")]
    flagged = [r for r in rows if r["technical_evidence_strength"] != "none"]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    names = {"unrelated: clean W": ("clean W", "clean W"),
             "unrelated: fresh init": ("untrained init", "untrained inits"),
             "unrelated: zk_model": ("zk_model (MNIST)", "zk_model (MNIST)"),
             "unrelated: third-party": ("third-party CIFAR-10 model", "third-party CIFAR-10 models")}
    parts = ", ".join(names.get(cls, (cls, cls))[0] if n == 1 else f"{n} {names.get(cls, (cls, cls))[1]}"
                      for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    alpha = record["params"]["detection_alpha"]
    return [Headline(
        "unrelated", "Unrelated models flagged",
        f"{len(flagged)} of {len(rows)}",
        f"models that carry no watermark graded above “no evidence” ({parts}). Detection level {alpha} for each test.",
        "hold", (path,))]


def _timestamp(record: dict, path: str) -> list[Headline]:
    blocks = [b for b in record["metrics"]["opentimestamps"]["chain_check"]["blocks"] if b["verified"]]
    if not blocks:
        return []
    first = min(blocks, key=lambda b: b["height"])
    when = datetime.fromisoformat(first["header_time_utc"])
    return [Headline(
        "timestamp", "Commitment timestamped",
        f"block {first['height']:,}",
        f"the published commitment existed by this Bitcoin block (header time {when.day} {when:%b %Y, %H:%M} UTC). "
        f"Checked against {len(first['explorers'])} block explorers, not a full node.",
        "hold", (path,))]


BUILDERS = {"robustness": _robustness, "auditor": _auditor, "timestamp": _timestamp}
ORDER = ("owner_model", "both_detected", "neither_detected", "distillation", "unrelated", "timestamp")


def build_headlines(records: dict[str, tuple[str, Any] | None]) -> list[Headline]:
    """Headlines from the records that loaded, keyed as `SOURCE_PREFIXES` (a missing record gives none)."""
    out: list[Headline] = []
    for name, loaded in records.items():
        if loaded is not None:
            path, record = loaded
            out += BUILDERS[name](record, path)
    return sorted(out, key=lambda h: ORDER.index(h.key))
