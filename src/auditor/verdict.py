"""The audit verdict object (P9.1), schema ``zk-crown/audit-verdict/v1``.

One `AuditVerdict` per suspect. Contents, in order:

1. **inputs**: SHA-256 of the record and of the publication; the suspect's
   label, file SHA-256 (when it came from a file), P5.1 fingerprint, state
   entry count, whether it loads strictly into ``main_model(width=32)`` and,
   if so, its parameter count; whether a trusted public key was supplied.
2. **record_valid** at the top level, and **record_status**: the P6.3
   `ProvenanceVerdict` dict, unchanged. It is run without the suspect, so the
   fingerprint comparison stays in its own slot. A failed record does not
   stop the audit.
3. **checks**: always five slots in this order: ``fingerprint``,
   ``behavioral``, ``weight``, ``commitment``, ``zk_proof``. Each is a
   `CheckResult`.
4. **secrets_used**: which of `K`, `S`, nonce a check that ran declared and
   was given. Empty when nothing secret was read (P9.6).
5. **grade**: always ``None`` here. Graded verdicts are P9.3; nothing in
   this object combines p-values or says "stolen".
6. **limitations**: fixed text attached to every verdict.
7. **run**: auditor version, UTC time, git snapshot, duration.

The object holds public values only; `src.auditor.secrets_guard` enforces
that on everything a check returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VERDICT_SCHEMA = "zk-crown/audit-verdict/v1"
AUDITOR_VERSION = "p9.2"
SLOTS = ("fingerprint", "behavioral", "weight", "commitment", "zk_proof")
OUTCOME_STATUSES = {
    "fingerprint": ("passed", "failed"),
    "behavioral": ("detected", "not_detected"),
    "weight": ("detected", "not_detected"),
    "commitment": ("passed", "failed"),
    "zk_proof": ("passed", "failed"),
}
COMMON_STATUSES = ("not_applicable", "not_run", "error")
REASON_REQUIRED = ("not_applicable", "not_run", "error")
P_VALUE_KINDS = ("exact", "upper_bound")
LEVELS = (0.05, 0.01, 1e-3, 1e-6, 1e-9)
NOT_WIRED = "not wired (P9.2)"
GRADE_NOTE = "Graded verdicts are P9.3. This verdict reports each check separately and combines nothing."
LIMITATIONS = (
    "Distillation into a fresh student removed both watermarks in every measured case (P4.7, P4.10); "
    "a distilled copy leaves no evidence for these checks.",
    "The weight test needs the owner's carrier layout. A physically channel-pruned or narrower suspect is "
    "not applicable unless re-aligned, which is not implemented (Icebox).",
    "The provenance record's own OpenTimestamps proof is pending; only the commitment publication has "
    "independent time evidence (Bitcoin block 969627, P5.5).",
    "This is a technical ownership verification demonstration, not legal evidence.",
)


def strongest_level_rejected(p_value: float | None) -> float | None:
    """The strictest of `LEVELS` that `p_value` rejects at. Reported only; not a verdict."""
    if p_value is None:
        return None
    rejected = [level for level in LEVELS if p_value <= level]
    return min(rejected) if rejected else None


@dataclass(frozen=True)
class CheckResult:
    """One slot of the verdict. ``statistic`` holds aggregates only."""

    slot: str
    status: str
    method: str | None = None
    statistic: dict[str, Any] = field(default_factory=dict)
    p_value: float | None = None
    p_value_kind: str | None = None
    reason: str | None = None
    duration_seconds: float | None = None
    guard_rejected: bool = False

    def __post_init__(self) -> None:
        if self.slot not in SLOTS:
            raise ValueError(f"unknown slot {self.slot!r}")
        if self.status not in OUTCOME_STATUSES[self.slot] + COMMON_STATUSES:
            raise ValueError(f"status {self.status!r} is not allowed for slot {self.slot!r}")
        if self.status in REASON_REQUIRED and not self.reason:
            raise ValueError(f"status {self.status!r} needs a reason")
        if self.p_value is not None:
            if not isinstance(self.p_value, float) or not 0.0 <= self.p_value <= 1.0:
                raise ValueError("p_value must be a float in [0, 1]")
            if self.p_value_kind not in P_VALUE_KINDS:
                raise ValueError(f"p_value_kind must be one of {P_VALUE_KINDS}")
        elif self.p_value_kind is not None:
            raise ValueError("p_value_kind without a p_value")

    @property
    def strongest_level_rejected(self) -> float | None:
        return strongest_level_rejected(self.p_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "status": self.status,
            "method": self.method,
            "statistic": dict(self.statistic),
            "p_value": self.p_value,
            "p_value_kind": self.p_value_kind,
            "strongest_level_rejected": self.strongest_level_rejected,
            "reason": self.reason,
            "duration_seconds": self.duration_seconds,
            "guard_rejected": self.guard_rejected,
        }


@dataclass(frozen=True)
class AuditVerdict:
    """The auditor's output for one suspect. See the module docstring."""

    inputs: dict[str, Any]
    record_status: dict[str, Any]
    checks: tuple[CheckResult, ...]
    secrets_used: tuple[str, ...]
    run: dict[str, Any]
    grade: None = None

    def __post_init__(self) -> None:
        if tuple(c.slot for c in self.checks) != SLOTS:
            raise ValueError(f"checks must be exactly the slots {SLOTS}, in order")
        if self.grade is not None:
            raise ValueError("grading is P9.3; grade must be None")

    @property
    def record_valid(self) -> bool:
        return bool(self.record_status.get("record_valid"))

    def check(self, slot: str) -> CheckResult:
        return self.checks[SLOTS.index(slot)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": VERDICT_SCHEMA,
            "record_valid": self.record_valid,
            "inputs": dict(self.inputs),
            "record_status": dict(self.record_status),
            "checks": [c.to_dict() for c in self.checks],
            "secrets_used": list(self.secrets_used),
            "grade": self.grade,
            "grade_note": GRADE_NOTE,
            "limitations": list(LIMITATIONS),
            "run": dict(self.run),
        }
