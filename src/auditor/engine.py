"""The auditor engine (P9.1): a suspect model plus the provenance record in, an `AuditVerdict` out.

    verdict = audit(suspect_state_dict, record, publication, trusted_public_key=...)

What runs in P9.1
-----------------
Only the record precondition: the P6.3 verifier on the record and its
publication (and the trusted key, when given). It is run without the suspect,
because comparing fingerprints is the ``fingerprint`` slot's job. Every one of
the five slots comes back ``not_run`` with the reason ``"not wired (P9.2)"``
unless a check is passed in ``checks``; the default registry is empty. An
invalid record does not stop the checks: ``record_valid`` is reported at the
top level and what it means is left to P9.3.

Check interface
---------------
A check is any object with:

- ``slot``: one of `SLOTS`;
- ``method``: the task it implements, e.g. ``"P2.8"``;
- ``requires_secrets``: a subset of ``("K", "S", "nonce")``;
- ``exception_is_rejection``: ``True`` for the ZK slots, where snarkjs and
  ezkl reject by raising (P7.8, P8.5), so an exception means ``failed``;
- ``run(context) -> CheckOutcome``.

The engine times it, gives it only the secrets it declared (asking for any
other raises inside the check), and passes its output through the no-secrets
guard before it reaches the verdict. A check never makes the audit raise:
its exception becomes ``error`` (or ``failed``, see above), and output the
guard refuses becomes ``error`` with ``guard_rejected`` set. A check that
needs secrets the caller did not supply is ``not_run``.

The engine raises only for malformed caller inputs (an unknown slot, a check
registered under the wrong slot, a malformed trusted key) and, as a last
line, if the finished verdict contains a supplied secret anywhere.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from src.auditor.secrets_guard import (
    MAX_REASON,
    SECRET_NAMES,
    OwnerSecrets,
    SecretLeak,
    check_statistic,
    check_text,
    find_needle,
)
from src.auditor.verdict import (
    AUDITOR_VERSION,
    NOT_WIRED,
    OUTCOME_STATUSES,
    SLOTS,
    AuditVerdict,
    CheckResult,
)
from src.crypto.fingerprint import fingerprint_state_dict
from src.crypto.provenance import record_sha256
from src.crypto.provenance_verifier import verify_provenance
from src.crypto.publication import artifact_sha256
from src.utils.results import _json_safe, git_info

MAIN_MODEL_ARCH = {"width": 32}


@dataclass(frozen=True)
class CheckOutcome:
    """What a check returns. Everything here passes through the guard."""

    status: str
    statistic: Mapping[str, Any] = field(default_factory=dict)
    p_value: float | None = None
    p_value_kind: str | None = None
    reason: str | None = None


class CheckContext:
    """What a check sees: the suspect, the public files, and only the secrets it declared."""

    def __init__(self, suspect_state: Mapping[str, Any], record: Any, publication: Any,
                 secrets: OwnerSecrets | None, allowed: tuple[str, ...]):
        self.suspect_state = suspect_state
        self.record = record
        self.publication = publication
        self._secrets = secrets
        self._allowed = allowed

    def __repr__(self) -> str:
        return f"CheckContext(allowed_secrets={self._allowed!r})"

    def secret(self, name: str) -> bytes:
        if name not in self._allowed or self._secrets is None:
            raise PermissionError(f"this check did not declare the secret {name!r}")
        return self._secrets.get(name)


class Check(Protocol):
    slot: str
    method: str
    requires_secrets: tuple[str, ...]
    exception_is_rejection: bool

    def run(self, context: CheckContext) -> CheckOutcome: ...


def _guarded(slot: str, method: str | None, outcome: Any, needles: tuple[str, ...]) -> CheckResult:
    """Turn a check's outcome into a `CheckResult`, or an ``error`` result naming the broken rule."""
    try:
        if not isinstance(outcome, CheckOutcome):
            raise SecretLeak("a check must return a CheckOutcome")
        statistic = check_statistic(outcome.statistic)
        if outcome.reason is not None:
            if not isinstance(outcome.reason, str):
                raise SecretLeak("reason must be a string")
            check_text("reason", outcome.reason, MAX_REASON)
        if needles and find_needle(json.dumps(statistic) + "\n" + (outcome.reason or ""), needles):
            raise SecretLeak("the output contains a supplied owner secret")
    except SecretLeak as rule:
        return CheckResult(slot=slot, status="error", method=method, guard_rejected=True,
                           reason=f"check output rejected by the no-secrets guard: {rule}")
    p_value = outcome.p_value
    if isinstance(p_value, int) and not isinstance(p_value, bool):
        p_value = float(p_value)
    try:
        return CheckResult(slot=slot, status=outcome.status, method=method, statistic=statistic,
                           p_value=p_value, p_value_kind=outcome.p_value_kind, reason=outcome.reason)
    except (ValueError, TypeError) as error:
        return CheckResult(slot=slot, status="error", method=method,
                           reason=f"check returned an invalid outcome: {error}")


def _run_check(check: Check, context_args: tuple, secrets: OwnerSecrets | None,
               needles: tuple[str, ...]) -> tuple[CheckResult, bool]:
    """Run one check. Returns its result and whether it ran (and so was given its secrets)."""
    slot, method = check.slot, getattr(check, "method", None)
    required = tuple(getattr(check, "requires_secrets", ()))
    if any(name not in SECRET_NAMES for name in required):
        return CheckResult(slot=slot, status="error", method=method,
                           reason=f"requires_secrets must be a subset of {SECRET_NAMES}"), False
    if required and secrets is None:
        return CheckResult(slot=slot, status="not_run", method=method,
                           reason=f"requires owner secrets ({', '.join(required)}), which were not supplied"), False
    start = time.perf_counter()
    try:
        outcome = check.run(CheckContext(*context_args, secrets, required))
        result = _guarded(slot, method, outcome, needles)
    except Exception as error:  # noqa: BLE001 -- a failing check is a verdict, never a crash
        message = f"{type(error).__name__}: {error}"
        try:
            check_text("exception message", message, MAX_REASON)
            if needles and find_needle(message, needles):
                raise SecretLeak("the exception message contains a supplied owner secret")
        except SecretLeak as rule:
            message = f"{type(error).__name__} (message withheld by the no-secrets guard: {rule})"
        if getattr(check, "exception_is_rejection", False):
            status = OUTCOME_STATUSES[slot][1]
            reason = f"rejected by raising: {message}"
        else:
            status, reason = "error", f"check raised {message}"
        result = CheckResult(slot=slot, status=status, method=method, reason=reason)
    elapsed = time.perf_counter() - start
    return CheckResult(**{**result.__dict__, "duration_seconds": round(elapsed, 6)}), True


def _suspect_inputs(state: Mapping[str, Any], label: str | None, file_sha256: str | None) -> dict[str, Any]:
    fingerprint = fingerprint_state_dict(state)
    inputs: dict[str, Any] = {
        "label": label,
        "file_sha256": file_sha256,
        "fingerprint": fingerprint.to_dict(),
        "state_entries": len(state),
        "loads_into_main_model": False,
        "parameter_count": None,
        "load_problem": None,
    }
    try:
        import torch

        from src.models.main_model import MainModel

        model = MainModel(**MAIN_MODEL_ARCH)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()}, strict=True)
        inputs["loads_into_main_model"] = True
        inputs["parameter_count"] = sum(p.numel() for p in model.parameters())
    except Exception as error:  # noqa: BLE001 -- recorded, not raised
        inputs["load_problem"] = type(error).__name__
    return inputs


def _sha_or_none(function, value) -> str | None:
    try:
        return function(value)
    except Exception:  # noqa: BLE001 -- a malformed file has no canonical hash
        return None


def audit(suspect_state: Mapping[str, Any], record: Any, publication: Any, *,
          trusted_public_key: bytes | str | None = None,
          checks: Mapping[str, Check] | None = None,
          owner_secrets: OwnerSecrets | None = None,
          suspect_label: str | None = None,
          suspect_file_sha256: str | None = None) -> AuditVerdict:
    """Audit one suspect against the provenance record. See the module docstring.

    Args:
        suspect_state: the suspect's state dict (tensors or arrays).
        record: the signed provenance record (``provenance/record.json``).
        publication: the commitment publication (``provenance/commitment.json``).
        trusted_public_key: the owner's Ed25519 key, obtained outside the record.
        checks: slot name to check. Missing slots are ``not_run``, "not wired (P9.2)".
        owner_secrets: needed only by checks that declare secrets; also arms the
            guard's value scan.
        suspect_label, suspect_file_sha256: descriptive, copied into ``inputs``.
    """
    start = time.perf_counter()
    if not isinstance(suspect_state, Mapping):
        raise TypeError("suspect_state must be a state dict")
    checks = dict(checks or {})
    for slot, check in checks.items():
        if slot not in SLOTS:
            raise ValueError(f"unknown slot {slot!r}; slots are {SLOTS}")
        if getattr(check, "slot", None) != slot:
            raise ValueError(f"the check registered under {slot!r} says its slot is {getattr(check, 'slot', None)!r}")
    if owner_secrets is not None and not isinstance(owner_secrets, OwnerSecrets):
        raise TypeError("owner_secrets must be an OwnerSecrets")
    needles = owner_secrets.needles() if owner_secrets is not None else ()

    record_status = verify_provenance(record, publication, trusted_public_key=trusted_public_key).to_dict()
    inputs = {
        "record_sha256": _sha_or_none(record_sha256, record),
        "publication_sha256": _sha_or_none(artifact_sha256, publication),
        "suspect": _suspect_inputs(suspect_state, suspect_label, suspect_file_sha256),
        "trusted_public_key_supplied": trusted_public_key is not None,
    }

    results, used = [], set()
    context_args = (suspect_state, record, publication)
    for slot in SLOTS:
        if slot not in checks:
            results.append(CheckResult(slot=slot, status="not_run", reason=NOT_WIRED))
            continue
        result, ran = _run_check(checks[slot], context_args, owner_secrets, needles)
        if ran:
            used.update(checks[slot].requires_secrets)
        results.append(result)

    verdict = AuditVerdict(
        inputs=_json_safe(inputs),
        record_status=record_status,
        checks=tuple(results),
        secrets_used=tuple(name for name in SECRET_NAMES if name in used),
        run={"auditor_version": AUDITOR_VERSION,
             "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "git": git_info(),
             "duration_seconds": round(time.perf_counter() - start, 6)},
    )
    if needles and find_needle(json.dumps(_json_safe(verdict.to_dict())), needles):
        raise SecretLeak("the built verdict contains a supplied owner secret; refusing to return it")
    return verdict
