"""The IP Auditor forensic verification engine (P9.1)."""

from src.auditor.engine import CheckContext, CheckOutcome, audit
from src.auditor.secrets_guard import OwnerSecrets, SecretLeak
from src.auditor.verdict import SLOTS, AuditVerdict, CheckResult

__all__ = ["CheckContext", "CheckOutcome", "audit", "OwnerSecrets", "SecretLeak", "SLOTS", "AuditVerdict",
           "CheckResult"]
