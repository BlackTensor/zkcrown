"""P9.1: run the auditor engine on the committed provenance files.

    python experiments/p9_1_audit_engine.py

CPU, a few seconds. Reads ``provenance/record.json``,
``provenance/commitment.json`` and the dual `W*` weights (checked against its
file SHA-256). Reads nothing under ``secrets/``: no check is wired yet, so
none needs a secret.

Two audits, each must give the stated verdict or the script stops:

1. **Genuine.** Dual `W*` as suspect, the committed record and publication,
   the P6.2 public key as trusted key: ``record_valid`` true, the public key
   trusted, all five slots ``not_run`` with "not wired (P9.2)", no secrets
   used, no grade.
2. **Tampered record.** The same with the record's model label changed (not
   re-signed): ``record_valid`` false, and the audit still returns all five
   slots, unchanged.

Nothing here is a watermark or fingerprint result; that is P9.2.
"""

from __future__ import annotations

import copy
import time

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from p6_3_verify_provenance import OWNER_PUBLIC_KEY, PUBLICATION_SHA256, RECORD_SHA256, W_DUAL_SHA256, load_state
from src.auditor import SLOTS, audit
from src.auditor.verdict import NOT_WIRED
from src.crypto.provenance import RECORD_PATH, read_record
from src.crypto.publication import ARTIFACT_PATH, read_publication
from src.utils.results import git_info, write_result
from src.utils.seeding import DEFAULT_SEED


def expect(verdict, *, record_valid: bool) -> dict:
    d = verdict.to_dict()
    problems = []
    if d["record_valid"] is not record_valid:
        problems.append(f"record_valid {d['record_valid']}, expected {record_valid}")
    if [c["slot"] for c in d["checks"]] != list(SLOTS):
        problems.append("slots out of order")
    if any(c["status"] != "not_run" or c["reason"] != NOT_WIRED for c in d["checks"]):
        problems.append("a slot is not 'not_run, not wired (P9.2)'")
    if d["secrets_used"] or d["grade"] is not None:
        problems.append("secrets used or a grade given")
    if problems:
        raise SystemExit("unexpected verdict: " + "; ".join(problems))
    return d


def main() -> dict:
    start = time.perf_counter()
    git = git_info()
    record, record_hash = read_record(RECORD_PATH)
    publication, publication_hash = read_publication(ARTIFACT_PATH)
    if (record_hash, publication_hash) != (RECORD_SHA256, PUBLICATION_SHA256):
        raise SystemExit("committed provenance files differ from P6.2 / P5.4")
    suspect = load_state("dual_W_star")

    genuine = expect(audit(suspect, record, publication, trusted_public_key=OWNER_PUBLIC_KEY,
                           suspect_label="dual_W_star (P3.6)", suspect_file_sha256=W_DUAL_SHA256), record_valid=True)
    if genuine["record_status"]["public_key_trusted"] is not True:
        raise SystemExit("owner key not trusted on the genuine record")
    if genuine["inputs"]["suspect"]["parameter_count"] != 307_946:
        raise SystemExit("dual W* did not load into main_model")

    tampered_record = copy.deepcopy(record)
    tampered_record["model"]["label"] = "not the model that was signed"
    tampered = expect(audit(suspect, tampered_record, publication, trusted_public_key=OWNER_PUBLIC_KEY,
                            suspect_label="dual_W_star (P3.6)", suspect_file_sha256=W_DUAL_SHA256), record_valid=False)

    path = write_result(
        "p9.1_audit_engine", DEFAULT_SEED,
        params={"record": RECORD_PATH, "record_sha256": record_hash, "publication": ARTIFACT_PATH,
                "publication_sha256": publication_hash, "suspect_file_sha256": W_DUAL_SHA256,
                "trusted_public_key_hex": OWNER_PUBLIC_KEY,
                "trusted_public_key_source": "the P6.2 result record (same repository; a mechanism check)"},
        metrics={"genuine": genuine, "tampered_record": tampered},
        task="P9.1", git=git, duration_seconds=time.perf_counter() - start,
        notes="Auditor engine with only the record precondition (P6.3) run. All five check slots are "
              "not_run, 'not wired (P9.2)'. No secrets read. No watermark or fingerprint result.")
    print(f"genuine: record_valid={genuine['record_valid']}; tampered: record_valid={tampered['record_valid']}; "
          f"slots all not_run; wrote {path}")
    return {"genuine": genuine, "tampered": tampered, "path": str(path)}


if __name__ == "__main__":
    main()
