"""Live audit logic (P9.7): replayed checks, the live commitment re-check, and the live grade.

What runs where, stated on the page for every step:

- **behavioral, weight, zk_proof: replayed** from the committed P9.2 verdicts.
  The watermark checks need the owner's key, which is never on this host; the
  proof checks need snarkjs and ezkl, which are not installed here.
- **fingerprint: live for the verbatim copy, replayed for the others** (P9.15).
  The one bundled model is the owner's dual W*, which is the verbatim-copy
  suspect, so its fingerprint is computed now from the bundled weights
  (`app.fingerprint_live`). The other suspects' weights are not bundled.
- **commitment: live.** Re-computed from ``provenance/record.json`` and
  ``provenance/commitment.json``: the record's ``C`` equals the published ``C``
  (decimal and hex agree), and the publication file's SHA-256 equals the hash
  the record names. The P9.2 commitment check also confirmed ``C`` is a valid
  BN254 field element and read the OpenTimestamps proof; those parts are shown
  as recorded, and labelled so.
- **grade: live.** `src/auditor/grading.py`, the auditor's own grading code,
  loaded by path (it needs only the standard library), applied to the replayed
  check results. The page compares it with the committed P9.3 grade.

Everything comes through the data layer, so each file is hash-checked first.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from app import repo_code
from app.data import DataIntegrityError, DataStore
from app.repo_code import GRADING_SOURCE

P9_2 = "results/p9.2_audit_suspects__"
P9_3 = "results/p9.3_graded_verdicts__"
RECORD = "provenance/record.json"
PUBLICATION = "provenance/commitment.json"
SLOTS = ("fingerprint", "behavioral", "weight", "commitment", "zk_proof")
REPLAYED_SLOTS = ("fingerprint", "behavioral", "weight", "zk_proof")
"""Slots replayed from the recorded audit, except where `is_live` says otherwise."""
LIVE_FINGERPRINT_SUSPECT = "verbatim"
"""The suspect whose weights are bundled (the owner's dual W*), so its fingerprint runs live (P9.15)."""


@dataclass(frozen=True)
class Suspect:
    prefix: str
    """Start of the suspect's name in the P9.2 verdicts; must pick out exactly one usable verdict."""
    title: str
    description: str


SUSPECTS: tuple[Suspect, ...] = (
    Suspect("verbatim", "Verbatim copy", "The owner's watermarked model, unchanged."),
    Suspect("int", "Quantized", "Conv and batch-norm fused, then quantized to integers; queried through its "
                                "integer runtime."),
    Suspect("prune_global", "Globally pruned", "Weights zeroed by global magnitude pruning, with no retraining."),
    Suspect("finetune", "Fine-tuned", "Retrained on the attacker's own images, at a rate that erases the "
                                      "trigger responses."),
    Suspect("distill", "Distilled student", "A fresh network of the same width, trained only on the model's "
                                            "outputs."),
    Suspect("clean_W", "Clean W", "The owner's own model trained without the key: no watermark."),
    Suspect("untrained", "Untrained model", "A freshly initialised network, unrelated to the owner."),
)

STEP_TITLES = {
    "fingerprint": "Model fingerprint",
    "behavioral": "Behavioral watermark",
    "weight": "Weight watermark",
    "commitment": "Commitment",
    "zk_proof": "Zero-knowledge proofs",
}
WHY_REPLAYED = {
    "fingerprint": "only the owner's own model is bundled with this demo, not this suspect's weights",
    "behavioral": "it needs the owner's secret key, which is never on this host",
    "weight": "it needs the owner's secret key, which is never on this host",
    "zk_proof": "snarkjs and ezkl are not installed on this host",
}


def grading_module() -> ModuleType:
    """`src/auditor/grading.py`, loaded on its own (the package import would pull in torch)."""
    return repo_code.grading()


def resolve(table: list[dict], verdicts: dict, prefix: str) -> str:
    """The one P9.2 suspect name starting with `prefix` whose weight test could be run.

    The second condition picks the width-32 distillation student over the width-16
    one, whose weight test is not applicable.
    """
    names = [r["suspect"] for r in table if r["suspect"].startswith(prefix)
             and _check(verdicts[r["suspect"]], "weight")["status"] != "not_applicable"]
    if len(names) != 1:
        raise DataIntegrityError(P9_2, f"expected exactly one suspect for {prefix!r}, found {names}")
    return names[0]


def _check(verdict: dict, slot: str) -> dict:
    return next(c for c in verdict["checks"] if c["slot"] == slot)


@dataclass
class AuditSources:
    p9_2_path: str
    p9_3_path: str
    verdicts: dict
    names: dict
    """Suspect prefix to P9.2 suspect name."""
    committed_grades: dict
    record: dict
    publication: dict
    publication_sha256: str


def load_sources(store: DataStore) -> AuditSources:
    p9_2_path, p9_3_path = store.latest(P9_2), store.latest(P9_3)
    p9_2, p9_3 = store.read_json(p9_2_path), store.read_json(p9_3_path)
    verdicts = p9_2["metrics"]["verdicts"]
    names = {s.prefix: resolve(p9_2["metrics"]["table"], verdicts, s.prefix) for s in SUSPECTS}
    grades = {s["suspect"]: s["grade"] for s in p9_3["metrics"]["suspects"]}
    publication_bytes = store.read_bytes(PUBLICATION)
    return AuditSources(p9_2_path, p9_3_path, verdicts, names, grades, store.read_json(RECORD),
                        store.read_json(PUBLICATION), hashlib.sha256(publication_bytes).hexdigest())


def live_commitment(sources: AuditSources) -> dict[str, Any]:
    """The live part of the commitment check, from the two provenance files."""
    record_c = sources.record["watermark_commitment"]["commitment"]
    published_c = sources.publication["commitment"]
    hex_agrees = int(record_c["decimal"]) == int.from_bytes(bytes.fromhex(record_c["hex"]), "big")
    facts = {
        "record_c_equals_published_c": record_c == published_c,
        "decimal_and_hex_agree": hex_agrees,
        "publication_hash_matches_record": sources.publication_sha256 ==
        sources.record["commitment_publication"]["sha256"],
    }
    return {"status": "passed" if all(facts.values()) else "failed", "facts": facts}


def checks_for_grading(sources: AuditSources, name: str) -> dict[str, dict]:
    """The check results the grade is computed from: the replayed P9.2 checks of this suspect."""
    return {c["slot"]: c for c in sources.verdicts[name]["checks"]}


def live_grade(sources: AuditSources, name: str) -> dict:
    verdict = sources.verdicts[name]
    return grading_module().grade(checks_for_grading(sources, name), verdict["record_status"])


def matches_committed(sources: AuditSources, name: str, grade: dict) -> bool:
    return sources.committed_grades.get(name) == grade


def is_live(slot: str, sources: AuditSources, name: str | None = None) -> bool:
    """Whether this step runs live on the host for this suspect."""
    if slot == "commitment":
        return True
    return slot == "fingerprint" and name is not None and name == sources.names.get(LIVE_FINGERPRINT_SUSPECT)


def audited_fingerprint(sources: AuditSources, name: str) -> str | None:
    """The suspect fingerprint the recorded audit measured, if it recorded one."""
    recorded = (sources.verdicts[name].get("inputs") or {}).get("suspect", {}).get("fingerprint") or {}
    return recorded.get("sha256")


def label_for(slot: str, sources: AuditSources, name: str | None = None) -> str:
    if slot == "commitment":
        return f"live · recomputed from {RECORD} and {PUBLICATION}"
    if is_live(slot, sources, name):
        from app.fingerprint_live import BUNDLE

        return f"live · computed from the bundled weights {BUNDLE}"
    return f"replayed from {sources.p9_2_path}"


def short_name(path: str) -> str:
    """A result file's task name, e.g. ``p9.2_audit_suspects`` for its dated file."""
    return path.rsplit("/", 1)[-1].split("__")[0]


def why_replayed(slot: str) -> str | None:
    return WHY_REPLAYED.get(slot)


# --- Private data revealed (P9.8) -----------------------------------------------------

P5_6 = "results/p5.6_opening_verifier__"
P7_7 = "results/p7.7_groth16_proof__"
GUARD_LIMIT = (
    "Known limit of the no-secrets guard (P9.1): it rejects arrays, byte strings, long hexadecimal strings and very "
    "large integers in check output, and searches for the owner's key, its halves, the signature and the nonce in "
    "hex, decimal and base64. It cannot catch a single small secret value, such as one target class or one bit of "
    "the signature S, and it does not catch fragments of a secret in encodings it does not search.")
VERDICT_EXCLUDES = (
    "Check outputs are flat maps of scalar statistics: no trigger images, target lists, projection matrices or "
    "weight arrays.",
    "No byte strings, long hexadecimal strings or very large integers, which is how a key or nonce would appear.",
    "Every output, and then the finished verdict as a whole, was searched for the key, its halves, the signature "
    "and the nonce in hex, decimal and base64 before it was written.",
)


@dataclass(frozen=True)
class PrivacyFacts:
    secrets_used: tuple[str, ...]
    """From the recorded verdict of this suspect."""
    files_read: tuple[str, ...]
    """Every file this page reads, all through the data layer."""
    p5_6_path: str
    opening_bytes: int
    opening_disclosed: bool
    p7_7_path: str
    proof_public_signals: int
    proof_public_signals_exactly_c: bool
    proof_bytes: int


def privacy_facts(store: DataStore, sources: AuditSources, name: str) -> PrivacyFacts:
    p5_6_path, p7_7_path = store.latest(P5_6), store.latest(P7_7)
    opening = store.read_json(p5_6_path)["metrics"]
    proof = store.read_json(p7_7_path)["metrics"]
    files = (sources.p9_2_path, sources.p9_3_path, RECORD, PUBLICATION, p5_6_path, p7_7_path)
    return PrivacyFacts(
        secrets_used=tuple(sources.verdicts[name]["secrets_used"]),
        files_read=files,
        p5_6_path=p5_6_path,
        opening_bytes=opening["secret_bytes_revealed_by_an_opening"],
        opening_disclosed=opening["opening_disclosed_to_anyone"],
        p7_7_path=p7_7_path,
        proof_public_signals=len(proof["public_signals"]),
        proof_public_signals_exactly_c=proof["public_signals_exactly_C"],
        proof_bytes=proof["proof_json_bytes"],
    )
