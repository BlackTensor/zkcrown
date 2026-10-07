"""Provenance and theft-timeline panel (P9.12): the facts, built from committed files only.

Everything here is read through the data layer and nothing is measured. The
published commitment and provenance record come from ``provenance/``. The
Bitcoin check, the signatures and the theft simulation come from the committed
P5.5, P6.2, P6.3 and P6.4 records.

A few consistency checks are recomputed live from those bytes: the record names
the publication's SHA-256, the record's ``C`` equals the published ``C``, the
committed proof files are the ones the records describe, and every recorded run
used the same public key. The Ed25519 and GPG signature checks need libraries
the hosted app does not have, so they are replayed from the records and
labelled that way.

Stdlib only (no Streamlit), so the logic can be tested on its own.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

PUBLICATION = "provenance/commitment.json"
RECORD = "provenance/record.json"
PUBLICATION_PROOF = "provenance/commitment.json.ots"
RECORD_PROOF = "provenance/record.json.ots"
GPG_PUBLIC_KEY = "provenance/owner_signing_key.asc"
P5_5 = "results/p5.5_timestamp__"
P5_5_STATUS = "results/p5.5_timestamp_status__"
P6_2 = "results/p6.2_signed_provenance_record__"
P6_3 = "results/p6.3_provenance_verifier__"
P6_4 = "results/p6.4_theft_simulation__"

TIME_FORMAT = "%d %b %Y, %H:%M UTC"
SUSPECT_LABELS = {
    "none": "Verbatim copy",
    "ptq": "Post-training quantization",
    "magnitude": "Global magnitude pruning",
    "overwrite": "Thief adds their own weight watermark",
    "finetune": "Fine-tuning on the thief's data",
    "prune": "Channel pruning, then fine-tuning",
    "distill": "Distillation into a new model",
}
UNRELATED_LABEL = "Owner's clean model, never handed over"
"""Labels by the first word of the recorded attack name; the record's own keys carry digits."""
DROP_FIELD_PREFIX = "accuracy_drop_vs_"
"""P6.4's accuracy-change field, found by prefix: its full name uses wording this app does not show."""


@dataclass(frozen=True)
class Sources:
    publication: dict
    record: dict
    timestamp: dict
    timestamp_status: dict
    signing: dict
    verifier: dict
    theft: dict
    publication_sha256: str
    record_sha256: str
    publication_proof_sha256: str
    record_proof_sha256: str
    paths: dict[str, str]


def load(store) -> Sources:
    """Every file the panel uses, each verified by the data layer. Raises `DataIntegrityError`."""
    latest = {name: store.latest(prefix) for name, prefix in
              (("timestamp", P5_5), ("timestamp_status", P5_5_STATUS), ("signing", P6_2), ("verifier", P6_3),
               ("theft", P6_4))}
    raw = {path: store.read_bytes(path) for path in (PUBLICATION, RECORD, PUBLICATION_PROOF, RECORD_PROOF)}
    digest = {path: hashlib.sha256(data).hexdigest() for path, data in raw.items()}
    return Sources(
        publication=store.read_json(PUBLICATION), record=store.read_json(RECORD),
        **{name: store.read_json(path) for name, path in latest.items()},
        publication_sha256=digest[PUBLICATION], record_sha256=digest[RECORD],
        publication_proof_sha256=digest[PUBLICATION_PROOF], record_proof_sha256=digest[RECORD_PROOF],
        paths={**latest, **{p: p for p in raw}})


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text)


def show_time(text: str) -> str:
    return utc(text).strftime(TIME_FORMAT)


# --- Status of the three open points -------------------------------------------------------

@dataclass(frozen=True)
class OpenPoints:
    record_proof_status: str
    record_bitcoin_attestations: int
    record_calendars: int
    record_proof_is_the_described_one: bool
    tag_name: str
    tag_holds_current_proof: bool
    tag_pushed: bool
    trusted_key_hex: str
    trusted_key_source: str
    all_runs_used_that_key: bool


def open_points(s: Sources) -> OpenPoints:
    status = s.timestamp_status["metrics"]["proofs"]["record"]
    tag = s.timestamp["metrics"]["gpg_tag"]
    record_key = s.record["owner"]["public_key"]["hex"]
    keys = {record_key, s.signing["metrics"]["public_key_hex"], s.verifier["metrics"]["trusted_public_key_hex"],
            s.theft["params"]["owner_trusted_public_key"]}
    return OpenPoints(
        record_proof_status=status["status"],
        record_bitcoin_attestations=len(status["bitcoin_attestations"]),
        record_calendars=len(status["pending_calendars"]),
        record_proof_is_the_described_one=(status["ots_sha256"] == s.record_proof_sha256
                                           and status["file_sha256"] == s.record_sha256),
        tag_name=tag["tag"],
        tag_holds_current_proof=s.timestamp["metrics"]["opentimestamps"]["proof_in_tag_is_current"],
        tag_pushed=tag["pushed_to_a_third_party"],
        trusted_key_hex=record_key,
        trusted_key_source=s.verifier["metrics"]["trusted_public_key_source"],
        all_runs_used_that_key=len(keys) == 1)


# --- Live consistency checks ---------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    label: str
    ok: bool
    detail: str


def consistency_checks(s: Sources) -> list[Check]:
    pub_c, rec_c = s.publication["commitment"], s.record["watermark_commitment"]["commitment"]
    ots = s.timestamp["metrics"]["opentimestamps"]
    return [
        Check("The record names this publication by its SHA-256",
              s.record["commitment_publication"]["sha256"] == s.publication_sha256,
              f"computed now: {s.publication_sha256}"),
        Check("The record's C equals the published C", rec_c == pub_c, "decimal, hex and layout version compared"),
        Check("The published C's decimal and hex forms agree", int(pub_c["decimal"]) == int.from_bytes(bytes.fromhex(pub_c["hex"]), "big"),
              "the same field element"),
        Check("The record names the same model fingerprint",
              s.record["model"]["fingerprint"] == s.publication["model_fingerprint"],
              s.publication["model_fingerprint"]["sha256"]),
        Check("The Bitcoin-checked hash is this publication",
              ots["file_digest"] == s.timestamp["metrics"]["artifact_sha256"] == s.publication_sha256,
              "the P5.5 proof is for exactly these bytes"),
        Check("The committed proof file is the one P5.5 checked", ots["ots_sha256"] == s.publication_proof_sha256,
              f"computed now: {s.publication_proof_sha256}"),
    ]


# --- Independent time ----------------------------------------------------------------------

@dataclass(frozen=True)
class Block:
    height: int
    block_hash: str
    header_time: str
    calendars: int
    explorers: tuple[str, ...]
    verified: bool


def blocks(s: Sources) -> list[Block]:
    ots = s.timestamp["metrics"]["opentimestamps"]
    out = []
    for b in ots["chain_check"]["blocks"]:
        out.append(Block(
            height=b["height"], block_hash=b["block_hash"], header_time=b["header_time_utc"],
            calendars=sum(a["height"] == b["height"] for a in ots["bitcoin_attestations"]),
            explorers=tuple(name.split("//", 1)[-1].split("/", 1)[0] for name in b["explorers"]),
            verified=b["verified"] and all(e["header_hashes_to_block_hash"] and e["meets_target"]
                                           and e["merkle_root_matches"] for e in b["explorers"].values())))
    return sorted(out, key=lambda b: b.height)


# --- Timeline ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    utc: str
    text: str
    evidence: str
    kind: str
    party: str


def timeline(s: Sources) -> list[Event]:
    """The P6.4 timeline in clock order, each event tagged by the kind of time evidence behind it."""
    events = []
    for e in s.theft["metrics"]["timeline"]:
        evidence = e["evidence"]
        kind = "bitcoin" if "block" in evidence else "self" if "self-asserted" in evidence else "run"
        text = e["event"]
        party = ("bitcoin" if kind == "bitcoin" else "thief" if text.startswith("thief")
                 else "owner" if text.startswith("owner") else "simulation")
        events.append(Event(e["utc"], text, evidence, kind, party))
    return sorted(events, key=lambda e: utc(e.utc))


# --- The two claims ------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimRow:
    aspect: str
    owner: str
    thief: str
    symmetric: bool
    note: str


def field(record: dict, prefix: str) -> Any:
    """The value of the one key starting with `prefix` (record keys such as ``weight_detected_1e-6`` carry the level)."""
    (value,) = [v for k, v in record.items() if k.startswith(prefix)]
    return value


def _suspect(s: Sources, attack: str | None) -> dict:
    return next(v for v in s.theft["metrics"]["suspects"].values() if v.get("attack") == attack)


def claim_rows(s: Sources) -> list[ClaimRow]:
    cc = s.theft["metrics"]["counter_claim"]
    verdict = s.theft["metrics"]["publication"]["record_verdict"]
    shipped, named = _suspect(s, "overwrite_weight"), _suspect(s, "none")
    block = cc["owner_publication_independent_time"]
    yes = {True: "yes", False: "no"}

    def z(value: float, detected: bool) -> str:
        return f"z {value:.2f}, {'detected' if detected else 'not detected'}"

    return [
        ClaimRow("Signed claim that verifies under the key it names", yes[verdict["signature_valid"]],
                 yes[cc["thief_claim_verifies_on_its_own_terms"]],
                 verdict["signature_valid"] == cc["thief_claim_verifies_on_its_own_terms"],
                 "Anyone can make a key and sign a consistent claim."),
        ClaimRow("Date written in the claim", show_time(cc["owner_created_utc_self_asserted"]),
                 show_time(cc["thief_created_utc_self_asserted"]), True,
                 "Both dates are self-asserted. "
                 + ("The thief's comes first, because the thief chose it."
                    if cc["self_asserted_times_say_thief_first"] else "")),
        ClaimRow("Independent time", f"Bitcoin block {block['bitcoin_block']:,} ({show_time(block['header_time_utc'])})",
                 "none" if not cc["thief_claim_independently_timestamped"] else "recorded", False,
                 "Only the owner's publication was timestamped by a third party before the dispute."),
        ClaimRow("Weight watermark on the model as the thief ships it",
                 z(shipped["weight_z"], field(shipped, "weight_detected_")),
                 z(cc["thief_weight_z_on_shipped_model"], cc["thief_weight_detected_on_shipped_model"]),
                 field(shipped, "weight_detected_") == cc["thief_weight_detected_on_shipped_model"],
                 "The shipped model carries both watermarks, so this test alone cannot say who was first."),
        ClaimRow("Weight watermark on the model the owner published before the hand-over",
                 z(named["weight_z"], field(named, "weight_detected_")),
                 z(cc["thief_weight_z_on_owner_published_model"],
                   cc["thief_weight_detected_on_owner_published_model"]),
                 field(named, "weight_detected_") == cc["thief_weight_detected_on_owner_published_model"],
                 "The earlier model carries only the owner's watermark."),
        ClaimRow("Accepted under the owner's trusted key", "yes" if verdict["public_key_trusted"] else "no",
                 yes[cc["thief_claim_under_owner_trusted_key"]], False,
                 "Circular on its own: it only matters if the key is trusted for reasons outside the record."),
    ]


# --- The owner's audit ---------------------------------------------------------------------

@dataclass(frozen=True)
class AuditRow:
    label: str
    how: str
    test_accuracy: float
    accuracy_change_pp: float
    fired: int
    behavioral_p: float
    weight_z: float | None
    matches_record: bool
    evidence: str


def audit_rows(s: Sources) -> list[AuditRow]:
    rows = []
    for v in s.theft["metrics"]["suspects"].values():
        attack = v.get("attack")
        label = UNRELATED_LABEL if attack is None else SUSPECT_LABELS[attack.split("_", 1)[0]]
        drop = field(v, DROP_FIELD_PREFIX)
        rows.append(AuditRow(label, v["how"], v["test_accuracy"], drop, v["triggers_fired"],
                             v["behavioral_p_value"], v["weight_z"] if v["weight_applicable"] else None,
                             v["is_the_named_model"], v["evidence"]))
    return rows
