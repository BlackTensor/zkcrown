"""Honest-limits panel (P9.14): each limit with the measured fact behind it, from committed files only.

Every limit is built from committed records through the data layer. The P4.9
master table supplies the attack limits, the P9.2 audit verdicts and the P5.6
record supply the audit limits, and the provenance and zero-knowledge logic
(`app.provenance_logic`, `app.zk_logic`) supply the rest. Each limit names the
files it was built from and the dashboard page where the evidence is shown.

Limits are built in groups. A group whose files fail their hash check is
reported as an error, and the other groups still show.

Stdlib only (no Streamlit), so the logic can be tested on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from app import audit_logic, provenance_logic, zk_logic
from app.attacks_logic import NEVER_BUILT

P4_9 = "results/p4.9_master_table__"
P9_2 = "results/p9.2_audit_suspects__"
P5_6 = "results/p5.6_opening_verifier__"
NOT_LEGAL_MARK = "not legal evidence"


@dataclass(frozen=True)
class Limit:
    key: str
    category: str
    title: str
    fact: str
    detail: str
    sources: tuple[str, ...]
    page: str
    page_label: str


def _row_ci_excludes_zero(row: dict) -> bool:
    low, high = next(v for k, v in row.items() if k.startswith("drop_ci"))
    return not low <= 0 <= high


def attack_limits(store) -> list[Limit]:
    path = store.latest(P4_9)
    record = store.read_json(path)
    summary, rows = record["metrics"]["summary"], record["metrics"]["rows"]
    alpha = record["params"]["detection_alpha"]
    out = []

    distill = summary["distill"]
    student = distill["highest_accuracy_neither_detected"]
    row = next(r for r in rows if r["family"] == "distill" and r["setting"] == student["setting"])
    out.append(Limit(
        "distillation", "Watermarks", "Distillation removes both watermarks",
        f"Neither watermark was detected in {distill['outcomes']['neither detected']} of {distill['rows']} distilled "
        f"students. The best kept {student['test_accuracy']:.2%} test accuracy ({student['setting']}), a drop of "
        f"{student['drop_pp']:+.2f} pp "
        + ("that is not distinguishable from zero." if not _row_ci_excludes_zero(row) else "that is distinguishable from zero."),
        f"The student only sees the model's outputs: it never queries a trigger and never sees the weights. Caveat: "
        f"{distill['note']}. A model with no evidence is not shown to be independent.",
        (path,), "attacks", "Attack lab"))

    channel = [r for r in rows if r["family"] == "prune_finetune_channel" and r["outcome"] == "neither detected"]
    fam = summary["prune_finetune_channel"]
    out.append(Limit(
        "channel_finetune", "Watermarks", "Channel pruning plus fine-tuning can remove both",
        f"{len(channel)} of {fam['rows']} runs left neither watermark detected at {alpha}, keeping "
        f"{min(r['test_accuracy'] for r in channel):.2%} to {max(r['test_accuracy'] for r in channel):.2%} test "
        f"accuracy (drops {min(r['drop_pp'] for r in channel):+.2f} to {max(r['drop_pp'] for r in channel):+.2f} pp).",
        "This is the only attack in the suite, besides distillation, that removed both watermarks and left a usable "
        "model, at a real accuracy cost. Its weight figures also assume re-alignment (next limit).",
        (path,), "attacks", "Attack lab"))

    noted = [f for f in record["params"]["families"] if f["note"] and "re-alignment" in f["note"]]
    noted_rows = sum(summary[f["id"]]["rows"] for f in noted)
    out.append(Limit(
        "layout", "Watermarks", "The weight test needs the owner's layout",
        f"{summary['all_attacks']['weight_not_applicable']} attacked models had no owner layout, so the weight test "
        f"could not run. The weight figures in {noted_rows} channel-pruning rows assume re-alignment, which was "
        "never built.",
        NEVER_BUILT + " A thief who physically deletes channels or changes the architecture ships a model the "
        "weight test cannot read; that is not evidence either way.",
        (path,), "attacks", "Attack lab"))

    seeds = sorted({m.group(1) for r in rows if (m := re.search(r"seed(\d+)", r["result_file"]))})
    settings = {(r["family"], r["setting"]) for r in rows}
    out.append(Limit(
        "single_runs", "Evidence scope", "One run per setting, one model, one key",
        f"{record['metrics']['row_count']} rows (the control plus the attacks), {len(settings)} distinct settings, each run once from one source "
        f"model ({record['params']['source']['name']} W*, {record['params']['source']['task']}) with one owner key "
        f"and seed {', '.join(seeds)}.",
        "Confidence intervals cover which test images were sampled, not other training seeds, keys or models. "
        "Counts over settings depend on which settings were chosen; they are not rates.",
        (path,), "attacks", "Attack lab"))
    return out


def crypto_limits(store) -> list[Limit]:
    z = zk_logic.load(store)
    a, x = zk_logic.track_a(z), zk_logic.blocked(z)
    setup, ptau, sizing = (z.paths[k] for k in ("setup", "ptau", "sizing"))
    return [
        Limit("single_contributor", "Cryptography", "The Groth16 setup had a single contributor",
              f"The proving key's own check lists {a.contributors} contribution to the circuit-specific setup, made by the "
              "owner.",
              "Groth16 is sound only if at least one contributor destroyed their secret. With one contributor, a "
              "verifier must trust the owner, who could otherwise forge proofs. Redoing it with independent "
              "contributors was not done.", (setup,), "zk", "Zero-knowledge"),
        Limit("hermez", "Cryptography", "The powers-of-tau file was not verified here",
              f"“{a.ptau_verify}”; the power-{a.ptau_power} file was accepted on “{a.ptau_accepted_on}”.",
              "The hash shows it is byte for byte the published ceremony file. It does not show the ceremony's "
              "contribution chain inside the file is consistent.", (ptau,), "zk", "Zero-knowledge"),
        Limit("trigger_circuit", "Cryptography", "Proving the triggers come from the key is blocked",
              f"Estimated at {x.total:,} constraints (lower bound), needing power {x.power_needed}; the file used "
              f"holds {x.ptau_capacity:,}. SHA-256 is {x.sha256_share:.1%} of it.",
              "So nothing proves that the triggers used in an audit derive from the committed key. A "
              "circuit-friendly redesign is an open choice, not started.", (sizing,), "zk", "Zero-knowledge"),
    ]


def provenance_limits(store) -> list[Limit]:
    s = provenance_logic.load(store)
    p = provenance_logic.open_points(s)
    status, timestamp, verifier = (s.paths[k] for k in ("timestamp_status", "timestamp", "verifier"))
    return [
        Limit("record_pending", "Provenance", "The provenance record has no independent time yet",
              f"Its Bitcoin proof is {p.record_proof_status}: {p.record_calendars} calendar promises, "
              f"{p.record_bitcoin_attestations} Bitcoin attestations.",
              "Only the commitment publication is Bitcoin-attested. The record, and the trigger set commitment and "
              "signing key it names, rest on the owner's clock.", (status, provenance_logic.RECORD_PROOF),
              "provenance", "Provenance"),
        Limit("stale_tag", "Provenance", "The signed tag holds the older proof",
              f"Tag {p.tag_name}: proof inside is "
              + ("current." if p.tag_holds_current_proof else "the earlier, pending one, not the Bitcoin-checked upgrade.")
              + (" Not pushed anywhere." if not p.tag_pushed else ""),
              "Both proofs cover the same file. The tag's own date is the signer's clock, so it is not time evidence.",
              (timestamp,), "provenance", "Provenance"),
        Limit("trusted_key", "Provenance", "The “trusted” key came from this repository",
              f"Recorded source: “{p.trusted_key_source}”.",
              "A forged claim fails against that key, which shows the check works. It does not show the key belongs "
              "to the owner; that has to come from outside the record.", (verifier,), "provenance", "Provenance"),
    ]


def audit_limits(store) -> list[Limit]:
    path, opening = store.latest(P9_2), store.latest(P5_6)
    record, p5_6 = store.read_json(path), store.read_json(opening)
    verdicts = record["metrics"]["verdicts"]
    with_key = sum("K" in v["secrets_used"] for v in verdicts.values())
    needs_key = [audit_logic.STEP_TITLES[s] for s in audit_logic.REPLAYED_SLOTS
                 if "secret key" in audit_logic.WHY_REPLAYED[s]]
    replayed = [audit_logic.STEP_TITLES[s] for s in audit_logic.REPLAYED_SLOTS]
    live = [audit_logic.STEP_TITLES[s] for s in audit_logic.SLOTS if s not in audit_logic.REPLAYED_SLOTS]
    not_legal = sorted({line for v in verdicts.values() for line in v["limitations"] if NOT_LEGAL_MARK in line})
    return [
        Limit("not_zero_knowledge", "Evidence scope", "The watermarks are not zero-knowledge",
              f"The {' and '.join(needs_key).lower()} tests read the owner's key: {with_key} of {len(verdicts)} "
              f"recorded audits used it. Opening the commitment would reveal "
              f"{p5_6['metrics']['secret_bytes_revealed_by_an_opening']} secret bytes.",
              "Only the Track A proof relation is zero-knowledge. A watermark test either runs on the owner's side "
              "or reveals the key, after which anyone can aim at the triggers.", (path, opening), "audit", "Live audit"),
        Limit("replayed", "This demo", "Most checks on this site are replayed",
              f"{len(replayed)} of {len(audit_logic.SLOTS)} audit checks are replayed from the recorded run "
              f"({', '.join(replayed).lower()}); {', '.join(live).lower()} and the grade are recomputed live.",
              "The hosted app holds no key, no suspect weights and no proving tools. The signature checks on the "
              "Provenance page are replayed too. Every replayed step names its source file.",
              (path,), "audit", "Live audit"),
        Limit("not_legal", "This demo", "Not legal evidence",
              not_legal[0] if len(not_legal) == 1 else "; ".join(not_legal),
              "Every grade is technical evidence strength from the measured tests. It is not a finding of ownership, "
              "theft or independence.", (path,), "evidence", "Evidence strength"),
    ]


GROUPS: tuple[tuple[str, Callable], ...] = (
    ("Attack results", attack_limits),
    ("Proof setup", crypto_limits),
    ("Provenance files", provenance_limits),
    ("Audit records", audit_limits),
)
CATEGORIES = ("Watermarks", "Evidence scope", "Cryptography", "Provenance", "This demo")
