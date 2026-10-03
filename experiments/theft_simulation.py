"""P6.4: the full theft timeline, end to end.

    python experiments/theft_simulation.py

CPU only, about ten minutes. Reads `K` (``secrets/K.bin``), because the audit
is the owner's, and CIFAR-10 from ``data/``. Contacts nobody.

The timeline, in order:

1. **Publication (before the theft).** The owner's commitment publication
   ``provenance/commitment.json`` (P5.4) and signed provenance record
   ``provenance/record.json`` (P6.2) already exist. Here they are re-read, the
   record is verified against the publication and the owner's public key
   (P6.3), and the publication's independent time is taken from the committed
   P5.5 chain check: Bitcoin block 969627. Nothing is re-published.
2. **Hand-off.** The thief receives the dual `W*` (P3.6), the exact model the
   record names. Every attack runs through the Phase 4 harness, whose
   `AttackContext` has no key field: the thief never sees `K`.
3. **The thief modifies it.** Seven thefts, from the cheapest to the
   strongest measured in Phase 4:
   - ``verbatim``: an exact copy (the P4.1 control);
   - ``int8``: static INT8 post-training quantization (P4.4), run here;
   - ``prune_global_50``: global magnitude pruning to 50% (P4.2), run here;
   - ``overwrite_weight_0.1``: the thief adds their own weight watermark at
     the owner's strength (P4.8), run here, and publishes a claim of their own
     (step 4);
   - ``finetune_lr0.05_e20``: fine-tuning on the attacker holdout, the
     cheapest removal of the behavioral watermark in P4.5;
   - ``channel50_finetune_lr0.1_e60``: channel pruning then fine-tuning, the
     P4.6 run that removed both watermarks;
   - ``distill_50k_w32``: distillation on 50,000 images into a fresh student,
     the P4.7 headline limitation.
   The three training attacks were run on Colab in Phase 4. Their returned
   weights are loaded here by the hash their apply records hold. Plus one
   suspect that is not a theft: clean `W` (P0.5), a model trained without `K`.
4. **The thief's counter-claim** (overwrite theft only). The thief builds a
   commitment publication and a signed provenance record of their own, with
   their own key, for the model they ship, and backdates both to 2026-09-01,
   before the owner's publication. Nothing of theirs is timestamped, because
   a real timestamp cannot be made for a time already past.
5. **The audit.** For each suspect, the owner runs the P6.3 verifier (does the
   suspect match the record?), the P2.8 behavioral test and the P3.7 weight
   test, each at 1e-6 as in Phase 4. For the counter-claim, the same verifier
   on the thief's files, the thief's watermark on the owner's published model,
   and the two timestamps.

Every live attack and every loaded model must reproduce its committed Phase 4
row (test images correct, triggers fired, weight z), or the script stops. So
the audit numbers here are the Phase 4 numbers, reached through the timeline.

There is no combined verdict: the two tests are reported separately, as in
Phase 4. Combining them is P9.3. This simulates the timeline after the fact:
no model was handed to anyone, and the thief is this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from run_attack_suite import Scorer, load_source, read_apply_record
from p2_4_measure_wdr import W_CLEAN_SHA256, sha256_file
from src.attacks.evaluation import build_row, evaluate_attacked
from src.attacks.harness import AttackConfig, AttackContext, apply_attack, get_attack, load_model
from src.attacks.overwrite import ATTACKER_OWNER_ID, attacker_key
from src.crypto.commitment import NONCE_BYTES, commit
from src.crypto.fingerprint import fingerprint_state_dict
from src.crypto.provenance import RECORD_PATH, build_unsigned_record, read_record
from src.crypto.provenance_verifier import verify_provenance
from src.crypto.publication import ARTIFACT_PATH, build_publication, read_publication
from src.crypto.signing import public_key_bytes, sign_record
from src.crypto.timestamping import describe_proof
from src.utils.results import git_info, read_result, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.watermark.signature import derive_signature

OWNER_PUBLIC_KEY = "94b0224bc4c2c7bdd80773cf142a3ec9c5c84ab6d55d8e808547a04f28b02d52"
"""The record signing key, from the P6.2 result record."""
P5_5_RESULT = "results/p5.5_timestamp__seed1337__20261003T092522+0000.json"
"""The chain check of the commitment publication's OpenTimestamps proof."""
OTS_PATH = ARTIFACT_PATH + ".ots"
BACKDATED_UTC = "2026-09-01T00:00:00+00:00"
"""The time the thief writes into their own claim: before the owner's publication."""
THIEF_TRIGGER_CLAIM = hashlib.sha256(b"zk-crown/theft-simulation/thief-claims-a-trigger-set-it-does-not-have").hexdigest()

ROWS = "results/attacks"
THEFTS = {
    "verbatim": {"live": ("none", 0, {}), "row": "p4.1_none_0__seed1337__20260914T093619+0000.json"},
    "int8": {"live": ("ptq_int8_static", 8, {"calibration_images": 1000}),
             "row": "p4.4_ptq_int8_static_8__seed1337__20260914T160859+0000.json"},
    "prune_global_50": {"live": ("magnitude_prune_global", 0.5, {}),
                        "row": "p4.2_magnitude_prune_global_0.5__seed1337__20260914T095915+0000.json"},
    "overwrite_weight_0.1": {"live": ("overwrite_weight", 0.1, {"attacker": "attacker-1"}),
                             "row": "p4.8_overwrite_weight_0.1__seed1337__20261002T134835+0000.json"},
    "finetune_lr0.05_e20": {
        "apply": "p4.5_apply/p4.5_apply_finetune_holdout_lr0.05_20__seed1337__20260914T173717+0000.json",
        "row": "p4.5_finetune_holdout_lr0.05_20__seed1337__20260914T180552+0000.json"},
    "channel50_finetune_lr0.1_e60": {
        "apply": "p4.6_apply/p4.6_apply_prune_finetune_channel-lr0.1-e60_0.5__seed1337__20260915T142730+0000.json",
        "row": "p4.6_prune_finetune_channel-lr0.1-e60_0.5__seed1337__20260915T144718+0000.json"},
    "distill_50k_w32": {
        "apply": "p4.7_apply/p4.7_apply_distill_train50k-T4-lr0.1_32__seed1337__20261002T124542+0000.json",
        "row": "p4.7_distill_train50k-T4-lr0.1_32__seed1337__20261002T132711+0000.json"},
}
CLEAN_W_FIRED = 3
"""Clean `W` on the owner's triggers, from P2.4 and P2.8."""


def publication_step() -> dict:
    """Step 1: the owner's published files, verified, and their independent time."""
    record, record_hash = read_record(repo_root() / RECORD_PATH)
    publication, publication_hash = read_publication(repo_root() / ARTIFACT_PATH)
    verdict = verify_provenance(record, publication, trusted_public_key=OWNER_PUBLIC_KEY)
    if not verdict.record_valid or not verdict.public_key_trusted:
        raise SystemExit(f"the owner's own record does not verify: {verdict.problems}")

    chain = read_result(repo_root() / P5_5_RESULT)["metrics"]
    if chain["artifact_sha256"] != publication_hash or not chain["opentimestamps"]["independent_time_evidence"]:
        raise SystemExit("the P5.5 chain check is not for this publication, or did not pass")
    proof = describe_proof(repo_root() / OTS_PATH, repo_root() / ARTIFACT_PATH)
    earliest = chain["opentimestamps"]["chain_check"]["earliest_verified_block"]
    if not proof["matches_file"] or earliest["height"] not in {a["height"] for a in proof["bitcoin_attestations"]}:
        raise SystemExit("the proof on disk does not carry the block the P5.5 chain check verified")
    return {
        "record": record, "publication": publication,
        "summary": {
            "commitment_publication": {"path": ARTIFACT_PATH, "sha256": publication_hash,
                                       "created_utc_self_asserted": publication["created_utc"]},
            "provenance_record": {"path": RECORD_PATH, "sha256": record_hash,
                                  "created_utc_self_asserted": record["timestamp"]["created_utc"],
                                  "independently_timestamped": False},
            "record_verdict": verdict.to_dict(),
            "independent_time": {"source": P5_5_RESULT, "bitcoin_block": earliest["height"],
                                 "block_hash": earliest["block_hash"], "header_time_utc": earliest["header_time_utc"],
                                 "meaning": "commitment.json existed by this block; header times are loose by hours"},
            "named_model_fingerprint": record["model"]["fingerprint"]["sha256"],
        },
    }


def committed_row(name: str) -> dict:
    return read_result(repo_root() / ROWS / THEFTS[name]["row"])["metrics"]["row"]


def reproduces(row: dict, committed: dict) -> None:
    """The audit row against the committed Phase 4 row. Stops on any difference."""
    checks = {
        "correct": (row["clean_accuracy"]["correct"], committed["clean_accuracy"]["correct"]),
        "fired": (row["behavioral"]["fired"], committed["behavioral"]["fired"]),
        "weight_applicable": (row["weight"]["applicable"], committed["weight"]["applicable"]),
    }
    if row["weight"]["applicable"]:
        checks["weight_z"] = (row["weight"]["z"], committed["weight"]["z"])
    wrong = {k: v for k, v in checks.items()
             if (abs(v[0] - v[1]) > 1e-9 if k == "weight_z" else v[0] != v[1])}
    if wrong:
        raise SystemExit(f"the audit does not reproduce the committed Phase 4 row: {wrong}")


def audit(state: dict, arch: dict, scorer: Scorer, source: dict, owner: dict, *, config, attacked: dict,
          runtime_model=None, theft: bool = True) -> tuple[dict, dict]:
    """Step 5 for one suspect: the record check and the two watermark tests. Returns (summary, full row).

    A theft is scored through `Scorer.row`, as in Phase 4. A suspect that is not a theft is scored with the same
    functions but without the harness's ``none`` control check, which requires the suspect to equal the source.
    """
    if theft:
        row = scorer.row(config, state, arch, source=source, attacked=attacked, runtime_model=runtime_model)
    else:
        evaluation = evaluate_attacked(state, arch, scorer.material, scorer.test_loader, scorer.reference["correct"],
                                       scorer.device)
        row = build_row(config, get_attack(config.attack), evaluation, source=source, attacked=attacked,
                        material=scorer.material)
    verdict = verify_provenance(owner["record"], owner["publication"], suspect=state, trusted_public_key=OWNER_PUBLIC_KEY)
    t = row["table"]
    return {
        "suspect_fingerprint": fingerprint_state_dict(state).sha256,
        "record_valid": verdict.record_valid,
        "is_the_named_model": verdict.fingerprint_matches,
        "test_accuracy": t["clean_accuracy"],
        "test_correct": row["clean_accuracy"]["correct"],
        "accuracy_drop_vs_stolen_pp": t["accuracy_drop_vs_source_pp"],
        "triggers_fired": row["behavioral"]["fired"],
        "behavioral_p_value": t["behavioral_p_value"],
        "behavioral_detected_1e-6": t["behavioral_detected"],
        "weight_applicable": t["weight_applicable"],
        "weight_z": t["weight_z"],
        "weight_p_value_bound": t["weight_p_value"],
        "weight_detected_1e-6": t["weight_detected"],
        "evidence": evidence(verdict.fingerprint_matches, t),
    }, row


def evidence(is_named_model: bool, table: dict) -> str:
    """Plain description of which tests found something. Not a graded verdict (P9.3)."""
    if is_named_model:
        return "exact copy of the model the record names"
    found = [name for name, hit in (("behavioral", table["behavioral_detected"]), ("weight", table["weight_detected"])) if hit]
    if not found:
        return "no watermark detected at 1e-6; a different set of weights"
    return f"{' and '.join(found)} watermark detected at 1e-6; a different set of weights"


def counter_claim(shipped_state: dict, owner: dict, overwrite_row: dict, source_state: dict) -> dict:
    """Step 4 and its audit: the thief's own publication and record, backdated, for the model they ship."""
    thief_key = attacker_key("attacker-1")  # the public demo key the overwrite attack used
    signature = derive_signature(thief_key, ATTACKER_OWNER_ID)
    thief_publication = build_publication(commit(thief_key, signature.value, os.urandom(NONCE_BYTES)),
                                          fingerprint_state_dict(shipped_state), ATTACKER_OWNER_ID,
                                          "thief's claim on the overwritten model", BACKDATED_UTC)
    signing_key = Ed25519PrivateKey.generate()
    thief_record = sign_record(build_unsigned_record(thief_publication, THIEF_TRIGGER_CLAIM, 100,
                                                     public_key_bytes(signing_key), BACKDATED_UTC), signing_key)

    on_own_terms = verify_provenance(thief_record, thief_publication, suspect=shipped_state)
    against_owner_key = verify_provenance(thief_record, thief_publication, suspect=shipped_state,
                                          trusted_public_key=OWNER_PUBLIC_KEY)
    owner_record_on_original = verify_provenance(owner["record"], owner["publication"], suspect=source_state,
                                                 trusted_public_key=OWNER_PUBLIC_KEY)
    weight = overwrite_row["attacked"]["info"]["weight"]
    owner_time = owner["summary"]["independent_time"]
    if not on_own_terms.valid or against_owner_key.valid or not owner_record_on_original.valid:
        raise SystemExit("the counter-claim did not give the expected verdicts")
    return {
        "thief_owner_id": ATTACKER_OWNER_ID,
        "thief_created_utc_self_asserted": BACKDATED_UTC,
        "owner_created_utc_self_asserted": owner["publication"]["created_utc"],
        "self_asserted_times_say_thief_first": BACKDATED_UTC < owner["publication"]["created_utc"],
        "thief_claim_verifies_on_its_own_terms": on_own_terms.valid,
        "thief_claim_names_the_shipped_model": on_own_terms.fingerprint_matches,
        "thief_claim_under_owner_trusted_key": against_owner_key.public_key_trusted,
        "thief_claim_independently_timestamped": False,
        "owner_publication_independent_time": owner_time,
        "thief_weight_z_on_shipped_model": weight["attacker_after"]["z"],
        "thief_weight_detected_on_shipped_model": weight["attacker_after"]["detected"],
        "thief_weight_z_on_owner_published_model": weight["attacker_before"]["z"],
        "thief_weight_detected_on_owner_published_model": weight["attacker_before"]["detected"],
        "owner_record_names_owner_published_model": owner_record_on_original.fingerprint_matches,
        "owner_watermarks_on_shipped_model": {"behavioral_detected_1e-6": overwrite_row["table"]["behavioral_detected"],
                                              "weight_detected_1e-6": overwrite_row["table"]["weight_detected"]},
        "reading": (
            "Both parties hold a signed claim that verifies under its own key, and the shipped model passes both "
            "parties' weight tests. The self-asserted dates favour the thief, because the thief chose theirs. "
            "Two things are not symmetric: the owner's publication has a Bitcoin-attested time and the thief's has "
            "none, and the model the owner published before the theft carries the owner's watermark but not the "
            "thief's, while the shipped model carries both."
        ),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--key", type=Path, default=repo_root() / "secrets" / "K.bin")
    parser.add_argument("--data-root", type=Path, default=repo_root() / "data")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    git = git_info()
    started = time.perf_counter()
    device = torch.device("cpu")

    # 1. Publication.
    owner = publication_step()

    # 2. Hand-off: the stolen model is exactly the one the record names.
    source_state, source = load_source("dual", None)
    handoff = verify_provenance(owner["record"], owner["publication"], suspect=source_state,
                                trusted_public_key=OWNER_PUBLIC_KEY)
    if not handoff.valid:
        raise SystemExit(f"the model handed to the thief is not the one the record names: {handoff.problems}")
    theft_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if theft_utc <= owner["summary"]["independent_time"]["header_time_utc"]:
        raise SystemExit("the simulated theft must come after the publication's attested time")

    scorer = Scorer(args.key, args.data_root, source_state, source, device)
    context = AttackContext(device=device, seed=args.seed, data_root=args.data_root)

    # 3 and 5. Each theft, then its audit.
    suspects, overwrite = {}, None
    for name, spec in THEFTS.items():
        if "live" in spec:
            attack, strength, params = spec["live"]
            config = AttackConfig(attack=attack, strength=strength, params=params)
            output = apply_attack(config, source_state, source["arch"], context)
            state, arch, runtime = output.state_dict, output.arch, output.runtime_model
            attacked = {"arch": arch, "info": output.info, "applied_in": "theft_simulation (live)"}
            how = "run here through the Phase 4 harness, no key"
        else:
            config, record, applied_source, weights = read_apply_record(repo_root() / ROWS / spec["apply"], None)
            if applied_source["weights_sha256"] != source["weights_sha256"]:
                raise SystemExit(f"{name}: the apply record was not applied to the dual W*")
            state = torch.load(weights, map_location="cpu", weights_only=True)
            arch, runtime = record["metrics"]["arch"], None
            attacked = {"arch": arch, "info": record["metrics"]["info"], "applied_in": "Colab apply (Phase 4)",
                        "weights_sha256": record["metrics"]["weights_sha256"]}
            how = f"trained on Colab in {record['task']}, loaded by hash {record['metrics']['weights_sha256'][:16]}..."
        summary, row = audit(state, arch, scorer, source, owner, config=config, attacked=attacked, runtime_model=runtime)
        reproduces(row, committed_row(name))
        summary.update({"attack": config.attack, "strength": config.strength, "how": how,
                        "reproduces_committed_row": THEFTS[name]["row"]})
        suspects[name] = summary
        if name == "overwrite_weight_0.1":
            overwrite = (state, row)
        print(f"{name:30s} named model={summary['is_the_named_model']!s:5}  acc {summary['test_accuracy']:.2%}  "
              f"fired {summary['triggers_fired']:3d}  z {summary['weight_z'] if summary['weight_z'] is None else round(summary['weight_z'], 2)}"
              f"  -> {summary['evidence']}")

    # The non-theft: a model trained without K.
    clean_path = repo_root() / "results" / "p0.5_clean_baseline_W.pt"
    if sha256_file(clean_path) != W_CLEAN_SHA256:
        raise SystemExit("clean W has the wrong hash")
    clean_state = torch.load(clean_path, map_location="cpu", weights_only=True)
    load_model(clean_state, {"width": 32})
    summary, row = audit(clean_state, {"width": 32}, scorer, source, owner,
                         config=AttackConfig(attack="none", strength=0, params={}),
                         attacked={"arch": {"width": 32}, "info": {}, "applied_in": "not a theft: clean W (P0.5)"},
                         theft=False)
    if summary["triggers_fired"] != CLEAN_W_FIRED or summary["behavioral_detected_1e-6"] or summary["weight_detected_1e-6"]:
        raise SystemExit(f"clean W did not reproduce P2.4 / P3.7: {summary}")
    summary.update({"attack": None, "how": "trained by the owner without K (P0.5); stands in for an unrelated model"})
    suspects["not_stolen_clean_W"] = summary
    print(f"{'not_stolen_clean_W':30s} named model=False  acc {summary['test_accuracy']:.2%}  "
          f"fired {summary['triggers_fired']:3d}  z {summary['weight_z']:.2f}  -> {summary['evidence']}")

    # 4. The thief's counter-claim, and its audit.
    claim = counter_claim(overwrite[0], owner, overwrite[1], source_state)
    print(f"counter-claim: verifies on its own terms={claim['thief_claim_verifies_on_its_own_terms']}, "
          f"self-asserted date first={claim['self_asserted_times_say_thief_first']}, independently timestamped=False; "
          f"thief z on owner's published model {claim['thief_weight_z_on_owner_published_model']:.2f}")

    audit_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    timeline = [
        {"step": 1, "event": "owner publishes commitment.json", "utc": owner["publication"]["created_utc"],
         "evidence": "self-asserted"},
        {"step": 1, "event": "commitment.json in Bitcoin block", "utc": owner["summary"]["independent_time"]["header_time_utc"],
         "evidence": f"block {owner['summary']['independent_time']['bitcoin_block']}, checked in P5.5"},
        {"step": 1, "event": "owner signs record.json", "utc": owner["record"]["timestamp"]["created_utc"],
         "evidence": "self-asserted; record proof pending"},
        {"step": 2, "event": "dual W* handed to the thief (simulated)", "utc": theft_utc, "evidence": "this run's clock"},
        {"step": 4, "event": "thief's backdated claim", "utc": BACKDATED_UTC, "evidence": "self-asserted, no timestamp"},
        {"step": 5, "event": "owner audits", "utc": audit_utc, "evidence": "this run's clock"},
    ]
    metrics = {"timeline": timeline, "publication": owner["summary"], "suspects": suspects, "counter_claim": claim,
               "detection_alpha": "1e-6",
               "counts": {
                   "thefts": len(THEFTS),
                   "exact_copy": sum(s["is_the_named_model"] for s in suspects.values()),
                   "behavioral_detected": sum(s["behavioral_detected_1e-6"] for k, s in suspects.items() if k in THEFTS),
                   "weight_detected": sum(s["weight_detected_1e-6"] for k, s in suspects.items() if k in THEFTS),
                   "neither_detected": sum(not (s["behavioral_detected_1e-6"] or s["weight_detected_1e-6"])
                                           for k, s in suspects.items() if k in THEFTS),
               }}
    path = write_result(
        name="p6.4_theft_simulation",
        seed=args.seed,
        task="P6.4",
        params={
            "stolen_model": {"name": "dual W* (P3.6)", "weights_sha256": source["weights_sha256"]},
            "thefts": THEFTS,
            "owner_trusted_public_key": OWNER_PUBLIC_KEY,
            "detection_alpha": "1e-6, per test, not combined (P9.3)",
            "thief_key": "public demo key of P4.8 (attacker-1); fresh Ed25519 signing key, discarded",
            "thief_backdated_utc": BACKDATED_UTC,
        },
        metrics=metrics,
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git,
        notes=(
            "A simulation after the fact: no model was handed to anyone. The audit numbers reproduce the committed "
            "Phase 4 rows exactly. Each watermark test is reported on its own at 1e-6; there is no combined verdict. "
            "The owner's record is not independently timestamped; the publication is (Bitcoin block 969627). This is "
            "a technical ownership verification demonstration, not legal evidence."
        ),
    )
    print("wrote", path)
    return {"path": path, "metrics": metrics}


if __name__ == "__main__":
    main()
