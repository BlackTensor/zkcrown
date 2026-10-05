"""P9.2: the auditor with all five checks, run on the P6.4 suspects.

    python experiments/p9_2_audit_suspects.py

CPU, a few minutes. Reads `K` and the commitment nonce from ``secrets/`` (the
watermark checks need `K`; all three secrets arm the no-secrets guard), CIFAR-10
from ``data/``, the committed provenance and ZK files, and the attacked weights
by the hashes their Phase 4 apply records hold.

Suspects:

- The P6.4 thefts, built the same way: ``verbatim`` (the dual `W*`), ``int8``,
  ``prune_global_50`` and ``overwrite_weight_0.1`` run live through the Phase
  4 harness (no key); ``finetune_lr0.05_e20``, ``channel50_finetune_lr0.1_e60``
  and ``distill_50k_w32`` loaded from Phase 4. The INT8 suspect is audited
  with its fused runtime as the queryable model, as shipped.
- ``distill_50k_w16``: the width-16 student from P4.7, a suspect without the
  owner's carrier layout, for the weight check's ``not_applicable`` path.
- Not stolen: clean `W` (P0.5, trained by the owner without `K`), and an
  untrained ``main_model`` from seed 20261005, unrelated to the owner.

Built-in checks (the script stops on any failure):

- every theft's fired count and weight z equal its committed Phase 4 row, and
  clean `W` gives P2.4's 3 fired and P3.7's z;
- neither not-stolen model is called watermarked by either test;
- only the verbatim copy passes the fingerprint check;
- the record is valid, the commitment passes and both ZK proofs verify, for
  every suspect;
- no check output was refused by the guard, and the written result file holds
  none of the secrets (decimal, hex, base64).

Each check is reported separately; nothing is graded (P9.3).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

import torch

from make_master_key import DEFAULT_KEY_PATH, load_key
from p2_4_measure_wdr import W_CLEAN_SHA256, sha256_file
from p5_4_publish_commitment import DEFAULT_NONCE_PATH, load_nonce
from run_attack_suite import load_source, read_apply_record
from theft_simulation import OWNER_PUBLIC_KEY, ROWS, THEFTS, committed_row
from src.attacks.harness import AttackConfig, AttackContext, apply_attack
from src.auditor import SLOTS, OwnerSecrets, audit
from src.auditor.checks import default_checks
from src.crypto.provenance import RECORD_PATH, read_record
from src.crypto.publication import ARTIFACT_PATH, read_publication
from src.models.main_model import MainModel
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.watermark.signature import PROJECT_OWNER_ID, derive_signature

EXTRA_W16 = {"apply": "p4.7_apply/p4.7_apply_distill_train50k-T4-lr0.1_16__seed1337__20261002T130528+0000.json",
             "row": "p4.7_distill_train50k-T4-lr0.1_16__seed1337__20261002T132651+0000.json"}
UNRELATED_SEED = 20261005
CLEAN_W = {"fired": 3, "z": 0.18}
"""From P2.4 and P3.7."""


def committed(rel: str) -> dict:
    from src.utils.results import read_result

    return read_result(repo_root() / ROWS / rel)["metrics"]["row"]


def summarise(name: str, kind: str, verdict) -> dict:
    d = verdict.to_dict()
    c = {x["slot"]: x for x in d["checks"]}
    beh, wgt = c["behavioral"], c["weight"]
    return {
        "suspect": name, "kind": kind, "record_valid": d["record_valid"],
        **{f"{slot}_status": c[slot]["status"] for slot in SLOTS},
        "fired": beh["statistic"].get("fired"), "behavioral_p_value": beh["p_value"],
        "weight_z": wgt["statistic"].get("z"), "weight_p_value_bound": wgt["p_value"],
        "weight_reason": wgt["reason"] if wgt["status"] == "not_applicable" else None,
        "guard_rejected": [x["slot"] for x in d["checks"] if x["guard_rejected"]],
        "secrets_used": d["secrets_used"],
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=repo_root() / "data")
    args = parser.parse_args(argv)
    git, started = git_info(), time.perf_counter()

    key, nonce = load_key(DEFAULT_KEY_PATH), load_nonce(DEFAULT_NONCE_PATH)
    secrets = OwnerSecrets(key, derive_signature(key, PROJECT_OWNER_ID).value, nonce)
    record, record_hash = read_record(RECORD_PATH)
    publication, publication_hash = read_publication(ARTIFACT_PATH)
    checks = default_checks(args.data_root)
    source_state, source = load_source("dual", None)
    context = AttackContext(device=torch.device("cpu"), seed=DEFAULT_SEED, data_root=args.data_root)

    suspects: list[tuple[str, str, dict, dict, object, dict | None, str | None]] = []
    for name, spec in THEFTS.items():
        if "live" in spec:
            attack, strength, params = spec["live"]
            out = apply_attack(AttackConfig(attack=attack, strength=strength, params=params), source_state,
                               source["arch"], context)
            suspects.append((name, "theft (live)", out.state_dict, out.arch, out.runtime_model,
                             committed_row(name), None))
        else:
            _, rec, _, weights = read_apply_record(repo_root() / ROWS / spec["apply"], None)
            suspects.append((name, "theft (Phase 4 weights)", torch.load(weights, map_location="cpu", weights_only=True),
                             rec["metrics"]["arch"], None, committed_row(name), sha256_file(weights)))
    _, rec, _, weights = read_apply_record(repo_root() / ROWS / EXTRA_W16["apply"], None)
    suspects.append(("distill_50k_w16", "theft (Phase 4 weights), no carrier layout",
                     torch.load(weights, map_location="cpu", weights_only=True), rec["metrics"]["arch"], None,
                     committed(EXTRA_W16["row"]), sha256_file(weights)))
    clean_path = repo_root() / "results" / "p0.5_clean_baseline_W.pt"
    if sha256_file(clean_path) != W_CLEAN_SHA256:
        raise SystemExit("clean W has the wrong hash")
    suspects.append(("clean_W", "not stolen (owner's model trained without K)",
                     torch.load(clean_path, map_location="cpu", weights_only=True), {"width": 32}, None, None,
                     W_CLEAN_SHA256))
    torch.manual_seed(UNRELATED_SEED)
    suspects.append(("untrained_unrelated", "not stolen (fresh init, seed 20261005)", MainModel(width=32).state_dict(),
                     {"width": 32}, None, None, None))

    verdicts, rows, problems = {}, [], []
    for name, kind, state, arch, runtime, row, file_hash in suspects:
        verdict = audit(state, record, publication, trusted_public_key=OWNER_PUBLIC_KEY, checks=checks,
                        owner_secrets=secrets, suspect_label=name, suspect_file_sha256=file_hash,
                        suspect_model=runtime, suspect_arch=arch)
        s = summarise(name, kind, verdict)
        verdicts[name], rows = verdict.to_dict(), rows + [s]
        if row is not None:
            if s["fired"] != row["behavioral"]["fired"]:
                problems.append(f"{name}: fired {s['fired']} vs Phase 4 {row['behavioral']['fired']}")
            if row["weight"]["applicable"] != (s["weight_status"] != "not_applicable"):
                problems.append(f"{name}: weight applicability differs from Phase 4")
            elif row["weight"]["applicable"] and abs(s["weight_z"] - row["weight"]["z"]) > 1e-9:
                problems.append(f"{name}: weight z {s['weight_z']} vs Phase 4 {row['weight']['z']}")
        if kind.startswith("not stolen") and "detected" in (s["behavioral_status"], s["weight_status"]):
            problems.append(f"{name}: a model that was not stolen was called watermarked")
        if (s["fingerprint_status"] == "passed") != (name == "verbatim"):
            problems.append(f"{name}: fingerprint status {s['fingerprint_status']}")
        if not (s["record_valid"] and s["commitment_status"] == "passed" and s["zk_proof_status"] == "passed"):
            problems.append(f"{name}: record, commitment or zk_proof not passing")
        if s["guard_rejected"]:
            problems.append(f"{name}: guard rejected {s['guard_rejected']}")
        print(f"{name:30s} fp={s['fingerprint_status']:7s} beh={s['behavioral_status']:13s} fired={s['fired']!s:>4} "
              f"wgt={s['weight_status']:14s} z={'n/a' if s['weight_z'] is None else round(s['weight_z'], 2)!s:>6} "
              f"commit={s['commitment_status']} zk={s['zk_proof_status']} secrets={s['secrets_used']}")
    clean = next(r for r in rows if r["suspect"] == "clean_W")
    if clean["fired"] != CLEAN_W["fired"] or round(clean["weight_z"], 2) != CLEAN_W["z"]:
        problems.append(f"clean W does not reproduce P2.4 / P3.7: {clean['fired']}, {clean['weight_z']}")
    if problems:
        raise SystemExit("P9.2 checks failed:\n  " + "\n  ".join(problems))

    path = write_result(
        "p9.2_audit_suspects", DEFAULT_SEED, task="P9.2", git=git, duration_seconds=time.perf_counter() - started,
        params={"record_sha256": record_hash, "publication_sha256": publication_hash,
                "trusted_public_key_hex": OWNER_PUBLIC_KEY, "detection_alpha": "1e-6",
                "checks": {slot: getattr(checks[slot], "method") for slot in SLOTS},
                "suspects": {n: {"kind": k, "arch": a, "runtime_model": r is not None, "file_sha256": h}
                             for n, k, _, a, r, _, h in suspects}},
        metrics={"table": rows, "verdicts": verdicts},
        notes="All five checks, each reported separately at 1e-6; nothing graded (P9.3). Every theft reproduces its "
              "Phase 4 row; neither not-stolen model is called watermarked.")
    text = Path(path).read_text(encoding="utf-8").lower()
    if any(needle in text for needle in secrets.needles()):
        raise SystemExit(f"{path} contains a secret")
    print(f"wrote {path}; no secret in it")
    return {"rows": rows, "path": str(path)}


if __name__ == "__main__":
    main()
