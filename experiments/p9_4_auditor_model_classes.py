"""P9.4: the auditor on three model classes, owner, attacked and unrelated.

    python experiments/p9_4_auditor_model_classes.py

CPU, about an hour. Reads `K` and the commitment nonce from ``secrets/`` (the
watermark checks need `K`; all three secrets arm the no-secrets guard), CIFAR-10
from ``data/``, the committed provenance and ZK files, and downloads the
third-party weights through ``torch.hub`` into ``data/torch_hub/`` (gitignored).

Models, all fixed before the run:

1. **The owner's model:** the dual `W*` (P3.6), the P4.1 control row.
2. **The 84 attacked variants:** every committed Phase 4 attack row (P4.2 to
   P4.8). Attacks run locally in Phase 4 are re-run here through the harness
   (no key); Colab ones are loaded from the weights their apply records hash.
   The INT8 and fused rows are audited through their runtime model, as shipped.
3. **Unrelated models:**
   - clean `W` (P0.5, trained by the owner without `K`);
   - 20 untrained ``main_model`` inits, seeds `FRESH_SEEDS`;
   - `zk_model` (P0.7, MNIST), through a grayscale, 28 x 28, MNIST-normalised
     input adapter;
   - the 19 non-ViT CIFAR-10 models of ``chenyaofo/pytorch-cifar-models``
     (`src.models.third_party.CATALOG`), each behind an input adapter for its
     own normalisation. Each must reach its published top-1 on the 10,000
     CIFAR-10 test images within `ACCURACY_TOLERANCE_PP` before it is audited;
     one that does not is excluded and listed.

Integrity checks (the script stops on any failure, before writing):

- every Phase 4 row reproduces its committed fired count, weight
  applicability and weight z (to 1e-9), and its grade equals the P9.3 grade of
  that row;
- clean `W` reproduces P2.4 / P3.7 (3 fired, z 0.18);
- the record is valid, the commitment passes and both ZK proofs verify for
  every model; no check output was refused by the guard.

Test criteria (recorded with their outcome; the task passes only if all hold):

- the dual `W*` is graded as an exact copy, with evidence;
- clean `W`, every fresh init and every included third-party model are graded
  ``none``, and each such verdict carries the no-exoneration caveat;
- `zk_model` is graded ``none`` too (an unrelated model of another domain);
- the weight test is ``not_applicable`` on every third-party model and on
  `zk_model`, and assessed on clean `W` and every fresh init.

The result file holds no secret (checked in decimal, hex and base64).
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

import torch

from make_master_key import DEFAULT_KEY_PATH, load_key
from p2_4_measure_wdr import W_CLEAN_SHA256, sha256_file
from p5_1_model_fingerprint import ZK_MODEL_SHA256
from p5_4_publish_commitment import DEFAULT_NONCE_PATH, load_nonce
from p9_3_grade_suspects import phase4_checks
from run_attack_suite import load_source, read_apply_record
from theft_simulation import OWNER_PUBLIC_KEY
from src.attacks.harness import AttackConfig, AttackContext, apply_attack
from src.auditor import SLOTS, OwnerSecrets, audit
from src.auditor.checks import default_checks
from src.auditor.grading import NO_EVIDENCE_CAVEAT, grade
from src.crypto.provenance import RECORD_PATH, read_record
from src.crypto.publication import ARTIFACT_PATH, read_publication
from src.data import CIFAR10_MEAN, CIFAR10_STD, MNIST_MEAN, MNIST_STD, cifar10_loaders
from src.models.adapters import InputAdapter
from src.models.main_model import MainModel
from src.models.third_party import (
    CATALOG,
    HUB_COMMIT,
    HUB_REPO,
    LICENSE,
    NORMALISATION_SOURCE,
    THIRD_PARTY_MEAN,
    THIRD_PARTY_STD,
    adapter_for,
    catalog_bytes,
    load_third_party,
)
from src.models.zk_model import ZKModel
from src.utils.results import git_info, read_result, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.watermark.signature import PROJECT_OWNER_ID, derive_signature

ROWS = "results/attacks"
P4_9_GLOB = "results/p4.9_master_table__*.json"
FRESH_SEEDS = tuple(range(20261006, 20261026))
"""20 untrained inits, seeds fixed before the run (not the project seed 1337, not P9.2's 20261005)."""
ACCURACY_TOLERANCE_PP = 1.0
"""A third-party model is audited only if its measured test top-1 is within this of its published figure."""
EXPECTED_PHASE4_GRADES = {"very strong": 70, "strong": 4, "moderate": 2, "weak": 1, "none": 8}
"""P9.3's survey of the 85 Phase 4 rows."""
CLEAN_W = {"fired": 3, "z": 0.18}
"""From P2.4 and P3.7."""
P0_5_RESULT = "results/p0.5_clean_baseline__seed1337__20260913T071152+0000.json"
GRADE_P_RTOL = 1e-6
WEIGHT_FPR_BASIS = (
    "The weight test cannot be run on the third-party models or zk_model: they do not have the owner's carrier "
    "layout, so the auditor reports it not_applicable and their grade rests on the behavioral test alone. The weight "
    "test's false-positive rate rests instead on: the P3.7 proof that P(z >= t) <= exp(-t^2/2) for any model "
    "independent of K; the P3.7 empirical null of 1,000 wrong keys on three models (z sd 0.97 to 1.01, KS vs N(0,1) "
    "p = 0.45 to 0.64, exceedances within their bounds); the owner's K on clean W (z = 0.18); and the 20 fresh "
    "main_model inits audited here.")


def phase4_expected() -> dict[str, dict]:
    """The P4.9 entry and its P9.3 grade, by row result file."""
    path = sorted(glob.glob(str(repo_root() / P4_9_GLOB)))[-1]
    rows = read_result(path)["metrics"]["rows"]
    return {r["result_file"]: {"entry": r, "grade": grade(phase4_checks(r), {"record_valid": True})["suspect"]}
            for r in rows}


def phase4_suspects(source_state: dict, source: dict, data_root: Path):
    """Yield (name, kind, state, arch, runtime_model, committed row, file sha256, result file) for the 85 rows."""
    context = AttackContext(device=torch.device("cpu"), seed=DEFAULT_SEED, data_root=data_root)
    for path in sorted((repo_root() / ROWS).glob("*.json")):
        record = read_result(path)
        row = record["metrics"]["row"]
        config = AttackConfig(**row["config"])
        name = f"{record['task']} {config.label()}"
        kind = "owner (dual W*)" if config.attack == "none" else "attacked"
        if row["attacked"].get("applied_in") == "apply":
            apply_file = row["attacked"]["apply_record"]["file"]
            hits = list((repo_root() / ROWS).glob(f"*_apply/{apply_file}"))
            if len(hits) != 1:
                raise SystemExit(f"{path.name}: apply record {apply_file} found {len(hits)} times")
            _, rec, _, weights = read_apply_record(hits[0], None)
            state = torch.load(weights, map_location="cpu", weights_only=True)
            yield name, kind, state, rec["metrics"]["arch"], None, row, sha256_file(weights), path.name
        else:
            out = apply_attack(config, source_state, source["arch"], context)
            yield name, kind, out.state_dict, out.arch, out.runtime_model, row, None, path.name


def test_accuracy(model: torch.nn.Module, loader) -> dict:
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            correct += int((model(x).argmax(dim=1) == y).sum())
            n += len(y)
    return {"correct": correct, "n": n, "accuracy": correct / n}


def summarise(name: str, kind: str, verdict, accuracy: float | None) -> dict:
    d = verdict.to_dict()
    c = {x["slot"]: x for x in d["checks"]}
    beh, wgt, g = c["behavioral"], c["weight"], d["grade"]["suspect"]
    return {
        "model": name, "class": kind, "test_accuracy": accuracy,
        "record_valid": d["record_valid"], **{f"{slot}_status": c[slot]["status"] for slot in SLOTS},
        "fired": beh["statistic"].get("fired"), "predicted_base_label": beh["statistic"].get("predicted_base_label"),
        "behavioral_p_value": beh["p_value"], "weight_z": wgt["statistic"].get("z"),
        "weight_p_value_bound": wgt["p_value"],
        "technical_evidence_strength": g["technical_evidence_strength"], "combined_p_value": g["combined_p_value"],
        "behavioral_tier": g["watermarks"]["behavioral"]["tier"], "weight_tier": g["watermarks"]["weight"]["tier"],
        "tests_assessed": g["tests_assessed"], "exact_copy": g["exact_copy"], "borderline": g["borderline"],
        "statement": g["statement"], "caveats": g["caveats"],
        "guard_rejected": [x["slot"] for x in d["checks"] if x["guard_rejected"]],
        "secrets_used": d["secrets_used"],
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=repo_root() / "data")
    parser.add_argument("--hub-dir", type=Path, default=repo_root() / "data" / "torch_hub")
    args = parser.parse_args(argv)
    git, started = git_info(), time.perf_counter()

    key, nonce = load_key(DEFAULT_KEY_PATH), load_nonce(DEFAULT_NONCE_PATH)
    secrets = OwnerSecrets(key, derive_signature(key, PROJECT_OWNER_ID).value, nonce)
    record, record_hash = read_record(RECORD_PATH)
    publication, publication_hash = read_publication(ARTIFACT_PATH)
    checks = default_checks(args.data_root)
    source_state, source = load_source("dual", None)
    expected = phase4_expected()
    test_loader = cifar10_loaders(args.data_root, eval_batch_size=500, num_workers=0, download=False)["test"]

    rows, verdicts, problems, third_party, excluded = [], {}, [], [], []

    def run(name, kind, state, arch, runtime, file_hash, accuracy):
        verdict = audit(state, record, publication, trusted_public_key=OWNER_PUBLIC_KEY, checks=checks,
                        owner_secrets=secrets, suspect_label=name, suspect_file_sha256=file_hash,
                        suspect_model=runtime, suspect_arch=arch)
        s = summarise(name, kind, verdict, accuracy)
        rows.append(s)
        verdicts[name] = verdict.to_dict()
        if not (s["record_valid"] and s["commitment_status"] == "passed" and s["zk_proof_status"] == "passed"):
            problems.append(f"{name}: record, commitment or zk_proof not passing")
        if s["guard_rejected"]:
            problems.append(f"{name}: guard rejected {s['guard_rejected']}")
        z = "n/a" if s["weight_z"] is None else f"{s['weight_z']:.2f}"
        print(f"{name:52s} {kind[:12]:12s} fired={s['fired']!s:>4} z={z:>6} grade={s['technical_evidence_strength']}",
              flush=True)
        return s

    # 1-2. The owner's model and the 84 attacked variants.
    for name, kind, state, arch, runtime, row, file_hash, result_file in phase4_suspects(source_state, source,
                                                                                         args.data_root):
        s = run(name, kind, state, arch, runtime, file_hash, row["clean_accuracy"]["accuracy"])
        s["phase4_result_file"] = result_file
        if s["fired"] != row["behavioral"]["fired"]:
            problems.append(f"{name}: fired {s['fired']} vs Phase 4 {row['behavioral']['fired']}")
        if row["weight"]["applicable"] != (s["weight_status"] != "not_applicable"):
            problems.append(f"{name}: weight applicability differs from Phase 4")
        elif row["weight"]["applicable"] and abs(s["weight_z"] - row["weight"]["z"]) > 1e-9:
            problems.append(f"{name}: weight z {s['weight_z']} vs Phase 4 {row['weight']['z']}")
        want = expected[result_file]["grade"]
        same_p = abs(s["combined_p_value"] - want["combined_p_value"]) <= GRADE_P_RTOL * want["combined_p_value"]
        if s["technical_evidence_strength"] != want["technical_evidence_strength"] or not same_p:
            problems.append(f"{name}: grade {s['technical_evidence_strength']} ({s['combined_p_value']}) vs P9.3 "
                            f"{want['technical_evidence_strength']} ({want['combined_p_value']})")
    phase4 = [r for r in rows if r["class"] in ("owner (dual W*)", "attacked")]
    if len(phase4) != 85 or sum(r["class"] == "attacked" for r in phase4) != 84:
        problems.append(f"expected 1 owner row and 84 attacked rows, got {len(phase4)} in all")
    if dict(Counter(r["technical_evidence_strength"] for r in phase4)) != EXPECTED_PHASE4_GRADES:
        problems.append("Phase 4 grade counts differ from P9.3")

    # 3. Unrelated models.
    clean_path = repo_root() / "results" / "p0.5_clean_baseline_W.pt"
    if sha256_file(clean_path) != W_CLEAN_SHA256:
        raise SystemExit("clean W has the wrong hash")
    s = run("clean W (P0.5)", "unrelated: clean W", torch.load(clean_path, map_location="cpu", weights_only=True),
            {"width": 32}, None, W_CLEAN_SHA256, read_result(repo_root() / P0_5_RESULT)["metrics"]["test_accuracy"])
    if s["fired"] != CLEAN_W["fired"] or round(s["weight_z"], 2) != CLEAN_W["z"]:
        problems.append(f"clean W does not reproduce P2.4 / P3.7: {s['fired']}, {s['weight_z']}")

    for seed in FRESH_SEEDS:
        torch.manual_seed(seed)
        run(f"fresh init seed {seed}", "unrelated: fresh init", MainModel(width=32).state_dict(), {"width": 32},
            None, None, None)

    zk_path = repo_root() / "results" / "p0.7_zk_model.pt"
    if sha256_file(zk_path) != ZK_MODEL_SHA256:
        raise SystemExit("zk_model has the wrong hash")
    zk = ZKModel()
    zk.load_state_dict(torch.load(zk_path, map_location="cpu", weights_only=True), strict=True)
    zk_adapter = InputAdapter(zk.eval(), source_mean=CIFAR10_MEAN, source_std=CIFAR10_STD, target_mean=MNIST_MEAN,
                              target_std=MNIST_STD, grayscale=True, size=28).eval()
    run("zk_model (P0.7, MNIST)", "unrelated: zk_model", zk.state_dict(), {"width": 32}, zk_adapter, ZK_MODEL_SHA256,
        None)

    for entry in CATALOG:
        model, info = load_third_party(entry, args.hub_dir)
        adapted = adapter_for(model, CIFAR10_MEAN, CIFAR10_STD)
        acc = test_accuracy(adapted, test_loader)
        info.update({"measured_top1_percent": round(acc["accuracy"] * 100, 2), "test_correct": acc["correct"],
                     "difference_pp": round(acc["accuracy"] * 100 - entry.published_top1, 2)})
        info["included"] = abs(info["difference_pp"]) <= ACCURACY_TOLERANCE_PP
        third_party.append(info)
        print(f"{entry.name}: test {info['measured_top1_percent']}% vs published {entry.published_top1}% "
              f"({info['difference_pp']:+.2f} pp) {'audit' if info['included'] else 'EXCLUDED'}", flush=True)
        if not info["included"]:
            excluded.append(entry.name)
            continue
        s = run(f"third-party {entry.name}", "unrelated: third-party", model.state_dict(), {"width": 32}, adapted,
                info["sha256"], acc["accuracy"])
        s["third_party"] = {k: info[k] for k in ("source_url", "license", "sha256", "published_top1_percent")}

    if problems:
        raise SystemExit("P9.4 integrity checks failed:\n  " + "\n  ".join(problems))

    # Test criteria.
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)
    owner = by_class["owner (dual W*)"][0]
    must_be_none = [r for r in rows if r["class"].startswith("unrelated")]
    criteria = {
        "owner_exact_copy_with_evidence": owner["exact_copy"] is True and owner["technical_evidence_strength"] not in (
            "none", "not assessed"),
        "unrelated_all_graded_none": all(r["technical_evidence_strength"] == "none" for r in must_be_none),
        "unrelated_all_carry_no_exoneration_caveat": all(NO_EVIDENCE_CAVEAT in r["caveats"] for r in must_be_none),
        "unrelated_none_exact_copy": all(r["exact_copy"] is False for r in must_be_none),
        "weight_not_applicable_on_third_party_and_zk_model": all(
            r["weight_status"] == "not_applicable" for r in must_be_none
            if r["class"] in ("unrelated: third-party", "unrelated: zk_model")),
        "weight_assessed_on_clean_w_and_fresh_inits": all(
            r["weight_status"] in ("detected", "not_detected") for r in must_be_none
            if r["class"] in ("unrelated: clean W", "unrelated: fresh init")),
        "third_party_models_audited": sum(r["class"] == "unrelated: third-party" for r in rows),
        "fresh_inits_audited": len(by_class["unrelated: fresh init"]),
    }
    passed = all(v for k, v in criteria.items() if isinstance(v, bool)) and criteria["fresh_inits_audited"] == 20 \
        and criteria["third_party_models_audited"] >= 1
    grade_counts = {cls: dict(Counter(r["technical_evidence_strength"] for r in rs)) for cls, rs in by_class.items()}

    path = write_result(
        "p9.4_auditor_model_classes", DEFAULT_SEED, task="P9.4", git=git,
        duration_seconds=time.perf_counter() - started,
        params={"record_sha256": record_hash, "publication_sha256": publication_hash,
                "trusted_public_key_hex": OWNER_PUBLIC_KEY, "detection_alpha": "1e-6",
                "checks": {slot: getattr(checks[slot], "method") for slot in SLOTS},
                "fresh_init_seeds": list(FRESH_SEEDS), "accuracy_tolerance_pp": ACCURACY_TOLERANCE_PP,
                "third_party": {"hub_repo": HUB_REPO, "hub_commit": HUB_COMMIT, "license": LICENSE,
                                "catalog_bytes": catalog_bytes(), "normalisation_mean": THIRD_PARTY_MEAN,
                                "normalisation_std": THIRD_PARTY_STD, "normalisation_source": NORMALISATION_SOURCE,
                                "weights_committed": False, "vit_excluded": True},
                "zk_model_adapter": "project CIFAR-10 normalisation undone, BT.601 luma, bilinear antialiased "
                                    "resize to 28 x 28, MNIST normalisation"},
        metrics={"passed": passed, "criteria": criteria, "grade_counts_by_class": grade_counts,
                 "third_party_models": third_party, "third_party_excluded": excluded,
                 "weight_test_false_positive_basis": WEIGHT_FPR_BASIS, "table": rows, "verdicts": verdicts},
        notes="The auditor on the owner's model, the 84 Phase 4 attacked variants and unrelated models (clean W, 20 "
              "fresh inits, zk_model, third-party CIFAR-10 models). Every Phase 4 row reproduces its committed "
              "statistics and P9.3 grade. Technical evidence strength only; not legal evidence.")
    text = Path(path).read_text(encoding="utf-8").lower()
    if any(needle in text for needle in secrets.needles()):
        raise SystemExit(f"{path} contains a secret")
    print(json.dumps(criteria, indent=1))
    print(f"grades by class: {grade_counts}")
    print(f"{'PASSED' if passed else 'FAILED'}; wrote {path}; no secret in it")
    return {"passed": passed, "rows": rows, "path": str(path)}


if __name__ == "__main__":
    main()
