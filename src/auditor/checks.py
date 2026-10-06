"""The five auditor checks (P9.2).

Each check fills one slot of the verdict and reports aggregates only; the
no-secrets guard in the engine enforces that. ``detected`` for the two
watermark tests means rejected at alpha 1e-6, the level fixed in P4.1 and used
throughout Phase 4. The checks are reported separately; grading is P9.3.

- `FingerprintCheck` (P5.1): the suspect's canonical fingerprint against the
  one the record names. ``passed`` means bit for bit the named model.
  ``failed`` means a different set of weights, which every Phase 4 attack
  produces, so on its own it says nothing about theft.
- `BehavioralCheck` (P2.4, P2.8): needs `K`. Regenerates the N = 100 owner
  triggers and targets, requires their bundle digest to equal the trigger set
  commitment in the record, queries the suspect once per trigger, and applies
  the exact P2.8 test. Needs a model that can be queried; otherwise
  ``not_applicable``.
- `WeightCheck` (P3.3, P3.7): needs `K` (`S` is derived from it). Blind
  extraction and the P3.7 bound. ``not_applicable`` when the suspect does not
  carry the owner's carrier layout (a different architecture or width, or
  channels physically removed); channel re-alignment is not implemented.
- `CommitmentCheck`: `C` in the record is well formed and is the `C` of the
  publication the record names. Reports what the OpenTimestamps proof says,
  read offline.
- `ZkProofCheck`: verifies the committed Track A Groth16 proof (P7.7), whose
  only public signal is compared with the record's `C` by value, and the
  committed Track B EZKL proof (P8.4). Any exception from snarkjs or ezkl is a
  rejection (P7.8, P8.5). Neither proof is about the suspect model.

Material derived from `K` (triggers, targets, `P_K`, `S`) is cached in memory
by `OwnerMaterialSource` for the life of the object, keyed by a hash of `K`, so
auditing several suspects derives it once. It never enters a verdict.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.auditor.engine import CheckContext, CheckOutcome
from src.crypto.fingerprint import fingerprint_state_dict
from src.crypto.publication import artifact_sha256, check_commitment_fields, validate_publication
from src.utils.results import repo_root

DETECTION_ALPHA = "1e-6"
P2_3_BUNDLE_SHA256 = "fbd65ec730a3555c5921f13d1a5d4840b6310261ddcaf31d3a724195122baec8"

# Committed ZK artifacts and the SHA-256 their tasks recorded.
GROTH16_VKEY = ("results/zk/p7.6/verification_key.json",
                "784df209f4a68d36ad442fddaad27771bcb6c1900ddf57fe29352e69b04523d1")
GROTH16_PROOF = ("results/zk/p7.7/proof.json", "3faf4c0520c97f6514c8cb84030908d55826f8111f5418c585008bdf00fb6765")
GROTH16_PUBLIC = "results/zk/p7.7/public.json"
EZKL_PROOF = ("results/zk/p8.4/proof.json", "f02b10439632891b85d431ee3cb7ea84bfd728e26614c79b6c89e748f7b8a40d")
EZKL_SETTINGS = ("results/zk/p8.3/settings.json", "f4480ce9bed9864545c0d764879b2c59e17a4d986a40830d4231b09d2633563f")
EZKL_VK = ("results/zk/p8.3/vk.key", "c582aec46db7e6bd4ea6aae5c7675882e58940b56d776d6a7797d623f7da8023")
EZKL_SRS = ("zk/ezkl/srs/kzg18.srs", "d0148475717a2ba269784a178cb0ab617bc77f16c58d4a3cbdfe785b591c7034")
OTS_PROOF = "provenance/commitment.json.ots"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get(obj: Any, *path: str) -> Any:
    for part in path:
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


# --- fingerprint ------------------------------------------------------------------


class FingerprintCheck:
    slot, method, requires_secrets, exception_is_rejection = "fingerprint", "P5.1", (), False

    def run(self, context: CheckContext) -> CheckOutcome:
        named = _get(context.record, "model", "fingerprint")
        if not isinstance(named, dict) or "sha256" not in named:
            return CheckOutcome("not_applicable", reason="the record names no readable model fingerprint")
        suspect = fingerprint_state_dict(context.suspect_state).to_dict()
        counts = ("tensors", "elements", "data_bytes")
        statistic = {"matches": suspect == named, "digest_equal": suspect["sha256"] == named.get("sha256"),
                     **{f"{k}_equal": suspect[k] == named.get(k) for k in counts}}
        if suspect == named:
            return CheckOutcome("passed", statistic,
                                reason="The suspect is bit for bit the model the record names (P5.1 fingerprint).")
        return CheckOutcome("failed", statistic, reason=(
            "The suspect is a different set of weights from the model the record names. Every modification, "
            "including conv-BN fusion and every Phase 4 attack, changes the fingerprint, so this alone says "
            "nothing about theft; the watermark checks address that."))


# --- owner material ---------------------------------------------------------------


class OwnerMaterialSource:
    """Derives and caches the owner's trigger set, targets, `P_K` and `S` from `K`. Secret; hidden ``repr``.

    `arrays` overrides the CIFAR-10 training images, labels and trigger pool
    (for tests). Otherwise they are read from `data_root` on first use.
    """

    def __init__(self, data_root: Path | str | None = None, owner_id: str | None = None,
                 arrays: Callable[[], tuple[np.ndarray, np.ndarray, list[int]]] | None = None):
        from src.watermark.signature import PROJECT_OWNER_ID

        self.data_root = Path(data_root) if data_root is not None else repo_root() / "data"
        self.owner_id = owner_id or PROJECT_OWNER_ID
        self._arrays = arrays
        self._cache: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"OwnerMaterialSource(owner_id={self.owner_id!r}, <cache hidden>)"

    def available(self) -> str | None:
        """None if the data is there, otherwise why not."""
        if self._arrays is not None or (self.data_root / "cifar-10-batches-py").is_dir():
            return None
        return f"CIFAR-10 training data not found under {self.data_root.name}/ (needed to regenerate the triggers)"

    def _load_arrays(self):
        if self._arrays is not None:
            return self._arrays()
        from torchvision.datasets import CIFAR10

        from src.data.cifar10 import cifar10_split_indices

        cifar = CIFAR10(str(self.data_root), train=True, download=False)
        pool, _ = cifar10_split_indices(len(cifar.data))
        return cifar.data, np.asarray(cifar.targets), pool

    def material(self, key: bytes):
        from src.attacks.evaluation import derive_owner_material

        tag = hashlib.sha256(b"zk-crown/auditor/material-cache\0" + key).hexdigest()
        if tag not in self._cache:
            images, labels, pool = self._load_arrays()
            self._cache = {tag: derive_owner_material(key, images, labels, pool, owner_id=self.owner_id)}
        return self._cache[tag]


def _committed_trigger_digest(record: Any) -> tuple[str, str]:
    digest = _get(record, "trigger_set_commitment", "sha256")
    if isinstance(digest, str):
        return digest, "the record's trigger set commitment"
    return P2_3_BUNDLE_SHA256, "the P2.3 bundle digest (the record's trigger set commitment is unreadable)"


# --- behavioral -------------------------------------------------------------------


class BehavioralCheck:
    slot, method, requires_secrets, exception_is_rejection = "behavioral", "P2.4 / P2.8 at 1e-6", ("K",), False

    def __init__(self, source: OwnerMaterialSource, device: str = "cpu"):
        self.source, self.device = source, torch.device(device)

    def run(self, context: CheckContext) -> CheckOutcome:
        from src.data import CIFAR10_MEAN, CIFAR10_STD
        from src.watermark.detection import measure_detection
        from src.watermark.significance import detection_test, detection_threshold

        missing = self.source.available()
        if missing:
            return CheckOutcome("not_run", reason=missing)
        model = context.model()
        if model is None:
            return CheckOutcome("not_applicable", reason=(
                f"the suspect cannot be queried: it does not load into main_model({context.arch_text}) "
                "and no runtime model was supplied"))
        material = self.source.material(context.secret("K"))
        expected, where = _committed_trigger_digest(context.record)
        if material.bundle_digest != expected:
            return CheckOutcome("error", reason=f"K does not regenerate the trigger set named by {where}; "
                                                "refusing to test against triggers that were not committed")
        result = measure_detection(model.to(self.device).eval(), material.triggers.images, material.responses,
                                   mean=CIFAR10_MEAN, std=CIFAR10_STD, device=self.device)
        test = detection_test(result.fired, result.n)
        k_star, _ = detection_threshold(result.n, DETECTION_ALPHA)
        s = result.summary()
        statistic = {k: s[k] for k in ("n", "fired", "wdr", "predicted_base_label", "predicted_other_class",
                                       "target_probability_mean", "target_probability_min")}
        statistic.update({"k_star_at_alpha": k_star, "detection_alpha": float(DETECTION_ALPHA),
                          "null_expected_fired_bound": float(test.summary()["null_expected_fired_bound"]),
                          "p_value_log10": float(test.summary()["p_value_log10"])})
        detected = test.rejects(DETECTION_ALPHA)
        reason = (f"{result.fired} of {result.n} owner triggers gave their keyed target; detection at 1e-6 needs "
                  f"{k_star}. Exact P2.8 test against a model independent of K, triggers regenerated from K and "
                  f"matched to {where}.")
        return CheckOutcome("detected" if detected else "not_detected", statistic, float(test.p_value), "exact",
                            reason)


# --- weight -----------------------------------------------------------------------


class WeightCheck:
    slot, method, requires_secrets, exception_is_rejection = "weight", "P3.3 / P3.7 at 1e-6", ("K",), False

    def __init__(self, source: OwnerMaterialSource):
        self.source = source

    def run(self, context: CheckContext) -> CheckOutcome:
        from src.watermark.weight_extraction import extract_weight_watermark
        from src.watermark.weight_significance import WeightDetectionTest, z_threshold

        missing = self.source.available()
        if missing:
            return CheckOutcome("not_run", reason=missing)
        material = self.source.material(context.secret("K"))
        state = context.state_tensors()
        try:
            material.layout.check(state)
        except (KeyError, ValueError, TypeError) as error:
            return CheckOutcome("not_applicable", reason=(
                f"the owner's carrier layout is not present ({error}). The suspect may be a different architecture "
                "or width, or have channels physically removed. Channel re-alignment is not implemented, so the "
                "weight test cannot be run on it; this is not evidence either way."))
        extraction = extract_weight_watermark(state, material.layout, material.projection, material.signature)
        test = WeightDetectionTest(extraction.correlation)
        statistic = {"correlation": extraction.correlation, "z": test.z, "bit_matches": extraction.bit_matches,
                     "rows": extraction.rows, "dim": extraction.dim, "amplitude": extraction.amplitude,
                     "projected_rms": extraction.projected_rms, "p_value_bound_log10": test.summary()["p_value_bound_log10"],
                     "z_threshold_at_alpha": z_threshold(DETECTION_ALPHA), "detection_alpha": float(DETECTION_ALPHA)}
        detected = test.rejects(DETECTION_ALPHA)
        reason = (f"Blind extraction with K over the owner's carrier: z = {test.z:.2f}, {extraction.bit_matches} of "
                  f"{extraction.rows} bits; detection at 1e-6 needs z >= {z_threshold(DETECTION_ALPHA):.3f}. P3.7 "
                  "bound, valid for any model independent of K. Zero-masked channels keep the layout and are tested.")
        return CheckOutcome("detected" if detected else "not_detected", statistic, float(test.p_value),
                            "upper_bound", reason)


# --- commitment -------------------------------------------------------------------


class CommitmentCheck:
    slot, method, requires_secrets, exception_is_rejection = "commitment", "P5.3 / P5.4 / P5.5", (), False

    def __init__(self, ots_path: Path | str | None = None):
        self.ots_path = Path(ots_path) if ots_path is not None else repo_root() / OTS_PROOF

    def _ots(self, publication: Any) -> dict[str, Any]:
        out = {"ots_proof_for_publication": None, "ots_bitcoin_attestations": None, "ots_earliest_block_height": None}
        try:
            from src.crypto.timestamping import describe_proof

            info = describe_proof(self.ots_path)
            heights = [a["height"] for a in info["bitcoin_attestations"]]
            out.update({"ots_proof_for_publication": info["file_digest"] == artifact_sha256(publication),
                        "ots_bitcoin_attestations": len(heights),
                        "ots_earliest_block_height": min(heights) if heights else None})
        except Exception:  # noqa: BLE001 -- reported as unknown, not a failure of the commitment
            pass
        return out

    def run(self, context: CheckContext) -> CheckOutcome:
        record, publication = context.record, context.publication
        problems, statistic = [], {}
        try:
            check_commitment_fields(_get(record, "watermark_commitment", "commitment"),
                                    _get(record, "watermark_commitment", "scheme"))
            statistic["c_well_formed"] = True
        except (ValueError, TypeError):
            statistic["c_well_formed"] = False
            problems.append("C in the record is not well formed")
        try:
            validate_publication(publication)
            statistic["publication_hash_matches_record"] = (
                artifact_sha256(publication) == _get(record, "commitment_publication", "sha256"))
            statistic["c_equals_published_c"] = (
                _get(record, "watermark_commitment", "commitment") == publication["commitment"])
        except (ValueError, TypeError):
            statistic.update({"publication_hash_matches_record": False, "c_equals_published_c": False})
            problems.append("the publication is malformed")
        if not statistic["publication_hash_matches_record"]:
            problems.append("the publication is not the one the record names")
        if not statistic["c_equals_published_c"]:
            problems.append("the record's C differs from the published C")
        if statistic["publication_hash_matches_record"]:
            statistic.update(self._ots(publication))
        scope = ("This shows only that the record commits to the published C. It does not show that anyone knows "
                 "an opening of C (see zk_proof), it does not tie C to the suspect model or to the triggers, and "
                 "the attestation heights are read from the proof offline, not checked against the chain here "
                 "(P5.5 did that).")
        if problems:
            return CheckOutcome("failed", statistic, reason="; ".join(problems) + ". " + scope)
        return CheckOutcome("passed", statistic, reason="C is well formed (layout v1, BN254) and equals the "
                                                        "published C. " + scope)


# --- zk proofs --------------------------------------------------------------------


def _field_value(signal: Any) -> int:
    """A public signal as an integer: decimal (leading zeros allowed) or 0x hex. Compared by value (P7.8)."""
    if not isinstance(signal, str):
        raise TypeError("a public signal is a string")
    text = signal.strip()
    return int(text[2:], 16) if text.lower().startswith("0x") else int(text, 10)


def _pinned(root: Path, entry: tuple[str, str]) -> Path:
    path = root / entry[0]
    if not path.is_file():
        raise FileNotFoundError(entry[0])
    if _sha256(path) != entry[1]:
        raise ValueError(f"{entry[0]} differs from the committed file")
    return path


class ZkProofCheck:
    slot, method, requires_secrets, exception_is_rejection = "zk_proof", "P7.7 (Groth16) / P8.4 (EZKL)", (), True

    SCOPE = ("Track A (Groth16, P7.7) proves knowledge of an opening of the published C (K, S, nonce in range); its "
             "only public signal is C. It is not about the suspect model, and its single-contributor setup is not "
             "sound against the owner (P7.6). Track B (EZKL, P8.4) proves inference of the MNIST zk_model on one "
             "public image; it concerns zk_model, not this suspect, and nothing about watermarks.")

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else repo_root()

    def run(self, context: CheckContext) -> CheckOutcome:
        from src.zk.toolchain import Toolchain, toolchain_available

        if not toolchain_available():
            return CheckOutcome("not_run", reason="the Track A toolchain (circom, snarkjs, node) is not installed")
        try:
            import ezkl  # noqa: F401
        except ImportError:
            return CheckOutcome("not_run", reason="ezkl is not installed")
        try:
            vkey, proof = _pinned(self.root, GROTH16_VKEY), _pinned(self.root, GROTH16_PROOF)
            public = self.root / GROTH16_PUBLIC
            ezkl_files = [_pinned(self.root, e) for e in (EZKL_PROOF, EZKL_SETTINGS, EZKL_VK, EZKL_SRS)]
        except FileNotFoundError as missing:
            return CheckOutcome("not_run", reason=f"a committed ZK file is missing: {missing}")
        except ValueError as changed:
            return CheckOutcome("failed", {"artifacts_match_committed": False}, reason=f"{changed}. {self.SCOPE}")

        statistic: dict[str, Any] = {"artifacts_match_committed": True}
        problems = []
        record_c = _get(context.record, "watermark_commitment", "commitment", "decimal")
        try:
            signals = json.loads(public.read_text(encoding="utf-8"))
            statistic["groth16_public_signals"] = len(signals) if isinstance(signals, list) else 0
            statistic["groth16_public_signal_equals_record_c"] = bool(
                isinstance(signals, list) and len(signals) == 1 and isinstance(record_c, str)
                and _field_value(signals[0]) == int(record_c, 10))
        except (ValueError, TypeError, OSError):
            statistic["groth16_public_signal_equals_record_c"] = False
        if not statistic["groth16_public_signal_equals_record_c"]:
            problems.append("the Groth16 public signal is not exactly the record's C (compared by value)")
        try:
            with tempfile.TemporaryDirectory() as work:
                statistic["groth16_verified"] = Toolchain(work).verify(vkey, public, proof, "auditor")
        except Exception as error:  # noqa: BLE001 -- snarkjs rejects by raising (P7.8)
            statistic["groth16_verified"] = False
            problems.append(f"snarkjs raised {type(error).__name__}, counted as a rejection")
        if not statistic["groth16_verified"]:
            problems.append("the Groth16 proof did not verify")
        try:
            import ezkl

            ok = ezkl.verify(*(str(p) for p in ezkl_files[:3]), srs_path=str(ezkl_files[3]))
            statistic["ezkl_verified"] = ok is True
        except Exception as error:  # noqa: BLE001 -- ezkl rejects by raising (P8.5)
            statistic["ezkl_verified"] = False
            problems.append(f"ezkl raised {type(error).__name__}, counted as a rejection")
        if not statistic["ezkl_verified"]:
            problems.append("the EZKL proof did not verify")
        if problems:
            return CheckOutcome("failed", statistic, reason="; ".join(dict.fromkeys(problems)) + ". " + self.SCOPE)
        return CheckOutcome("passed", statistic, reason="Both committed proofs verify. " + self.SCOPE)


def default_checks(data_root: Path | str | None = None, device: str = "cpu") -> dict[str, Any]:
    """The P9.2 registry: one check per slot, sharing one owner material cache."""
    source = OwnerMaterialSource(data_root)
    return {"fingerprint": FingerprintCheck(), "behavioral": BehavioralCheck(source, device),
            "weight": WeightCheck(source), "commitment": CommitmentCheck(), "zk_proof": ZkProofCheck()}
