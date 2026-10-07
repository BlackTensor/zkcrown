"""Zero-knowledge panel (P9.13): the two proof tracks, built from committed files only.

Track A (Circom and Groth16) comes from the P7.4 to P7.9 records and the
committed proof and verification key. Track B (EZKL) comes from the P8.3 to
P8.6 records and the committed EZKL proof and verification key. Nothing is
proved or verified here. A few facts are recomputed live from the committed
bytes: proof sizes, the Groth16 public signal against the published ``C``, and
the EZKL instance count.

Files whose paths carry task numbers are found through the records that name
them (by recorded path, or by the SHA-256 the record gives), not typed in.

Stdlib only (no Streamlit), so the logic can be tested on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

PUBLICATION = "provenance/commitment.json"
PREFIXES = {
    "ptau": "results/p7.4_ptau__",
    "circuit": "results/p7.5_commitment_circuit__",
    "setup": "results/p7.6_groth16_setup__",
    "proof": "results/p7.7_groth16_proof__",
    "negatives": "results/p7.8_negative_tests__",
    "sizing": "results/p7.9_sizing_estimate__",
    "ezkl_setup": "results/p8.3_ezkl_setup__",
    "ezkl_prove": "results/p8.4_ezkl_prove__",
    "ezkl_verify": "results/p8.5_ezkl_verify__",
    "fidelity": "results/p8.6_ezkl_fidelity__",
}
MACHINE = "this machine, not Colab"
"""Every timing and memory figure on the panel carries this label."""

EZKL_FAMILIES = {
    "output": "One output score changed",
    "input": "One input pixel changed",
    "proof_bits": "One bit flipped in the proof bytes",
    "other_image": "Another image's public values",
    "structure": "Wrong number of public values",
}
SIZING_PARTS = {
    "keystream_sha256": "Key-derived stream (HMAC-SHA-256)",
    "bundle_digest_sha256": "Published trigger digest (SHA-256 over every pixel)",
    "pixel_perturbation": "Pixel perturbation and clipping",
    "dataset_membership": "Base images belong to a committed dataset",
}


@dataclass(frozen=True)
class Sources:
    records: dict[str, dict]
    publication: dict
    groth16_proof: bytes
    groth16_public: list
    groth16_vkey: bytes
    ezkl_proof: bytes
    ezkl_vkey: bytes
    paths: dict[str, str]


def _by_sha256(store, digest: str) -> str:
    """The one manifest path whose recorded SHA-256 is `digest`."""
    (path,) = [name for name, entry in store.files.items() if entry["sha256"] == digest]
    return path


def _path(field: str) -> str:
    """A recorded path field such as ``results/... (committed, SHA-256 checked)`` without its note."""
    return field.split(" ", 1)[0]


def load(store) -> Sources:
    """Every file the panel uses, each verified by the data layer. Raises `DataIntegrityError`."""
    paths = {name: store.latest(prefix) for name, prefix in PREFIXES.items()}
    records = {name: store.read_json(path) for name, path in paths.items()}
    proof_path = _by_sha256(store, records["proof"]["metrics"]["proof_sha256"])
    public_path = proof_path.rsplit("/", 1)[0] + "/public.json"
    vkey_path = _path(records["setup"]["metrics"]["verification_key_committed"])
    ezkl_proof_path = records["ezkl_verify"]["params"]["proof"]["path"]
    ezkl_vkey_path = records["ezkl_verify"]["params"]["vk"]
    paths.update(publication=PUBLICATION, groth16_proof=proof_path, groth16_public=public_path,
                 groth16_vkey=vkey_path, ezkl_proof=ezkl_proof_path, ezkl_vkey=ezkl_vkey_path)
    return Sources(
        records=records, publication=store.read_json(PUBLICATION),
        groth16_proof=store.read_bytes(proof_path), groth16_public=store.read_json(public_path),
        groth16_vkey=store.read_bytes(vkey_path), ezkl_proof=store.read_bytes(ezkl_proof_path),
        ezkl_vkey=store.read_bytes(ezkl_vkey_path), paths=paths)


def field_element(text: str) -> int:
    """A public signal as a field element, so encodings of the same value compare equal (P7.8).

    Plain decimal first (leading zeros allowed), then Python's prefixed forms such as hex.
    """
    try:
        return int(text)
    except ValueError:
        return int(text, 0)


# --- Track A -------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackA:
    statement: str
    constraints: int
    private_inputs: int
    public_inputs: int
    proof_bytes: int
    proof_bytes_recorded: int
    public_signals: list
    public_signal_is_published_c: bool
    vkey_bytes: int
    proving_key_bytes: int
    contributors: int
    zkey_verify: str
    ptau_power: int
    ptau_verify: str
    ptau_accepted_on: str
    prove_seconds: float
    verify_median: float
    verify_min: float
    verify_max: float
    verify_runs: int
    node_baseline_median: float
    prove_peak_bytes: int
    verify_peak_bytes: int
    setup_seconds: float
    setup_peak_bytes: int
    negatives_tried: int
    negatives_accepted: int
    positive_controls: int
    positive_controls_ok: bool
    crashes: int
    malleable: bool


def track_a(s: Sources) -> TrackA:
    circ, setup, proof = s.records["circuit"]["metrics"], s.records["setup"]["metrics"], s.records["proof"]["metrics"]
    neg, ptau = s.records["negatives"]["metrics"], s.records["ptau"]["metrics"]
    published = field_element(s.publication["commitment"]["decimal"])
    crashes = sum(count for fam in neg["families"].values() for reason, count in fam["refused_at"].items()
                  if "crashed" in reason)
    return TrackA(
        statement=s.records["circuit"]["params"]["statement"],
        constraints=circ["r1cs"]["constraints"], private_inputs=circ["r1cs"]["private_inputs"],
        public_inputs=circ["r1cs"]["public_inputs"],
        proof_bytes=len(s.groth16_proof), proof_bytes_recorded=proof["proof_json_bytes"],
        public_signals=s.groth16_public,
        public_signal_is_published_c=[field_element(v) for v in s.groth16_public] == [published],
        vkey_bytes=len(s.groth16_vkey), proving_key_bytes=setup["file_bytes"]["zkey_final"],
        contributors=len(setup["contributions_listed_by_zkey_verify"]), zkey_verify=setup["zkey_verify"],
        ptau_power=ptau["power"], ptau_verify=ptau["snarkjs_powersoftau_verify"], ptau_accepted_on=ptau["accepted_on"],
        prove_seconds=proof["prove_seconds"], verify_median=proof["verify_seconds"]["median"],
        verify_min=proof["verify_seconds"]["min"], verify_max=proof["verify_seconds"]["max"],
        verify_runs=proof["verify_seconds"]["n"], node_baseline_median=proof["node_baseline_seconds"]["median"],
        prove_peak_bytes=proof["prove_peak_working_set_bytes"],
        verify_peak_bytes=proof["verify_peak_working_set_bytes_max"], setup_seconds=setup["setup_seconds_total"],
        setup_peak_bytes=setup["setup_peak_working_set_bytes"],
        negatives_tried=neg["cases_tried"], negatives_accepted=neg["cases_accepted"],
        positive_controls=neg["positive_controls"]["count"], positive_controls_ok=neg["positive_controls"]["all_verified"],
        crashes=crashes, malleable=neg["malleability_neg_a_neg_b_verified"])


@dataclass(frozen=True)
class Family:
    name: str
    tried: int
    accepted: int
    refused: str


def track_a_families(s: Sources) -> list[Family]:
    out = []
    for key, fam in s.records["negatives"]["metrics"]["families"].items():
        name = key.split(" ", 1)[-1]
        refused = "; ".join(f"{count} × {reason}" for reason, count in fam["refused_at"].items())
        out.append(Family(name[:1].upper() + name[1:], fam["tried"], fam["accepted"], refused))
    return out


# --- Track B -------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackB:
    rows: int
    logrows: int
    input_scale: int
    param_scale: int
    check_mode: str
    image: str
    proof_bytes: int
    proof_bytes_recorded: int
    proof_file_bytes: int
    public_instances: int
    vkey_bytes: int
    proving_key_bytes: int
    srs_bytes: int
    setup_seconds: float
    setup_peak_bytes: int
    prove_seconds: float
    prove_peak_bytes: int
    verify_median: float
    verify_min: float
    verify_max: float
    verify_runs: int
    verify_peak_bytes: int
    negatives_tried: int
    negatives_accepted: int
    negatives_error: int
    negatives_false: int
    controls_tried: int
    controls_accepted: int
    fidelity_n: int
    fidelity_agree: int
    fidelity_accuracy_pytorch: float
    fidelity_accuracy_circuit: float
    fidelity_max_diff: float
    fidelity_mean_diff: float
    proved_subset: int
    proved_subset_ok: bool


def _stage(stages: list, name: str) -> dict:
    (stage,) = [s for s in stages if s["stage"] == name]
    return stage


def track_b(s: Sources) -> TrackB:
    setup, prove = s.records["ezkl_setup"]["metrics"], s.records["ezkl_prove"]["metrics"]
    verify, fid = s.records["ezkl_verify"]["metrics"], s.records["fidelity"]["metrics"]
    run = setup["settings_calibrated"]["run_args"]
    proof = json.loads(s.ezkl_proof)
    image = s.records["ezkl_prove"]["params"]["test_image"]
    summary = verify["summary"]
    times = verify["verify_seconds"]
    return TrackB(
        rows=setup["settings_calibrated"]["num_rows"], logrows=setup["logrows"],
        input_scale=run["input_scale"], param_scale=run["param_scale"], check_mode=run["check_mode"],
        image=f"{image['split']}, index {image['index']}",
        proof_bytes=len(bytes.fromhex(proof["hex_proof"].split("x")[-1])), proof_bytes_recorded=prove["proof_bytes"],
        proof_file_bytes=len(s.ezkl_proof), public_instances=sum(len(group) for group in proof["instances"]),
        vkey_bytes=len(s.ezkl_vkey), proving_key_bytes=setup["file_bytes"]["pk"], srs_bytes=setup["file_bytes"]["srs"],
        setup_seconds=_stage(setup["stages"], "setup")["call_seconds"],
        setup_peak_bytes=_stage(setup["stages"], "setup")["peak_working_set_bytes"],
        prove_seconds=_stage(prove["stages"], "prove")["call_seconds"],
        prove_peak_bytes=_stage(prove["stages"], "prove")["peak_working_set_bytes"],
        verify_median=verify["verify_seconds_median"], verify_min=min(times), verify_max=max(times),
        verify_runs=len(times), verify_peak_bytes=max(verify["verify_peak_working_set_bytes"]),
        negatives_tried=summary["negatives_tried"], negatives_accepted=summary["negatives_accepted"],
        negatives_error=summary["negatives_error"], negatives_false=summary["negatives_rejected_false"],
        controls_tried=summary["controls_tried"], controls_accepted=summary["controls_accepted"],
        fidelity_n=fid["n"], fidelity_agree=_prefixed(fid, "top", "_agree"), fidelity_accuracy_pytorch=fid["pytorch_accuracy"],
        fidelity_accuracy_circuit=fid["circuit_accuracy"], fidelity_max_diff=fid["abs_logit_diff"]["max"],
        fidelity_mean_diff=fid["abs_logit_diff"]["mean"], proved_subset=len(fid["proved_subset"]),
        proved_subset_ok=fid["proved_subset_all_verify_and_equal_bulk"])


def track_b_families(s: Sources) -> list[Family]:
    out = []
    for key, fam in s.records["ezkl_verify"]["metrics"]["by_family"].items():
        messages = sorted({m.split("[halo2] ", 1)[-1] for m in fam["error_messages"]})
        refused = f"{fam['error']} × verifier error ({'; '.join(messages)})"
        if fam["rejected_false"]:
            refused += f"; {fam['rejected_false']} × returned False"
        out.append(Family(EZKL_FAMILIES.get(key, key), fam["tried"], fam["accepted"], refused))
    return out


# --- P7.9, blocked -------------------------------------------------------------------------

@dataclass(frozen=True)
class Blocked:
    total: int
    power_needed: int
    ptau_capacity: int
    ptau_power: int
    sha256_share: float
    fits_largest_hermez: bool
    single_block: int
    times_over: float
    parts: list[tuple[str, int]]


def _prefixed(record: dict, prefix: str, suffix: str = "") -> object:
    """The value of the one key with this prefix and suffix (P7.9 keys carry the power they refer to)."""
    (value,) = [v for k, v in record.items() if k.startswith(prefix) and k.endswith(suffix)]
    return value


def blocked(s: Sources) -> Blocked:
    m = s.records["sizing"]["metrics"]
    parts = [(SIZING_PARTS[k], v) for k, v in m["parts"].items() if k in SIZING_PARTS]
    parts.append(("Commitment opening (the Track A circuit)", _prefixed(m["parts"], "commitment_opening")))
    capacity = _prefixed(m, "ptau_", "_capacity")
    return Blocked(
        total=m["total_estimate"], power_needed=m["power_needed"], ptau_capacity=capacity,
        ptau_power=s.records["ptau"]["metrics"]["power"], sha256_share=_prefixed(m, "sha", "_share"),
        fits_largest_hermez=_prefixed(m, "fits_largest_hermez"),
        single_block=m["single_keystream_block_with_shared_states"],
        times_over=m["total_estimate"] / capacity, parts=parts)
