"""P7.8: helpers of the proof-level negative tests, and the committed record."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

from src.crypto.poseidon import BN254_SCALAR_FIELD as P
from src.zk.toolchain import toolchain_available

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
try:
    import p7_8_negative_tests as neg  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "experiments"))

RECORDS = sorted(glob.glob(str(REPO_ROOT / "results" / "p7.8_negative_tests__*.json")))
needs_toolchain = pytest.mark.skipif(not toolchain_available(), reason="Track A toolchain not installed (zk/README.md)")


def test_script_reads_no_owner_secret():
    src = (REPO_ROOT / "experiments" / "p7_8_negative_tests.py").read_text(encoding="utf-8")
    for forbidden in ("K.bin", "commitment_nonce", "load_key", "load_nonce", "derive_signature", "secrets"):
        assert forbidden not in src


def test_derived_values_are_fixed_and_in_range():
    for bits in (128, 248, 253):
        vals = [neg.derived("t", i, bits) for i in range(20)]
        assert vals == [neg.derived("t", i, bits) for i in range(20)]
        assert all(0 <= v < 2**bits for v in vals) and len(set(vals)) == 20
    assert 2**253 < P


def test_demo_vector_is_consistent():
    private, c = neg.demo()
    assert neg.host_c(private) == c


def test_wtns_round_trip(tmp_path):
    n8 = 32
    values = [1, 5, P - 1, 0]
    header = (b"wtns" + (2).to_bytes(4, "little") + (2).to_bytes(4, "little")
              + (1).to_bytes(4, "little") + (4 + n8 + 4).to_bytes(8, "little")
              + n8.to_bytes(4, "little") + P.to_bytes(n8, "little") + len(values).to_bytes(4, "little")
              + (2).to_bytes(4, "little") + (n8 * len(values)).to_bytes(8, "little"))
    f = tmp_path / "w.wtns"
    neg.write_wtns(f, header, n8, values)
    h, n, got = neg.read_wtns(f)
    assert (h, n, got) == (header, n8, values)
    neg.write_wtns(f, header, n8, [P + 3, 0, 0, 0])  # reduced mod p on write
    assert neg.read_wtns(f)[2][0] == 3


def test_wire_index_parses_sym(tmp_path):
    sym = tmp_path / "c.sym"
    sym.write_text("1,1,76,main.C\n2,2,76,main.K_hi\n6,-1,75,main.h.out\n", encoding="utf-8")
    assert neg.wire_index(sym) == {"main.C": 1, "main.K_hi": 2, "main.h.out": -1}


@pytest.mark.skipif(not RECORDS, reason="P7.8 record not committed yet")
def test_committed_record_has_zero_accepted_negatives():
    m = json.loads(Path(RECORDS[-1]).read_text(encoding="utf-8"))["metrics"]
    assert m["cases_accepted"] == 0 and m["cases_tried"] == len(m["cases"]) > 0
    assert not any(c["accepted"] for c in m["cases"])
    assert m["positive_controls"]["all_verified"]


@needs_toolchain
@pytest.mark.skipif(not (neg.OUT_DIR / "demo_proof.json").exists(), reason="P7.8 demo proof not committed yet")
def test_committed_demo_proof_verifies_only_against_demo_c(tmp_path):
    from src.zk.toolchain import Toolchain

    _, c = neg.demo()
    tc = Toolchain(tmp_path)
    assert json.loads((neg.OUT_DIR / "demo_public.json").read_text(encoding="utf-8")) == [str(c)]
    assert tc.verify(neg.VKEY, neg.OUT_DIR / "demo_public.json", neg.OUT_DIR / "demo_proof.json", "demo")
    wrong = tmp_path / "w.json"
    wrong.write_text(json.dumps([str(c + 1)]), encoding="utf-8")
    assert not tc.verify(neg.VKEY, wrong, neg.OUT_DIR / "demo_proof.json", "wrong")


def test_verify_reason_classifies_snarkjs_output():
    assert neg.verify_reason("[ERROR] snarkJS: Invalid proof").startswith("invalid proof")
    assert neg.verify_reason("[ERROR] snarkJS: Proof commitments are not valid.").startswith("proof points")
    assert neg.verify_reason("[ERROR] snarkJS: Public inputs are not valid.").startswith("public inputs")
    assert neg.verify_reason("TypeError: Cannot read properties").startswith("verifier crashed")
    assert neg.verify_reason("something else") == "verifier error (other)"
