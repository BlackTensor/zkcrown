"""P7.3: the Track A toolchain wrapper and the toy Poseidon-preimage loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.crypto.poseidon import BN254_SCALAR_FIELD, poseidon
from src.zk.toolchain import CIRCUITS_DIR, Toolchain, _decimal, toolchain_available

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
try:
    import p7_3_toy_poseidon_proof as toy  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "experiments"))

TOY_RESULTS = REPO_ROOT / "results" / "zk" / "p7.3_toy"
needs_toolchain = pytest.mark.skipif(not toolchain_available(), reason="Track A toolchain not installed (zk/README.md)")


def test_decimal_encodes_field_elements_as_strings():
    assert _decimal({"a": [1, 2], "b": 3}) == {"a": ["1", "2"], "b": "3"}
    assert _decimal(BN254_SCALAR_FIELD - 1) == str(BN254_SCALAR_FIELD - 1)
    with pytest.raises(TypeError):
        _decimal(True)


def test_demo_preimage_is_fixed_and_in_field():
    a = toy.demo_preimage()
    assert a == toy.demo_preimage()
    assert len(a) == 2 and a[0] != a[1]
    assert all(0 <= x < 2**248 for x in a)


@pytest.mark.parametrize("constraints,expected", [(1, 2), (517, 10), (1022, 10), (1023, 11)])
def test_ptau_power_covers_constraints_and_public_inputs(constraints, expected):
    assert toy.ptau_power({"constraints": constraints, "public_inputs": 1, "outputs": 0}) == expected


def test_circuit_constrains_the_public_hash():
    src = (CIRCUITS_DIR / "toy_poseidon_preimage.circom").read_text(encoding="utf-8")
    assert "hash === h.out;" in src
    assert "public [hash]" in src
    assert "<--" not in src


@needs_toolchain
@pytest.mark.skipif(not (TOY_RESULTS / "proof.json").exists(), reason="committed toy proof not present")
def test_committed_toy_proof_verifies_and_matches_host_poseidon(tmp_path):
    public = json.loads((TOY_RESULTS / "public.json").read_text(encoding="utf-8"))
    assert public == [str(poseidon(toy.demo_preimage()))]
    tc = Toolchain(tmp_path)
    assert tc.verify(TOY_RESULTS / "verification_key.json", TOY_RESULTS / "public.json", TOY_RESULTS / "proof.json",
                     "committed")
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps([str(int(public[0]) + 1)]), encoding="utf-8")
    assert not tc.verify(TOY_RESULTS / "verification_key.json", wrong, TOY_RESULTS / "proof.json", "wrong")


@needs_toolchain
def test_full_toy_loop_end_to_end(tmp_path):
    r = toy.run(tmp_path)
    assert r["verified"] is True
    assert r["wrong_hash_witness_exit"] != 0 and r["wrong_hash_failed_on_constraint"]
    assert r["wrong_public_verified"] is False
    assert r["hash"] == poseidon(r["preimage"])
    assert r["info"]["public_inputs"] == 1 and r["info"]["private_inputs"] == 2
