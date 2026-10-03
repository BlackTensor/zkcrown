"""P7.5: the commitment-opening circuit matches the P5.3 layout. Public demo values only."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from src.crypto.commitment import DOMAIN_ELEMENT, INPUT_NAMES
from src.crypto.poseidon import BN254_SCALAR_FIELD
from src.zk.toolchain import toolchain_available

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
try:
    import p7_5_commitment_circuit as script  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "experiments"))

SOURCE = script.CIRCUIT.read_text(encoding="utf-8")
CODE = script.code_without_comments(SOURCE)
needs_toolchain = pytest.mark.skipif(not toolchain_available(), reason="Track A toolchain not installed (zk/README.md)")


def test_domain_literal_equals_host_domain():
    assert script.circuit_domain(SOURCE) == DOMAIN_ELEMENT
    assert DOMAIN_ELEMENT == int.from_bytes(b"zk-crown/commitment/v1", "big")


def test_no_left_arrow_outside_comments():
    assert "<--" not in CODE
    assert script.code_without_comments("a <-- b; // <-- x\n/* <-- */ c") == "a <-- b; \n c"


def test_c_is_the_only_public_input_and_is_constrained():
    assert "component main {public [C]}" in CODE
    assert "C === h.out;" in CODE


def test_poseidon_inputs_follow_the_p53_order():
    order = re.findall(r"h\.inputs\[(\d)\] <== (\w+);", CODE)
    assert [name for _, name in sorted(order)] == ["DOMAIN", *INPUT_NAMES[1:]]
    assert "Poseidon(5)" in CODE


def test_every_private_input_is_range_checked_to_its_width():
    checks = dict((sig, int(n)) for n, sig in re.findall(
        r"component \w+ = Num2Bits\((\d+)\);\s*\w+\.in <== (\w+);", CODE))
    assert checks == {"K_hi": 128, "K_lo": 128, "S": 128, "nonce": 248}
    assert script.BITS == checks


def test_demo_vector_is_public_and_consistent():
    v = json.loads(script.VECTOR.read_text(encoding="utf-8"))
    assert "PUBLIC DEMO" in v["warning"]
    d = script.demo_vector()
    assert script.host_c(d["private"]) == d["C"]


def test_range_cases_cover_both_sides_of_each_bound():
    cases = script.range_cases({k: 1 for k in script.PRIVATE})
    assert len(cases) == 12
    for name, private, accept in cases:
        k = next(p for p in script.PRIVATE if name.startswith(p + "_"))
        v = private[k]
        assert accept == (v < 2 ** script.BITS[k])
        assert v < BN254_SCALAR_FIELD


def test_real_witness_dir_is_gitignored():
    assert script.REAL_WORK_DIR.relative_to(REPO_ROOT).parts[0] == "secrets"


@needs_toolchain
def test_circuit_end_to_end_with_demo_values(tmp_path):
    r = script.run(tmp_path, real=False)
    assert r["info"]["public_inputs"] == 1 and r["info"]["private_inputs"] == 4
    assert r["need"] <= script.PTAU_POWER
    assert all(c["accepted"] == (c["expected"] == "accept") for c in r["ranges"].values())
    assert all(c["refused_by_num2bits"] for c in r["ranges"].values() if c["expected"] == "refuse")
    assert r["wrong_c_refused_on_c_line"]
    assert r["real"] is None
