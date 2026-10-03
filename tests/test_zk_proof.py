"""P7.7: the real Groth16 proof of the commitment opening."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.crypto.publication import ARTIFACT_PATH, read_publication
from src.zk.toolchain import Toolchain, toolchain_available

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
try:
    import p7_7_prove_commitment as prove  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "experiments"))

PROOF = prove.OUT_DIR / "proof.json"
PUBLIC = prove.OUT_DIR / "public.json"
needs_toolchain = pytest.mark.skipif(not toolchain_available(), reason="Track A toolchain not installed (zk/README.md)")
needs_proof = pytest.mark.skipif(not PROOF.exists(), reason="P7.7 proof not committed yet")
needs_secrets = pytest.mark.skipif(not (REPO_ROOT / "secrets" / "K.bin").exists()
                                   or not (REPO_ROOT / "secrets" / "commitment_nonce.bin").exists(),
                                   reason="owner secrets not present")


def published_c() -> int:
    return int(read_publication(ARTIFACT_PATH)[0]["commitment"]["decimal"])


def test_public_signals_must_be_exactly_c():
    prove.check_public_signals(["7"], 7)
    for bad in [["7", "1"], ["8"], [], [7]]:
        with pytest.raises(SystemExit):
            prove.check_public_signals(bad, 7)


def test_find_secrets_catches_decimal_and_hex(tmp_path):
    value = 0xABCDEF0123456789ABCDEF0123456789
    needles = [str(value), format(value, "x")]
    clean, dec, hexed = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    clean.write_text('{"x": "1"}', encoding="utf-8")
    dec.write_text(f'{{"x": "{value}"}}', encoding="utf-8")
    hexed.write_text(f'{{"x": "0x{format(value, "X")}"}}', encoding="utf-8")
    assert prove.find_secrets([clean, dec, hexed], needles) == ["b.json", "c.json"]


def test_script_refuses_to_replace_the_proof(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "proof.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(prove, "OUT_DIR", out)
    with pytest.raises(SystemExit, match="refusing to replace"):
        prove.run(tmp_path / "work", 1)


@needs_proof
def test_committed_public_signals_are_only_the_published_c():
    assert json.loads(PUBLIC.read_text(encoding="utf-8")) == [str(published_c())]


@needs_proof
def test_committed_proof_shape():
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    assert proof["protocol"] == "groth16" and proof["curve"] == "bn128"
    assert set(proof) == {"pi_a", "pi_b", "pi_c", "protocol", "curve"}


@needs_toolchain
@needs_proof
def test_committed_proof_verifies_and_fails_against_c_plus_1(tmp_path):
    tc = Toolchain(tmp_path)
    assert tc.verify(prove.VKEY, PUBLIC, PROOF, "committed")
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps([str(published_c() + 1)]), encoding="utf-8")
    assert not tc.verify(prove.VKEY, wrong, PROOF, "wrong")


@needs_proof
@needs_secrets
def test_no_secret_in_committed_p7_7_files(monkeypatch):
    import glob

    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))

    records = [Path(p) for p in glob.glob(str(REPO_ROOT / "results" / "p7.7_groth16_proof__*.json"))]
    assert prove.find_secrets([PROOF, PUBLIC, *records], prove.secret_needles()) == []


def test_secret_work_dir_is_gitignored_and_absent():
    assert prove.SECRET_WORK_DIR.parent.name == "secrets"
    assert not prove.SECRET_WORK_DIR.exists()
