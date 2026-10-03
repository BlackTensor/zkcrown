"""P7.4: the powers-of-tau table, sizing rule and hash-checked fetch."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pytest

import src.zk.ptau as ptau
from src.zk.ptau import PtauEntry, PtauHashMismatch, fetch, read_table, required_power

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
try:
    import p7_4_fetch_ptau as script  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "experiments"))

needs_readme = pytest.mark.skipif(not ptau.SNARKJS_README.exists(), reason="snarkjs not installed (zk/README.md)")


@needs_readme
def test_readme_table_has_every_power_and_pinned_hash():
    table = read_table()
    assert sorted(table) == list(range(8, 29))
    e = table[script.CHOSEN_POWER]
    assert e.blake2b_512 == script.CHOSEN_BLAKE2B_512
    assert e.filename == "powersOfTau28_hez_final_11.ptau"
    assert e.url == "https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_11.ptau"
    assert all(len(x.blake2b_512) == 128 for x in table.values())


@pytest.mark.parametrize("constraints,public,expected", [(1471, 1, 11), (2046, 1, 11), (2047, 1, 12), (517, 1, 10)])
def test_required_power(constraints, public, expected):
    assert required_power(constraints, public) == expected


def _entry(data: bytes) -> PtauEntry:
    return PtauEntry(99, "x", "fake.ptau", "https://example.invalid/fake.ptau", hashlib.blake2b(data, digest_size=64).hexdigest())


def test_fetch_downloads_and_verifies(tmp_path, monkeypatch):
    data = b"powers of tau" * 1000
    monkeypatch.setattr(ptau.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(data))
    path, downloaded = fetch(_entry(data), tmp_path)
    assert downloaded and path.read_bytes() == data
    # second call re-hashes the existing file and does not download
    monkeypatch.setattr(ptau.urllib.request, "urlopen", lambda *a, **k: pytest.fail("should not download"))
    assert fetch(_entry(data), tmp_path) == (path, False)


def test_fetch_refuses_corrupted_download_and_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ptau.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"tampered"))
    with pytest.raises(PtauHashMismatch):
        fetch(_entry(b"genuine"), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_refuses_corrupted_existing_file(tmp_path):
    (tmp_path / "fake.ptau").write_bytes(b"tampered")
    with pytest.raises(PtauHashMismatch):
        fetch(_entry(b"genuine"), tmp_path, download=False)


def test_missing_file_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fetch(_entry(b"x"), tmp_path, download=False)


def test_probe_multiplicity_matches_p53_layout():
    # K_hi, K_lo, S are 128-bit; the nonce is 248-bit; one Poseidon(5).
    assert script.PROBE_MULTIPLICITY == {"poseidon5": 1, "num2bits128": 3, "num2bits248": 1}
    assert "Poseidon(5)" in script.PROBES["poseidon5"]
