"""P7.6: wrapper timeouts, entropy redaction, peak memory, and the committed verification key."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.zk import toolchain
from src.zk.toolchain import DEFAULT_TIMEOUT_SECONDS, SETUP_TIMEOUT_SECONDS, Toolchain, _redact, toolchain_available

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
try:
    import p7_6_groth16_setup as setup  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "experiments"))

VKEY = REPO_ROOT / "results" / "zk" / "p7.6" / "verification_key.json"
needs_toolchain = pytest.mark.skipif(not toolchain_available(), reason="Track A toolchain not installed (zk/README.md)")


def test_setup_timeout_is_raised_above_the_default():
    assert DEFAULT_TIMEOUT_SECONDS == 1800
    assert SETUP_TIMEOUT_SECONDS == 7200


def test_redact_hides_entropy_only():
    argv = ["node", "cli", "zkey", "contribute", "a", "b", "--name=x", "-e=deadbeef"]
    assert _redact(argv) == ["node", "cli", "zkey", "contribute", "a", "b", "--name=x", "-e=<redacted>"]


def test_entropy_is_fresh_each_call():
    a, b = toolchain._entropy(), toolchain._entropy()
    assert a != b and len(a) == 64


def test_contribution_hash_parsing():
    out = "[INFO]  snarkJS: Circuit Hash: \n...\n[INFO]  snarkJS: Contribution Hash: \n\t\t1a2b3c4d 5e6f7a8b\n\t\t9c0d1e2f 3a4b5c6d\n"
    assert setup.contribution_hash(out) == "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d"
    assert setup.contribution_hash("nothing") is None


def test_check_vkey_rejects_wrong_shape():
    good = {"protocol": "groth16", "curve": "bn128", "nPublic": 1, "IC": [[], []]}
    setup.check_vkey(good)
    for bad in [{**good, "protocol": "plonk"}, {**good, "nPublic": 2}, {**good, "IC": [[]]}]:
        with pytest.raises(SystemExit):
            setup.check_vkey(bad)


@needs_toolchain
def test_timeout_kills_the_command(tmp_path, monkeypatch):
    import subprocess

    tc = Toolchain(tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        tc._run("sleep", [tc.node, "-e", "setTimeout(()=>{}, 20000)"], timeout=1)


@needs_toolchain
@pytest.mark.skipif(sys.platform != "win32", reason="peak memory is read from Windows only")
def test_peak_memory_is_measured_on_windows(tmp_path):
    step = Toolchain(tmp_path)._run("node_alloc", [toolchain.shutil.which("node"), "-e",
                                                   "const b = Buffer.alloc(64*1024*1024, 1); console.log(b[0])"])
    assert step.peak_working_set_bytes >= 64 * 2**20
    assert step.peak_private_bytes >= 64 * 2**20


@needs_toolchain
def test_recorded_argv_never_holds_entropy(tmp_path):
    tc = Toolchain(tmp_path)
    step = tc._run("echo", [tc.node, "-e", "console.log(process.argv.join(' '))", "--", "-e=abc123"])
    assert "abc123" in step.output  # the tool did receive it
    assert "-e=<redacted>" in step.argv and not any("abc123" in a for a in step.argv)


@pytest.mark.skipif(not VKEY.exists(), reason="P7.6 verification key not committed yet")
def test_committed_verification_key_shape():
    vkey = json.loads(VKEY.read_text(encoding="utf-8"))
    setup.check_vkey(vkey)


@needs_toolchain
@pytest.mark.skipif(not (VKEY.exists() and setup.ZKEY_PATH.exists()), reason="P7.6 keys not present locally")
def test_committed_vkey_is_exported_from_the_local_zkey(tmp_path):
    tc = Toolchain(tmp_path)
    out = tmp_path / "vk.json"
    tc.snarkjs("export", "zkey", "export", "verificationkey", str(setup.ZKEY_PATH), str(out))
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(VKEY.read_text(encoding="utf-8"))


def test_script_refuses_to_replace_existing_keys(tmp_path, monkeypatch):
    existing = tmp_path / "k.zkey"
    existing.write_bytes(b"x")
    monkeypatch.setattr(setup, "ZKEY_PATH", existing)
    with pytest.raises(SystemExit, match="refusing to replace"):
        setup.run(tmp_path / "work")
