"""Tests for P5.5: OpenTimestamps proofs and the signed-tag check. No network, no real key."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest

pytest.importorskip("opentimestamps")

from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation  # noqa: E402
from opentimestamps.core.op import OpAppend, OpSHA256  # noqa: E402
from opentimestamps.core.timestamp import Timestamp  # noqa: E402

from src.crypto import timestamping  # noqa: E402
from src.crypto.timestamping import describe_proof, parse_verify_tag, stamp_file, upgrade_proof  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ALICE = "https://alice.btc.calendar.opentimestamps.org"
BOB = "https://bob.btc.calendar.opentimestamps.org"


class FakeCalendar:
    """Stands in for a calendar server. Records what it was sent."""

    seen: list = []
    confirmed = False

    def __init__(self, url):
        self.url = url

    def submit(self, digest, timeout=None):
        FakeCalendar.seen.append((self.url, bytes(digest)))
        if "down" in self.url:
            raise ConnectionError("unreachable")
        stamp = Timestamp(digest)
        stamp.attestations.add(PendingAttestation(ALICE if "a." in self.url else BOB))
        return stamp

    def get_timestamp(self, commitment, timeout=None):
        if not FakeCalendar.confirmed:
            raise KeyError("Pending confirmation in Bitcoin blockchain")
        stamp = Timestamp(commitment)
        stamp.ops.add(OpAppend(b"\x01" * 4)).ops.add(OpSHA256()).attestations.add(BitcoinBlockHeaderAttestation(900000))
        return stamp


@pytest.fixture(autouse=True)
def reset():
    FakeCalendar.seen, FakeCalendar.confirmed = [], False


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "commitment.json"
    path.write_bytes(b'{"public": "artifact"}\n')
    return path


def stamp(artifact, calendars=("https://a.test", "https://b.test")):
    return stamp_file(artifact, calendars, calendar_factory=FakeCalendar)


# --- stamping -----------------------------------------------------------------


def test_stamp_writes_a_pending_proof_for_the_file(artifact):
    outcome = stamp(artifact)
    ots = Path(outcome["ots_path"])
    assert ots.name == "commitment.json.ots" and outcome["calendars_accepted"] == ["https://a.test", "https://b.test"]
    proof = describe_proof(ots, artifact)
    assert proof["file_digest"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert proof["matches_file"] is True and proof["status"] == "pending" and proof["hash_op"] == "OpSHA256"
    assert proof["pending_calendars"] == [ALICE, BOB] and proof["bitcoin_attestations"] == []
    assert proof["ots_sha256"] == hashlib.sha256(ots.read_bytes()).hexdigest()


def test_calendars_never_see_the_files_own_hash(artifact):
    stamp(artifact)
    digest = hashlib.sha256(artifact.read_bytes()).digest()
    sent = {d for _, d in FakeCalendar.seen}
    assert len(sent) == 1 and digest not in sent and len(next(iter(sent))) == 32  # one nonced digest, to all


def test_proof_is_never_replaced(artifact):
    ots = Path(stamp(artifact)["ots_path"])
    before = ots.read_bytes()
    with pytest.raises(FileExistsError):
        stamp(artifact)
    assert ots.read_bytes() == before


def test_stamp_needs_two_calendars(artifact):
    with pytest.raises(RuntimeError, match="only 1 calendar"):
        stamp(artifact, ("https://a.test", "https://down.test"))
    assert not artifact.with_name("commitment.json.ots").exists()
    outcome = stamp(artifact, ("https://a.test", "https://down.test", "https://b.test"))
    assert list(outcome["calendars_failed"]) == ["https://down.test"]


def test_a_proof_does_not_match_a_changed_file(artifact):
    ots = Path(stamp(artifact)["ots_path"])
    artifact.write_bytes(artifact.read_bytes() + b" ")
    assert describe_proof(ots, artifact)["matches_file"] is False
    assert describe_proof(ots)["matches_file"] is None


# --- upgrading ----------------------------------------------------------------


def test_upgrade_before_confirmation_changes_nothing(artifact):
    ots = Path(stamp(artifact)["ots_path"])
    before = ots.read_bytes()
    outcome = upgrade_proof(ots, calendar_factory=FakeCalendar)
    assert outcome["changed"] is False and outcome["status"] == "pending"
    assert all(v.startswith("not ready") for v in outcome["calendars"].values()) and ots.read_bytes() == before


def test_upgrade_after_confirmation_adds_a_bitcoin_attestation(artifact):
    ots = Path(stamp(artifact)["ots_path"])
    FakeCalendar.confirmed = True
    outcome = upgrade_proof(ots, calendar_factory=FakeCalendar)
    assert outcome["changed"] is True and outcome["status"] == "bitcoin-attested (unverified)"
    proof = describe_proof(ots, artifact)
    assert proof["matches_file"] is True and {a["height"] for a in proof["bitcoin_attestations"]} == {900000}
    assert all(len(a["merkle_root"]) == 64 for a in proof["bitcoin_attestations"])


def test_upgrade_only_contacts_whitelisted_calendars(artifact, monkeypatch):
    class Evil(FakeCalendar):
        def submit(self, digest, timeout=None):
            stamp = Timestamp(digest)
            stamp.attestations.add(PendingAttestation("https://evil.example.com"))
            return stamp

    ots = Path(stamp_file(artifact, ("https://a.test", "https://b.test"), calendar_factory=Evil)["ots_path"])
    FakeCalendar.confirmed = True
    outcome = upgrade_proof(ots, calendar_factory=FakeCalendar)
    assert outcome["calendars"] == {"https://evil.example.com": "skipped: not on the calendar whitelist"}
    assert outcome["changed"] is False


# --- the signed tag -----------------------------------------------------------

FPR = "C7301BA7D92FC2A65257BFC2A759F8EC04BF66E7"
GOOD = (
    "[GNUPG:] NEWSIG\n"
    f"[GNUPG:] GOODSIG A759F8EC04BF66E7 owner <o@example.com>\n"
    f"[GNUPG:] VALIDSIG {FPR} 2026-10-02 1790964000 0 4 0 22 10 00 {FPR}\n"
    "[GNUPG:] TRUST_ULTIMATE 0 pgp\n"
)


def test_parse_good_signature():
    status = parse_verify_tag(GOOD)
    assert status == {"good": True, "fingerprint": FPR, "signed_date": "2026-10-02", "signed_unix": 1790964000}


@pytest.mark.parametrize("output", [
    "",
    "[GNUPG:] BADSIG A759F8EC04BF66E7 owner\n",
    "[GNUPG:] ERRSIG A759F8EC04BF66E7 22 10 00 1790964000 9 -\n[GNUPG:] NO_PUBKEY A759F8EC04BF66E7\n",
    GOOD.replace("GOODSIG", "EXPKEYSIG"),
    GOOD + "[GNUPG:] REVKEYSIG A759F8EC04BF66E7 owner\n",
    GOOD.replace("[GNUPG:] VALIDSIG", "[GNUPG:] SOMETHING"),
    "error: no signature found\n",
])
def test_parse_refuses_anything_but_a_good_signature(output):
    assert parse_verify_tag(output)["good"] is False


# --- the script ---------------------------------------------------------------


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p5_5_timestamp")


def test_script_constants(script):
    assert script.OTS_PATH == "provenance/commitment.json.ots" and script.PUBLIC_KEY_PATH.startswith("provenance/")
    assert len(script.OWNER_KEY_FINGERPRINT) == 40 and script.TAG == "provenance-commitment-v1"
    assert timestamping.MIN_CALENDARS == 2 and len(timestamping.DEFAULT_CALENDARS) == 3


def test_unsigned_and_missing_tags_are_not_good(script, tmp_path):
    import subprocess

    def run(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    run("init", "-q")
    (tmp_path / "f.txt").write_text("x")
    run("add", "f.txt")
    run("-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "c")
    run("-c", "user.name=t", "-c", "user.email=t@example.com", "tag", "-a", "unsigned", "-m", "no signature")
    unsigned = script.verify_tag("unsigned", cwd=tmp_path)
    assert unsigned["good"] is False and unsigned["object_type"] == "tag" and unsigned["verify_exit_code"] != 0
    assert script.verify_tag("absent", cwd=tmp_path)["good"] is False
    assert script.tagged_blob_sha256("unsigned", "f.txt", cwd=tmp_path) == hashlib.sha256(b"x").hexdigest()
    assert script.tagged_blob_sha256("unsigned", "missing.txt", cwd=tmp_path) is None


def test_committed_proof_is_for_the_committed_artifact():
    artifact, ots = REPO_ROOT / "provenance" / "commitment.json", REPO_ROOT / "provenance" / "commitment.json.ots"
    if not ots.exists():
        pytest.skip("P5.5 has not stamped the artifact")
    proof = describe_proof(ots, artifact)
    assert proof["matches_file"] is True and proof["status"] != "no attestation"
