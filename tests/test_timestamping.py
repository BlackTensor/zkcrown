"""Tests for P5.5: OpenTimestamps proofs and the signed-tag check. No network, no real key."""

from __future__ import annotations

import hashlib
import json
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


def test_script_record_target(script):
    """The record gets its own proof next to it."""
    assert script.TARGETS["record"] == ("provenance/record.json", "provenance/record.json.ots")
    assert script.TARGETS["commitment"] == (script.ARTIFACT_PATH, script.OTS_PATH)


@pytest.mark.skipif(not (REPO_ROOT / "provenance" / "record.json.ots").exists(), reason="no record proof")
@pytest.mark.parametrize("verified", [True, False])
def test_record_target_chain_check_is_recorded_pass_or_fail(script, monkeypatch, tmp_path, verified):
    """`record --target record` chain-checks the record's proof and writes a result even when a block fails."""
    proof = describe_proof(REPO_ROOT / "provenance" / "record.json.ots", REPO_ROOT / "provenance" / "record.json")
    if not proof["bitcoin_attestations"]:
        pytest.skip("the record proof has no Bitcoin attestation yet")
    seen = {}

    def fake(attestations, explorers=(), **_):
        seen["heights"] = sorted({a["height"] for a in attestations})
        blocks = [{"height": h, "verified": verified} for h in seen["heights"]]
        return {"explorers": list(explorers), "blocks": blocks, "all_verified": verified,
                "earliest_verified_block": None}

    monkeypatch.setattr(script, "check_bitcoin_attestations", fake)
    out = script.main(["record", "--target", "record", "--out-dir", str(tmp_path)])
    record = json.loads(open(out["path"], encoding="utf-8").read())
    ots = record["metrics"]["opentimestamps"]
    assert seen["heights"] == sorted({a["height"] for a in proof["bitcoin_attestations"]})
    assert ots["independent_time_evidence"] is verified and ots["chain_check"]["all_verified"] is verified
    assert record["task"] == "P6.2" and record["metrics"]["record_sha256"] == script.P6_2_RECORD_SHA256


@pytest.mark.skipif(not (REPO_ROOT / "provenance" / "record.json").exists(), reason="no committed record")
def test_script_reads_the_committed_record_target(script):
    assert script.read_target("record") == script.P6_2_RECORD_SHA256


def test_script_refuses_a_record_that_is_not_the_p6_2_one(script, monkeypatch):
    monkeypatch.setattr(script, "P6_2_RECORD_SHA256", "00" * 32)
    if not (REPO_ROOT / "provenance" / "record.json").exists():
        pytest.skip("no committed record")
    with pytest.raises(SystemExit):
        script.read_target("record")


# --- the chain check (no network: a fake explorer) ----------------------------

GENESIS_HEADER = bytes.fromhex(
    "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3"
    "888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c")
GENESIS_HASH = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
GENESIS_ROOT = "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"
# Block 969627, the block the P5.5 proof attests to, as served by blockstream.info on 2026-10-03.
B969627_HEADER = bytes.fromhex(
    "0000002490e70f57a93955780d0b50b88e540fe7fcd8b6d3554e000000000000000000002ff417df88d9c3faffba479026050cd1896037d7"
    "0014bdcb9f8f5edd22f70e37fd07c06ac51e0217c2d63618")
B969627_HASH = "000000000000000000000d8e10dd470dd91b1125eaef9ac881f931e3152b426d"
B969627_ROOT = "370ef722dd5e8f9fcbbd1400d7376089d10c05269047bafffac3d988df17f42f"


def test_parse_block_header_known_answers():
    genesis = timestamping.parse_block_header(GENESIS_HEADER)
    assert genesis["block_hash"] == GENESIS_HASH and genesis["merkle_root"] == GENESIS_ROOT
    assert genesis["time"] == 1231006505 and genesis["meets_target"]
    block = timestamping.parse_block_header(B969627_HEADER)
    assert block["block_hash"] == B969627_HASH and block["merkle_root"] == B969627_ROOT and block["meets_target"]


def test_header_without_work_fails_the_target():
    forged = bytearray(B969627_HEADER)
    forged[76:80] = b"\x00\x00\x00\x00"  # a different nonce: same root, but the hash no longer meets the target
    assert not timestamping.parse_block_header(bytes(forged))["meets_target"]
    with pytest.raises(ValueError):
        timestamping.parse_block_header(B969627_HEADER[:79])


def explorer(chain: dict, *, down=(), lie=None):
    """A fake explorer API: `chain` maps height to header bytes. `lie` serves another header for one base."""

    def fetch(url, timeout):
        base = url.rsplit("/block", 1)[0]
        if base in down:
            raise ConnectionError("unreachable")
        if "/block-height/" in url:
            height = int(url.rsplit("/", 1)[1])
            header = lie[1] if lie and lie[0] == base else chain[height]
            return timestamping.parse_block_header(header)["block_hash"]
        block_hash = url.split("/block/")[1].split("/")[0]
        for header in [*chain.values(), *([lie[1]] if lie else [])]:
            if timestamping.parse_block_header(header)["block_hash"] == block_hash:
                return header.hex()
        raise KeyError(block_hash)

    return fetch


EXPLORERS = ("https://e1.test/api", "https://e2.test/api")
ATTESTED = [{"height": 969627, "merkle_root": B969627_ROOT}, {"height": 969627, "merkle_root": B969627_ROOT}]


def check(attestations=ATTESTED, **kwargs):
    return timestamping.check_bitcoin_attestations(attestations, EXPLORERS, fetch=explorer({969627: B969627_HEADER},
                                                                                          **kwargs))


def test_matching_block_verifies_on_both_explorers():
    result = check()
    assert result["all_verified"] and len(result["blocks"]) == 1
    block = result["blocks"][0]
    assert block["block_hash"] == B969627_HASH and block["header_time_utc"] == "2026-10-02T19:37:33+00:00"
    assert result["earliest_verified_block"]["height"] == 969627


def test_wrong_merkle_root_is_not_verified():
    assert not check([{"height": 969627, "merkle_root": "00" * 32}])["all_verified"]


def test_one_explorer_down_is_not_verified():
    result = check(down=(EXPLORERS[1],))
    assert not result["all_verified"] and "error" in result["blocks"][0]["explorers"][EXPLORERS[1]]


def test_one_explorer_serving_another_block_is_not_verified():
    assert not check(lie=(EXPLORERS[0], GENESIS_HEADER))["all_verified"]


def test_a_single_explorer_is_not_enough():
    result = timestamping.check_bitcoin_attestations(ATTESTED, EXPLORERS[:1],
                                                     fetch=explorer({969627: B969627_HEADER}))
    assert not result["all_verified"]


def test_two_roots_for_one_height_and_no_attestations_are_not_verified():
    assert not check([{"height": 969627, "merkle_root": B969627_ROOT},
                      {"height": 969627, "merkle_root": GENESIS_ROOT}])["all_verified"]
    assert not check([])["all_verified"]
