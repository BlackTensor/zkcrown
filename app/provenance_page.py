"""Provenance and theft-timeline page (P9.12).

Shows the published commitment, its Bitcoin timestamp, the two signatures, and
the P6.4 theft simulation with the thief's backdated counter-claim. Every value
comes from a committed file through the data layer (`app.provenance_logic`).
Nothing here reads a secret: the hosted app has none, and the committed files
contain none.
"""

from __future__ import annotations

import html

import streamlit as st

from app import provenance_logic as logic
from app import ui
from app.data import DataIntegrityError

KIND_LABELS = {"bitcoin": "Bitcoin-attested", "self": "Self-asserted", "run": "Simulation clock"}
PARTY_LABELS = {"owner": "Owner", "thief": "Thief", "simulation": "Simulation", "bitcoin": "Third party"}


def _e(value: object) -> str:
    return html.escape(str(value))


def _section(title: str, lede: str) -> None:
    st.html(f'<h2 class="zk-section">{_e(title)}</h2><p class="zk-section-lede">{_e(lede)}</p>')


def _field(label: str, value: str, mono: bool = True, note: str = "") -> str:
    cls = "zk-pv-value zk-mono" if mono else "zk-pv-value"
    extra = f'<div class="zk-pv-note">{_e(note)}</div>' if note else ""
    return f'<div class="zk-pv-field"><div class="zk-pv-label">{_e(label)}</div><div class="{cls}">{_e(value)}</div>{extra}</div>'


def _card(title: str, chip: str, body: str, tone: str = "") -> str:
    return (f'<article class="zk-pv-card {tone}"><div class="zk-pv-head"><span class="zk-pv-title">{_e(title)}</span>'
            f'<span class="zk-chip">{_e(chip)}</span></div>{body}</article>')


def _open_points(points: logic.OpenPoints) -> None:
    proof = (f"**The provenance record's own Bitcoin proof is still {points.record_proof_status}.** "
             f"Its OpenTimestamps proof holds {points.record_calendars} calendar promises and "
             f"{points.record_bitcoin_attestations} Bitcoin attestations, so the record "
             "(and the trigger set commitment and signing key it names) has no independent time yet. "
             "Only the commitment publication is Bitcoin-attested.")
    if not points.record_proof_is_the_described_one:
        proof += " The committed proof file differs from the one the status record describes; its state is not known here."
    tag = (f"**The signed git tag `{points.tag_name}` holds the older, pending proof** of the commitment, not the "
           "upgraded one checked against Bitcoin. Both cover the same file. The tag's date is the signer's own clock, "
           + ("and the tag has not been pushed anywhere." if not points.tag_pushed else "and it has been pushed."))
    key = ("**The “trusted” key in the recorded runs came from this repository** (recorded source: "
           f"“{points.trusted_key_source}”). A forged claim fails against that key, which shows the check works; it "
           "does not show the key belongs to the owner. That trust has to come from outside the record.")
    if not points.all_runs_used_that_key:
        key += " The recorded runs do not all name the same key."
    st.warning("\n\n".join((proof, tag, key)), icon=":material/report:")


def _commitment(s: logic.Sources, checks: list[logic.Check]) -> None:
    _section("The published commitment",
             "The owner committed to the secret key before handing the model over. The commitment hides the key; "
             "the file below is what was published.")
    pub = s.publication
    c, fp = pub["commitment"], pub["model_fingerprint"]
    left = "".join((
        _field("Commitment C (decimal)", c["decimal"]),
        _field("Commitment C (hex)", c["hex"]),
        _field("Hash and inputs", f"{pub['commitment_scheme']['hash']} over {', '.join(pub['commitment_scheme']['inputs'])}",
               note="Only C is public. The key, signature S and nonce are not in any committed file."),
    ))
    right = "".join((
        _field("Model", pub["model"], mono=False),
        _field("Model fingerprint (SHA-256)", fp["sha256"],
               note=f"{fp['tensors']:,} tensors, {fp['elements']:,} elements. Matches only an exact copy."),
        _field("Owner id", pub["owner_id"]),
        _field("Date written in the file", logic.show_time(pub["created_utc"]), mono=False,
               note="Self-asserted by the owner's clock. The Bitcoin block below is the independent time."),
        _field("Publication file SHA-256", s.publication_sha256),
    ))
    a, b = st.columns([1, 1], gap="medium")
    a.html(_card("Commitment", pub["commitment"]["version"], left))
    b.html(_card("What it names", pub["schema"], right))
    rows = "".join(f'<li class="{"zk-ok" if ch.ok else "zk-bad"}"><b>{"✓" if ch.ok else "✗"} {_e(ch.label)}</b>'
                   f'<span>{_e(ch.detail)}</span></li>' for ch in checks)
    st.html(f'<div class="zk-pv-checks"><div class="zk-pv-label">Checked live on this page, from the committed '
            f'bytes</div><ul>{rows}</ul></div>')
    if not all(ch.ok for ch in checks):
        st.error("A consistency check failed. The files on this page do not agree with each other.",
                 icon=":material/gpp_bad:")


def _bitcoin(s: logic.Sources, blocks: list[logic.Block]) -> None:
    _section("Independent time: Bitcoin",
             "The publication's SHA-256 was submitted to OpenTimestamps calendars and later anchored in Bitcoin. "
             "Each block below was looked up on two block explorers and its header checked locally.")
    first = blocks[0]
    tiles = []
    for b in blocks:
        status = "header hashes to the block, meets its work target, Merkle root matches" if b.verified else "NOT verified"
        tiles.append(
            f'<article class="zk-tile{"" if b is first else " zk-tile-muted"}">'
            f'<div class="zk-tile-label">{"Earliest attestation" if b is first else "Later attestation"} · '
            f'{b.calendars} calendar{"s" if b.calendars != 1 else ""}</div>'
            f'<div class="zk-tile-value">Block {b.height:,}</div>'
            f'<p class="zk-tile-detail">Header time {_e(logic.show_time(b.header_time))}<br>'
            f'<span class="zk-mono zk-wrap">{_e(b.block_hash)}</span><br>{_e(status)} on {_e(" and ".join(b.explorers))}.</p>'
            f'<div class="zk-tile-source">{_e(s.paths["timestamp"].rsplit("/", 1)[-1].split("__")[0])}</div></article>')
    st.html(f'<div class="zk-tiles zk-tiles-two">{"".join(tiles)}</div>')
    st.caption(f"So the publication existed by block {first.height:,}, around {logic.show_time(first.header_time)}. "
               "Header times are set by miners and are loose by hours. The check trusts the two explorers' view of "
               "which block sits at each height; it is not a full node.")


def _signatures(s: logic.Sources, points: logic.OpenPoints) -> None:
    _section("Signatures",
             "Two separate keys vouch for two separate things. Both checks below need libraries this hosted app does "
             "not have, so they are replayed from the recorded runs.")
    rec, m2, m3 = s.record, s.signing["metrics"], s.verifier["metrics"]
    tamper = m2["tampering"]
    record_body = "".join((
        _field("Signs", f"{logic.RECORD} (C, model fingerprint, trigger set commitment, owner key)", mono=False),
        _field("Algorithm and public key", f"{rec['signature']['algorithm']} · {rec['owner']['public_key']['hex']}"),
        _field("Signature", rec["signature"]["hex"]),
        _field("Trigger set commitment", f"{rec['trigger_set_commitment']['sha256']} "
               f"({rec['trigger_set_commitment']['triggers']} triggers)",
               note="A digest of the real trigger set. The triggers themselves are never published."),
        _field("Result in the recorded runs",
               f"{'valid' if m2['signature_verifies'] and m3['genuine_verdict']['signature_valid'] else 'INVALID'}; "
               f"{tamper['field_changes_tried'] - tamper['field_changes_accepted']} of {tamper['field_changes_tried']} "
               f"field changes and {m3['signature_bit_flips_tried'] - m3['signature_bit_flips_accepted']:,} of "
               f"{m3['signature_bit_flips_tried']:,} signature bit flips rejected", mono=False,
               note="Replayed from P6.2 and P6.3. The signature shows the record was not changed after signing; it "
                    "does not show who holds the key."),
        _field("Independent time of the record", f"none yet: proof {points.record_proof_status}", mono=False),
    ))
    tag = s.timestamp["metrics"]["gpg_tag"]
    tag_body = "".join((
        _field("Tag", f"{tag['tag']} on commit {tag['commit']}"),
        _field("Signing key (GPG)", tag["fingerprint"], note=f"Public key committed at {logic.GPG_PUBLIC_KEY}."),
        _field("Signature", "good" if tag["good"] and tag["verify_exit_code"] == 0 else "NOT good", mono=False),
        _field("Date in the tag", tag["signed_date"], mono=False, note="The signer's own clock: not time evidence."),
        _field("Tagged commit holds the publication and the key",
               "yes" if tag["tagged_commit_holds_artifact"] and tag["tagged_commit_holds_public_key"] else "no",
               mono=False),
        _field("Proof inside the tag", "current" if points.tag_holds_current_proof else
               "the older, pending proof (the upgraded one is in a later commit)", mono=False),
        _field("Pushed to a third party", "yes" if tag["pushed_to_a_third_party"] else "no", mono=False),
    ))
    a, b = st.columns([1, 1], gap="medium")
    a.html(_card("Provenance record", f"{rec['signature']['algorithm']} · replayed", record_body))
    b.html(_card("Git tag over the publication", "GPG · replayed", tag_body))


def _timeline(events: list[logic.Event]) -> None:
    _section("The simulated theft, in clock order",
             "The P6.4 simulation handed the model to a scripted thief, who modified it and then wrote a claim of "
             "their own with an earlier date. Sorted by the dates written down, the thief's claim comes first.")
    items = "".join(
        f'<li class="zk-tl-{e.kind} zk-tl-party-{e.party}"><div class="zk-tl-time">{_e(logic.show_time(e.utc))}</div>'
        f'<div class="zk-tl-body"><div class="zk-tl-text">{_e(e.text)}</div>'
        f'<div class="zk-tl-tags"><span class="zk-tl-tag">{_e(PARTY_LABELS[e.party])}</span>'
        f'<span class="zk-tl-tag zk-tl-kind">{_e(KIND_LABELS[e.kind])}</span>'
        + (f'<span class="zk-tl-ev">{_e(e.evidence)}</span>' if e.evidence != KIND_LABELS[e.kind].lower() else "")
        + '</div></div></li>' for e in events)
    st.html(f'<ol class="zk-tl">{items}</ol>')
    st.caption("Only the green entry is backed by a third party. Self-asserted dates can be any date the writer "
               "chooses; the simulation clock is the time this run executed, after the fact.")


def _claims(rows: list[logic.ClaimRow]) -> None:
    _section("Two claims to one model",
             "After the thief adds their own weight watermark, both parties hold a claim that checks out on its own "
             "terms. What separates them is what is not symmetric.")
    body = "".join(
        f'<tr class="{"" if r.symmetric else "zk-asym"}"><td>{_e(r.aspect)}</td><td>{_e(r.owner)}</td>'
        f'<td>{_e(r.thief)}</td><td>{"same for both" if r.symmetric else "<b>differs</b>"}</td>'
        f'<td class="zk-claim-note">{_e(r.note)}</td></tr>' for r in rows)
    st.html('<div class="zk-claims-wrap"><table class="zk-claims"><thead><tr><th>Aspect</th><th>Owner</th><th>Thief</th>'
            f'<th>Symmetric?</th><th>Why it matters</th></tr></thead><tbody>{body}</tbody></table></div>')


def _audit(s: logic.Sources, rows: list[logic.AuditRow]) -> None:
    _section("What the owner's audit found",
             "Each model handed back was tested with both watermark tests at "
             f"{s.theft['metrics']['detection_alpha']}, each on its own. The full step-by-step audit is on the Live "
             "audit page.")
    st.dataframe(
        [{"Model": r.label, "Test accuracy": f"{r.test_accuracy:.2%}", "Accuracy change (pp)": r.accuracy_change_pp,
          "Triggers fired": r.fired, "Trigger test p": f"{r.behavioral_p:.2g}", "Weight z": r.weight_z,
          "Exact copy": "yes" if r.matches_record else "no", "Result": r.evidence,
          "How it was made": r.how} for r in rows],
        hide_index=True, width="stretch",
        column_config={"Accuracy change (pp)": st.column_config.NumberColumn(format="%+.2f"),
                       "Weight z": st.column_config.NumberColumn(format="%.2f"),
                       "Exact copy": st.column_config.TextColumn(help="The suspect's fingerprint equals the one the "
                                                                          "record names.")})
    st.caption("Accuracy change is against the model the record names; positive means lower accuracy. Two of the "
               "modified models leave no evidence on either test (channel pruning plus fine-tuning, and "
               "distillation). A model with no evidence is not shown to be independent; see Honest limits.")


def render_provenance(spec) -> None:
    ui.header(spec)
    store = ui.store()
    if store is None:
        return
    try:
        s = logic.load(store)
    except DataIntegrityError as error:
        ui.data_error(error)
        return
    points = logic.open_points(s)
    _open_points(points)
    _commitment(s, logic.consistency_checks(s))
    _bitcoin(s, logic.blocks(s))
    _signatures(s, points)
    _timeline(logic.timeline(s))
    _claims(logic.claim_rows(s))
    _audit(s, logic.audit_rows(s))
    with st.expander("Where each value comes from", icon=":material/description:"):
        st.dataframe([{"Committed file": path, "SHA-256 (verified on read)": store.recorded_sha256(path)}
                      for path in s.paths.values()], hide_index=True, width="stretch")
