"""Live audit page (P9.7): pick a suspect, watch the five checks, read the graded verdict.

Each step says whether it is live or replayed, and from which file. The grade is
recomputed live by the auditor's own grading code from the replayed check
results, and compared with the committed P9.3 grade. Owner evidence is shown
in its own section and never feeds the suspect's grade.
"""

from __future__ import annotations

import html
import time

import streamlit as st

from app import audit_logic as al
from app import ui
from app.data import DataIntegrityError

OUTCOME = {
    "detected": ("Detected", "green"),
    "not_detected": ("Not detected", "gray"),
    "passed": ("Passed", "green"),
    "failed": ("Failed", "red"),
    "not_applicable": ("Not applicable", "orange"),
    "not_run": ("Not run", "orange"),
    "error": ("Error", "red"),
}
TIER_CLASS = {"very strong": "vstrong", "strong": "strong", "moderate": "moderate", "weak": "weak",
              "marginal": "marginal", "none": "none", "not assessed": "none"}


def _yes(flag) -> str:
    return "yes" if flag else "no"


def _p(value) -> str:
    return "n/a" if value is None else f"{value:.2g}"


def _metrics(items) -> None:
    for col, (label, value, tip) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=tip)


def _details(slot: str, check: dict, sources: al.AuditSources) -> None:
    stat = check.get("statistic") or {}
    if slot == "fingerprint":
        st.markdown("Bit for bit the model the record names." if check["status"] == "passed"
                    else "A different set of weights from the model the record names. On its own this says "
                         "nothing about where the model came from.")
    elif slot == "behavioral":
        _metrics((("Triggers giving the keyed target", f"{stat['fired']} / {stat['n']}", None),
                  ("Needed at the detection level", f"{stat['k_star_at_alpha']}", None),
                  ("p-value (exact)", _p(check["p_value"]), None)))
    elif slot == "weight":
        _metrics((("Correlation z", f"{stat['z']:.2f}", "Blind extraction over the owner's carrier."),
                  ("Needed at the detection level", f"{stat['z_threshold_at_alpha']:.2f}", None),
                  ("p-value (upper bound)", _p(check["p_value"]), None)))
        st.caption(f"{stat['bit_matches']} of {stat['rows']} signature bits recovered.")
    elif slot == "commitment":
        live = al.live_commitment(sources)
        facts = live["facts"]
        st.markdown(
            f"- The record's commitment equals the published one: **{_yes(facts['record_c_equals_published_c'])}**\n"
            f"- Its decimal and hex forms agree: **{_yes(facts['decimal_and_hex_agree'])}**\n"
            f"- The publication's SHA-256 is the one the record names: "
            f"**{_yes(facts['publication_hash_matches_record'])}**")
        st.caption(f"Recorded in the replayed audit ({sources.p9_2_path}): commitment is a valid BN254 field "
                   f"element: {_yes(stat.get('c_well_formed'))}; earliest Bitcoin attestation in the "
                   f"OpenTimestamps proof: block {stat.get('ots_earliest_block_height')}. This shows the record "
                   "commits to the published value; it does not tie it to the suspect.")
    elif slot == "zk_proof":
        st.markdown(
            f"- Groth16 proof of knowledge of an opening of the commitment verifies: "
            f"**{_yes(stat.get('groth16_verified'))}**; its only public signal equals the record's commitment: "
            f"**{_yes(stat.get('groth16_public_signal_equals_record_c'))}**\n"
            f"- EZKL proof of the small MNIST model's inference verifies: **{_yes(stat.get('ezkl_verified'))}**")
        st.caption("Neither proof is about the suspect model.")
    if check.get("reason") and slot not in ("commitment",):
        with st.expander("Auditor's note", icon=":material/notes:"):
            st.write(check["reason"])


def _step(slot: str, check: dict, sources: al.AuditSources, animate: bool, progress, index: int) -> None:
    title = al.STEP_TITLES[slot]
    status = al.live_commitment(sources)["status"] if slot == "commitment" else check["status"]
    text, color = OUTCOME.get(status, (status, "gray"))
    live = slot == "commitment"
    mode = "live" if live else f"replayed from {al.short_name(sources.p9_2_path)}"
    with st.status(f"{title} · {mode}", state="running" if animate else "complete", expanded=animate) as box:
        if animate:
            working = st.empty()
            working.caption("Recomputing from the provenance files…" if live else "Reading the recorded result…")
            time.sleep(0.6)
            working.empty()
        cols = st.columns([1, 1])
        with cols[0]:
            st.badge("live" if live else "replayed", icon=":material/bolt:" if live else ":material/history:",
                     color="green" if live else "blue")
        cols[1].markdown(f":{color}-badge[{text}]")
        st.caption(al.label_for(slot, sources) + (f" — {al.why_replayed(slot)}." if not live else "."))
        _details(slot, check, sources)
        if live and status != check["status"]:
            st.error(f"The live result ({status}) differs from the recorded one ({check['status']}).")
        box.update(label=f"{title} · {text} · {mode}",
                   state="error" if status == "error" else "complete", expanded=False)
    progress.progress((index + 1) / (len(al.SLOTS) + 1),
                      text=f"Step {index + 1} of {len(al.SLOTS) + 1} · {title}: {text.lower()}")
    if animate:
        time.sleep(0.25)


def _grade_card(grade: dict, matches: bool, sources: al.AuditSources) -> None:
    s = grade["suspect"]
    tier = s["technical_evidence_strength"]
    marks = s["watermarks"]
    rows = "".join(
        f'<div class="zk-gw"><span>{html.escape(al.STEP_TITLES[slot])}</span>'
        f'<span class="zk-gw-tier">{html.escape(marks[slot]["tier"])}</span>'
        f'<span class="zk-gw-p">p {html.escape(_p(marks[slot]["p_value"]))}</span></div>'
        for slot in ("behavioral", "weight"))
    caveats = "".join(f"<li>{html.escape(c)}</li>" for c in s["caveats"])
    borderline = "".join(f"<li>{html.escape(b)}</li>" for b in s["borderline"])
    st.html(
        f'<section class="zk-grade zk-grade-{TIER_CLASS.get(tier, "none")}">'
        '<div class="zk-grade-kicker">Technical evidence strength · suspect</div>'
        f'<div class="zk-grade-tier">{html.escape(tier)}</div>'
        f'<div class="zk-grade-p">combined p = {html.escape(_p(s["combined_p_value"]))} '
        f'· {s["tests_assessed"]} of {len(marks)} tests assessed</div>'
        f'<p class="zk-grade-statement">{html.escape(s["statement"])}</p>'
        f'<div class="zk-gws">{rows}</div>'
        f'<p class="zk-grade-fp">{html.escape(s["fingerprint_statement"])}</p>'
        + (f'<div class="zk-grade-sub">Caveats</div><ul class="zk-grade-list">{caveats}</ul>' if caveats else "")
        + (f'<div class="zk-grade-sub">Borderline</div><ul class="zk-grade-list">{borderline}</ul>'
           if borderline else "")
        + f'<div class="zk-grade-method">{html.escape(s["combination"])}</div>'
        '</section>')
    if matches:
        st.success(f"Recomputed live with {al.GRADING_SOURCE}; identical to the committed grade in "
                   f"{sources.p9_3_path}.", icon=":material/verified:")
    else:
        st.error(f"The live grade differs from the committed grade in {sources.p9_3_path}. Treat this verdict as "
                 "unverified.", icon=":material/gpp_bad:")


def _owner_evidence(grade: dict) -> None:
    o = grade["owner_evidence"]
    items = (("Signed provenance record verifies", o["record_valid"]),
             ("Signing key matches the trusted owner key", o["public_key_trusted"]),
             ("Commitment check", o["commitment"]),
             ("Zero-knowledge proofs", o["zk_proof"]))
    rows = "".join(f'<div class="zk-oe-row"><span>{html.escape(k)}</span><span class="zk-oe-v">'
                   f'{html.escape(v if isinstance(v, str) else _yes(v))}</span></div>' for k, v in items)
    st.html('<section class="zk-oe"><div class="zk-grade-kicker">Owner evidence · separate from the grade</div>'
            f'{rows}<p class="zk-oe-note">{html.escape(o["statement"])}</p></section>')


def render_audit(spec) -> None:
    ui.header(spec)
    store = ui.store()
    if store is None:
        return
    try:
        sources = al.load_sources(store)
    except DataIntegrityError as error:
        ui.data_error(error)
        return
    st.info(f"**Replay mode.** The behavioral and weight checks need the owner's secret key, which is never on this "
            f"host, so they are replayed from the committed audit `{sources.p9_2_path}`. The fingerprint and proof "
            "checks are replayed too. The commitment check and the grade run live. Each step says which.",
            icon=":material/history:")

    by_prefix = {s.prefix: s for s in al.SUSPECTS}
    choice = st.pills("Suspect", [s.prefix for s in al.SUSPECTS], format_func=lambda p: by_prefix[p].title,
                      default=al.SUSPECTS[0].prefix, key="zk_suspect")
    if choice is None:
        st.caption("Pick a suspect to audit.")
        return
    suspect, name = by_prefix[choice], sources.names[choice]
    st.html(f'<div class="zk-suspect"><div class="zk-suspect-title">{html.escape(suspect.title)}</div>'
            f'<div class="zk-suspect-desc">{html.escape(suspect.description)}</div>'
            f'<div class="zk-suspect-id">recorded as <code>{html.escape(name)}</code></div></div>')

    done = st.session_state.setdefault("zk_audited", set())
    run = st.button("Run the audit", type="primary", icon=":material/play_arrow:", key=f"zk_run_{choice}")
    if not run and name not in done:
        return
    animate = run
    progress = st.progress(0.0, text="Starting the audit…")
    checks = al.checks_for_grading(sources, name)
    st.html('<h2 class="zk-section">Checks</h2>')
    for index, slot in enumerate(al.SLOTS):
        _step(slot, checks[slot], sources, animate, progress, index)

    with st.status("Grading · live", state="running" if animate else "complete", expanded=False) as box:
        if animate:
            time.sleep(0.6)
        grade = al.live_grade(sources, name)
        st.badge("live", icon=":material/bolt:", color="green")
        st.caption(f"live · {al.GRADING_SOURCE} applied to the replayed p-values")
        box.update(label=f"Grading · {grade['suspect']['technical_evidence_strength']} · live", state="complete")
    progress.progress(1.0, text="Audit complete")
    done.add(name)
    if animate:
        st.toast(f"{suspect.title}: {grade['suspect']['technical_evidence_strength']}", icon=":material/fact_check:")

    st.html('<h2 class="zk-section">Verdict</h2>')
    _grade_card(grade, al.matches_committed(sources, name, grade), sources)
    _owner_evidence(grade)
