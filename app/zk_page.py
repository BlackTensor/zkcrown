"""Zero-knowledge page (P9.13): what each proof track proves, what it costs, and what it does not prove.

Every value comes from a committed file through `app.zk_logic`. Nothing is
proved or verified on the host. Every timing and memory figure is labelled as
measured on the owner's machine, not on Colab.
"""

from __future__ import annotations

import html

import streamlit as st

from app import ui
from app import zk_logic as logic
from app.data import DataIntegrityError

M = logic.MACHINE


def _e(value: object) -> str:
    return html.escape(str(value))


def _section(title: str, lede: str) -> None:
    st.html(f'<h2 class="zk-section">{_e(title)}</h2><p class="zk-section-lede">{_e(lede)}</p>')


def _tile(label: str, value: str, detail: str, source: str, limit: bool = False, small: bool = False) -> str:
    return (f'<article class="zk-tile{" zk-tile-limit" if limit else ""}"><div class="zk-tile-label">{_e(label)}</div>'
            f'<div class="zk-tile-value{" zk-tile-value-sm" if small else ""}">{_e(value)}</div><p class="zk-tile-detail">{_e(detail)}</p>'
            f'<div class="zk-tile-source">{_e(source)}</div></article>')


def _src(s: logic.Sources, *names: str) -> str:
    return ", ".join(s.paths[n].rsplit("/", 1)[-1].split("__")[0] for n in names)


def _limits(a: logic.TrackA, b: logic.TrackB, x: logic.Blocked) -> None:
    st.warning("\n\n".join((
        "**What these proofs do not show.**",
        "- **Track A proves knowledge of an opening of the published commitment C, and nothing about any model.** It "
        "does not show that a model carries a watermark, that S comes from the key, or that the audit's triggers do.",
        "- **Track B is plain inference of a small MNIST model on a public input.** It says nothing about watermarks, "
        "ownership, or the CIFAR-10 model the commitment names.",
        f"- **The Groth16 setup had a single contributor** ({a.contributors} contribution to the circuit-specific setup, by the owner). The "
        "proving key is not sound against its own creator, who could forge proofs if they kept the setup secret.",
        f"- **The Hermez powers-of-tau check did not finish**: “{a.ptau_verify}”. The file was accepted on its "
        f"published hash only ({a.ptau_accepted_on}).",
        f"- **Binding the triggers to the key in a circuit is blocked** at an estimated {x.total:,} constraints "
        f"(P7.9), about {x.times_over:,.0f} times what the power-{x.ptau_power} file holds.",
    )), icon=":material/report:")


def _tracks(s: logic.Sources, a: logic.TrackA, b: logic.TrackB) -> None:
    left, right = st.columns([1, 1], gap="medium")
    left.html(
        '<article class="zk-pv-card"><div class="zk-pv-head"><span class="zk-pv-title">Track A · Circom and Groth16</span>'
        '<span class="zk-chip">commitment opening</span></div>'
        f'<div class="zk-pv-field"><div class="zk-pv-label">Proves</div><div class="zk-pv-value zk-mono">{_e(a.statement)}</div>'
        '<div class="zk-pv-note">The prover knows the four private values behind C. Only C is public.</div></div>'
        '<div class="zk-pv-field"><div class="zk-pv-label">Does not prove</div><div class="zk-pv-value">Anything about a '
        'model or a watermark; that S derives from the key; when C was published; who is presenting the proof.</div></div>'
        '</article>')
    right.html(
        '<article class="zk-pv-card"><div class="zk-pv-head"><span class="zk-pv-title">Track B · EZKL (Halo2)</span>'
        '<span class="zk-chip">plain inference</span></div>'
        '<div class="zk-pv-field"><div class="zk-pv-label">Proves</div><div class="zk-pv-value">The small MNIST model, '
        'its weights fixed in the committed verification key, produces the stated outputs on a public input '
        f'({_e(b.image)}), up to quantisation at scale {b.input_scale}.</div></div>'
        '<div class="zk-pv-field"><div class="zk-pv-label">Does not prove</div><div class="zk-pv-value">Anything about '
        'watermarks, ownership or the CIFAR-10 model; zero-knowledge of anything (the input and outputs are public); '
        f'agreement with PyTorch beyond the {b.fidelity_n:,} images measured.</div></div>'
        f'<div class="zk-pv-field"><div class="zk-pv-label">Settings</div><div class="zk-pv-value zk-mono">'
        f'{b.rows:,} rows, logrows {b.logrows}, scales {b.input_scale}/{b.param_scale}, check_mode {_e(b.check_mode)}'
        '</div></div></article>')


def _numbers(s: logic.Sources, a: logic.TrackA, b: logic.TrackB) -> None:
    _section("Track A in numbers", "Proof size and public signal are measured live from the committed files.")
    signal = "C" if a.public_signal_is_published_c else "not C"
    st.html('<div class="zk-tiles zk-tiles-four">' + "".join((
        _tile("Proof size", f"{a.proof_bytes:,} bytes",
              f"snarkjs JSON, measured from the committed file (recorded: {a.proof_bytes_recorded:,}).",
              _src(s, "proof")),
        _tile("Public signals", f"{len(a.public_signals)}: {signal}",
              "Compared by value with the published commitment. " + (
                  "It is exactly C." if a.public_signal_is_published_c else "It does NOT equal the published C."),
              _src(s, "proof"), limit=not a.public_signal_is_published_c),
        _tile("Tampered cases accepted", f"{a.negatives_accepted} of {a.negatives_tried:,}",
              f"{a.positive_controls} honest controls in the same run: "
              + ("all verified." if a.positive_controls_ok else "NOT all verified."), _src(s, "negatives")),
        _tile("Verify time", f"{a.verify_median:.2f} s",
              f"Median of {a.verify_runs} ({a.verify_min:.2f}–{a.verify_max:.2f} s), one snarkjs process each including "
              f"Node start-up · {M}.", _src(s, "proof")),
    )) + "</div>")
    st.caption(f"Circuit: {a.constraints:,} constraints, {a.private_inputs} private inputs, {a.public_inputs} public. "
               f"Verification key {a.vkey_bytes:,} bytes (committed); proving key {a.proving_key_bytes:,} bytes (not "
               "committed).")

    _section("Track B in numbers", "Proof size and instance count are measured live from the committed proof file.")
    st.html('<div class="zk-tiles zk-tiles-four">' + "".join((
        _tile("Proof size", f"{b.proof_bytes:,} bytes",
              f"Measured from the committed proof (recorded: {b.proof_bytes_recorded:,}). The JSON file is "
              f"{b.proof_file_bytes:,} bytes, mostly its {b.public_instances:,} public values.", _src(s, "ezkl_prove")),
        _tile("Verify time", f"{b.verify_median:.3f} s",
              f"Median of {b.verify_runs} ezkl.verify calls ({b.verify_min:.3f}–{b.verify_max:.3f} s) · {M}.",
              _src(s, "ezkl_verify")),
        _tile("Tampered proofs accepted", f"{b.negatives_accepted} of {b.negatives_tried}",
              f"{b.controls_accepted} of {b.controls_tried} honest controls accepted in the same run.",
              _src(s, "ezkl_verify")),
        _tile("Same top class as PyTorch", f"{b.fidelity_agree:,} / {b.fidelity_n:,}",
              f"All MNIST test images. Accuracy {b.fidelity_accuracy_circuit:.2%} in the circuit and "
              f"{b.fidelity_accuracy_pytorch:.2%} in PyTorch; largest output difference {b.fidelity_max_diff:.4f}. "
              f"{b.proved_subset} images also proved and verified"
              + (" with equal outputs." if b.proved_subset_ok else "; NOT all matched."), _src(s, "fidelity"), small=True),
    )) + "</div>")


def _rejections(s: logic.Sources, a: logic.TrackA, b: logic.TrackB) -> None:
    _section("Tampered proofs that were rejected",
             "Each case changes one thing about an honest proof or its inputs. These are the cases tried; they are "
             "evidence, not a soundness proof.")
    st.info(
        "**A verifier error counts as a rejection.** "
        f"In Track A, {a.crashes} altered-signal case made snarkjs crash instead of answering; it is counted as refused. "
        f"In Track B, all {b.negatives_error} of {b.negatives_tried} rejections were raised as errors by ezkl, and "
        f"{b.negatives_false} returned a plain False. Any code calling these verifiers has to treat an error as a "
        "rejection, never as a crash to retry or ignore.", icon=":material/rule:")
    tab_a, tab_b = st.tabs([f"Track A · {a.negatives_tried:,} cases", f"Track B · {b.negatives_tried} cases"])
    for tab, families in ((tab_a, logic.track_a_families(s)), (tab_b, logic.track_b_families(s))):
        tab.dataframe([{"What was changed": f.name, "Tried": f.tried, "Accepted": f.accepted, "How it was refused": f.refused}
                       for f in families], hide_index=True, width="stretch")
    notes = []
    if a.malleable:
        notes.append("In Track A, negating two proof points gives a second valid proof of the same statement. That is "
                     "expected of Groth16, so a proof's bytes do not identify it.")
    notes.append("Public signals are compared by value: the same C written with a leading zero or in hex verifies.")
    st.caption(" ".join(notes))


def _costs(a: logic.TrackA, b: logic.TrackB) -> None:
    _section("What it costs", f"Every time and memory figure was measured once on the owner's computer ({M}).")
    rows = [
        ("Track A", "Groth16 setup, all steps", a.setup_seconds, a.setup_peak_bytes),
        ("Track A", "Prove", a.prove_seconds, a.prove_peak_bytes),
        ("Track A", f"Verify (median of {a.verify_runs}; memory is the largest)", a.verify_median, a.verify_peak_bytes),
        ("Track A", f"Node start-up alone (median of {a.verify_runs})", a.node_baseline_median, None),
        ("Track B", "EZKL setup", b.setup_seconds, b.setup_peak_bytes),
        ("Track B", "Prove", b.prove_seconds, b.prove_peak_bytes),
        ("Track B", f"Verify (median of {b.verify_runs}; memory is the largest)", b.verify_median, b.verify_peak_bytes),
    ]
    st.dataframe([{"Track": t, "Step": step, f"Seconds ({M})": sec,
                   f"Peak memory, bytes ({M})": f"{mem:,}" if mem is not None else "not recorded"}
                  for t, step, sec, mem in rows], hide_index=True, width="stretch",
                 column_config={f"Seconds ({M})": st.column_config.NumberColumn(format="%.3f")})
    st.caption(f"Key sizes: Track A proving key {a.proving_key_bytes:,} bytes, verification key {a.vkey_bytes:,}. "
               f"Track B proving key {b.proving_key_bytes:,} bytes, verification key {b.vkey_bytes:,}, structured "
               f"reference string {b.srs_bytes:,}. Peak memory is per process (Windows working set).")


def _blocked(s: logic.Sources, x: logic.Blocked) -> None:
    _section("Blocked: binding the triggers to the key",
             "Proving that the audit's triggers really come from the committed key would close the loop between the "
             "cryptography and the watermark. Sized before writing any circuit, it does not fit.")
    bars = "".join(
        f'<div class="zk-bar-row"><div class="zk-bar-label">{_e(name)}</div><div class="zk-bar-track">'
        f'<div class="zk-bar-fill" style="width:{count / x.total:.2%}"></div></div>'
        f'<div class="zk-bar-value">{count:,}</div></div>' for name, count in x.parts)
    st.html('<div class="zk-tiles zk-tiles-two">'
            + _tile("Estimated constraints (lower bound)", f"{x.total:,}",
                    f"Needs power {x.power_needed}; the file used holds {x.ptau_capacity:,} (power {x.ptau_power}). "
                    + ("" if x.fits_largest_hermez else "More than the largest published Hermez file. ")
                    + f"SHA-256 is {x.sha256_share:.1%} of it.", _src(s, "sizing"), limit=True)
            + _tile("Even one keystream block", f"{x.single_block:,}",
                    f"Constraints for a single block, already more than the {x.ptau_capacity:,} the file holds.",
                    _src(s, "sizing"), limit=True)
            + f'</div><div class="zk-bars">{bars}</div>')
    st.caption("Why: the current trigger derivation and the published trigger digest use SHA-256, which is costly inside a "
               "prime-field circuit. A circuit-friendly version is an open design choice, not started.")


def render_zk(spec) -> None:
    ui.header(spec)
    store = ui.store()
    if store is None:
        return
    try:
        s = logic.load(store)
    except DataIntegrityError as error:
        ui.data_error(error)
        return
    a, b, x = logic.track_a(s), logic.track_b(s), logic.blocked(s)
    _limits(a, b, x)
    _tracks(s, a, b)
    _numbers(s, a, b)
    _rejections(s, a, b)
    _costs(a, b)
    _blocked(s, x)
    with st.expander("Where each value comes from", icon=":material/description:"):
        st.dataframe([{"Committed file": path, "SHA-256 (verified on read)": store.recorded_sha256(path)}
                      for path in s.paths.values()], hide_index=True, width="stretch")
