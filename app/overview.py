"""Landing page (P9.6): hero, the five-stage pipeline, headline figures and their sources.

Every figure comes from `app.headlines`, which reads committed result records
through the data layer. A record that fails its SHA-256 check is shown as an
error and its figures are left out; nothing stands in for them.
"""

from __future__ import annotations

import html

import streamlit as st

from app import ui
from app.data import DataIntegrityError
from app.headlines import SOURCE_PREFIXES, build_headlines
from app.pages import PageSpec

PITCH = "Watermark a model. Attack it like a thief. Audit it against evidence published first."
LEDE = ("Two watermarks, an attack laboratory, a commitment timestamped in Bitcoin, zero-knowledge proofs, and "
        "an auditor that grades technical evidence strength, including the cases where it finds none.")

STAGES = (
    ("Watermark", "triggers", "Embed a keyed identity: trigger responses plus a spread-spectrum weight signal."),
    ("Attack", "attacks", "Prune, quantize, fine-tune, distill and overwrite the model, as a thief would."),
    ("Commit", "provenance", "Publish a Poseidon commitment to the key and timestamp it in Bitcoin."),
    ("Prove", "zk", "Show knowledge of the committed key in zero knowledge, without revealing it."),
    ("Audit", "audit", "Test a suspect against the owner's key and grade the evidence, or its absence."),
)


def _hero() -> None:
    st.html(
        '<section class="zk-hero">'
        '<div class="zk-kicker">zk-Crown · neural watermarking and provenance</div>'
        f'<h1 class="zk-hero-title">{html.escape(PITCH)}</h1>'
        f'<p class="zk-hero-lede">{html.escape(LEDE)}</p>'
        '</section>'
    )
    with st.container(horizontal=True, key="zk_cta"):
        st.page_link("views/audit.py", label="Open the live audit", icon=":material/policy:")
        st.page_link("views/limits.py", label="See where it fails", icon=":material/report:")


def _pipeline() -> None:
    nodes = "".join(
        f'<a class="zk-stage" href="{slug}" target="_self">'
        f'<span class="zk-stage-name">{html.escape(name)}</span>'
        f'<span class="zk-stage-text">{html.escape(text)}</span>'
        f'<span class="zk-stage-go">Open →</span>'
        '</a>'
        for name, slug, text in STAGES)
    st.html('<h2 class="zk-section">How it works</h2>'
            f'<nav class="zk-pipeline" aria-label="Pipeline">{nodes}</nav>')


def _load_records() -> dict:
    store = ui.store()
    records: dict = {}
    for name, prefix in SOURCE_PREFIXES.items():
        records[name] = None
        if store is None:
            continue
        try:
            path = store.latest(prefix)
            records[name] = (path, store.read_json(path))
        except DataIntegrityError as error:
            ui.data_error(error)
    return records


def _headlines(records: dict) -> list:
    headlines = build_headlines(records)
    tiles = "".join(
        f'<article class="zk-tile zk-tile-{h.tone}">'
        f'<div class="zk-tile-label">{html.escape(h.label)}</div>'
        f'<div class="zk-tile-value">{html.escape(h.value)}</div>'
        f'<p class="zk-tile-detail">{html.escape(h.detail)}</p>'
        f'<div class="zk-tile-source">{html.escape(", ".join(p.rsplit("/", 1)[-1].split("__")[0] for p in h.sources))}'
        '</div></article>'
        for h in headlines)
    st.html('<h2 class="zk-section">Headline results</h2>'
            '<p class="zk-section-lede">What survives, what does not, and how the auditor treats models that carry '
            'no watermark. Counts are over the attack settings that were run, so they are not rates.</p>'
            f'<div class="zk-tiles">{tiles}</div>')
    return headlines


def _sources(records: dict, headlines: list) -> None:
    store = ui.store()
    with st.expander("Where each figure comes from", icon=":material/description:"):
        rows = []
        for name, loaded in records.items():
            if loaded is None:
                continue
            path = loaded[0]
            rows.append({"Figures": ", ".join(h.label for h in headlines if path in h.sources),
                         "Committed file": path, "SHA-256 (verified on read)": store.recorded_sha256(path)})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Each file is listed in app/manifest.json and re-hashed before it is read. If a hash does not "
                   "match, the page shows the error and leaves that file's figures out.")


def render_overview(spec: PageSpec) -> None:
    _hero()
    st.html(ui.mode_line(spec))
    _pipeline()
    records = _load_records()
    headlines = _headlines(records)
    if any(records.values()):
        _sources(records, headlines)
