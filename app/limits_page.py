"""Honest-limits page (P9.14): where the watermarks fail and what the cryptography does not cover.

Each limit comes from `app.limits_logic`, which reads committed files through
the data layer. Each card names its source files and links to the page where
the evidence is shown.
"""

from __future__ import annotations

import html

import streamlit as st

from app import limits_logic as logic
from app import ui
from app.data import DataIntegrityError

def _short(path: str) -> str:
    return path.rsplit("/", 1)[-1].split("__")[0]


def _card(store, limit: logic.Limit) -> None:
    with st.container(border=True):
        st.html(f'<div class="zk-lim-title">{html.escape(limit.title)}</div>'
                f'<p class="zk-lim-fact">{html.escape(limit.fact)}</p>'
                f'<p class="zk-lim-detail">{html.escape(limit.detail)}</p>')
        st.page_link(f"views/{limit.page}.py", label=f"See the evidence on {limit.page_label}",
                     icon=":material/arrow_forward:")
        st.caption(f"source · {', '.join(_short(p) for p in limit.sources)}",
                   help="\n\n".join(f"{p} · SHA-256 {store.recorded_sha256(p)}" for p in limit.sources))


def render_limits(spec) -> None:
    ui.header(spec)
    store = ui.store()
    if store is None:
        return
    st.markdown("Each limit below is stated with the measured fact behind it, read from a committed result. None "
                "of them is hidden elsewhere on this site; this page puts them in one place.")
    limits: list[logic.Limit] = []
    for name, build in logic.GROUPS:
        try:
            limits += build(store)
        except DataIntegrityError as error:
            st.error(f"**{name}:** these limits could not be shown.", icon=":material/gpp_bad:")
            ui.data_error(error)
    present = [c for c in logic.CATEGORIES if any(l.category == c for l in limits)]
    counts = " · ".join(f"{c}: {sum(l.category == c for l in limits)}" for c in present)
    st.caption(f"{len(limits)} limits. {counts}.")
    for category in present:
        st.html(f'<h2 class="zk-section">{html.escape(category)}</h2>')
        cols: list = []
        for limit in (l for l in limits if l.category == category):
            if not cols:
                cols = list(st.columns([1, 1], gap="medium"))
            with cols.pop(0):
                _card(store, limit)
    st.html(f'<div class="zk-lim-end">{html.escape(ui.NOT_LEGAL)}</div>')
