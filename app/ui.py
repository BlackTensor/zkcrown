"""Shared page parts (P9.5): theme injection, page header, placeholder card, data errors, footer."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st

from app.data import DataIntegrityError, DataStore
from app.pages import PageSpec

NOT_LEGAL = "This is a technical ownership verification demonstration, not legal evidence."
"""Must equal `src.auditor.grading.NOT_LEGAL` (a test checks it)."""
THEME_CSS = Path(__file__).resolve().parent / "theme.css"


def inject_theme() -> None:
    st.html(f"<style>{THEME_CSS.read_text(encoding='utf-8')}</style>")


@st.cache_resource(show_spinner=False)
def _cached_store() -> DataStore:
    return DataStore()


def store() -> DataStore | None:
    """The data store (manifest loaded once; every read re-hashes), or None after showing why not."""
    try:
        return _cached_store()
    except DataIntegrityError as error:
        data_error(error)
        return None


def data_error(error: DataIntegrityError) -> None:
    """The one way a data problem is shown: visibly, with the file and the reason, and no substitute value."""
    st.error(f"**Data check failed for `{error.path}`.** {error.problem}. Nothing from this file is shown.",
             icon=":material/gpp_bad:")


def read_json(path: str) -> Any | None:
    """A verified JSON file, or None after showing the error. Callers must render nothing in its place."""
    s = store()
    if s is None:
        return None
    try:
        return s.read_json(path)
    except DataIntegrityError as error:
        data_error(error)
        return None


def header(spec: PageSpec) -> None:
    st.html(
        f'<div class="zk-kicker">{html.escape(spec.section)}</div>'
        f'<h1 class="zk-title">{html.escape(spec.title)}</h1>'
        f'<p class="zk-lede">{html.escape(spec.summary)}</p>'
        + mode_line(spec)
    )


def mode_line(spec: PageSpec) -> str:
    """The page's live / replayed statement (P9.15), as HTML."""
    return f'<p class="zk-mode"><span class="zk-mode-tag">On this page</span>{html.escape(spec.mode)}</p>'


def placeholder(spec: PageSpec) -> None:
    items = "".join(f"<li>{html.escape(line)}</li>" for line in spec.plan)
    st.html(
        '<div class="zk-card zk-placeholder">'
        f'<div class="zk-chip">Planned · {html.escape(spec.tasks)}</div>'
        '<div class="zk-card-title">This panel is not built yet</div>'
        f'<ul class="zk-list">{items}</ul>'
        '<div class="zk-note">Every figure it shows will be read from committed results, checked against '
        'their recorded SHA-256.</div>'
        '</div>'
    )


def footer() -> None:
    st.html(f'<footer class="zk-footer"><span class="zk-footer-mark">zk-Crown</span>'
            f'<span>{html.escape(NOT_LEGAL)}</span></footer>')
