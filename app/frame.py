"""What every page draws around its panel (P9.5): the header, then the panel or its placeholder.

The not-legal-evidence footer is drawn by the entry point after the page runs,
so no page can leave it out.
"""

from __future__ import annotations

from app import ui
from app.pages import PAGES


def render_page(slug: str) -> None:
    spec = next(p for p in PAGES if p.slug == slug)
    ui.header(spec)
    if slug == "integrity":
        from app.integrity import render_integrity

        render_integrity()
    else:
        ui.placeholder(spec)
