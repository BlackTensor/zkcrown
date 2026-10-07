"""What every page draws around its panel (P9.5): the header, then the panel or its placeholder.

The overview (P9.6) draws its own hero instead of the standard header.

The not-legal-evidence footer is drawn by the entry point after the page runs,
so no page can leave it out.
"""

from __future__ import annotations

from app import ui
from app.pages import PAGES


def render_page(slug: str) -> None:
    spec = next(p for p in PAGES if p.slug == slug)
    if slug == "overview":
        from app.overview import render_overview

        render_overview(spec)
        return
    if slug == "triggers":
        from app.triggers_page import render_triggers

        render_triggers(spec)
        return
    if slug == "attacks":
        from app.attacks_page import render_attacks

        render_attacks(spec)
        return
    if slug == "evidence":
        from app.evidence_page import render_evidence

        render_evidence(spec)
        return
    if slug == "zk":
        from app.zk_page import render_zk

        render_zk(spec)
        return
    if slug == "provenance":
        from app.provenance_page import render_provenance

        render_provenance(spec)
        return
    if slug == "audit":
        from app.audit_page import render_audit

        render_audit(spec)
        return
    ui.header(spec)
    if slug == "integrity":
        from app.integrity import render_integrity

        render_integrity()
    else:
        ui.placeholder(spec)
