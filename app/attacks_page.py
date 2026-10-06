"""Attack-lab explorer (P9.10): every Phase 4 setting, filterable, with its cost and its evidence tier.

Interactive version of the P4.9 heatmap, plus a sortable table and a scatter of
accuracy cost against evidence strength. Data from `app.attacks_logic`; chart
styling from ``app/chart_style.json``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from app import attacks_logic as lab_logic
from app import repo_code, ui
from app.data import DataIntegrityError

STYLE = json.loads((Path(__file__).resolve().parent / "chart_style.json").read_text(encoding="utf-8"))
OUTCOME_LABELS = {
    "both detected": "Both detected",
    "behavioral lost, weight detected": "Only the weight watermark",
    "weight lost, behavioral detected": "Only the trigger watermark",
    "neither detected": "Neither watermark detected",
}
ALL_OUTCOMES = "All settings"


def _p(value) -> str:
    return "n/a" if value is None else f"{value:.2g}"


def _outcome_tiles(lab: lab_logic.AttackLab) -> None:
    tiles = "".join(
        f'<article class="zk-tile zk-tile-{"limit" if key == "neither detected" else "hold"}">'
        f'<div class="zk-tile-label">{html.escape(OUTCOME_LABELS.get(key, key))}</div>'
        f'<div class="zk-tile-value">{count} of {lab.settings_run}</div>'
        f'<p class="zk-tile-detail">attack settings run</p></article>'
        for key, count in lab.outcomes.items())
    st.html(f'<div class="zk-tiles zk-tiles-four">{tiles}</div>')
    st.caption(f"Counts over the {lab.settings_run} attack settings that were run (the unattacked control is not "
               "counted). They depend on which settings were chosen, so they are not rates. Each watermark's own "
               "test, at the auditor's detection level; the two are not combined here.")


def _filters(lab: lab_logic.AttackLab) -> list[lab_logic.Row]:
    titles = dict(lab.families)
    outcome = st.segmented_control(
        "Outcome", [ALL_OUTCOMES, *lab.outcomes], default=ALL_OUTCOMES, key="zk_lab_outcome",
        format_func=lambda o: o if o == ALL_OUTCOMES else OUTCOME_LABELS.get(o, o),
        help="One click on “Neither watermark detected” shows the settings that leave no evidence for either test.")
    left, right = st.columns([1, 1])
    families = left.multiselect("Attack family", [k for k, _ in lab.families], format_func=titles.get,
                                placeholder="All families", key="zk_lab_families")
    in_family = [r for r in lab.rows if not families or r.family in families]
    settings = right.multiselect("Setting", [f"{r.family_title} · {r.setting}" for r in in_family],
                                 placeholder="All settings in the chosen families", key="zk_lab_settings")
    chosen = None if outcome in (None, ALL_OUTCOMES) else outcome
    return lab_logic.filter_rows(lab.rows, families, chosen, settings)


def heatmap_cells(rows: list[lab_logic.Row]) -> list[dict]:
    """One cell per row and metric, with its shade, its text and whether that watermark was lost."""
    acc_hi = max(r.test_accuracy for r in rows)
    z_hi = max([r.weight_z for r in rows if r.weight_applicable] or [None]) or None
    cells = []
    for r in rows:
        label = f"{r.family_title} · {r.setting}"
        cells += [
            {"row": label, "metric": "Test accuracy", "shade": r.test_accuracy / acc_hi,
             "text": f"{r.test_accuracy:.2%}", "lost": False},
            {"row": label, "metric": "Triggers fired", "shade": r.fired / r.n_triggers,
             "text": f"{r.fired} / {r.n_triggers}", "lost": not r.behavioral_detected},
            {"row": label, "metric": "Weight z",
             "shade": max(r.weight_z, 0) / z_hi if r.weight_applicable and z_hi else None,
             "text": (f"{r.weight_z:.2f}" if r.weight_applicable else "not applicable")
             + (" ⚠" if r.channel_pruned else ""),
             "lost": not r.weight_detected},
        ]
    return cells


def _heatmap(rows: list[lab_logic.Row]) -> None:
    cells = heatmap_cells(rows)
    spec = {
        "height": len(rows) * STYLE["row_height"],
        "data": {"values": cells},
        "encoding": {"y": {"field": "row", "type": "nominal", "sort": None, "title": None,
                           "axis": {"labelLimit": STYLE["row_label_limit"], "labelOverlap": False}},
                     "x": {"field": "metric", "type": "nominal", "sort": None, "title": None,
                           "axis": {"orient": "top", "labelAngle": 0}}},
        "layer": [
            {"mark": {"type": "rect", "stroke": STYLE["cell_stroke"]},
             "encoding": {"color": {"field": "shade", "type": "quantitative", "legend": None,
                                    "scale": {"range": STYLE["heat_range"], "domainMin": 0, "domainMax": 1}}}},
            {"transform": [{"filter": "datum.lost"}],
             "mark": {"type": "rect", "fill": None, "stroke": STYLE["floor_color"],
                      "strokeWidth": STYLE["suspect_rule_width"]}},
            {"mark": {"type": "text", "fontSize": STYLE["label_font_size"]},
             "encoding": {"text": {"field": "text"},
                          "color": {"condition": {"test": "datum.lost", "value": STYLE["suspect_color"]},
                                    "value": STYLE["bound_color"]}}},
        ],
    }
    st.vega_lite_chart(spec, width="stretch", theme="streamlit")
    st.caption("Each row is one setting. Shading is relative within each column of the rows shown. An orange frame "
               "and orange text mark a watermark that its own test did not detect at the auditor's level; "
               "“not applicable” means the weight test could not be run because the owner's layout is absent. "
               "⚠ marks channel-pruning rows, whose weight figure assumes a re-alignment step that was never built.")


def _table(rows: list[lab_logic.Row]) -> None:
    st.dataframe(
        [{"Family": r.family_title, "Setting": r.setting, "Test accuracy": r.test_accuracy,
          "Accuracy cost (pp)": r.drop_pp, "Triggers fired": r.fired, "Behavioral test": r.behavioral_status,
          "Weight test": r.weight_status, "Weight z": r.weight_z if r.weight_applicable else None,
          "Weight caveat": r.weight_caveat,
          "Evidence tier": r.tier, "Combined p": r.combined_p, "Note": r.note or "", "Task": r.task}
         for r in rows],
        hide_index=True, width="stretch",
        column_config={
            "Test accuracy": st.column_config.NumberColumn(format="percent"),
            "Accuracy cost (pp)": st.column_config.NumberColumn(
                format="%.2f", help="Drop in test accuracy against the unattacked dual W*, in percentage points."),
            "Weight z": st.column_config.NumberColumn(
                format="%.2f", help="Empty only where the Weight test column says “not applicable”."),
            "Combined p": st.column_config.NumberColumn(format="%.1e", help="Bonferroni over the two tests (P9.3)."),
        })


def _cheapest(rows: list[lab_logic.Row], none: str) -> str:
    clean = sorted((r for r in rows if r.tier == none), key=lambda r: r.drop_pp)
    if not clean:
        return "every setting shown leaves some evidence."
    r = clean[0]
    return (f"of the settings shown, {len(clean)} leave no evidence; the cheapest is {r.family_title.lower()}, "
            f"{r.setting}, at an accuracy cost of {r.drop_pp:+.2f} pp.")


def _scatter(rows: list[lab_logic.Row]) -> None:
    grading = repo_code.grading()
    tiers = [name for _, name in grading.TIERS] + [grading.NONE]
    points = [{"cost": r.drop_pp, "p": r.combined_p, "tier": r.tier, "setting": f"{r.family_title} · {r.setting}",
               "outcome": OUTCOME_LABELS.get(r.outcome, r.outcome), "accuracy": f"{r.test_accuracy:.2%}",
               "caveat": r.weight_caveat or lab_logic.NO_CAVEAT}
              for r in rows if r.combined_p is not None]
    spec = {
        "height": STYLE["scatter_height"],
        "data": {"values": points},
        "transform": [{"calculate": "random()", "as": "jitter"}],
        "mark": {"type": "point", "filled": True, "size": STYLE["point_size"], "opacity": STYLE["bar_opacity"]},
        "encoding": {
            "x": {"field": "cost", "type": "quantitative", "scale": {"nice": False, "zero": True},
                  "title": "accuracy cost to the attacker (percentage points of test accuracy)"},
            "y": {"field": "tier", "type": "ordinal", "sort": tiers,
                  "scale": {"domain": tiers, "paddingInner": STYLE["band_padding"]},
                  "title": "evidence tier left behind (P9.3)"},
            "yOffset": {"field": "jitter", "type": "quantitative", "scale": {"domain": [0, 1]}},
            "color": {"field": "tier", "type": "nominal", "scale": {"domain": tiers, "range": STYLE["tier_colors"]},
                      "legend": None},
            "shape": {"field": "caveat", "type": "nominal", "title": None,
                      "scale": {"domain": [lab_logic.CAVEAT_SHORT, lab_logic.NO_CAVEAT],
                                "range": STYLE["caveat_shapes"]},
                      "legend": {"orient": "top", "labelLimit": STYLE["legend_label_limit"]}},
            "tooltip": [{"field": "setting"}, {"field": "accuracy", "title": "test accuracy"},
                        {"field": "cost", "title": "cost (pp)", "format": ".2f"},
                        {"field": "p", "title": "combined p", "format": ".1e"}, {"field": "tier"},
                        {"field": "outcome"}, {"field": "caveat", "title": "note"}]},
    }
    st.vega_lite_chart(spec, width="stretch", theme="streamlit")
    st.caption(
        "Each dot is one attack setting: how much test accuracy it cost, and how strong the evidence left behind "
        "is (one band per tier, strongest at the top; dots are spread within a band so they do not hide each "
        "other, and the tooltip gives the exact combined p-value). Triangles are channel-pruning settings: their "
        "weight figure assumes a re-alignment step that was never built. An attacker wants a dot at the bottom left: "
        f"little cost, no evidence. **Conclude:** {_cheapest(rows, grading.NONE)} **Not shown:** run-to-run variation "
        "(each setting was run once), attacks outside this deliberately chosen grid, or attackers with more data. "
        "The cloud is not a rate or a frontier.")


def render_attacks(spec) -> None:
    ui.header(spec)
    store = ui.store()
    if store is None:
        return
    try:
        lab = lab_logic.load(store)
    except DataIntegrityError as error:
        ui.data_error(error)
        return
    _outcome_tiles(lab)
    if lab.grade_mismatches:
        st.error(f"The evidence tier recomputed for {len(lab.grade_mismatches)} rows differs from the committed "
                 f"grade in {lab.p9_4_path}. Treat those tiers as unverified.", icon=":material/gpp_bad:")
    else:
        st.success(f"Evidence tiers recomputed live with {repo_code.GRADING_SOURCE} from each row's p-values; all "
                   f"{len(lab.rows)} equal the committed grades in {lab.p9_4_path}.", icon=":material/verified:")

    st.html('<h2 class="zk-section">Explore</h2>')
    rows = _filters(lab)
    attack_rows = [r for r in rows if r.family != lab_logic.CONTROL]
    shown = lab_logic.outcome_counts(rows, list(lab.outcomes))
    st.caption(f"Showing {len(rows)} of {len(lab.rows)} rows. Outcomes among the attack settings shown: "
               + "; ".join(f"{OUTCOME_LABELS.get(k, k).lower()} {v}" for k, v in shown.items()) + ".")
    if not rows:
        st.info("No setting matches these filters.", icon=":material/filter_alt_off:")
        return
    if any(r.channel_pruned for r in rows):
        st.warning(f"**Channel-pruning rows:** {lab_logic.NEVER_BUILT}", icon=":material/construction:")

    heat, table = st.tabs(["Heatmap", "Table"])
    with heat:
        _heatmap(rows)
    with table:
        _table(rows)
        st.caption("Click a column header to sort. “Accuracy cost” is the drop against the unattacked model, so the "
                   "control row is zero. The Weight caveat column marks the channel-pruning rows, whose weight figure "
                   "assumes a re-alignment step that was never built; the Note column carries P4.9's family notes.")

    st.html('<h2 class="zk-section">The attacker\'s trade-off</h2>')
    if attack_rows:
        _scatter(attack_rows)
    with st.expander("Where each figure comes from", icon=":material/description:"):
        st.dataframe([{"Used for": "rows, outcome counts, family notes", "Committed file": lab.p4_9_path,
                       "SHA-256 (verified on read)": store.recorded_sha256(lab.p4_9_path)},
                      {"Used for": "committed grade of every row", "Committed file": lab.p9_4_path,
                       "SHA-256 (verified on read)": store.recorded_sha256(lab.p9_4_path)}],
                     hide_index=True, width="stretch")
