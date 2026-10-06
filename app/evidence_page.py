"""Evidence-strength page (P9.9): why a suspect's statistic is, or is not, beyond chance.

Three charts, all drawn with Streamlit's built-in Vega-Lite charting from
committed records (see `app.evidence_logic`). Styling comes from
``app/chart_style.json``; every value shown comes from a record or is computed
by the repo's own statistics code.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app import audit_logic as al
from app import evidence_logic as ev
from app import ui
from app.data import DataIntegrityError
from app.suspect_picker import pick

STYLE = json.loads((Path(__file__).resolve().parent / "chart_style.json").read_text(encoding="utf-8"))


def _p(value) -> str:
    return "n/a" if value is None else f"{value:.2g}"


def _rules(values: list[dict], field: str, style: dict) -> list[dict]:
    """Dashed threshold rules with a label at the top of the plot."""
    return [
        {"data": {"values": values},
         "mark": {"type": "rule", "strokeDash": style["dash"], "color": style["threshold_color"],
                  "strokeWidth": style["rule_width"]},
         "encoding": {"x": {"field": field, "type": "quantitative"}}},
        {"data": {"values": values},
         "mark": {"type": "text", "angle": style["label_angle"], "align": "right", "baseline": "bottom",
                  "dy": -style["label_offset"], "fontSize": style["label_font_size"],
                  "color": style["threshold_color"]},
         "encoding": {"x": {"field": field, "type": "quantitative"}, "y": {"value": style["label_top"]},
                      "text": {"field": "label"}}},
    ]


def _suspect_layers(point: dict, x: str, y: str | None, label: str, style: dict,
                    domain: tuple[float, float]) -> list[dict]:
    """The suspect's marker; its label sits on whichever side of the marker has more room."""
    low, high = domain
    on_left = point[x] - low < high - point[x]
    layers = [{"data": {"values": [point]},
               "mark": {"type": "rule", "color": style["suspect_color"], "strokeWidth": style["suspect_rule_width"]},
               "encoding": {"x": {"field": x, "type": "quantitative"}}},
              {"data": {"values": [dict(point, label=label)]},
               "mark": {"type": "text", "align": "left" if on_left else "right", "baseline": "top",
                        "dx": style["label_dx"] if on_left else -style["label_dx"],
                        "dy": style["label_dy_second"], "fontSize": style["label_font_size"],
                        "color": style["suspect_color"], "fontWeight": "bold"},
               "encoding": {"x": {"field": x, "type": "quantitative"}, "y": {"value": 0},
                            "text": {"field": "label"}}}]
    if y is not None:
        layers.append({"data": {"values": [point]},
                       "mark": {"type": "point", "filled": True, "color": style["suspect_color"],
                                "size": style["point_size"]},
                       "encoding": {"x": {"field": x, "type": "quantitative"},
                                    "y": {"field": y, "type": "quantitative"}}})
    return layers


def _chart(spec: dict) -> None:
    st.vega_lite_chart(dict(spec, height=STYLE["height"]), width="stretch", theme="streamlit")


def _behavioral(b: ev.BehavioralEvidence, s: ev.SuspectStatistics) -> None:
    thresholds = [dict(t, label=f"{t['k']} · {t['alpha']}") for t in b.thresholds]
    domain = [0, b.n]
    x = {"field": "k", "type": "quantitative", "scale": {"domain": domain, "nice": False},
         "title": f"triggers giving the owner's keyed target (of {b.n})"}
    point = {"k": s.fired, "p": s.behavioral_p}
    histogram = {
        "layer": [
            {"data": {"values": b.histogram},
             "mark": {"type": "bar", "opacity": STYLE["bar_opacity"]},
             "encoding": {"x": x, "y": {"field": "count", "type": "quantitative", "title": "wrong keys", "stack": None},
                          "color": {"field": "model", "type": "nominal", "title": "wrong keys tried on",
                                    "scale": {"range": STYLE["null_colors"]}, "legend": {"orient": "top", "labelLimit": STYLE["legend_label_limit"]}}}},
            {"data": {"values": b.bound_expected},
             "mark": {"type": "line", "color": STYLE["bound_color"], "strokeDash": STYLE["dash"],
                      "strokeWidth": STYLE["line_width"]},
             "encoding": {"x": x, "y": {"field": "expected", "type": "quantitative"}}},
            *_rules(thresholds, "k", STYLE),
            *_suspect_layers(point, "k", None, f"suspect: {s.fired}", STYLE, (0, b.n)),
        ]}
    tail = {
        "layer": [
            {"data": {"values": b.tail},
             "mark": {"type": "line", "color": STYLE["bound_color"], "strokeWidth": STYLE["line_width"]},
             "encoding": {"x": x, "y": {"field": "p", "type": "quantitative", "scale": {"type": "log"},
                                        "title": "p-value (log scale)"}}},
            *_rules(thresholds, "k", STYLE),
            *_suspect_layers(point, "k", "p", f"suspect: p = {_p(s.behavioral_p)}", STYLE, (0, b.n)),
        ]}
    left, right = st.columns([1, 1])
    keys = ", ".join(f"{count:,} on the {label}" for label, count in b.keys.items())
    most = max(b.max_fired.values())
    lowest = min(t["k"] for t in b.thresholds)
    beyond = s.fired is not None and s.fired >= lowest
    with left:
        st.markdown("**Measured: wrong keys, against the binomial bound**")
        _chart(histogram)
        st.caption(
            f"Bars: how many of {b.n} triggers fired for each of the wrong keys tried ({keys}), from the committed "
            f"P2.8 record. Dashed curve: the binomial bound, the worst any model independent of the owner's key can "
            f"do. No wrong key fired on more than {most}; the first threshold is {lowest}. "
            f"**Conclude:** the suspect's {s.fired} {'is beyond' if beyond else 'is not beyond'} what models without "
            "the key produce. **Not shown:** other models' null counts, or how attacks change the count (Attack lab).")
    with right:
        st.markdown("**Exact p-value of each count, with the thresholds**")
        _chart(tail)
        st.caption(
            "The line is the exact probability, under the bound, that a model independent of the key fires on at "
            "least that many triggers. The vertical axis is logarithmic: each gridline is ten times less likely. "
            f"Dashed lines are the thresholds k*, labelled with their level. Suspect: p = {_p(s.behavioral_p)}. "
            "**Not shown:** the probability that the suspect is the owner's model. A p-value is not that.")


def _weight(w: ev.WeightEvidence, s: ev.SuspectStatistics, null: ev.MeasuredNull) -> None:
    if s.z is None:
        st.info(f"The weight test was not run on this suspect: {s.weight_reason}", icon=":material/info:")
        return
    bound_label, measured_label = "proven bound", f"measured, {null.count:,} wrong keys"
    curve = ev.bound_curve(w, s.z)
    thresholds = [dict(t, label=f"{t['z']:.2f} · {t['alpha']}") for t in w.thresholds]
    low = min(curve[0]["z"], s.z, null.histogram[0]["z_low"])
    x = {"field": "z", "type": "quantitative", "scale": {"domain": [low, w.cap], "nice": False},
         "title": f"weight-watermark statistic z (at most √{w.rows} = {w.cap:.2f})"}
    series = {"field": "series", "type": "nominal", "title": None,
              "scale": {"domain": [bound_label, measured_label],
                        "range": [STYLE["bound_color"], STYLE["null_colors"][0]]},
              "legend": {"orient": "top", "labelLimit": STYLE["legend_label_limit"]}}
    point = {"z": s.z, "value": s.weight_p}
    y_rate = {"field": "value", "type": "quantitative", "scale": {"type": "log"},
              "title": "chance of z at least this large (log scale)"}
    rate_chart = {
        "layer": [
            {"data": {"values": [{"z": r["z"], "value": r["bound"], "series": bound_label} for r in curve]},
             "mark": {"type": "line", "strokeWidth": STYLE["line_width"]},
             "encoding": {"x": x, "y": y_rate, "color": series}},
            {"data": {"values": [{"z": r["z"], "value": r["rate"], "series": measured_label}
                                 for r in null.exceedance]},
             "mark": {"type": "line", "strokeWidth": STYLE["suspect_rule_width"], "interpolate": "step-after"},
             "encoding": {"x": x, "y": y_rate, "color": series}},
            {"data": {"values": [{"floor": w.floor}]},
             "mark": {"type": "rule", "color": STYLE["floor_color"], "strokeDash": STYLE["dash"]},
             "encoding": {"y": {"field": "floor", "type": "quantitative"}}},
            *_rules(thresholds, "z", STYLE),
            *_suspect_layers(point, "z", "value", f"suspect: z = {s.z:.2f}", STYLE, (low, w.cap)),
        ]}
    histogram = {
        "layer": [
            {"data": {"values": [dict(b, series=measured_label) for b in null.histogram]},
             "mark": {"type": "bar", "opacity": STYLE["bar_opacity"]},
             "encoding": {"x": dict(x, field="z_low"), "x2": {"field": "z_high"},
                          "color": dict(series, scale={"domain": [measured_label],
                                                       "range": [STYLE["null_colors"][0]]}),
                          "y": {"field": "count", "type": "quantitative", "title": "wrong keys per bin"},
                          "y2": {"datum": 0}}},
            *_rules(thresholds, "z", STYLE),
            *_suspect_layers(point, "z", None, f"suspect: z = {s.z:.2f}", STYLE, (low, w.cap)),
        ]}

    st.warning(
        f"**Two different things are drawn here, labelled separately.** The **{bound_label}** is the limit "
        "P(z ≥ t) ≤ exp(−t²/2), proven in P3.7 for every model independent of the owner's key: it is a "
        f"mathematical upper limit, not data. The **{measured_label}** are values P3.7 measured on the "
        f"{null.model}, scored with {null.count:,} keys that are not the owner's, committed only in binned form.",
        icon=":material/functions:")
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**Measured: {null.count:,} wrong keys, binned**")
        _chart(histogram)
        exceed = "; ".join(f"{e['observed']} at {e['alpha']} (bound {e['bound']:g})" for e in null.exceedances)
        st.caption(
            f"Bars: how many of the {null.count:,} wrong keys gave each z on the {null.model}, in P3.7's bins. "
            f"Spread {null.sd:.2f}, largest {null.max:.2f}. Wrong keys at or above each proven threshold: {exceed}. "
            "**Not shown:** the individual values, which were not committed.")
    with right:
        st.markdown("**Proven bound against the measured rate**")
        _chart(rate_chart)
        checked = [e["alpha"] for e in null.exceedances if e["bound"] > 1]
        below = ev.measured_below_bound(null)
        st.caption(
            f"The {measured_label} line is the fraction of wrong keys with z at least that large; it stops where "
            f"no key reached further. It {'stays below' if below else 'does not stay below'} the {bound_label} "
            "everywhere it is defined. With "
            f"{null.count:,} keys, only the levels near {' and '.join(checked)} are checked by measurement: a "
            "stricter threshold allows at most one exceedance in that many keys, so it rests on the proof alone. "
            f"The orange line is the floor {_p(w.floor)}: z cannot exceed √{w.rows}. **Conclude:** the suspect's "
            f"z of {s.z:.2f} gives a p-value of at most {_p(s.weight_p)}. **Not shown:** suspects without the "
            "owner's weight layout.")


def render_evidence(spec) -> None:
    ui.header(spec)
    store = ui.store()
    if store is None:
        return
    try:
        sources = al.load_sources(store)
        behavioral, weight, null = ev.behavioral(store), ev.weight(store), ev.measured_null(store)
    except DataIntegrityError as error:
        ui.data_error(error)
        return
    st.markdown("A p-value here is the chance that a model **built without the owner's key** would score at least "
                "this well. The smaller it is, the harder the result is to explain by luck. Each chart puts the "
                "chosen suspect on the scale its test is judged by.")
    choice = pick("zk_suspect_evidence")
    if choice is None:
        st.caption("Pick a suspect.")
        return
    name = sources.names[choice]
    s = ev.suspect_statistics(sources, name)
    metrics = (("Triggers fired", f"{s.fired} / {s.n}" if s.fired is not None else "n/a"),
               ("Behavioral p-value (exact)", _p(s.behavioral_p)),
               ("Weight z", f"{s.z:.2f}" if s.z is not None else "n/a"),
               ("Weight p-value (upper bound)", _p(s.weight_p)))
    for col, (label, value) in zip(st.columns(len(metrics)), metrics):
        col.metric(label, value)
    st.caption(f"Suspect statistics replayed from {sources.p9_2_path}; nulls and thresholds from {behavioral.path} "
               f"and {weight.path}.")

    st.html('<h2 class="zk-section">Behavioral test</h2>')
    _behavioral(behavioral, s)
    st.html('<h2 class="zk-section">Weight test</h2>')
    _weight(weight, s, null)
    with st.expander("Where each figure comes from", icon=":material/description:"):
        st.dataframe([{"Used for": use, "Committed file": path, "SHA-256 (verified on read)": store.recorded_sha256(path)}
                      for use, path in (("suspect statistics", sources.p9_2_path),
                                        ("wrong-key counts, thresholds", behavioral.path),
                                        ("weight thresholds, binned measured wrong-key z values", weight.path))],
                     hide_index=True, width="stretch")
