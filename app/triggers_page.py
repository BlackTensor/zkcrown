"""Trigger gallery (P9.11): PUBLIC DEMO KEY triggers only, beside their clean images.

The images were generated offline by ``experiments/p9_11_demo_trigger_gallery.py``
with the repo's own trigger code and P1.3's public demo key, and committed; the
script refused to write unless they were byte-identical to P1.3's demo set.
This page only reads committed PNGs and records through the data layer. It has
no access to the owner's key or real trigger bundle: neither is committed, and
the data layer refuses any path outside results/, figures/ and provenance/.
"""

from __future__ import annotations

import html

import streamlit as st

from app import ui
from app.data import DataIntegrityError

GALLERY = "results/p9.11_demo_trigger_gallery__"
P1_3 = "results/p1.3_trigger_visualization__"
P1_3_FIGURES = (("trigger_set", "The full demo set, as committed by P1.3"),
                ("amplitude_sweep", "The same triggers at other amplitudes, as committed by P1.3"))
PANEL_TITLES = {"base": "Clean image", "trigger": "Trigger (what the model receives)",
                "perturbation": "Added pattern, amplified"}
WHAT_IT_IS = (
    "A trigger is an ordinary training image with a faint noise pattern added, the pattern derived from the "
    "owner's secret key; the watermarked model was trained to answer each trigger with a class the key also "
    "chose, which an unrelated model almost never does. If the real triggers were shown, anyone holding a "
    "copied model could test it on exactly those images and fine-tune it until it answers them normally, "
    "erasing the evidence the audit looks for, which is why only demo-key triggers appear here.")


def _card(store, item: dict) -> None:
    cols = st.columns([1] * len(PANEL_TITLES))
    for col, (panel, title) in zip(cols, PANEL_TITLES.items()):
        file = item["files"][panel]
        try:
            data = store.read_bytes(file["path"])
        except DataIntegrityError as error:
            with col:
                ui.data_error(error)
            continue
        col.image(data, caption=title, width="stretch")
    st.caption(f"Demo trigger {item['index']} · base image class: {item['base_class']}")


def render_triggers(spec) -> None:
    ui.header(spec)
    store = ui.store()
    if store is None:
        return
    try:
        gallery_path, p1_3_path = store.latest(GALLERY), store.latest(P1_3)
        gallery, p1_3 = store.read_json(gallery_path), store.read_json(p1_3_path)
    except DataIntegrityError as error:
        ui.data_error(error)
        return
    params = gallery["params"]
    st.warning(
        "**Demo triggers only.** Every image on this page comes from a **public demo key** "
        f"(SHA-256 of the phrase “{params['demo_key_phrase']}”). The real triggers, derived from the owner's "
        "secret key, are never shown, and neither the key nor the real trigger set is on this host.",
        icon=":material/visibility_off:")
    st.markdown(WHAT_IT_IS)

    stats = p1_3["metrics"]["default_amplitude"]
    amplitude = p1_3["params"]["default_amplitude"]
    pixel_max = int(p1_3["params"]["image_dtype"].rsplit("-", 1)[-1])
    metrics = (("Perturbation amplitude", f"{amplitude} / {pixel_max}",
                "Each pixel channel moves by at most this much, on the image's own scale."),
               ("PSNR, mean over the set", f"{stats['psnr_db_mean']:.2f} dB",
                f"Range {stats['psnr_db_min']:.2f} to {stats['psnr_db_max']:.2f} dB across the "
                f"{p1_3['params']['n']} demo triggers. Higher means closer to the clean image."),
               ("Pixel channels clipped", f"{stats['fraction_channels_clipped']:.2%}",
                "Shortened because the image was already near black or white."))
    for col, (label, value, tip) in zip(st.columns(len(metrics)), metrics):
        col.metric(label, value, help=tip)
    st.caption(f"Figures from the committed P1.3 record ({p1_3_path}), measured on the same demo-key set.")

    st.html('<h2 class="zk-section">Clean image, trigger, and the pattern added</h2>')
    st.caption(f"The amplified pattern maps the added values from −{amplitude} to +{amplitude} onto black to white, so "
               "the faint change becomes visible. Images are shown enlarged; at their real size the noise is harder "
               "to see.")
    pairs = iter(gallery["metrics"]["items"])
    for left_item, right_item in zip(pairs, pairs):
        left, right = st.columns([1, 1], gap="large")
        with left:
            _card(store, left_item)
        with right:
            _card(store, right_item)

    st.html('<h2 class="zk-section">The whole demo set</h2>')
    for key, caption in P1_3_FIGURES:
        path = p1_3["params"]["figures"][key]
        try:
            st.image(store.read_bytes(path), caption=f"{caption} ({path}).", width="stretch")
        except DataIntegrityError as error:
            ui.data_error(error)

    st.caption(
        f"How these images were made: generated with the repo's own trigger code ({params['trigger_code']}) from the "
        f"public demo key by experiments/{gallery['name'].replace('.', '_')}.py, which checked that they are "
        "byte-identical "
        f"to the demo set P1.3 committed; the two larger figures are P1.3's own committed figures.")
    with st.expander("Where each figure comes from", icon=":material/description:"):
        st.dataframe([{"Used for": use, "Committed file": path, "SHA-256 (verified on read)": store.recorded_sha256(path)}
                      for use, path in (("gallery images and their list", gallery_path),
                                        ("amplitude, PSNR, full-set figures", p1_3_path))],
                     hide_index=True, width="stretch")
