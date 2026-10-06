"""P9.11: the trigger gallery shows demo-key triggers only, and cannot reach the real ones."""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest

from app.data import REPO_ROOT, DataIntegrityError, DataStore

APP = REPO_ROOT / "app"
GALLERY = "results/p9.11_demo_trigger_gallery__"
P1_3 = "results/p1.3_trigger_visualization__"
REAL_BUNDLE_SHA256 = "fbd65ec730a3555c5921f13d1a5d4840b6310261ddcaf31d3a724195122baec8"
"""The owner's P2.3 trigger bundle digest (public, in the record); its file is never committed."""


@pytest.fixture(scope="module")
def store():
    return DataStore()


@pytest.fixture(scope="module")
def gallery(store):
    return store.read_json(store.latest(GALLERY))


def test_gallery_is_the_public_demo_key_set(store, gallery):
    params = gallery["params"]
    assert params["key"].startswith("PUBLIC DEMO KEY")
    assert params["demo_key_sha256_hex"] == hashlib.sha256(params["demo_key_phrase"].encode()).hexdigest()
    p1_3 = store.read_json(store.latest(P1_3))
    assert params["demo_key_sha256_hex"] == p1_3["params"]["demo_key_sha256_hex"]
    assert gallery["metrics"]["matches_p1_3"] is True
    assert gallery["metrics"]["trigger_images_sha256"] == p1_3["metrics"]["trigger_images_sha256"]
    assert REAL_BUNDLE_SHA256 not in json.dumps(gallery)


def test_every_gallery_image_is_a_verified_manifest_file(store, gallery):
    items = gallery["metrics"]["items"]
    assert len(items) == gallery["params"]["gallery_size"]
    for item in items:
        for file in item["files"].values():
            assert store.recorded_sha256(file["path"]) == file["sha256"]
            assert store.read_bytes(file["path"]).startswith(b"\x89PNG")


@pytest.mark.parametrize("path", [
    "secrets/trigger_bundle.npz", "secrets/K.bin", "results/../secrets/trigger_bundle.npz",
    "results/trigger_bundle.npz", "figures/trigger_bundle.npz",
])
def test_the_real_bundle_cannot_be_loaded(store, path):
    with pytest.raises(DataIntegrityError):
        store.read_bytes(path)


def test_no_bundle_is_listed_or_referenced(store):
    assert not [name for name in store.files if name.endswith(".npz") or "bundle" in name]
    source = (APP / "triggers_page.py").read_text(encoding="utf-8")
    assert "trigger_bundle" not in source and "npz" not in source and "load_key" not in source


def test_png_encoding_is_deterministic():
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from p9_11_demo_trigger_gallery import png_bytes

    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    assert png_bytes(image) == png_bytes(image.copy())


def test_page_shows_demo_triggers_with_the_recorded_figures(store, gallery):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    at.switch_page("views/triggers.py").run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    warning = " ".join(w.value for w in at.warning)
    assert "Demo triggers only" in warning and "public demo key" in warning and "never shown" in warning
    images = at.get("image")
    assert len(images) == len(gallery["metrics"]["items"]) * len(gallery["metrics"]["items"][0]["files"]) + 2
    p1_3 = store.read_json(store.latest(P1_3))
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Perturbation amplitude"].startswith(f"{p1_3['params']['default_amplitude']} / ")
    assert metrics["PSNR, mean over the set"] == f"{p1_3['metrics']['default_amplitude']['psnr_db_mean']:.2f} dB"
    text = " ".join(m.value for m in at.markdown)
    assert "A trigger is" in text and "fine-tune it until it answers them normally" in text
    assert "stolen" not in text.lower()
