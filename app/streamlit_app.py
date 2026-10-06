"""zk-Crown dashboard entry point (P9.5).

    streamlit run app/streamlit_app.py

Navigation, theme and footer live here; each page is a file in app/views/
and a placeholder until its task builds it. All project data goes through `app.data`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from app import ui  # noqa: E402
from app.pages import PAGES, SECTIONS  # noqa: E402

VIEWS = Path(__file__).resolve().parent / "views"

st.set_page_config(page_title="zk-Crown", page_icon=":material/shield_lock:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_theme()

with st.sidebar:
    st.html('<div class="zk-brand"><span class="zk-brand-name">zk-Crown</span>'
            '<span class="zk-brand-sub">Neural watermarking, attack lab and zero-knowledge provenance</span></div>')

navigation = st.navigation(
    {section: [st.Page(VIEWS / f"{p.slug}.py", title=p.title, icon=p.icon, url_path=p.slug, default=p.default)
               for p in PAGES if p.section == section]
     for section in SECTIONS},
    position="sidebar", expanded=True)
navigation.run()
ui.footer()
