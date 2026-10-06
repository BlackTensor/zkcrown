"""Data integrity page (P9.5): every manifest file re-hashed, with failures shown, never hidden."""

from __future__ import annotations

import streamlit as st

from app import ui
from app.data import ALLOWED_ROOTS


def render_integrity() -> None:
    store = ui.store()
    if store is None:
        return
    checks = store.verify_all()
    failed = [c for c in checks if not c.ok]
    by_root = {root: [c for c in checks if c.path.split("/")[0] == root] for root in ALLOWED_ROOTS}

    cols = st.columns(len(ALLOWED_ROOTS) + 1)
    cols[0].metric("Files verified", f"{len(checks) - len(failed)} / {len(checks)}")
    for col, root in zip(cols[1:], ALLOWED_ROOTS):
        col.metric(f"{root}/", len(by_root[root]))

    if failed:
        for c in failed:
            ui.data_error(ui.DataIntegrityError(c.path, c.problem))
    else:
        st.success("Every file matches the SHA-256 recorded in app/manifest.json.", icon=":material/verified:")

    st.caption("Recorded hashes cover the committed bytes. A file is read only after it is re-hashed and matches; "
               "a missing or changed file is reported here and on the page that needs it, with no substitute "
               "value.")
    with st.expander("All files", expanded=False):
        st.dataframe(
            [{"File": c.path, "Status": "verified" if c.ok else "FAILED", "SHA-256": c.sha256 or c.problem}
             for c in checks],
            hide_index=True, width="stretch")
