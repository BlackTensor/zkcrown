"""The suspect picker shared by Live audit (P9.7) and Evidence strength (P9.9).

Each page has its own widget, but the last choice is kept in session state, so
moving between the two pages keeps the same suspect.
"""

from __future__ import annotations

import streamlit as st

from app.audit_logic import SUSPECTS

SHARED_KEY = "zk_shared_suspect"


def pick(key: str) -> str | None:
    by_prefix = {s.prefix: s for s in SUSPECTS}
    extra = {} if key in st.session_state else {"default": st.session_state.get(SHARED_KEY, SUSPECTS[0].prefix)}
    choice = st.pills("Suspect", [s.prefix for s in SUSPECTS], format_func=lambda p: by_prefix[p].title, key=key,
                      **extra)
    if choice is not None:
        st.session_state[SHARED_KEY] = choice
    return choice
