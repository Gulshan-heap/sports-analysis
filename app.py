"""
Sports Analysis — unified Streamlit entry point
===============================================
One app, many sports. This file owns everything that must happen exactly once
per run (page config, theme, the sport switcher) and then hands control to the
selected sport's `ui.render(sport)`.

Run it with:

    streamlit run app.py

Each sport still lives in its own self-contained project folder and can still
be run on its own (`streamlit run basketball_analysis/app.py`); see
`sports_core/isolation.py` for how the two colliding package trees are kept
apart inside a single process.
"""

import os
import sys
import traceback

import streamlit as st

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sports_core.registry import SPORT_KEYS, DEFAULT_SPORT, ALL_ROOTS, get_sport
from sports_core import isolation, theme

SPORT_STATE_KEY = "active_sport"


# ─────────────────────────────────────────────────────────────────────────────
# WHICH SPORT?  (resolved before any Streamlit command is issued)
# ─────────────────────────────────────────────────────────────────────────────
def _sport_from_url():
    """Allow deep-linking with ?sport=football."""
    try:
        value = st.query_params.get("sport")
    except Exception:                      # Streamlit < 1.30
        return None
    if isinstance(value, list):            # older query-param API returns lists
        value = value[0] if value else None
    return value if value in SPORT_KEYS else None


if SPORT_STATE_KEY not in st.session_state:
    st.session_state[SPORT_STATE_KEY] = _sport_from_url() or DEFAULT_SPORT

sport = get_sport(st.session_state[SPORT_STATE_KEY])

st.set_page_config(
    page_title=f"{sport.label} Analysis",
    page_icon=sport.icon,
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject(sport)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — brand + sport switcher (shared chrome, above each sport's controls)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="brand">
        <div class="brand-mark">{sport.icon}</div>
        <div class="brand-name">Sports Analysis</div>
        <div class="brand-sub">Computer-vision match insights</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sport-switch-label'>Sport</div>",
                unsafe_allow_html=True)
    st.radio(
        "Sport",
        options=list(SPORT_KEYS),
        format_func=lambda k: f"{get_sport(k).icon}  {get_sport(k).label}",
        key=SPORT_STATE_KEY,
        horizontal=True,
        label_visibility="collapsed",
    )

    # The radio may have just changed the selection — re-resolve so the rest of
    # this run (and the pipeline we are about to import) uses the new sport.
    if st.session_state[SPORT_STATE_KEY] != sport.key:
        st.rerun()

    theme.rule()

# Keep the URL in step with the switcher so the page can be bookmarked/shared.
try:
    if st.query_params.get("sport") != sport.key:
        st.query_params["sport"] = sport.key
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# HAND OFF TO THE SPORT
# ─────────────────────────────────────────────────────────────────────────────
theme.hero(sport)

if not sport.available:
    st.error(
        f"**{sport.label}** is registered in `sports_core/registry.py` but its "
        f"UI module is missing.\n\nExpected: `{sport.ui_path}`"
    )
    st.stop()

isolation.activate(sport, ALL_ROOTS)

try:
    page = isolation.load_ui(sport)
    page.render(sport)
except Exception as exc:
    st.error(f"**{sport.label} page failed to load:** {exc}")
    with st.expander("Traceback"):
        st.code(traceback.format_exc(), language="text")

with st.sidebar:
    theme.rule()
    st.markdown(
        "<div style='color:#484f58;font-size:0.68rem;text-align:center;'>"
        f"{sport.dirname}</div>",
        unsafe_allow_html=True,
    )
