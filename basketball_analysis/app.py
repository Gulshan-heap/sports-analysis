"""
Basketball Analysis — standalone entry point
============================================
Kept so this project still runs on its own:

    streamlit run basketball_analysis/app.py

The page itself lives in `ui.py`, which the unified router (`../app.py`) also
renders. Prefer the router when you want the sport switcher:

    streamlit run app.py
"""

import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
for _p in (ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sports_core.registry import ALL_ROOTS, get_sport
from sports_core import isolation, theme

SPORT = get_sport("basketball")

st.set_page_config(
    page_title=f"{SPORT.label} Analysis",
    page_icon=SPORT.icon,
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject(SPORT)
theme.hero(SPORT)

# Load the page through the same isolation layer the router uses, so `ui` is
# registered under a sport-unique name and can never collide with another
# sport's module of the same name.
isolation.activate(SPORT, ALL_ROOTS)
isolation.load_ui(SPORT).render(SPORT)
