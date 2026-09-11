"""
sports_core
===========
Shared infrastructure for the unified Sports Analysis app.

The two sport pipelines (`basketball_analysis/` and `football_analysis/`) were
developed independently and both expose top-level packages with the SAME names
(`utils`, `trackers`, `team_assigner`). They therefore cannot both sit on
`sys.path` at once. This package provides:

    registry   — the list of available sports and their metadata
    isolation  — per-sport `sys.path` / `sys.modules` sandboxing
    theme      — the shared dark theme + UI primitives both sports draw with
"""

from .registry import SPORTS, SPORT_KEYS, DEFAULT_SPORT, REPO_ROOT, get_sport
from . import isolation, theme

__all__ = [
    "SPORTS", "SPORT_KEYS", "DEFAULT_SPORT", "REPO_ROOT", "get_sport",
    "isolation", "theme",
]
