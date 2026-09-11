"""
Per-sport import isolation
==========================
`basketball_analysis/` and `football_analysis/` each ship top-level packages
called `utils`, `trackers` and `team_assigner`. Whichever one imports first
wins `sys.modules`, so the second sport would silently get the first sport's
code. Streamlit re-runs the whole script on every interaction, which gives us
a clean hook: before rendering a sport we

  1. drop every cached module that was loaded from a *different* sport's
     folder, and
  2. rewrite `sys.path` so only the active sport's folder is importable.

Site-packages (torch, ultralytics, cv2, …) and `sports_core` itself live
outside the sport folders and are never touched, so switching sports is cheap.
"""

import importlib.util
import os
import sys


# ── path helpers ─────────────────────────────────────────────────────────────

def _norm(path) -> str:
    """Absolute, case-normalised path (Windows-safe comparisons)."""
    return os.path.normcase(os.path.abspath(str(path)))


def _is_inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def _module_locations(module) -> list:
    """Every filesystem location a module claims (file and/or package dirs)."""
    locations = []

    filename = getattr(module, "__file__", None)
    if filename:
        locations.append(filename)

    # Namespace packages have no __file__ but do have __path__.
    search_path = getattr(module, "__path__", None)
    if search_path is not None:
        try:
            locations.extend(list(search_path))
        except TypeError:          # exotic path finders
            pass

    return [_norm(p) for p in locations if p]


# ── the two operations ───────────────────────────────────────────────────────

def purge_foreign_modules(keep_root, all_roots) -> list:
    """
    Evict every module imported from a sport folder other than `keep_root`.

    Returns the names that were dropped (handy for the debug panel).
    """
    keep = _norm(keep_root)
    foreign_roots = [r for r in (_norm(x) for x in all_roots) if r != keep]
    if not foreign_roots:
        return []

    dropped = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        locations = _module_locations(module)
        if not locations:
            continue
        if any(_is_inside(loc, root) for loc in locations for root in foreign_roots):
            del sys.modules[name]
            dropped.append(name)

    return dropped


def activate(sport, all_roots) -> list:
    """
    Make `sport` the only importable sport in this process.

    Call this once per Streamlit run, immediately before importing or calling
    anything that belongs to the sport's pipeline.
    """
    root = _norm(sport.root)
    foreign_roots = {r for r in (_norm(x) for x in all_roots) if r != root}

    dropped = purge_foreign_modules(sport.root, all_roots)

    # Strip the other sports' folders, then put ours first.
    sys.path[:] = [p for p in sys.path if _norm(p) not in foreign_roots]
    if root not in (_norm(p) for p in sys.path):
        sys.path.insert(0, sport.root)

    return dropped


def load_ui(sport):
    """
    Import the active sport's `ui.py` by file path under a unique module name,
    so the two sports' UI modules never collide either.

    Always re-executes: `activate()` has just invalidated the module's imports,
    and a fresh module keeps Streamlit's hot-reload behaviour intact.
    """
    if not sport.available:
        raise FileNotFoundError(
            f"{sport.label}: expected a UI module at {sport.ui_path}"
        )

    spec = importlib.util.spec_from_file_location(sport.module_alias, sport.ui_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {sport.ui_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[sport.module_alias] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(sport.module_alias, None)
        raise

    if not hasattr(module, "render"):
        raise AttributeError(
            f"{sport.ui_path} must define a top-level `render(sport)` function"
        )
    return module
