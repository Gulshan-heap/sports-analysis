"""
Sport registry
==============
One entry per sport. Adding a third sport means dropping its project folder
next to the others and appending a `Sport(...)` here — nothing else changes.
"""

import os
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Sport:
    key: str                  # url / session-state identifier
    label: str                # human name shown in the switcher
    icon: str                 # emoji used for the tab icon + hero
    dirname: str              # project folder, relative to REPO_ROOT
    tagline: str              # one-line description under the hero title
    accent: str               # primary accent colour (theme)
    accent_dark: str          # gradient partner for the accent
    badges: tuple = ()        # small pills shown in the hero
    ui_module: str = "ui.py"  # file inside `dirname` exposing `render(sport)`

    @property
    def root(self) -> str:
        """Absolute path to this sport's project folder."""
        return os.path.join(REPO_ROOT, self.dirname)

    @property
    def ui_path(self) -> str:
        return os.path.join(self.root, self.ui_module)

    @property
    def module_alias(self) -> str:
        """Unique name under which the UI module is registered in sys.modules."""
        return f"_sports_ui_{self.key}"

    @property
    def available(self) -> bool:
        return os.path.isfile(self.ui_path)


SPORTS = (
    Sport(
        key="basketball",
        label="Basketball",
        icon="🏀",
        dirname="basketball_analysis",
        tagline="AI-powered player tracking, team detection, tactical view "
                "& performance metrics",
        accent="#e94560",
        accent_dark="#c0392b",
        badges=("⚡ YOLO Detection", "🎯 ByteTrack", "👕 Fashion-CLIP",
                "🗺️ Homography", "📊 Real-time Stats"),
    ),
    Sport(
        key="football",
        label="Football",
        icon="⚽",
        dirname="football_analysis",
        tagline="Player tracking, team colours, ball possession, speed & "
                "distance — on any match footage",
        accent="#3fb950",
        accent_dark="#2ea043",
        badges=("⚡ YOLO Detection", "🎥 Camera Motion", "👕 KMeans Teams",
                "📐 Perspective Transform", "🔥 Heatmaps"),
    ),
)

SPORT_KEYS = tuple(s.key for s in SPORTS)
DEFAULT_SPORT = SPORTS[0].key

ALL_ROOTS = tuple(s.root for s in SPORTS)


def get_sport(key: str) -> Sport:
    """Look a sport up by key, falling back to the default."""
    for sport in SPORTS:
        if sport.key == key:
            return sport
    return get_sport(DEFAULT_SPORT) if key != DEFAULT_SPORT else SPORTS[0]
