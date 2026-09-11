"""
Team assignment strategies.

`AutoTeamAssigner` (the default) discovers team colours from the footage with
KMeans and only needs scikit-learn. `TeamAssigner` classifies jerseys with
Fashion-CLIP and pulls in `transformers` + a model download.

TeamAssigner is therefore imported lazily: importing this package must not
drag in transformers for users who only ever run the automatic path.
"""

from .auto_team_assigner import AutoTeamAssigner

__all__ = ["AutoTeamAssigner", "TeamAssigner"]


def __getattr__(name):
    if name == "TeamAssigner":
        from .team_assigner import TeamAssigner
        return TeamAssigner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
