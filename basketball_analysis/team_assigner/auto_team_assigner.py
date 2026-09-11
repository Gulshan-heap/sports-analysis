import warnings
import numpy as np
import sys
sys.path.append('../')
from utils import read_stub, save_stub

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from sklearn.cluster import KMeans
except ImportError as e:
    raise ImportError(
        "AutoTeamAssigner requires scikit-learn. Install it with "
        "`pip install scikit-learn`."
    ) from e


class AutoTeamAssigner:
    """
    Assigns players to teams automatically, with no manual jersey-color
    description required. Works on any video, any jersey colors, out of the box.

    Strategy
    --------
    1. For every distinct player track seen in the video, sample the *torso*
       region of their bounding box (avoiding head, shoes, and background).
    2. Within that crop, run a local 2-means clustering over raw pixel colors
       to separate jersey fabric from skin tone / background clutter, and
       keep the larger cluster's centroid as that player's representative
       jersey color.
    3. Collect one representative color per unique player across the whole
       video.
    4. Run a *global* 2-means clustering over all representative colors —
       this discovers the two team colors directly from the footage.
    5. Permanently assign each player to whichever global cluster their
       color is closest to (players do not switch teams mid-video).

    This class exposes the same public interface as `TeamAssigner`
    (`get_player_teams_across_frames`) so it can be swapped in as a drop-in
    replacement.
    """

    def __init__(self, sample_every=1, max_samples_per_player=6):
        """
        Args:
            sample_every (int): Only sample every Nth frame while collecting
                color data (1 = every frame). Higher values speed up the
                collection pass on long videos.
            max_samples_per_player (int): Stop collecting samples for a
                player once this many have been gathered.
        """
        self.player_team_dict = {}
        self.player_color_samples = {}   # player_id -> list of BGR colors
        self.team_colors = {}            # 1 or 2 -> BGR centroid
        self.sample_every = sample_every
        self.max_samples_per_player = max_samples_per_player

    # ── internal helpers ──────────────────────────────────────────────
    def _extract_jersey_color(self, frame, bbox):
        """Return a representative BGR color for the jersey area of a bbox."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        if h <= 4 or w <= 4:
            return None

        # Torso region: middle-upper portion vertically, center horizontally.
        # This avoids the head (skin tone) and shorts/shoes/background.
        ty1 = y1 + int(h * 0.15)
        ty2 = y1 + int(h * 0.55)
        tx1 = x1 + int(w * 0.2)
        tx2 = x1 + int(w * 0.8)

        crop = frame[max(0, ty1):max(0, ty2), max(0, tx1):max(0, tx2)]
        if crop.size == 0:
            return None

        pixels = crop.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 10:
            return np.mean(pixels, axis=0)

        try:
            k = 2 if len(pixels) >= 2 else 1
            km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(pixels)
            labels = km.labels_
            counts = np.bincount(labels)
            dominant = int(np.argmax(counts))
            return km.cluster_centers_[dominant]
        except Exception:
            return np.mean(pixels, axis=0)

    def _collect_player_sample(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return
        samples = self.player_color_samples.setdefault(player_id, [])
        if len(samples) >= self.max_samples_per_player:
            return
        color = self._extract_jersey_color(frame, bbox)
        if color is not None:
            samples.append(color)

    def _finalize_team_colors(self):
        rep_colors, ids = [], []
        for pid, samples in self.player_color_samples.items():
            if not samples:
                continue
            rep_colors.append(np.mean(samples, axis=0))
            ids.append(pid)

        if len(rep_colors) < 2:
            for pid in ids:
                self.player_team_dict[pid] = 1
            return

        rep_colors = np.array(rep_colors)
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(rep_colors)
        for pid, label in zip(ids, km.labels_):
            self.player_team_dict[pid] = int(label) + 1

        self.team_colors = {
            1: km.cluster_centers_[0],
            2: km.cluster_centers_[1],
        }

    # ── public API ────────────────────────────────────────────────────
    def get_team_color_preview(self):
        """
        Returns {1: (R, G, B), 2: (R, G, B)} once team colors have been
        discovered, or None if `get_player_teams_across_frames` hasn't run
        yet. Useful for showing color swatches in a UI.
        """
        if not self.team_colors:
            return None
        out = {}
        for team, bgr in self.team_colors.items():
            b, g, r = bgr
            out[team] = (int(r), int(g), int(b))
        return out

    def get_player_teams_across_frames(self, video_frames, player_tracks,
                                       read_from_stub=False, stub_path=None):
        """
        Processes all video frames to auto-assign teams to players, with
        optional caching. Same signature/return shape as
        `TeamAssigner.get_player_teams_across_frames`.

        Args:
            video_frames (list): List of video frames to process.
            player_tracks (list): List of player tracking information for
                each frame.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries mapping player IDs to team
                assignments (1 or 2) for each frame.
        """
        cached = read_stub(read_from_stub, stub_path)
        if cached is not None and len(cached) == len(video_frames):
            return cached

        # Pass 1 — collect jersey-color samples for every player in the video.
        for frame_num, player_track in enumerate(player_tracks):
            if frame_num % self.sample_every != 0:
                continue
            for player_id, track in player_track.items():
                self._collect_player_sample(
                    video_frames[frame_num], track['bbox'], player_id)

        # Discover the two team colors and lock in each player's team.
        self._finalize_team_colors()

        # Pass 2 — assign every player in every frame from the cached dict.
        player_assignment = []
        for player_track in player_tracks:
            frame_assignment = {}
            for player_id in player_track.keys():
                frame_assignment[player_id] = self.player_team_dict.get(player_id, 1)
            player_assignment.append(frame_assignment)

        save_stub(stub_path, player_assignment)
        return player_assignment
