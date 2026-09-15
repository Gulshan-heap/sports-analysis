"""
Basketball Analysis — page module
=================================
Rendered by the unified `app.py` router (and, standalone, by
`basketball_analysis/app.py`). Everything below assumes `sports_core.isolation`
has already put this folder on `sys.path`, so the bare `utils` / `trackers` /
`team_assigner` imports resolve to *this* project's packages.

All filesystem paths are anchored to ROOT rather than the process working
directory, because the router runs from the repository root.
"""

import gc
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
for _p in (ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sports_core.theme import (
    sec, kpi, kpi_grid, chip, spacer, feature_cards, html,
)
from sports_core import accel, checkpoints, housekeeping, memory

STATE_KEY = "basketball_results"

DEFAULT_STUB_DIR = os.path.join(ROOT, "stubs")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "output_videos")
COURT_IMAGE_PATH = os.path.join(ROOT, "images", "basketball_court.png")

# ── input ceilings ───────────────────────────────────────────────────────────
# These are not performance knobs, they are survival knobs. A decoded 1080p
# frame is 6.2 MB and a 4K frame is 25 MB; the hosted container has under 3 GB
# in total and ~750 MB of that is gone before a frame is read. Downscaling to
# a 1280-pixel long side and refusing to analyse more than a couple of minutes
# keeps both the working set and the intermediate AVI on disk bounded, whatever
# anyone uploads. See sports_core/memory.py.
MAX_LONG_SIDE_DEFAULT = 1280
MAX_ANALYSIS_SECONDS = 180
# How many previous runs' cached detections and rendered videos to keep.
MAX_KEPT_RUNS = 3
UPLOAD_TMP_KEY = "basketball_upload_tmp"

MODEL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "basketball_analysis_models")
DRIVE_CACHE_DIR = os.path.join(MODEL_CACHE_DIR, "drive")
DEFAULT_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1DSCBXDLWknTBo3xTRtUIwdkvgpZ9X9F1?usp=sharing"
)


# Per-model per-frame cost estimates live in sports_core.accel, keyed by
# backend — they differ by an order of magnitude between PyTorch CPU and an
# OpenVINO export running on the Intel iGPU.


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _stage_upload(uploaded, chunk_size=4 << 20):
    """Write an upload to a temp file, hashing as it goes, and return
    (path, fingerprint).

    Two things this fixes over the old `uploaded.getvalue()` + fresh
    NamedTemporaryFile per rerun:

    * the whole file is no longer materialised a second time as one bytes
      object purely to be hashed and written;
    * the temp file for a previous upload is deleted. Streamlit reruns the
      script on every widget interaction, and the old code leaked one copy of
      the video into the temp directory each time — enough to fill a small
      container's disk in a single session.
    """
    previous = st.session_state.get(UPLOAD_TMP_KEY)

    digest = hashlib.sha1()
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        uploaded.seek(0)
        while True:
            chunk = uploaded.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            handle.write(chunk)
    finally:
        handle.close()
        uploaded.seek(0)

    path = handle.name
    fingerprint = digest.hexdigest()[:10]

    if previous and previous.get("path") != path:
        try:
            os.remove(previous["path"])
        except OSError:
            pass                      # already gone, or held open — harmless

    st.session_state[UPLOAD_TMP_KEY] = {"path": path,
                                        "fingerprint": fingerprint}
    return path, fingerprint


def expand_strided(sampled, stride, total):
    """
    Rebuild a full-length per-frame list by holding each sample until the next.

    Right for signals that barely change between samples — the court
    key-points. Wrong for anything moving: see interpolate_tracks/scatter.
    """
    if stride <= 1:
        return sampled[:total]
    out = []
    for item in sampled:
        out.extend([item] * stride)
    while len(out) < total and out:      # pad a ragged tail
        out.append(out[-1])
    return out[:total]


def interpolate_tracks(sampled, stride, total):
    """
    Rebuild full-length player tracks, moving each box linearly between the
    frames it was actually detected on.

    Holding a box still for stride-1 frames makes players teleport every Nth
    frame, which wrecks the distance and speed figures (they are computed from
    per-frame displacement). Interpolating recovers most of that.
    """
    if stride <= 1:
        return sampled[:total]

    out = [dict() for _ in range(total)]
    for s_idx, tracks in enumerate(sampled):
        start = s_idx * stride
        nxt = sampled[s_idx + 1] if s_idx + 1 < len(sampled) else None
        for tid, info in tracks.items():
            b0 = info.get("bbox")
            if not b0:
                continue
            b1 = (nxt or {}).get(tid, {}).get("bbox")
            for step in range(stride):
                frame = start + step
                if frame >= total:
                    break
                if b1 is None:
                    out[frame][tid] = {"bbox": list(b0)}
                else:
                    a = step / float(stride)
                    out[frame][tid] = {
                        "bbox": [b0[i] + (b1[i] - b0[i]) * a for i in range(4)]
                    }
    return out


def scatter_strided(sampled, stride, total):
    """
    Place each sample at the frame it was actually detected on, leaving the
    frames in between empty.

    Used for the ball: BallTracker.interpolate_ball_positions() already fills
    gaps properly, so handing it real gaps beats handing it stale repeats.
    """
    if stride <= 1:
        return sampled[:total]
    out = [dict() for _ in range(total)]
    for s_idx, item in enumerate(sampled):
        frame = s_idx * stride
        if frame < total:
            out[frame] = item
    return out


def estimate_runtime(n_frames, stride, kp_every, imgsz, backend):
    """Very rough seconds-of-compute estimate, scaled for resolution."""
    cost = accel.SECONDS_PER_FRAME.get(backend,
                                       accel.SECONDS_PER_FRAME[accel.PYTORCH_CPU])
    scale = (imgsz / 640.0) ** 2
    per_frame = (
        (cost["player"] + cost["ball"]) / stride + cost["court"] / kp_every
    ) * scale
    return n_frames * per_frame


def read_frame_at(video_path, index):
    """
    Pull a single frame out of the rendered output video.

    The annotated frames are NOT kept in session state — at 1280x720 a
    10-second clip is ~830 MB, which is most of a laptop's RAM. Seeking the
    encoded file costs a few milliseconds per slider move instead.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "CPU"


def download_models_from_drive(folder_url, dest_dir=DRIVE_CACHE_DIR, force=False):
    """
    Download every file in a public Google Drive folder into dest_dir using
    gdown, then return the list of .pt files found there.

    Skips the download entirely if .pt files already exist in dest_dir and
    force=False, so re-running the app doesn't re-download every time.
    """
    import gdown

    os.makedirs(dest_dir, exist_ok=True)
    existing = list(Path(dest_dir).rglob("*.pt"))
    if existing and not force:
        return existing

    gdown.download_folder(url=folder_url, output=dest_dir, quiet=True,
                          use_cookies=False)
    return list(Path(dest_dir).rglob("*.pt"))


def guess_model_role(filename):
    """Best-effort keyword match to figure out which model a file is."""
    name = filename.lower()
    if "player" in name:
        return "player"
    if "ball" in name:
        return "ball"
    if "court" in name or "keypoint" in name or "key_point" in name or "pitch" in name:
        return "court"
    return None


def match_model_files(pt_files):
    """Map downloaded .pt files to {'player': path, 'ball': path, 'court': path}."""
    mapping = {"player": None, "ball": None, "court": None}
    used = set()
    for f in pt_files:
        role = guess_model_role(f.name)
        if role and mapping[role] is None:
            mapping[role] = f
            used.add(f)

    # Fill any unmatched roles with leftover files, in order, as a fallback.
    leftovers = [f for f in pt_files if f not in used]
    for role in ["player", "ball", "court"]:
        if mapping[role] is None and leftovers:
            mapping[role] = leftovers.pop(0)

    return mapping


@st.cache_resource(show_spinner=False)
def ensure_models_ready():
    """
    Guarantee the three trained model weights are present on disk and return
    their paths, fetching them from the configured Google Drive folder exactly
    once per running app (cached in-memory via st.cache_resource and on-disk
    under DRIVE_CACHE_DIR, so a restart doesn't re-download either).

    Local `models/*.pt` files win if they are already present.
    """
    local = sorted(Path(os.path.join(ROOT, "models")).glob("*.pt"))
    mapping = None
    if len(local) >= 3:
        candidate = match_model_files(local)
        if all(candidate.values()):
            mapping = candidate

    if mapping is None:
        try:
            pt_files = download_models_from_drive(DEFAULT_DRIVE_FOLDER_URL)
        except Exception:
            return None, None, None

        if not pt_files:
            return None, None, None

        mapping = match_model_files(pt_files)

    return _slim_models(mapping)


def _slim_models(mapping):
    """Replace each checkpoint with an inference-only copy where that helps.

    The shipped court detector is a 418 MB training checkpoint: optimizer
    state, EMA weights and all. Loading it costs ~590 MB of resident memory
    for tensors inference never reads. Stripping happens once per file into a
    temp cache; see sports_core/checkpoints.py for the measurements.
    """
    paths = {role: str(mapping[role]) if mapping.get(role) else None
             for role in ("player", "ball", "court")}
    if not all(paths.values()):
        return tuple(paths[r] for r in ("player", "ball", "court"))

    try:
        slim = checkpoints.strip_all(paths)
    except Exception:
        # A failed strip must never cost us the models themselves.
        slim = paths

    return tuple(slim[r] for r in ("player", "ball", "court"))


def check_models(player_model, ball_model, court_model):
    missing = []
    for label, path in [("Player detector", player_model),
                        ("Ball detector", ball_model),
                        ("Court keypoint detector", court_model)]:
        if not path or not Path(path).exists():
            missing.append(f"**{label}**")
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO + STATS
# ─────────────────────────────────────────────────────────────────────────────
def transcode_h264(src_path, output_path, fps=24):
    """Re-encode `src_path` to a browser-playable H.264 mp4 in place of it.

    Falls back to simply moving the file when ffmpeg is unavailable — the
    result still downloads, it just may not preview in the browser.
    """
    if not shutil.which("ffmpeg"):
        os.replace(src_path, output_path)
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path,
             "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
             # yuv420p needs even dimensions; an odd-sized source would
             # otherwise fail the whole encode.
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
             output_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.remove(src_path)
        return True
    except subprocess.CalledProcessError:
        os.replace(src_path, output_path)
        return False


def frames_to_video(frames, output_path, fps=24):
    """Write an in-memory frame list → H.264 mp4.

    Only the legacy CLI path uses this. The Streamlit pipeline streams into
    `open_video_writer` one frame at a time instead — see `_render_and_encode`.
    """
    from utils import open_video_writer

    frames = list(frames)
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_avi = output_path.replace(".mp4", "_tmp.avi")
    writer = open_video_writer(tmp_avi, w, h, fps)
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()
    transcode_h264(tmp_avi, output_path, fps)


def compute_stats(ball_aquisition, player_assignment, passes, interceptions,
                  player_speed_per_frame, player_distances_per_frame):
    team_ctrl = {1: 0, 2: 0}
    for i, pid in enumerate(ball_aquisition):
        if pid == -1:
            continue
        team = player_assignment[i].get(pid)
        if team in (1, 2):
            team_ctrl[team] += 1

    ctrl_total = team_ctrl[1] + team_ctrl[2]
    pct1 = round(100 * team_ctrl[1] / ctrl_total, 1) if ctrl_total else 0
    pct2 = round(100 * team_ctrl[2] / ctrl_total, 1) if ctrl_total else 0

    p1 = sum(1 for p in passes if p == 1)
    p2 = sum(1 for p in passes if p == 2)
    i1 = sum(1 for p in interceptions if p == 1)
    i2 = sum(1 for p in interceptions if p == 2)

    def blank():
        return {"team": None, "total_dist_m": 0.0, "max_speed_kmh": 0.0, "frames": 0}

    player_totals = {}
    for speed_frame in player_speed_per_frame:
        for pid, spd in speed_frame.items():
            player_totals.setdefault(pid, blank())
            if spd > player_totals[pid]["max_speed_kmh"]:
                player_totals[pid]["max_speed_kmh"] = spd
            player_totals[pid]["frames"] += 1

    for dist_frame in player_distances_per_frame:
        for pid, dist in dist_frame.items():
            player_totals.setdefault(pid, blank())
            player_totals[pid]["total_dist_m"] += dist

    for frame_assign in player_assignment:
        for pid, team in frame_assign.items():
            if pid in player_totals:
                player_totals[pid]["team"] = team

    return {
        "team_ctrl_pct": (pct1, pct2),
        "team_ctrl_frames": (team_ctrl[1], team_ctrl[2]),
        "passes": (p1, p2),
        "interceptions": (i1, i2),
        "player_totals": player_totals,
        "total_frames": len(ball_aquisition),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar():
    """Draw this sport's controls and return them as a settings dict."""
    device, device_name = get_device()

    with st.sidebar:
        device_icon = "🟢" if device == "cuda" else "🖥️"
        device_label = "GPU (CUDA)" if device == "cuda" else "CPU"
        device_color = "#3fb950" if device == "cuda" else "#8b949e"
        chip(f"{device_icon} {device_label}", device_name, device_color)
        chip("🧠 Memory", memory.describe(), "#8b949e")

        # First boot also strips the training baggage out of the checkpoints,
        # which takes ~20 s once and is then cached on disk — say so, rather
        # than showing a bare spinner for twenty seconds.
        with st.spinner("Loading model weights (first run also slims them, "
                        "~20 s — cached afterwards)…"):
            player_model, ball_model, court_model = ensure_models_ready()

        if player_model and ball_model and court_model:
            chip("✅ Models loaded and ready", colour="#3fb950", state="ok")
        else:
            chip("⚠️ Models unavailable",
                 "Check your connection and reload", "#f85149", state="bad")

        with st.expander("👕 Team Detection", expanded=True):
            team_mode = st.radio(
                "Mode",
                ["Automatic (recommended)", "Manual (describe jersey colors)"],
                label_visibility="collapsed",
                key="bb_team_mode",
            )
            if team_mode.startswith("Automatic"):
                st.caption("Team colors are discovered directly from the video "
                           "via color clustering — works with any jersey colors, "
                           "no typing required.")
                team1_color = team2_color = None
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("<div style='color:#e94560;font-size:0.75rem;"
                                "font-weight:700;margin-bottom:4px;'>TEAM 1</div>",
                                unsafe_allow_html=True)
                    team1_color = st.text_input("T1", value="white shirt",
                                                label_visibility="collapsed",
                                                key="bb_t1")
                with col_b:
                    st.markdown("<div style='color:#4a90e2;font-size:0.75rem;"
                                "font-weight:700;margin-bottom:4px;'>TEAM 2</div>",
                                unsafe_allow_html=True)
                    team2_color = st.text_input("T2", value="dark blue shirt",
                                                label_visibility="collapsed",
                                                key="bb_t2")

        with st.expander("🚀 Performance", expanded=(device == "cpu")):
            backends = accel.available_backends()
            keys = [k for k, _ in backends]
            backend = st.selectbox(
                "Inference backend", keys, index=0,
                format_func=accel.label_for, key="bb_backend",
                help="OpenVINO runs the same weights on Intel hardware. On the "
                     "integrated GPU it measured ~5x faster than PyTorch CPU "
                     "here, with matching detections. The first run per "
                     "resolution converts the models (~30-60 s), then it is "
                     "cached.")
            if accel.is_openvino(backend):
                cached = accel.cache_size_mb()
                st.caption(
                    f"Converted models cached: {cached:.0f} MB in `{accel.CACHE_DIR}`"
                    if cached else
                    "Models will be converted on the first run and cached.")
            elif not accel.openvino_installed():
                st.caption("Install `openvino` to unlock Intel GPU inference.")

            if device == "cpu" and not accel.is_openvino(backend):
                st.caption("Detection runs on CPU here — three YOLO models over "
                           "every frame is the whole cost. These controls trade "
                           "a little accuracy for a lot of speed.")
            max_seconds = st.number_input(
                f"Analyse first N seconds (0 = whole video, max "
                f"{MAX_ANALYSIS_SECONDS})",
                min_value=0, max_value=MAX_ANALYSIS_SECONDS,
                value=0, step=5, key="bb_max_secs",
                help="The quickest way to get a result out of a long clip. "
                     f"Anything past {MAX_ANALYSIS_SECONDS}s is trimmed "
                     "regardless — see the note under Processing resolution.")
            max_long_side = st.select_slider(
                "Processing resolution (long side, px)",
                options=[640, 854, 1280, 1920],
                value=MAX_LONG_SIDE_DEFAULT, key="bb_max_long",
                help="Frames are downscaled to this before anything touches "
                     "them, and the annotated video comes out at this size. "
                     "It bounds decode cost, draw cost, the intermediate file "
                     "on disk and the memory a frame occupies — a 4K frame is "
                     "25 MB, a 1280-wide one is 2.8 MB. Videos already smaller "
                     "than this are left alone.")
            detect_stride = st.slider(
                "Detect every Nth frame", 1, 10, 1, key="bb_stride",
                help="Players and the ball are detected on every Nth frame; "
                     "boxes are interpolated in between. Roughly Nx faster, "
                     "but ByteTrack sees larger jumps between frames and "
                     "splits players into more IDs (measured: 13 IDs at 1, "
                     "18 at 3 on a 117-frame clip), which skews per-player "
                     "distance. Good for a quick look, not for final numbers.")
            keypoint_every = st.slider(
                "Detect court key-points every Nth frame", 1, 60, 30,
                key="bb_kp_every",
                help="The court barely moves, but its detector is the slowest "
                     "of the three (~1.7-2.3 s/frame on CPU). Running it every "
                     "30th frame costs almost no accuracy and removes about "
                     "half the total detection time.")
            imgsz = st.select_slider(
                "Detection resolution", options=[320, 416, 512, 640, 960],
                value=640, key="bb_imgsz",
                help="YOLO input size. 640 is what the models were trained at; "
                     "lower is quadratically faster but misses small/distant "
                     "objects — the ball especially.")

        with st.expander("⚙️ Processing", expanded=False):
            use_stubs = st.checkbox("Reuse cache for this video", value=True,
                                    key="bb_use_stubs",
                                    help="Cached detections are keyed by video "
                                         "content and settings, so re-running "
                                         "the same clip is instant.")
            stub_dir = st.text_input("Cache directory", value=DEFAULT_STUB_DIR,
                                     key="bb_stub_dir")
            output_fps = st.slider("Output FPS", 12, 60, 24, key="bb_fps")
            batch_size = st.slider(
                "Detection batch size", 1, 32,
                4 if memory.is_constrained() else 8, key="bb_batch",
                help="Frames sent to YOLO at once — the knob that sets peak "
                     "memory now that nothing else buffers. Measured on CPU "
                     "with these models: batch 1 adds 188 MB and runs at "
                     "3.4 fps, batch 4 adds 263 MB at 4.1 fps, batch 8 adds "
                     "330 MB at 4.3 fps, batch 20 adds 574 MB and is *slower* "
                     "at 3.2 fps. Past about 8 it is all cost and no gain.")

        with st.expander("🎨 Visualisation Layers", expanded=False):
            layers = {
                "players":   st.checkbox("Player tracks", True, key="bb_l_players"),
                "ball":      st.checkbox("Ball track", True, key="bb_l_ball"),
                "keypoints": st.checkbox("Court key-points", True, key="bb_l_kp"),
                "ball_ctrl": st.checkbox("Team ball control", True, key="bb_l_ctrl"),
                "frame_num": st.checkbox("Frame numbers", True, key="bb_l_fn"),
                "passes":    st.checkbox("Passes & interceptions", True, key="bb_l_pass"),
                "speed":     st.checkbox("Speed & distance", True, key="bb_l_speed"),
                "tactical":  st.checkbox("Tactical view overlay", True, key="bb_l_tac"),
            }

        if STATE_KEY in st.session_state:
            if st.button("🗑️ Clear results", use_container_width=True,
                         key="bb_clear"):
                del st.session_state[STATE_KEY]
                st.rerun()

    return {
        "device": device,
        "device_name": device_name,
        "player_model": player_model,
        "ball_model": ball_model,
        "court_model": court_model,
        "team_mode": team_mode,
        "team1_color": team1_color,
        "team2_color": team2_color,
        "use_stubs": use_stubs,
        "stub_dir": stub_dir or DEFAULT_STUB_DIR,
        "output_fps": output_fps,
        "layers": layers,
        "max_seconds": int(max_seconds),
        "detect_stride": int(detect_stride),
        "keypoint_every": int(keypoint_every),
        "imgsz": int(imgsz),
        "backend": backend,
        "max_long_side": int(max_long_side),
        "batch_size": int(batch_size),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def _at(seq, index, default=None):
    """`seq[index]` when it exists, else `default`.

    Container frame counts are advisory: some files report more frames than
    they decode. Every per-frame list is therefore read defensively so a
    ragged tail cannot take down a run that is otherwise finished.
    """
    if seq is None or index >= len(seq):
        return default
    return seq[index]


def _run_pipeline(video_path, cfg):
    """Execute the full basketball pipeline, returning the results dict.

    The pipeline makes several lazy passes over the file rather than decoding
    it into a list once. Decoding is cheap next to YOLO inference (sub-second
    per pass against minutes of detection), and it is the difference between a
    peak of a few megabytes of pixels and a peak of several gigabytes:

      1. players      — every `stride`-th frame
      2. ball         — every `stride`-th frame
      3. court        — every `kp_every`-th frame
      4. team colours — every frame
      5. render + encode — every frame, straight into the video writer

    At no point is more than one batch of frames alive.
    """
    from utils import frame_reader, get_video_info, scaled_size
    from trackers import PlayerTracker, BallTracker
    # TeamAssigner is imported lazily inside the manual branch — it pulls in
    # transformers + Fashion-CLIP, which the automatic path never needs.
    from team_assigner import AutoTeamAssigner
    from court_keypoint_detector import CourtKeypointDetector
    from ball_aquisition import BallAquisitionDetector
    from pass_and_interception_detector import PassAndInterceptionDetector
    from tactical_view_converter import TacticalViewConverter
    from speed_and_distance_calculator import SpeedAndDistanceCalculator
    from drawers import (
        PlayerTracksDrawer, BallTracksDrawer, CourtKeypointDrawer,
        TeamBallControlDrawer, FrameNumberDrawer, PassInterceptionDrawer,
        TacticalViewDrawer, SpeedAndDistanceDrawer,
    )

    device = cfg["device"]
    use_stubs = cfg["use_stubs"]
    layers = cfg["layers"]
    stride = max(1, cfg["detect_stride"])
    kp_every = max(1, cfg["keypoint_every"])
    imgsz = cfg["imgsz"]

    # ── what, exactly, are we analysing? ─────────────────────────────────────
    src_w, src_h, src_fps, src_count = get_video_info(video_path)
    scale = memory.fit_scale(src_w, src_h, cfg["max_long_side"])
    proc_w, proc_h = scaled_size(src_w, src_h, scale)

    total_frames = src_count or 0
    if cfg["max_seconds"]:
        capped = int(cfg["max_seconds"] * src_fps)
        total_frames = min(total_frames, capped) if total_frames else capped
    if not total_frames:
        # Unreported frame count: fall back to the duration ceiling so the
        # lazy readers still terminate.
        total_frames = int(MAX_ANALYSIS_SECONDS * src_fps)
    total_frames = min(total_frames, int(MAX_ANALYSIS_SECONDS * src_fps))

    def frames(step=1, count=None):
        """A fresh lazy reader over the analysed slice, at processing scale."""
        return (frame for _, frame in frame_reader(
            video_path, max_frames=count, scale=scale, stride=step))

    n_det = -(-total_frames // stride)       # ceil
    n_kp = -(-total_frames // kp_every)

    # Cache detections per (video content, settings) so re-running the same clip
    # is instant, and so two different videos can never share a cache entry.
    stub_dir = os.path.join(
        cfg["stub_dir"],
        f"{cfg['fingerprint']}_s{stride}k{kp_every}i{imgsz}"
        f"w{proc_w}_{cfg['backend']}")
    os.makedirs(stub_dir, exist_ok=True)

    # Every distinct (video, settings) combination gets its own cache
    # directory, so without pruning the disk fills with pickles nobody will
    # look at again. Same for rendered outputs.
    housekeeping.prune(cfg["stub_dir"], keep=MAX_KEPT_RUNS, protect=(stub_dir,))
    housekeeping.prune(DEFAULT_OUTPUT_DIR, keep=MAX_KEPT_RUNS,
                       suffixes=(".mp4", ".avi"))

    prog = st.progress(0)
    status = st.empty()
    timings = {}

    def upd(pct, msg):
        prog.progress(min(100, max(0, int(pct))))
        status.markdown(
            f"<div style='color:#8b949e;font-size:0.85rem;margin-top:0.3rem;'>{msg}</div>",
            unsafe_allow_html=True)

    def stage(name):
        """Context-free stopwatch: stage('x') ... done('x') records elapsed."""
        timings[name] = time.time()

    def done(name):
        timings[name] = round(time.time() - timings[name], 1)

    def detect_progress(label, base, span, expected, step):
        """Live per-batch progress for a detection pass.

        Detection is the long pole; without this the bar sat frozen for
        minutes and the run looked hung.
        """
        def report(count, _expected):
            frac = (count / float(expected)) if expected else 0.0
            upd(base + span * frac,
                f"{label} — frame {min(count * step, total_frames)} "
                f"of {total_frames}")
        return report

    t0 = time.time()

    # ── 1. detectors ─────────────────────────────────────────────────────────
    # Resolve the three checkpoints for the chosen backend. OpenVINO converts
    # on first use (slow once, then cached); if that fails for any reason we
    # fall back to PyTorch rather than failing the run.
    backend = cfg["backend"]
    if accel.is_openvino(backend):
        resolved = {}
        try:
            for role, pt in (("player", cfg["player_model"]),
                             ("ball", cfg["ball_model"]),
                             ("court", cfg["court_model"])):
                if not accel.is_exported(pt, imgsz):
                    upd(6, f"⚙️ Converting {role} model to OpenVINO "
                           f"(one-off, ~30 s)…")
                resolved[role] = accel.resolve_model(pt, imgsz, backend)
        except Exception as exc:
            st.warning(f"OpenVINO conversion failed ({exc}). "
                       f"Falling back to PyTorch CPU.")
            backend = accel.PYTORCH_CPU
            resolved = None
    else:
        resolved = None

    if resolved is None:
        resolved = {"player": (cfg["player_model"], None),
                    "ball": (cfg["ball_model"], None),
                    "court": (cfg["court_model"], None)}

    pred_device = accel.predict_device(backend)
    batch = accel.batch_size_for(backend, default=cfg["batch_size"])

    # ── 2. detection passes ──────────────────────────────────────────────────
    # One detector is constructed, used and released at a time. Holding all
    # three cost 1.4 GB of resident memory; sequentially, with the stripped
    # checkpoints, the same three cost ~460 MB. See sports_core/checkpoints.py.
    # A detector whose stub is already cached is never constructed at all.
    def stub(name):
        return os.path.join(stub_dir, name)

    def cached(path, expected):
        """The cached result for this pass, or None if it has to be computed."""
        if not use_stubs:
            return None
        from utils import read_stub
        result = read_stub(True, path)
        if result is not None and len(result) == expected:
            return result
        return None

    def detection_pass(stub_name, expected, step, make_detector, method,
                       label, base, span):
        """Run one detection pass over every `step`-th frame.

        The detector is constructed only on a cache miss, and released as soon
        as the pass is done, so its weights are never resident alongside
        another detector's.
        """
        path = stub(stub_name)
        result = cached(path, expected)
        if result is not None:
            return result

        detector = make_detector()
        try:
            return getattr(detector, method)(
                frames(step, expected), read_from_stub=False, stub_path=path,
                expected_count=expected,
                on_progress=detect_progress(label, base, span, expected, step))
        finally:
            del detector
            gc.collect()

    upd(10, f"🏃 Tracking players ({n_det} of {total_frames} frames)…")
    stage("players")
    player_tracks = interpolate_tracks(
        detection_pass(
            "player_track_stubs.pkl", n_det, stride,
            lambda: PlayerTracker(resolved["player"][0], device=pred_device,
                                  imgsz=imgsz, task=resolved["player"][1],
                                  batch_size=batch),
            "get_object_tracks", "🏃 Tracking players", 10, 18),
        stride, total_frames)
    done("players")

    upd(28, f"🏀 Tracking ball ({n_det} of {total_frames} frames)…")
    stage("ball")
    ball_tracks = scatter_strided(
        detection_pass(
            "ball_track_stubs.pkl", n_det, stride,
            lambda: BallTracker(resolved["ball"][0], device=pred_device,
                                imgsz=imgsz, task=resolved["ball"][1],
                                batch_size=batch),
            "get_object_tracks", "🏀 Tracking ball", 28, 10),
        stride, total_frames)
    done("ball")

    upd(38, "🧹 Cleaning ball track…")
    # Pure post-processing on the track list — static, so no detector (and no
    # 400 MB of weights) has to be alive for it.
    ball_tracks = BallTracker.remove_wrong_detections(ball_tracks)
    ball_tracks = BallTracker.interpolate_ball_positions(ball_tracks)

    upd(40, f"🔑 Detecting court key-points ({n_kp} of {total_frames} frames)…")
    stage("court")
    court_kp = expand_strided(
        detection_pass(
            "court_key_points_stub.pkl", n_kp, kp_every,
            lambda: CourtKeypointDetector(resolved["court"][0],
                                          device=pred_device, imgsz=imgsz,
                                          task=resolved["court"][1],
                                          batch_size=batch),
            "get_court_keypoints", "🔑 Court key-points", 40, 8),
        kp_every, total_frames)
    done("court")
    gc.collect()

    # ── 3. team assignment ───────────────────────────────────────────────────
    team_colors = None
    stage("teams")
    if cfg["team_mode"].startswith("Automatic"):
        upd(50, "👕 Discovering team colors automatically…")
        team_assigner = AutoTeamAssigner()
        player_assignment = team_assigner.get_player_teams_across_frames(
            frames(1, total_frames), player_tracks, read_from_stub=use_stubs,
            stub_path=os.path.join(stub_dir, "auto_player_assignment_stub.pkl"),
            expected_count=total_frames)
        team_colors = team_assigner.get_team_color_preview()
    else:
        upd(50, "👕 Assigning teams from jersey descriptions…")
        from team_assigner import TeamAssigner
        team_assigner = TeamAssigner(
            team_1_class_name=cfg["team1_color"],
            team_2_class_name=cfg["team2_color"],
            device=device)
        player_assignment = team_assigner.get_player_teams_across_frames(
            frames(1, total_frames), player_tracks, read_from_stub=use_stubs,
            stub_path=os.path.join(stub_dir, "player_assignment_stub.pkl"),
            expected_count=total_frames)
    done("teams")
    del team_assigner
    gc.collect()

    # ── 4. derived signals (all cheap, all per-frame dicts) ──────────────────
    upd(58, "🤝 Detecting ball possession…")
    ball_aquisition = BallAquisitionDetector().detect_ball_possession(
        player_tracks, ball_tracks)

    upd(62, "📊 Detecting passes & interceptions…")
    pi_det = PassAndInterceptionDetector()
    passes = pi_det.detect_passes(ball_aquisition, player_assignment)
    interceptions = pi_det.detect_interceptions(ball_aquisition, player_assignment)

    upd(66, "🗺️ Computing tactical view…")
    tactical_conv = TacticalViewConverter(court_image_path=COURT_IMAGE_PATH)
    court_kp = tactical_conv.validate_keypoints(court_kp)
    tactical_pos = tactical_conv.transform_players_to_tactical_view(
        court_kp, player_tracks)

    upd(70, "⚡ Calculating speed & distance…")
    speed_calc = SpeedAndDistanceCalculator(
        tactical_conv.width, tactical_conv.height,
        tactical_conv.actual_width_in_meters, tactical_conv.actual_height_in_meters)
    dist_per_frame = speed_calc.calculate_distance(tactical_pos)
    speed_per_frame = speed_calc.calculate_speed(dist_per_frame)

    # ── 5. render + encode, one frame at a time ──────────────────────────────
    stage("draw")
    out_path = os.path.join(DEFAULT_OUTPUT_DIR,
                            f"output_{cfg['fingerprint']}.mp4")
    written = _render_and_encode(
        out_path, cfg, layers,
        reader=frames(1, total_frames),
        total_frames=total_frames,
        size=(proc_w, proc_h),
        player_tracks=player_tracks,
        ball_tracks=ball_tracks,
        court_kp=court_kp,
        player_assignment=player_assignment,
        ball_aquisition=ball_aquisition,
        passes=passes,
        interceptions=interceptions,
        dist_per_frame=dist_per_frame,
        speed_per_frame=speed_per_frame,
        tactical_conv=tactical_conv,
        tactical_pos=tactical_pos,
        drawers=(PlayerTracksDrawer, BallTracksDrawer, CourtKeypointDrawer,
                 FrameNumberDrawer, TeamBallControlDrawer,
                 PassInterceptionDrawer, SpeedAndDistanceDrawer,
                 TacticalViewDrawer),
        on_progress=lambda i: upd(74 + 22 * (i / float(total_frames or 1)),
                                  f"🎨 Rendering frame {i} of {total_frames}…"),
    )
    done("draw")

    # `written` is authoritative: the container's frame count is only a hint,
    # so the stats below describe what actually made it into the video.
    total_frames = written or total_frames

    prog.progress(100)
    status.empty()
    gc.collect()

    elapsed = time.time() - t0

    return {
        "out_video_path": out_path,
        "ball_aquisition": ball_aquisition[:total_frames],
        "player_assignment": player_assignment[:total_frames],
        "passes": passes[:total_frames],
        "interceptions": interceptions[:total_frames],
        "player_speed_per_frame": speed_per_frame[:total_frames],
        "player_distances_per_frame": dist_per_frame[:total_frames],
        "player_tracks": player_tracks[:total_frames],
        "total_frames": total_frames,
        "team_colors": team_colors,
        "elapsed": elapsed,
        "timings": timings,
        "settings": {
            "detect_stride": stride,
            "keypoint_every": kp_every,
            "imgsz": imgsz,
            "backend": accel.label_for(backend),
            "device": cfg["device_name"],
            "frames_detected": n_det,
            "keypoint_frames": n_kp,
            "source_size": f"{src_w}x{src_h}",
            "processed_size": f"{proc_w}x{proc_h}",
        },
    }


def _render_and_encode(out_path, cfg, layers, reader, total_frames, size,
                       player_tracks, ball_tracks, court_kp, player_assignment,
                       ball_aquisition, passes, interceptions, dist_per_frame,
                       speed_per_frame, tactical_conv, tactical_pos, drawers,
                       on_progress=None):
    """Draw every enabled layer onto each frame and write it straight out.

    This replaces the old chain of eight whole-video list transforms. Each of
    those allocated a fresh copy of every frame, so with the original list and
    the one under construction the peak was three full videos of pixels —
    ~2 GB for an eight-second 720p clip, on a container with under 3 GB total.
    Here exactly one frame is alive at a time and it goes to the encoder as
    soon as it is drawn.

    Returns the number of frames actually written.
    """
    from utils import open_video_writer

    (PlayerTracksDrawer, BallTracksDrawer, CourtKeypointDrawer,
     FrameNumberDrawer, TeamBallControlDrawer, PassInterceptionDrawer,
     SpeedAndDistanceDrawer, TacticalViewDrawer) = drawers

    proc_w, proc_h = size
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_avi = out_path.replace(".mp4", "_tmp.avi")

    players_d = PlayerTracksDrawer() if layers["players"] else None
    ball_d = BallTracksDrawer() if layers["ball"] else None
    court_d = CourtKeypointDrawer() if layers["keypoints"] else None
    number_d = FrameNumberDrawer() if layers["frame_num"] else None
    control_d = TeamBallControlDrawer() if layers["ball_ctrl"] else None
    pass_d = PassInterceptionDrawer() if layers["passes"] else None
    speed_d = SpeedAndDistanceDrawer() if layers["speed"] else None
    tactical_d = TacticalViewDrawer() if layers["tactical"] else None

    # Computed once instead of per frame: the possession array and the court
    # image were both rebuilt on every frame the old drawers touched.
    team_ball_control = (control_d.get_team_ball_control(
        player_assignment, ball_aquisition) if control_d else None)
    court_image = (tactical_d.court_image(
        tactical_conv.court_image_path, tactical_conv.width,
        tactical_conv.height) if tactical_d else None)

    writer = open_video_writer(tmp_avi, proc_w, proc_h, cfg["output_fps"])
    written = 0
    try:
        for index, frame in enumerate(reader):
            if index >= total_frames:
                break

            if players_d:
                players_d.draw_frame(
                    frame,
                    _at(player_tracks, index, {}),
                    _at(player_assignment, index, {}),
                    _at(ball_aquisition, index, -1))
            if ball_d:
                ball_d.draw_frame(frame, _at(ball_tracks, index, {}))
            if court_d:
                keypoints = _at(court_kp, index)
                if keypoints is not None:
                    court_d.draw_frame(frame, keypoints)
            if number_d:
                number_d.draw_frame(frame, index)
            if control_d:
                control_d.draw_frame(frame, index, team_ball_control)
            if pass_d:
                pass_d.draw_frame(frame, index, passes, interceptions)
            if speed_d:
                speed_d.draw_frame(
                    frame,
                    _at(player_tracks, index, {}),
                    _at(dist_per_frame, index, {}),
                    _at(speed_per_frame, index, {}))
            if tactical_d:
                tactical_d.draw_frame(
                    frame, court_image, tactical_conv.width,
                    tactical_conv.height, tactical_conv.key_points,
                    _at(tactical_pos, index, {}),
                    _at(player_assignment, index, {}),
                    _at(ball_aquisition, index, -1))

            writer.write(frame)
            written += 1
            # `frame` is rebound on the next iteration and the reader hands out
            # a fresh buffer, so the drawn frame is released immediately.
            if on_progress is not None and index % 20 == 0:
                on_progress(index)
    finally:
        writer.release()

    transcode_h264(tmp_avi, out_path, cfg["output_fps"])
    return written


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def _render_results(r):
    import altair as alt

    sec("📹", "Analysed Video")
    if os.path.exists(r["out_video_path"]):
        st.video(r["out_video_path"])
        with open(r["out_video_path"], "rb") as vf:
            st.download_button("⬇️  Download MP4", data=vf,
                               file_name="basketball_analysis.mp4",
                               mime="video/mp4", key="bb_dl_video")

    if r.get("team_colors"):
        (r1, g1, b1), (r2, g2, b2) = r["team_colors"][1], r["team_colors"][2]
        html(f"""
        <div style="display:flex;gap:1rem;margin:0.5rem 0;">
          <div style="display:flex;align-items:center;gap:0.4rem;">
            <div style="width:16px;height:16px;border-radius:4px;
                        background:rgb({r1},{g1},{b1});border:1px solid #30363d;"></div>
            <span style="color:#8b949e;font-size:0.8rem;">Team 1 detected color</span>
          </div>
          <div style="display:flex;align-items:center;gap:0.4rem;">
            <div style="width:16px;height:16px;border-radius:4px;
                        background:rgb({r2},{g2},{b2});border:1px solid #30363d;"></div>
            <span style="color:#8b949e;font-size:0.8rem;">Team 2 detected color</span>
          </div>
        </div>""")

    stats = compute_stats(
        r["ball_aquisition"], r["player_assignment"],
        r["passes"], r["interceptions"],
        r["player_speed_per_frame"], r["player_distances_per_frame"],
    )
    pct1, pct2 = stats["team_ctrl_pct"]
    p1, p2 = stats["passes"]
    i1, i2 = stats["interceptions"]
    pt = stats["player_totals"]

    sec("📊", "Match Overview")
    kpi_grid(
        kpi("Team 1 Ball Control", f"{pct1}%",
            f"{stats['team_ctrl_frames'][0]} frames", "red"),
        kpi("Team 2 Ball Control", f"{pct2}%",
            f"{stats['team_ctrl_frames'][1]} frames", "blue"),
        kpi("Total Frames", f"{r['total_frames']:,}", "", "gray"),
        kpi("Total Players", str(len(pt)), "tracked", "green"),
    )

    if pct1 + pct2 > 0:
        html(f"""
        <div class="team-strip">
            <div class="team-strip-header">
                <div class="t1">🔴 Team 1 — {pct1}%</div>
                <div class="label">BALL CONTROL</div>
                <div class="t2">{pct2}% — Team 2 🔵</div>
            </div>
            <div class="control-bar">
                <div class="cb-t1" style="width:{pct1}%"></div>
                <div class="cb-t2" style="width:{pct2}%"></div>
            </div>
            <div class="stat-row">
                <span class="stat-val red">{p1}</span>
                <span class="stat-name">PASSES</span>
                <span class="stat-val blue">{p2}</span>
            </div>
            <div class="stat-row">
                <span class="stat-val red">{i1}</span>
                <span class="stat-name">INTERCEPTIONS</span>
                <span class="stat-val blue">{i2}</span>
            </div>
            <div class="stat-row">
                <span class="stat-val red">{p1 + i1}</span>
                <span class="stat-name">TOTAL ACTIONS</span>
                <span class="stat-val blue">{p2 + i2}</span>
            </div>
        </div>
        """)

    # ── ball control over time ───────────────────────────────────────────────
    sec("⏱️", "Ball Control Over Time")
    ba, pa = r["ball_aquisition"], r["player_assignment"]
    timeline_rows, window = [], 30
    for i in range(0, len(ba) - window, window):
        seg = ba[i:i + window]
        t1 = sum(1 for j, pid in enumerate(seg) if pid != -1 and pa[i + j].get(pid) == 1)
        t2 = sum(1 for j, pid in enumerate(seg) if pid != -1 and pa[i + j].get(pid) == 2)
        tot = t1 + t2 or 1
        timeline_rows.append({"frame": i,
                              "Team 1": round(100 * t1 / tot),
                              "Team 2": round(100 * t2 / tot)})

    if timeline_rows:
        tdf = pd.DataFrame(timeline_rows).melt("frame", var_name="team",
                                               value_name="pct")
        st.altair_chart(
            alt.Chart(tdf)
            .mark_area(opacity=0.75, interpolate="monotone")
            .encode(
                x=alt.X("frame:Q", title="Frame"),
                y=alt.Y("pct:Q", stack="normalize",
                        axis=alt.Axis(format="%", title="Ball Control")),
                color=alt.Color("team:N",
                    scale=alt.Scale(domain=["Team 1", "Team 2"],
                                    range=["#e94560", "#4a90e2"]),
                    legend=alt.Legend(orient="top-right")),
                tooltip=["frame:Q", "team:N", "pct:Q"],
            )
            .properties(height=200)
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#21262d", labelColor="#8b949e",
                            titleColor="#8b949e", domainColor="#21262d")
            .configure_legend(labelColor="#8b949e", titleColor="#8b949e",
                              fillColor="#0d1117", strokeColor="#21262d", padding=8),
            use_container_width=True,
        )

    # ── player performance ───────────────────────────────────────────────────
    df = None
    if pt:
        sec("🏃", "Player Performance")
        rows = []
        for pid, data in sorted(pt.items()):
            team_label = ("🔴 Team 1" if data["team"] == 1
                          else "🔵 Team 2" if data["team"] == 2 else "Unknown")
            rows.append({
                "Player ID": pid,
                "Team": team_label,
                "Distance (m)": round(data["total_dist_m"], 2),
                "Max Speed (km/h)": round(data["max_speed_kmh"], 1),
                "Active Frames": data["frames"],
            })
        df = (pd.DataFrame(rows)
              .sort_values("Distance (m)", ascending=False)
              .reset_index(drop=True))
        st.dataframe(
            df.style
              .background_gradient(subset=["Distance (m)"], cmap="RdYlGn")
              .background_gradient(subset=["Max Speed (km/h)"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )

        cdf = df.rename(columns={"Player ID": "pid", "Distance (m)": "dist",
                                 "Team": "team", "Max Speed (km/h)": "spd"})
        cdf["pid"] = cdf["pid"].astype(str)

        sec("📈", "Distance by Player")
        st.altair_chart(
            alt.Chart(cdf)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("pid:N", sort="-y", title="Player ID",
                        axis=alt.Axis(labelAngle=0, labelColor="#8b949e",
                                      titleColor="#8b949e")),
                y=alt.Y("dist:Q", title="Distance (m)",
                        axis=alt.Axis(labelColor="#8b949e", titleColor="#8b949e",
                                      gridColor="#21262d")),
                color=alt.Color("team:N",
                    scale=alt.Scale(domain=["🔴 Team 1", "🔵 Team 2"],
                                    range=["#e94560", "#4a90e2"]),
                    legend=alt.Legend(orient="top-right")),
                tooltip=[
                    alt.Tooltip("pid:N", title="Player"),
                    alt.Tooltip("team:N", title="Team"),
                    alt.Tooltip("dist:Q", title="Distance (m)", format=".2f"),
                    alt.Tooltip("spd:Q", title="Max Speed (km/h)", format=".1f"),
                ],
            )
            .properties(height=280)
            .configure_view(strokeWidth=0, fill="#0d1117")
            .configure_axis(domainColor="#21262d")
            .configure_legend(labelColor="#8b949e", titleColor="#8b949e",
                              fillColor="#0d1117", strokeColor="#21262d", padding=8),
            use_container_width=True,
        )

        sec("⚡", "Max Speed by Player")
        st.altair_chart(
            alt.Chart(cdf)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("pid:N", sort="-y", title="Player ID",
                        axis=alt.Axis(labelAngle=0, labelColor="#8b949e",
                                      titleColor="#8b949e")),
                y=alt.Y("spd:Q", title="Max Speed (km/h)",
                        axis=alt.Axis(labelColor="#8b949e", titleColor="#8b949e",
                                      gridColor="#21262d")),
                color=alt.Color("team:N",
                    scale=alt.Scale(domain=["🔴 Team 1", "🔵 Team 2"],
                                    range=["#e94560", "#4a90e2"]),
                    legend=None),
                tooltip=[alt.Tooltip("pid:N", title="Player"),
                         alt.Tooltip("spd:Q", title="Max Speed (km/h)",
                                     format=".1f")],
            )
            .properties(height=230)
            .configure_view(strokeWidth=0, fill="#0d1117")
            .configure_axis(domainColor="#21262d"),
            use_container_width=True,
        )

    # ── pipeline timing ──────────────────────────────────────────────────────
    t = r.get("timings") or {}
    s = r.get("settings") or {}
    if t:
        sec("⚙️", "Where The Time Went")
        detect_s = sum(t.get(k, 0) for k in ("players", "ball", "court"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total run", f"{r['elapsed']:.0f} s")
        m2.metric("Detection", f"{detect_s:.0f} s",
                  help="The three YOLO models — almost always the bottleneck.")
        m3.metric("Per analysed frame",
                  f"{detect_s / max(1, s.get('frames_detected', 1)):.2f} s")
        m4.metric("Device", s.get("device", "—"))
        resolution = ""
        if s.get("source_size") and s.get("processed_size"):
            resolution = (f" Source {s['source_size']}, processed and written "
                          f"at {s['processed_size']}.")
        st.caption(
            f"Detected on {s.get('frames_detected', 0)} of {r['total_frames']} "
            f"frames (stride {s.get('detect_stride', 1)}), court key-points on "
            f"{s.get('keypoint_frames', 0)} (every {s.get('keypoint_every', 1)}), "
            f"at imgsz {s.get('imgsz', 640)}.{resolution}"
        )
        with st.expander("Per-stage breakdown"):
            st.dataframe(
                pd.DataFrame(
                    [{"Stage": k, "Seconds": v} for k, v in t.items()]
                ).sort_values("Seconds", ascending=False),
                use_container_width=True, hide_index=True)

    # ── frame explorer ───────────────────────────────────────────────────────
    sec("🔎", "Frame Explorer")
    if os.path.exists(r["out_video_path"]) and r["total_frames"] > 0:
        max_f = r["total_frames"] - 1
        frame_idx = st.slider("Scrub through frames", 0, max_f, 0,
                              label_visibility="collapsed", key="bb_scrub")
        frame = read_frame_at(r["out_video_path"], frame_idx)
        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     use_container_width=True)

        pid_holding = ba[frame_idx] if frame_idx < len(ba) else -1
        team_holding = pa[frame_idx].get(pid_holding, "?") if pid_holding != -1 else "–"
        pass_ev = r["passes"][frame_idx] if frame_idx < len(r["passes"]) else -1
        int_ev = r["interceptions"][frame_idx] if frame_idx < len(r["interceptions"]) else -1

        event_str = ("🟢 Pass (T" + str(pass_ev) + ")" if pass_ev != -1
                     else "🟠 Interception (T" + str(int_ev) + ")" if int_ev != -1
                     else "—")
        team_str = ("🔴 Team 1" if team_holding == 1
                    else "🔵 Team 2" if team_holding == 2 else "—")

        html(f"""
        <div class="frame-meta">
            <div class="fm-item"><div class="fm-label">Frame</div>
                <div class="fm-value">{frame_idx} / {max_f}</div></div>
            <div class="fm-item"><div class="fm-label">Ball held by</div>
                <div class="fm-value">{"Player " + str(pid_holding) if pid_holding != -1 else "—"}</div></div>
            <div class="fm-item"><div class="fm-label">Team in possession</div>
                <div class="fm-value">{team_str}</div></div>
            <div class="fm-item"><div class="fm-label">Event</div>
                <div class="fm-value">{event_str}</div></div>
        </div>""")

    # ── export ───────────────────────────────────────────────────────────────
    if df is not None:
        sec("⬇️", "Export")
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.download_button("📄  Download Player Stats CSV",
                               data=df.to_csv(index=False),
                               file_name="basketball_player_stats.csv",
                               mime="text/csv", use_container_width=True,
                               key="bb_dl_csv")
        with ecol2:
            summary = {
                "team1_ball_control_pct": pct1,
                "team2_ball_control_pct": pct2,
                "team1_passes": p1, "team2_passes": p2,
                "team1_interceptions": i1, "team2_interceptions": i2,
                "total_frames": r["total_frames"],
            }
            st.download_button("📋  Download Match Summary JSON",
                               data=pd.Series(summary).to_json(),
                               file_name="basketball_match_summary.json",
                               mime="application/json", use_container_width=True,
                               key="bb_dl_json")


def _render_landing():
    sec("🏆", "What This App Does")
    feature_cards([
        ("🏃", "Player Tracking", "ByteTrack-powered multi-player detection and tracking across every frame."),
        ("🏀", "Ball Tracking", "YOLO-based ball detection with interpolation for missing frames."),
        ("👕", "Team Detection", "Fashion-CLIP classifies jersey colours to auto-assign teams."),
        ("🗺️", "Tactical View", "Homography maps all players onto a top-down court diagram."),
        ("⚡", "Speed & Distance", "Real-world speed (km/h) and total distance (m) per player."),
        ("🤝", "Pass Detection", "Automatic pass & interception events attributed per team."),
        ("🔑", "Court Key-points", "Neural-network landmark detection for court calibration."),
        ("📊", "Match Statistics", "Ball control %, player charts, frame explorer & CSV export."),
    ])

    spacer()
    sec("🚀", "Getting Started")
    html("""
    <div class="steps">
        <b>1.</b> Upload a basketball video above<br>
        <b>2.</b> Model weights load automatically — nothing to configure<br>
        <b>3.</b> Team colors are detected automatically from the footage<br>
        <b>4.</b> Hit <span class="hl">Run Analysis</span> and wait for the pipeline to finish<br>
        <b>5.</b> Explore the annotated video, stats charts, and frame explorer
    </div>""")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def render(sport=None):
    """Render the whole Basketball Analysis page. Called by the router."""
    cfg = _sidebar()

    sec("📁", "Upload Video")
    uploaded = st.file_uploader(
        "Drop a basketball video here",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
        key="bb_upload",
    )

    video_path = None
    fingerprint = None
    if uploaded is None:
        html("""
        <div class="upload-zone">
            <div style="font-size:2.5rem;margin-bottom:0.6rem;">📹</div>
            <div style="color:#e6edf3;font-weight:600;font-size:1rem;">Drop your video here</div>
            <div style="color:#8b949e;font-size:0.8rem;margin-top:0.3rem;">MP4 · AVI · MOV · MKV</div>
        </div>""")
    else:
        video_path, fingerprint = _stage_upload(uploaded)
        st.video(video_path)

    if video_path:
        cfg["fingerprint"] = fingerprint
        spacer("0.5rem")
        missing_models = check_models(cfg["player_model"], cfg["ball_model"],
                                      cfg["court_model"])

        # Tell the user what they're committing to before they commit to it.
        from utils import get_video_info, scaled_size

        src_w, src_h, src_fps, src_count = get_video_info(video_path)
        n_frames = src_count or 0
        if cfg["max_seconds"]:
            capped = int(cfg["max_seconds"] * src_fps)
            n_frames = min(n_frames, capped) if n_frames else capped
        hard_cap = int(MAX_ANALYSIS_SECONDS * src_fps)
        trimmed = bool(n_frames and n_frames > hard_cap)
        n_frames = min(n_frames or hard_cap, hard_cap)

        scale = memory.fit_scale(src_w, src_h, cfg["max_long_side"])
        proc_w, proc_h = scaled_size(src_w, src_h, scale)
        if scale != 1.0:
            st.caption(f"Source is {src_w}x{src_h}; it will be processed and "
                       f"written out at {proc_w}x{proc_h} "
                       f"({memory.frame_mb(proc_w, proc_h):.1f} MB per frame "
                       f"instead of {memory.frame_mb(src_w, src_h):.1f} MB).")
        if trimmed:
            st.info(f"Only the first {MAX_ANALYSIS_SECONDS}s will be analysed "
                    f"— that ceiling is what keeps the hosted app inside its "
                    f"memory and disk budget.")

        est = estimate_runtime(n_frames, cfg["detect_stride"],
                               cfg["keypoint_every"], cfg["imgsz"],
                               cfg["backend"])

        left, right = st.columns([1, 3])
        with left:
            run_analysis = st.button("🚀  Run Analysis", type="primary",
                                     use_container_width=True,
                                     disabled=bool(missing_models),
                                     key="bb_run")
        with right:
            if missing_models:
                st.warning("⚠️ Model weights could not be loaded automatically. "
                           "Please check your connection and reload the app.")
            elif est > 240:
                st.warning(
                    f"⏱️ ~{est/60:.0f} min of CPU work for {n_frames} frames. "
                    f"Raise **Detect every Nth frame**, lower **Detection "
                    f"resolution**, or cap **Analyse first N seconds** in the "
                    f"sidebar to bring this down.")
            else:
                eta = f"{est/60:.1f} min" if est > 60 else f"{est:.0f} s"
                st.success(f"✅ Models loaded — {n_frames} frames on "
                           f"**{accel.label_for(cfg['backend'])}**, "
                           f"roughly {eta}")

        if run_analysis:
            # Drop the previous run before starting a new one: its per-frame
            # track lists are the largest thing left in session state.
            st.session_state.pop(STATE_KEY, None)
            gc.collect()
            try:
                results = _run_pipeline(video_path, cfg)
            except ImportError as exc:
                st.error(f"Import error: {exc}")
                return
            except MemoryError:
                gc.collect()
                st.error(
                    "Ran out of memory. Lower **Processing resolution**, cap "
                    "**Analyse first N seconds**, or reduce **Detection batch "
                    "size** in the sidebar, then try again.")
                return
            st.session_state[STATE_KEY] = results
            st.success(f"✅ Analysis complete in **{results['elapsed']:.1f}s** · "
                       f"{results['total_frames']} frames processed")

    if STATE_KEY in st.session_state:
        _render_results(st.session_state[STATE_KEY])
    else:
        _render_landing()
