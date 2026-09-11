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
from sports_core import accel

STATE_KEY = "basketball_results"

DEFAULT_STUB_DIR = os.path.join(ROOT, "stubs")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "output_videos")
COURT_IMAGE_PATH = os.path.join(ROOT, "images", "basketball_court.png")

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
def video_fingerprint(data):
    """Short content hash of an uploaded video, used to key the stub cache."""
    return hashlib.sha1(data).hexdigest()[:10]


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
    if len(local) >= 3:
        mapping = match_model_files(local)
        if all(mapping.values()):
            return tuple(str(mapping[r]) for r in ("player", "ball", "court"))

    try:
        pt_files = download_models_from_drive(DEFAULT_DRIVE_FOLDER_URL)
    except Exception:
        return None, None, None

    if not pt_files:
        return None, None, None

    mapping = match_model_files(pt_files)
    return tuple(str(mapping[r]) if mapping[r] else None
                 for r in ("player", "ball", "court"))


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
def frames_to_video(frames, output_path, fps=24):
    """Write frames → temp AVI → re-encode to H.264 mp4 via ffmpeg."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_avi = output_path.replace(".mp4", "_tmp.avi")
    writer = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    if shutil.which("ffmpeg"):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_avi,
                 "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                 output_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            os.remove(tmp_avi)
            return
        except subprocess.CalledProcessError:
            pass
    os.replace(tmp_avi, output_path)


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

        with st.spinner("Loading model weights…"):
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
                "Analyse first N seconds (0 = whole video)",
                min_value=0, value=0, step=5, key="bb_max_secs",
                help="The quickest way to get a result out of a long clip.")
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
    }


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def _run_pipeline(video_path, cfg):
    """Execute the full basketball pipeline, returning the results dict."""
    from utils import read_video
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

    # Cache detections per (video content, settings) so re-running the same clip
    # is instant, and so two different videos can never share a cache entry.
    stub_dir = os.path.join(
        cfg["stub_dir"],
        f"{cfg['fingerprint']}_s{stride}k{kp_every}i{imgsz}_{cfg['backend']}")
    os.makedirs(stub_dir, exist_ok=True)

    prog = st.progress(0)
    status = st.empty()
    timings = {}

    def upd(pct, msg):
        prog.progress(pct)
        status.markdown(
            f"<div style='color:#8b949e;font-size:0.85rem;margin-top:0.3rem;'>{msg}</div>",
            unsafe_allow_html=True)

    def stage(name):
        """Context-free stopwatch: stage('x') ... records elapsed under 'x'."""
        timings[name] = time.time()

    def done(name):
        timings[name] = round(time.time() - timings[name], 1)

    t0 = time.time()

    upd(5, "📽️ Reading video frames…")
    stage("read")
    video_frames = read_video(video_path)
    if cfg["max_seconds"]:
        video_frames = video_frames[:int(cfg["max_seconds"] * cfg["output_fps"])]
    total_frames = len(video_frames)
    done("read")

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
                    upd(10, f"⚙️ Converting {role} model to OpenVINO "
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
    batch = accel.batch_size_for(backend)

    upd(10, f"🔍 Initialising detectors — {accel.label_for(backend)}…")
    player_tracker = PlayerTracker(resolved["player"][0], device=pred_device,
                                   imgsz=imgsz, task=resolved["player"][1],
                                   batch_size=batch)
    ball_tracker = BallTracker(resolved["ball"][0], device=pred_device,
                               imgsz=imgsz, task=resolved["ball"][1],
                               batch_size=batch)
    court_kp_det = CourtKeypointDetector(resolved["court"][0], device=pred_device,
                                         imgsz=imgsz, task=resolved["court"][1],
                                         batch_size=batch)

    # Detect on a subsampled frame list, then hold each result until the next
    # sample. The court detector gets its own, much coarser interval.
    det_frames = video_frames[::stride]
    kp_frames = video_frames[::kp_every]

    upd(15, f"🏃 Tracking players ({len(det_frames)} of {total_frames} frames)…")
    stage("players")
    player_tracks = interpolate_tracks(
        player_tracker.get_object_tracks(
            det_frames, read_from_stub=use_stubs,
            stub_path=os.path.join(stub_dir, "player_track_stubs.pkl")),
        stride, total_frames)
    done("players")

    upd(30, f"🏀 Tracking ball ({len(det_frames)} of {total_frames} frames)…")
    stage("ball")
    ball_tracks = scatter_strided(
        ball_tracker.get_object_tracks(
            det_frames, read_from_stub=use_stubs,
            stub_path=os.path.join(stub_dir, "ball_track_stubs.pkl")),
        stride, total_frames)
    done("ball")

    upd(38, f"🔑 Detecting court key-points ({len(kp_frames)} of {total_frames} frames)…")
    stage("court")
    court_kp = expand_strided(
        court_kp_det.get_court_keypoints(
            kp_frames, read_from_stub=use_stubs,
            stub_path=os.path.join(stub_dir, "court_key_points_stub.pkl")),
        kp_every, total_frames)
    done("court")

    upd(42, "🧹 Cleaning ball track…")
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    team_colors = None
    stage("teams")
    if cfg["team_mode"].startswith("Automatic"):
        upd(48, "👕 Discovering team colors automatically…")
        team_assigner = AutoTeamAssigner()
        player_assignment = team_assigner.get_player_teams_across_frames(
            video_frames, player_tracks, read_from_stub=use_stubs,
            stub_path=os.path.join(stub_dir, "auto_player_assignment_stub.pkl"))
        team_colors = team_assigner.get_team_color_preview()
    else:
        upd(48, "👕 Assigning teams from jersey descriptions…")
        from team_assigner import TeamAssigner
        team_assigner = TeamAssigner(
            team_1_class_name=cfg["team1_color"],
            team_2_class_name=cfg["team2_color"],
            device=device)
        player_assignment = team_assigner.get_player_teams_across_frames(
            video_frames, player_tracks, read_from_stub=use_stubs,
            stub_path=os.path.join(stub_dir, "player_assignment_stub.pkl"))

    done("teams")

    upd(55, "🤝 Detecting ball possession…")
    ball_aquisition = BallAquisitionDetector().detect_ball_possession(
        player_tracks, ball_tracks)

    upd(60, "📊 Detecting passes & interceptions…")
    pi_det = PassAndInterceptionDetector()
    passes = pi_det.detect_passes(ball_aquisition, player_assignment)
    interceptions = pi_det.detect_interceptions(ball_aquisition, player_assignment)

    upd(65, "🗺️ Computing tactical view…")
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

    upd(78, "🎨 Drawing visualisations…")
    stage("draw")
    out_frames = video_frames.copy()
    if layers["players"]:
        out_frames = PlayerTracksDrawer().draw(
            out_frames, player_tracks, player_assignment, ball_aquisition)
    if layers["ball"]:
        out_frames = BallTracksDrawer().draw(out_frames, ball_tracks)
    if layers["keypoints"]:
        out_frames = CourtKeypointDrawer().draw(out_frames, court_kp)
    if layers["frame_num"]:
        out_frames = FrameNumberDrawer().draw(out_frames)
    if layers["ball_ctrl"]:
        out_frames = TeamBallControlDrawer().draw(
            out_frames, player_assignment, ball_aquisition)
    if layers["passes"]:
        out_frames = PassInterceptionDrawer().draw(out_frames, passes, interceptions)
    if layers["speed"]:
        out_frames = SpeedAndDistanceDrawer().draw(
            out_frames, player_tracks, dist_per_frame, speed_per_frame)
    if layers["tactical"]:
        out_frames = TacticalViewDrawer().draw(
            out_frames, tactical_conv.court_image_path,
            tactical_conv.width, tactical_conv.height,
            tactical_conv.key_points, tactical_pos,
            player_assignment, ball_aquisition)

    done("draw")

    upd(90, "💾 Encoding video (H.264)…")
    stage("encode")
    out_path = os.path.join(DEFAULT_OUTPUT_DIR,
                            f"output_{cfg['fingerprint']}.mp4")
    frames_to_video(out_frames, out_path, fps=cfg["output_fps"])
    done("encode")

    prog.progress(100)
    status.empty()

    elapsed = time.time() - t0

    # Deliberately NOT keeping `out_frames` — see read_frame_at(). Holding the
    # annotated frames would cost ~830 MB for a 10-second 720p clip.
    del out_frames

    return {
        "out_video_path": out_path,
        "ball_aquisition": ball_aquisition,
        "player_assignment": player_assignment,
        "passes": passes,
        "interceptions": interceptions,
        "player_speed_per_frame": speed_per_frame,
        "player_distances_per_frame": dist_per_frame,
        "player_tracks": player_tracks,
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
            "frames_detected": len(det_frames),
            "keypoint_frames": len(kp_frames),
        },
    }


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
        st.caption(
            f"Detected on {s.get('frames_detected', 0)} of {r['total_frames']} "
            f"frames (stride {s.get('detect_stride', 1)}), court key-points on "
            f"{s.get('keypoint_frames', 0)} (every {s.get('keypoint_every', 1)}), "
            f"at imgsz {s.get('imgsz', 640)}."
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
        data = uploaded.getvalue()
        fingerprint = video_fingerprint(data)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(data)
        tmp.close()
        video_path = tmp.name
        st.video(video_path)

    if video_path:
        cfg["fingerprint"] = fingerprint
        spacer("0.5rem")
        missing_models = check_models(cfg["player_model"], cfg["ball_model"],
                                      cfg["court_model"])

        # Tell the user what they're committing to before they commit to it.
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        if cfg["max_seconds"]:
            n_frames = min(n_frames, int(cfg["max_seconds"] * src_fps))
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
            try:
                results = _run_pipeline(video_path, cfg)
            except ImportError as exc:
                st.error(f"Import error: {exc}")
                return
            st.session_state[STATE_KEY] = results
            st.success(f"✅ Analysis complete in **{results['elapsed']:.1f}s** · "
                       f"{results['total_frames']} frames processed")

    if STATE_KEY in st.session_state:
        _render_results(st.session_state[STATE_KEY])
    else:
        _render_landing()
