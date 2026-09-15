"""
Football Analysis — page module
===============================
Rendered by the unified `app.py` router (and, standalone, by
`football_analysis/app.py`).

The heavy lifting runs as a subprocess (`main.py`) with `cwd=ROOT`, so this
page only needs `utils` for the H.264 re-encode; `sports_core.isolation`
guarantees that import resolves to *this* project's `utils` package.
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
for _p in (ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sports_core.theme import (
    sec, kpi, kpi_grid, chip, spacer, feature_cards, html,
)
from sports_core import accel, housekeeping, weights

STATE_KEY = "football_results"

UPLOAD_DIR = os.path.join(ROOT, "app_uploads")
OUTPUT_DIR = os.path.join(ROOT, "output_videos")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_MODEL = os.path.join(MODEL_DIR, "best.pt")

# `models/` is gitignored, so a clone or a cloud deploy has no best.pt. The
# detector is fetched from here on first use and cached, the way basketball's
# weights are — nothing to configure for the app to work on a fresh deploy.
#
# Override without touching code by setting `football_model_url` in
# .streamlit/secrets.toml, or the FOOTBALL_MODEL_URL environment variable, or
# by pasting a link in the sidebar.
MODEL_URL_KEY = "football_model_url"
DEFAULT_MODEL_URL = (
    "https://drive.google.com/file/d/"
    "1bzmKYxV4I89zbA28eHdN8LJswiMR3P36/view?usp=sharing"
)


@st.cache_resource(show_spinner=False)
def ensure_model_ready(local_path, url):
    """
    Guarantee a detector checkpoint exists, returning (path, source).

    Cached per (local_path, url) so a rerun never re-downloads.
    """
    return weights.resolve_single(local_path, url, cache_name="football",
                                  filename="best.pt")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
# Uploads land in a directory inside the repo, so without pruning they
# accumulate for the life of the container — a handful of match clips is
# enough to fill a small host's disk.
MAX_KEPT_UPLOADS = 3

# Long edge of the in-browser preview. st.video() serves the whole file
# through Streamlit's media server, which holds it in memory — a 30-second
# 1080p annotation is ~22 MB per viewer. The full-resolution render is
# untouched and still offered under Export.
PREVIEW_LONG_SIDE = 1280


def save_upload(uploaded_file):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dst = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    _prune_uploads(keep=dst)
    return dst


def _prune_uploads(keep=None, limit=MAX_KEPT_UPLOADS):
    """Keep only the most recent `limit` uploads, plus `keep`."""
    housekeeping.prune(UPLOAD_DIR, keep=limit, protect=(keep,))


def read_first_frame(video_path, max_h=720):
    """Return (downscaled_preview_frame, scale) for the first frame."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return np.zeros((400, 700, 3), np.uint8), 1.0
    h, w = frame.shape[:2]
    scale = 1.0
    if h > max_h:
        scale = max_h / float(h)
        frame = cv2.resize(frame, (int(w * scale), max_h))
    return frame, scale


def build_calibration_json(corners_pixels, region_length, region_width, out_path):
    """corners_pixels must already be in ORIGINAL frame coords (TL, TR, BR, BL)."""
    data = {
        "pixel_vertices": [[int(x), int(y)] for (x, y) in corners_pixels],
        "region_length": float(region_length),
        "region_width": float(region_width),
    }
    with open(out_path, "w") as f:
        json.dump(data, f)
    return out_path


def build_command(input_path, output_raw, args, calib_path):
    cmd = [
        sys.executable, os.path.join(ROOT, "main.py"),
        "--input", input_path,
        "--output", output_raw,
        "--model", args["model"],
    ]
    if calib_path:
        cmd += ["--calib", calib_path]
    if args.get("use_cache"):
        cmd += ["--use_cache"]
    if args.get("fps"):
        cmd += ["--fps", str(args["fps"])]
    if args.get("max_frames"):
        cmd += ["--max-frames", str(args["max_frames"])]
    if args.get("scale"):
        cmd += ["--scale", str(args["scale"])]
    if args.get("detect_stride") and args["detect_stride"] > 1:
        cmd += ["--detect-stride", str(args["detect_stride"])]
    if args.get("imgsz"):
        cmd += ["--imgsz", str(args["imgsz"])]
    if args.get("device"):
        cmd += ["--device", str(args["device"])]
    if args.get("task"):
        cmd += ["--task", str(args["task"])]
    cmd += ["--progress-every", "10"]
    if args.get("export_csv"):
        cmd += ["--export-csv", args["export_csv"]]
    if args.get("heatmaps_dir"):
        cmd += ["--heatmaps-dir", args["heatmaps_dir"]]
    return cmd


def parse_result_json(lines):
    """Pull the pipeline's machine-readable summary out of its stdout."""
    for line in reversed(lines):
        if line.startswith("RESULT_JSON: "):
            try:
                return json.loads(line[len("RESULT_JSON: "):])
            except json.JSONDecodeError:
                return None
    return None


# Library chatter the user can do nothing about — never surfaced in the status
# panel. The full log is still kept and shown on demand / on failure.
_NOISE = (
    "FutureWarning", "DeprecationWarning", "UserWarning", "RuntimeWarning",
    "warnings.warn", "self.tracker = sv.ByteTrack()", "TracerWarning",
)


def is_noise(line):
    stripped = line.strip()
    if not stripped:
        return True
    return any(token in line for token in _NOISE)


@st.cache_data(show_spinner=False)
def detect_pitch(video_path, stamp):
    """
    Automatic pitch detection, cached per (video, mtime).

    `stamp` is only there to key the cache on file identity; detection reads
    several frames, so it must not rerun on every widget interaction. It is
    deliberately not underscore-prefixed — Streamlit drops underscore-prefixed
    arguments from the cache key, which made the stamp a no-op and returned a
    previous video's corners whenever an upload reused a filename.
    """
    import pitch_calibration as pitch
    corners, confidence, info = pitch.detect_from_video(video_path, samples=5)

    frame = None
    if corners is not None:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(info.get("frame_index", 0)))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            frame = None
    return corners, confidence, info, frame


def estimate_runtime(n_frames, stride, imgsz, backend):
    """Rough seconds of compute for the football pipeline (one detector)."""
    cost = accel.SECONDS_PER_FRAME.get(backend,
                                       accel.SECONDS_PER_FRAME[accel.PYTORCH_CPU])
    scale = (imgsz / 640.0) ** 2
    return n_frames * (cost["player"] / max(1, stride)) * scale


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        # Resolve the detector: local file, else a configured/pasted URL that
        # gets downloaded once and cached.
        configured = weights.configured_url(MODEL_URL_KEY, DEFAULT_MODEL_URL)
        pasted = st.session_state.get("fb_model_url", "").strip()
        with st.spinner("Fetching detector weights (first run only)…"):
            resolved, source, why = ensure_model_ready(DEFAULT_MODEL,
                                                       pasted or configured)

        if source == "local":
            chip("✅ Detector found", os.path.relpath(resolved, ROOT),
                 "#3fb950", state="ok")
        elif source == "downloaded":
            size = os.path.getsize(resolved) / 1e6
            chip("✅ Detector downloaded", f"{size:.0f} MB, cached",
                 "#3fb950", state="ok")
        else:
            chip("⚠️ Detector weights unavailable",
                 "see the reason below", "#f85149", state="bad")

        if source == "missing":
            if why:
                st.caption(f"Download failed — {why}")
            st.text_input(
                "Weights URL (.pt or Google Drive link)",
                key="fb_model_url", placeholder="https://…",
                help="The built-in download could not be reached. Paste a link "
                     "here, or set `football_model_url` in Streamlit secrets / "
                     "FOOTBALL_MODEL_URL to override it permanently.")

        with st.expander("🎯 Detection", expanded=True):
            model_path = st.text_input("YOLO model (.pt)",
                                       value=resolved or DEFAULT_MODEL,
                                       key="fb_model")
            use_cache = st.checkbox("Use cached tracking (same video re-runs)",
                                    value=False, key="fb_cache")
            fps = st.number_input("FPS override (0 = auto)", min_value=0, value=0,
                                  step=1, key="fb_fps")
            do_heatmaps = st.checkbox("Generate player heatmaps", value=False,
                                      key="fb_heatmaps")

        with st.expander("🚀 Performance", expanded=True):
            backends = accel.available_backends()
            backend = st.selectbox(
                "Inference backend", [k for k, _ in backends], index=0,
                format_func=accel.label_for, key="fb_backend",
                help="OpenVINO runs the same weights on Intel hardware. On an "
                     "integrated GPU it measured ~5x faster than PyTorch CPU, "
                     "with matching detections. The first run per resolution "
                     "converts the model (~20 s), then it is cached.")
            if accel.is_openvino(backend):
                cached = accel.cache_size_mb()
                st.caption(f"Converted models cached: {cached:.0f} MB"
                           if cached else
                           "The model is converted on first run, then cached.")
            elif not accel.openvino_installed():
                st.caption("Install `openvino` to unlock Intel GPU inference.")

            max_frames = st.number_input(
                "Max frames (0 = all)", min_value=0, value=0, step=50,
                key="fb_maxframes",
                help="The quickest way to get a result out of a long clip.")
            detect_stride = st.number_input(
                "Detect every Nth frame", min_value=1, value=1, step=1,
                key="fb_stride",
                help="Skipped frames reuse the last known boxes. Roughly Nx "
                     "faster, but ByteTrack sees bigger jumps and splits "
                     "players into more IDs — good for a quick look, not for "
                     "final numbers.")
            imgsz = st.select_slider(
                "Detection resolution", options=[320, 416, 512, 640, 960],
                value=640, key="fb_imgsz",
                help="YOLO input size. 640 is what the model was trained at; "
                     "lower is quadratically faster but misses small/distant "
                     "objects — the ball especially.")
            scale = st.number_input(
                "Output frame scale (1.0 = original)", min_value=0.1,
                max_value=1.0, value=1.0, step=0.1, key="fb_scale",
                help="Downscales the video itself, so both detection and "
                     "encoding get cheaper.")

        if STATE_KEY in st.session_state:
            if st.button("🗑️ Clear results", use_container_width=True,
                         key="fb_clear"):
                del st.session_state[STATE_KEY]
                st.rerun()

    return {
        # The text input keeps whatever it was first rendered with, so if the
        # weights only arrived on a later run, fall back to the resolved path
        # rather than handing the pipeline a path that does not exist.
        "model": model_path if os.path.exists(model_path)
                 else (resolved or model_path),
        "use_cache": use_cache,
        "max_frames": int(max_frames) if max_frames else None,
        "scale": scale if scale and scale != 1.0 else None,
        "detect_stride": int(detect_stride) if detect_stride else None,
        "fps": fps if fps else None,
        "do_heatmaps": do_heatmaps,
        "imgsz": int(imgsz),
        "backend": backend,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION UI
# ─────────────────────────────────────────────────────────────────────────────
def _calibration_corners(frame0, preview_scale):
    """Collect the 4 pitch corners, returning them in ORIGINAL frame coords."""
    h, w = frame0.shape[:2]
    with st.expander("Pitch corners — top-left, top-right, bottom-right, "
                     "bottom-left", expanded=True):
        col = st.columns(4)
        tl_x = col[0].number_input("TL x", 0, w, int(w * 0.15), key="fb_tlx")
        tl_y = col[0].number_input("TL y", 0, h, int(h * 0.10), key="fb_tly")
        tr_x = col[1].number_input("TR x", 0, w, int(w * 0.85), key="fb_trx")
        tr_y = col[1].number_input("TR y", 0, h, int(h * 0.10), key="fb_try")
        br_x = col[2].number_input("BR x", 0, w, int(w * 0.88), key="fb_brx")
        br_y = col[2].number_input("BR y", 0, h, int(h * 0.55), key="fb_bry")
        bl_x = col[3].number_input("BL x", 0, w, int(w * 0.12), key="fb_blx")
        bl_y = col[3].number_input("BL y", 0, h, int(h * 0.55), key="fb_bly")

        c1, c2 = st.columns(2)
        region_length = c1.number_input("Visible region length (x-span, m)",
                                        1.0, 300.0, value=105.0, key="fb_rl")
        region_width = c2.number_input("Visible region width (y-span, m)",
                                       1.0, 300.0, value=68.0, key="fb_rw")

        overlay = frame0.copy()
        pts = np.array([(tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)],
                       np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)
        for label, (x, y) in [("TL", (tl_x, tl_y)), ("TR", (tr_x, tr_y)),
                              ("BR", (br_x, br_y)), ("BL", (bl_x, bl_y))]:
            cv2.circle(overlay, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(overlay, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption="Calibration quadrilateral (TL, TR, BR, BL)",
                 use_container_width=True)

    # scale preview coords back to ORIGINAL frame resolution for the pipeline
    corners = [
        (int(tl_x / preview_scale), int(tl_y / preview_scale)),
        (int(tr_x / preview_scale), int(tr_y / preview_scale)),
        (int(br_x / preview_scale), int(br_y / preview_scale)),
        (int(bl_x / preview_scale), int(bl_y / preview_scale)),
    ]
    return corners, region_length, region_width


def _auto_calibration(input_path, preview_scale):
    """
    Detect the pitch corners automatically and let the user accept or adjust.

    Returns (corners, region_length, region_width) in ORIGINAL frame
    coordinates — the same shape `_calibration_corners` returns — or None when
    detection failed.
    """
    import pitch_calibration as pitch

    stamp = os.path.getmtime(input_path) if os.path.exists(input_path) else 0
    with st.spinner("Detecting the pitch…"):
        corners, confidence, info, frame = detect_pitch(input_path, stamp)

    if corners is None:
        st.warning(f"⚠️ Could not detect the pitch — "
                   f"{info.get('reason') or 'no pitch-like surface found'}. "
                   f"Switch to **Manual corners**, or run without calibration.")
        return None

    shape = frame.shape if frame is not None else (1, 1, 3)
    clipped = pitch.clipped_edges(corners, shape)
    length, width, note = pitch.guess_region_size(corners, shape)

    left, right = st.columns([3, 2])
    with left:
        st.image(cv2.cvtColor(pitch.draw_overlay(frame, corners, confidence),
                              cv2.COLOR_BGR2RGB),
                 caption=f"Detected pitch region — {info.get('method')} "
                         f"(frame {info.get('frame_index', 0)})",
                 use_container_width=True)

    with right:
        grade = ("Good" if confidence >= 0.7 else
                 "Usable" if confidence >= 0.45 else "Low")
        colour = ("#3fb950" if confidence >= 0.7 else
                  "#d29922" if confidence >= 0.45 else "#f85149")
        kpi_grid(
            kpi("Confidence", f"{confidence:.0%}", grade, "accent"),
            kpi("Pitch corners seen", f"{4 - clipped}/4",
                "rest run off-frame" if clipped else "all in shot",
                "green" if clipped == 0 else "gray"),
        )
        html(f"<div style='color:{colour};font-size:0.8rem;font-weight:600;'>"
             f"{grade} detection</div>")

        st.markdown("**Real-world span of that region**")
        c1, c2 = st.columns(2)
        length = c1.number_input("Length (m)", 1.0, 300.0, value=float(length),
                                 key="fb_auto_len")
        width = c2.number_input("Width (m)", 1.0, 300.0, value=float(width),
                                key="fb_auto_wid")

    if note == "region-clipped":
        st.warning(
            f"⚠️ Only {4 - clipped} of the 4 pitch corners are in shot — the "
            f"pitch continues past the frame edge. The *shape* above is right, "
            f"but its real-world size cannot be measured from the video, and "
            f"**speed and distance scale directly with the two numbers above**. "
            f"105 x 68 m is the full-pitch default; for a view like this, enter "
            f"the span you can actually see (one half is roughly 52 x 68 m)."
        )
    else:
        st.success("✅ All four pitch corners are in shot, so 105 x 68 m is a "
                   "safe default for a standard pitch.")

    # Seed the manual inputs from this detection, so switching to
    # "Manual corners" starts from the automatic result instead of from
    # scratch. Manual works in preview coordinates, hence the scaling.
    for key, value in zip(("fb_tlx", "fb_tly", "fb_trx", "fb_try",
                           "fb_brx", "fb_bry", "fb_blx", "fb_bly"),
                          [c for xy in corners for c in xy]):
        st.session_state.setdefault(key, int(value * preview_scale))

    return corners, length, width


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def _run_pipeline(input_path, video_name, cfg, calib_corners):
    run_dir = os.path.join(OUTPUT_DIR, f"app_{video_name}")
    os.makedirs(run_dir, exist_ok=True)

    raw_out = os.path.join(run_dir, f"{video_name}_raw.mp4")
    h264_out = os.path.join(run_dir, f"{video_name}_output.mp4")
    csv_path = os.path.join(run_dir, f"{video_name}_data.csv")
    heatmaps_dir = os.path.join(run_dir, "heatmaps")

    calib_path = None
    if calib_corners is not None:
        corners, region_length, region_width = calib_corners
        calib_path = os.path.join(run_dir, f"{video_name}_calib.json")
        build_calibration_json(corners, region_length, region_width, calib_path)

    # Resolve the detector for the chosen backend (OpenVINO converts once,
    # then is cached). Fall back to PyTorch rather than failing the run.
    backend = cfg["backend"]
    model_path, task, device = cfg["model"], None, accel.predict_device(backend)
    if accel.is_openvino(backend):
        try:
            if not accel.is_exported(cfg["model"], cfg["imgsz"]):
                st.info(f"Converting the detector to OpenVINO for "
                        f"imgsz {cfg['imgsz']} — one-off, about 20 s.")
            model_path, task = accel.resolve_model(cfg["model"], cfg["imgsz"],
                                                   backend)
        except Exception as exc:
            st.warning(f"OpenVINO conversion failed ({exc}). "
                       f"Falling back to PyTorch CPU.")
            backend = accel.PYTORCH_CPU
            device = accel.predict_device(backend)

    args = dict(cfg)
    args["model"] = model_path
    args["task"] = task
    args["device"] = device
    args["export_csv"] = csv_path
    args["heatmaps_dir"] = heatmaps_dir if cfg["do_heatmaps"] else None
    cmd = build_command(input_path, raw_out, args, calib_path)

    # -u so the child's stdout reaches us line by line; warnings off so the
    # status panel shows pipeline stages, not library deprecation notices.
    cmd = [cmd[0], "-u"] + cmd[1:]
    env = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")

    progress = st.progress(0)
    status_line = st.empty()
    lines = []

    def show(pct, msg, detail=""):
        progress.progress(min(100, max(0, int(pct))))
        extra = (f"<span style='color:var(--faint);'> — {detail}</span>"
                 if detail else "")
        html(f"<div style='color:var(--muted);font-size:0.85rem;"
             f"margin-top:0.3rem;'>{msg}{extra}</div>", target=status_line)

    show(2, "🚀 Starting pipeline…")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env)
    stage_msg = "Starting pipeline…"
    for raw_line in iter(proc.stdout.readline, ""):
        line = raw_line.rstrip("\n")
        lines.append(line)

        if line.startswith("@STAGE "):
            try:
                d = json.loads(line[len("@STAGE "):])
                stage_msg = d.get("msg", stage_msg)
                show(d.get("pct", 0), f"⚙️ {stage_msg}")
            except json.JSONDecodeError:
                pass
        elif line.startswith("@PROGRESS "):
            try:
                d = json.loads(line[len("@PROGRESS "):])
                frame, total = d.get("frame", 0), d.get("total") or 0
                pct = 12 + (frame / total * 74) if total else 50
                detail = (f"frame {frame:,}/{total:,} · {d.get('fps', 0)} fps"
                          + (f" · ~{d.get('eta', 0)}s left" if d.get("eta") else ""))
                show(pct, "🏃 Tracking players, ball & possession", detail)
            except json.JSONDecodeError:
                pass

    proc.wait()
    if proc.returncode != 0:
        progress.empty()
        status_line.empty()
        real = [x for x in lines if not is_noise(x)]
        st.error("Pipeline exited with an error.")
        with st.expander("Pipeline log", expanded=True):
            st.code("\n".join(real[-60:]) or "\n".join(lines[-60:]), language="text")
        return None

    show(96, "🎬 Encoding H.264 for browser playback…")
    from utils import convert_to_h264, is_browser_playable, probe_codec

    # The pipeline writes raw_out with OpenCV's "mp4v" fourcc, i.e. MPEG-4
    # Part 2. That is a valid .mp4 that no browser will decode, so falling
    # back to it on a failed transcode gave a player that loaded and then
    # showed nothing. Now the preview is only offered once it is verified
    # playable, and the failure is stated instead of being shown as a blank
    # video element.
    preview_error = None
    try:
        convert_to_h264(raw_out, h264_out, fps=cfg.get("fps"),
                        max_long_side=PREVIEW_LONG_SIDE)
    except Exception as exc:
        preview_error = str(exc)

    playable = is_browser_playable(h264_out)
    if not playable:
        preview_error = preview_error or (
            f"the encoder produced a {probe_codec(h264_out) or 'unreadable'} "
            f"file, which browsers cannot play")

    progress.progress(100)
    progress.empty()
    status_line.empty()

    return {
        "raw_out": raw_out,
        "h264_out": h264_out,
        "playable": playable,
        "preview_error": preview_error,
        "csv_path": csv_path,
        "calib_path": calib_path,
        "heatmaps_dir": heatmaps_dir if cfg["do_heatmaps"] else None,
        "stats": parse_result_json(lines),
        "log": lines,
        "settings": {
            "backend": accel.label_for(backend),
            "imgsz": cfg["imgsz"],
            "detect_stride": cfg.get("detect_stride") or 1,
            # Carried so a later preview rebuild encodes at the same rate.
            "fps": cfg.get("fps"),
        },
    }


# A footballer's top speed is ~37 km/h (elite sprinters reach ~36 in a match).
# We compare against a generous ceiling and use the 90th percentile rather than
# the max: the extreme tail is dominated by ByteTrack identity switches, which
# teleport a player and produce a huge instantaneous speed. Blaming calibration
# for that would be wrong, so the message names both possible causes.
HUMAN_SPRINT_CEILING_KMH = 45.0


def _speed_sanity_check(df):
    """Flag speeds that are not physically possible, and say what to do."""
    if "speed" not in df.columns:
        return
    speeds = pd.to_numeric(df["speed"], errors="coerce").dropna()
    speeds = speeds[speeds > 0]
    if len(speeds) < 20:
        return

    median, p90 = float(speeds.median()), float(speeds.quantile(0.90))
    if p90 <= HUMAN_SPRINT_CEILING_KMH:
        st.caption(f"Speed sanity check: median {median:.1f} km/h, "
                   f"90th percentile {p90:.1f} km/h — physically plausible.")
        return

    st.warning("\n\n".join([
        f"⚠️ Speeds look too high: 90th percentile is **{p90:.0f} km/h** "
        f"(a footballer tops out near 37). Two things cause this, and both "
        f"are worth checking:",

        "1. **The calibrated span is too large.** Speed scales directly with "
        "the metres you entered under Calibration — try the span you can "
        "actually see rather than a full 105 x 68 m pitch.",

        "2. **Tracking is fragmenting IDs.** If **Detect every Nth frame** is "
        "above 1, ByteTrack sees bigger jumps and swaps identities, which "
        "registers as a teleport. Set it back to 1 for final numbers.",
    ]))


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# PER-PLAYER ANALYTICS
#
# Everything below is derived from the per-frame CSV the pipeline already
# writes, so it costs one file read rather than keeping tracking state in
# session memory. Column semantics, from analytics.export_frame_data_csv:
#
#   frame            index into the processed frames — the same index as the
#                    annotated video, so the frame explorer can seek by it
#   speed            km/h over a rolling 5-frame window, repeated across the
#                    frames of that window
#   distance         CUMULATIVE metres for that player up to this frame, not a
#                    per-frame delta. A player's total is therefore max(),
#                    never sum() — summing overcounts by the window length.
# ─────────────────────────────────────────────────────────────────────────────
TEAM_COLORS = ["#e94560", "#4a90e2"]
TEAM_LABELS = ["🔴 Team 1", "🔵 Team 2"]


@st.cache_data(show_spinner=False)
def _load_tracking(csv_path, mtime):
    """Read the per-frame CSV once per (path, modification time).

    Streamlit reruns the whole script on every widget interaction, and these
    panels all need the same frame; without the cache, dragging the frame
    explorer would re-parse a 15k-row CSV on every step. `mtime` is unused in
    the body but must NOT be underscore-prefixed: Streamlit excludes
    underscore-prefixed arguments from the cache key, so `_mtime` would be
    ignored and a re-run of the same video would serve the previous run's data.
    """
    return pd.read_csv(csv_path)


def load_tracking(csv_path):
    """Cached tracking frame for `csv_path`, or None when it is unusable."""
    try:
        return _load_tracking(csv_path, os.path.getmtime(csv_path))
    except Exception:
        return None


def _team_label(team):
    try:
        return TEAM_LABELS[int(team) - 1]
    except (ValueError, TypeError, IndexError):
        return "Unknown"


def player_summary(df):
    """One row per player: team, total distance, top speed, frames tracked."""
    if df is None or df.empty or "player_id" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("player_id").agg(
        team=("team", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        dist=("distance", "max"),      # cumulative column — see the note above
        spd=("speed", "max"),
        frames=("frame", "nunique"),
    ).reset_index()

    grouped["dist"] = grouped["dist"].fillna(0.0)
    grouped["spd"] = grouped["spd"].fillna(0.0)
    grouped["team_label"] = grouped["team"].map(_team_label)
    return grouped.sort_values("dist", ascending=False).reset_index(drop=True)


def _dark(chart, height):
    """The app's chart styling, applied identically to every panel."""
    return (chart
            .properties(height=height)
            .configure_view(strokeWidth=0, fill="#0d1117")
            .configure_axis(gridColor="#21262d", labelColor="#8b949e",
                            titleColor="#8b949e", domainColor="#21262d")
            .configure_legend(labelColor="#8b949e", titleColor="#8b949e",
                              fillColor="#0d1117", strokeColor="#21262d",
                              padding=8))


def read_frame_at(video_path, index):
    """Pull a single frame out of a rendered video.

    Seeking the encoded file per slider move costs a few milliseconds and
    keeps nothing in memory — the alternative, holding the decoded frames, is
    what made the basketball page unusable on a small container.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


def _scrub_source(r):
    """Which file the frame explorer should seek.

    The browser preview is downscaled, so the full-resolution render is
    preferred for scrubbing; its MPEG-4 codec is no obstacle to OpenCV, only
    to browsers.
    """
    for key in ("raw_out", "h264_out"):
        path = r.get(key)
        if path and os.path.exists(path):
            return path
    return None


def _render_ball_control(df):
    """Rolling share of possession across the match."""
    import altair as alt

    per_frame = df.groupby("frame")["team_in_control"].first()
    if per_frame.empty:
        return

    total = int(per_frame.index.max()) + 1
    # Aim for ~100 points whatever the clip length, so a three-minute match
    # does not turn into a few thousand slivers.
    window = max(10, total // 100)

    rows = []
    for start in range(0, total, window):
        segment = per_frame.loc[start:start + window - 1]
        t1 = int((segment == 1).sum())
        t2 = int((segment == 2).sum())
        if t1 + t2 == 0:
            continue
        rows.append({"frame": start,
                     "Team 1": round(100 * t1 / (t1 + t2)),
                     "Team 2": round(100 * t2 / (t1 + t2))})
    if not rows:
        return

    sec("⏱️", "Ball Control Over Time")
    melted = pd.DataFrame(rows).melt("frame", var_name="team", value_name="pct")
    st.altair_chart(
        _dark(
            alt.Chart(melted)
            .mark_area(opacity=0.75, interpolate="monotone")
            .encode(
                x=alt.X("frame:Q", title="Frame"),
                y=alt.Y("pct:Q", stack="normalize",
                        axis=alt.Axis(format="%", title="Ball Control")),
                color=alt.Color("team:N",
                                scale=alt.Scale(domain=["Team 1", "Team 2"],
                                                range=TEAM_COLORS),
                                legend=alt.Legend(orient="top-right")),
                tooltip=["frame:Q", "team:N", "pct:Q"],
            ), 200),
        use_container_width=True)
    st.caption(f"Possession share per {window}-frame window.")


def _render_player_panels(summary):
    """Player table plus the distance and speed bar charts."""
    import altair as alt

    sec("🏃", "Player Performance")

    # ByteTrack splits a player into a new ID whenever it loses them, so a
    # match typically yields far more IDs than players — most of them
    # fragments seen for a handful of frames. Ranking by distance and showing
    # the top N keeps the charts readable; the full set is in the CSV.
    total_players = len(summary)
    default_n = min(15, total_players)
    top_n = default_n
    if total_players > 5:
        top_n = st.slider(
            "Players shown (ranked by distance covered)",
            5, total_players, default_n, key="fb_top_n",
            help="Tracking assigns a new ID whenever a player is lost and "
                 "re-found, so most IDs are short fragments. The full, "
                 "unfiltered data is in the CSV under Export.")

    shown = summary.head(top_n).copy()

    table = pd.DataFrame({
        "Player ID": shown["player_id"],
        "Team": shown["team_label"],
        "Distance (m)": shown["dist"].round(2),
        "Max Speed (km/h)": shown["spd"].round(1),
        "Active Frames": shown["frames"],
    }).reset_index(drop=True)

    st.dataframe(
        table.style
             .background_gradient(subset=["Distance (m)"], cmap="RdYlGn")
             .background_gradient(subset=["Max Speed (km/h)"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(shown)} of {total_players} tracked IDs.")

    chart_df = shown.assign(pid=shown["player_id"].astype(str))

    sec("📈", "Distance by Player")
    st.altair_chart(
        _dark(
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("pid:N", sort="-y", title="Player ID",
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("dist:Q", title="Distance (m)"),
                color=alt.Color("team_label:N", title="Team",
                                scale=alt.Scale(domain=TEAM_LABELS,
                                                range=TEAM_COLORS),
                                legend=alt.Legend(orient="top-right")),
                tooltip=[
                    alt.Tooltip("pid:N", title="Player"),
                    alt.Tooltip("team_label:N", title="Team"),
                    alt.Tooltip("dist:Q", title="Distance (m)", format=".2f"),
                    alt.Tooltip("spd:Q", title="Max Speed (km/h)", format=".1f"),
                ],
            ), 280),
        use_container_width=True)

    sec("⚡", "Max Speed by Player")
    st.altair_chart(
        _dark(
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("pid:N", sort="-y", title="Player ID",
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("spd:Q", title="Max Speed (km/h)"),
                color=alt.Color("team_label:N",
                                scale=alt.Scale(domain=TEAM_LABELS,
                                                range=TEAM_COLORS),
                                legend=None),
                tooltip=[alt.Tooltip("pid:N", title="Player"),
                         alt.Tooltip("spd:Q", title="Max Speed (km/h)",
                                     format=".1f")],
            ), 230),
        use_container_width=True)


def _render_frame_explorer(r, df):
    """Scrub the annotated video and read off that frame's state."""
    source = _scrub_source(r)
    if not source or df is None or df.empty:
        return

    frames = df["frame"].max()
    if pd.isna(frames) or frames < 1:
        return
    max_f = int(frames)

    sec("🔎", "Frame Explorer")
    frame_idx = st.slider("Scrub through frames", 0, max_f, 0,
                          label_visibility="collapsed", key="fb_scrub")

    frame = read_frame_at(source, frame_idx)
    if frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    rows = df[df["frame"] == frame_idx]
    control = rows["team_in_control"].iloc[0] if not rows.empty else None
    holders = rows[rows["has_ball"] == True]["player_id"].tolist()  # noqa: E712

    holder_str = f"Player {int(holders[0])}" if holders else "—"
    control_str = _team_label(control) if control in (1, 2) else "—"
    fastest = "—"
    if not rows.empty and rows["speed"].notna().any():
        top = rows.loc[rows["speed"].idxmax()]
        fastest = f"Player {int(top['player_id'])} · {top['speed']:.1f} km/h"

    html(f"""
    <div class="frame-meta">
        <div class="fm-item"><div class="fm-label">Frame</div>
            <div class="fm-value">{frame_idx} / {max_f}</div></div>
        <div class="fm-item"><div class="fm-label">Players on frame</div>
            <div class="fm-value">{len(rows)}</div></div>
        <div class="fm-item"><div class="fm-label">Ball held by</div>
            <div class="fm-value">{holder_str}</div></div>
        <div class="fm-item"><div class="fm-label">Team in possession</div>
            <div class="fm-value">{control_str}</div></div>
        <div class="fm-item"><div class="fm-label">Fastest this frame</div>
            <div class="fm-value">{fastest}</div></div>
    </div>""")


def ensure_browser_preview(r):
    """Return (playable_path, error) for this result, transcoding if needed.

    Rather than giving up when the stored preview is not H.264, build one now
    from whichever render exists. That covers a first-pass transcode that
    failed, and results still sitting in session state from before the preview
    was verified at all — those recorded the pipeline's raw MPEG-4 output as
    the preview, which no browser will play.

    The rebuilt file is written next to the others and reused on later reruns,
    so the transcode happens at most once per run.
    """
    from utils import convert_to_h264, is_browser_playable

    stored = r.get("h264_out")
    if stored and is_browser_playable(stored):
        return stored, None

    sources = [p for p in (r.get("raw_out"), stored) if p and os.path.exists(p)]
    if not sources:
        return None, "no rendered video was produced"

    source = sources[0]
    repaired = os.path.splitext(source)[0] + "_web.mp4"

    # Reuse an earlier repair unless the render has moved on since.
    if is_browser_playable(repaired) and \
            os.path.getmtime(repaired) >= os.path.getmtime(source):
        return repaired, None

    try:
        with st.spinner("Preparing a browser-playable preview…"):
            convert_to_h264(source, repaired,
                            fps=(r.get("settings") or {}).get("fps"),
                            max_long_side=PREVIEW_LONG_SIDE)
    except Exception as exc:
        return None, str(exc)

    if not is_browser_playable(repaired):
        return None, "the encoder produced a file the browser cannot decode"
    return repaired, None


def _render_video(r):
    """Show the annotated video, rebuilding the preview if it is not playable."""
    path, error = ensure_browser_preview(r)

    if path:
        st.video(path)
        st.caption(f"{os.path.getsize(path) / 1e6:.1f} MB preview · "
                   f"full-resolution render under Export below.")
        return

    reason = error or r.get("preview_error") or "the preview could not be built"
    st.warning(
        f"The annotated video could not be prepared for in-browser playback "
        f"({reason}).\n\n"
        f"The render itself is fine — download it under **Export** below and "
        f"it will play in VLC or any desktop player. In-browser playback "
        f"needs H.264, which requires `ffmpeg` on the server: check that "
        f"`ffmpeg` is listed in `packages.txt`.")


def _render_results(r):
    stats = r.get("stats") or {}

    sec("📹", "Analysed Video")
    _render_video(r)

    if stats:
        sec("📊", "Match Overview")
        pct1 = stats.get("possession_team1_pct", 0)
        pct2 = stats.get("possession_team2_pct", 0)
        kpi_grid(
            kpi("Team 1 Possession", f"{pct1}%", "", "red"),
            kpi("Team 2 Possession", f"{pct2}%", "", "blue"),
            kpi("Players Tracked", stats.get("players_tracked", 0), "", "green"),
            kpi("Frames Analysed", f"{stats.get('frames_processed', 0):,}", "", "gray"),
        )

        if pct1 + pct2 > 0:
            html(f"""
            <div class="team-strip">
                <div class="team-strip-header">
                    <div class="t1">🔴 Team 1 — {pct1}%</div>
                    <div class="label">BALL POSSESSION</div>
                    <div class="t2">{pct2}% — Team 2 🔵</div>
                </div>
                <div class="control-bar">
                    <div class="cb-t1" style="width:{pct1}%"></div>
                    <div class="cb-t2" style="width:{pct2}%"></div>
                </div>
            </div>""")

        sec("⚙️", "Pipeline Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Processing speed", f"{stats.get('processing_fps', 0)} fps",
                  help="Frames analysed per second (whole pipeline)")
        m2.metric("Detection speed", f"{stats.get('detection_fps', 0)} fps",
                  help=f"YOLO frames/sec at stride {stats.get('detect_stride', 1)}")
        m3.metric("Run time", f"{stats.get('total_time_s', 0)} s")
        m4.metric("Video fps", stats.get("video_fps", 0))

        cfg_used = r.get("settings") or {}
        if cfg_used:
            st.caption(
                f"{cfg_used.get('backend', '—')} · imgsz "
                f"{cfg_used.get('imgsz', 640)} · detect stride "
                f"{cfg_used.get('detect_stride', 1)}")

    # ── per-player analytics, all derived from the one CSV read ─────────────
    df = None
    if r.get("csv_path") and os.path.exists(r["csv_path"]):
        df = load_tracking(r["csv_path"])

    if df is not None and not df.empty:
        _speed_sanity_check(df)
        _render_ball_control(df)

        summary = player_summary(df)
        if not summary.empty:
            _render_player_panels(summary)

        _render_frame_explorer(r, df)

        sec("📈", "Tracked Data")
        st.dataframe(df.head(500), use_container_width=True, hide_index=True)
        st.caption(f"Showing the first {min(len(df), 500):,} of {len(df):,} rows "
                   f"— download the full CSV below.")
    elif r.get("csv_path") and os.path.exists(r["csv_path"]):
        st.warning("Could not read the tracked data CSV, so the per-player "
                   "panels are unavailable.")

    # ── downloads ────────────────────────────────────────────────────────────
    sec("⬇️", "Export")
    d1, d2, d3 = st.columns(3)
    if os.path.exists(r["raw_out"]):
        with open(r["raw_out"], "rb") as fh:
            d1.download_button("🎬  Download raw video", fh,
                               file_name=os.path.basename(r["raw_out"]),
                               use_container_width=True, key="fb_dl_raw")
    if os.path.exists(r["csv_path"]):
        with open(r["csv_path"], "rb") as fh:
            d2.download_button("📄  Download CSV", fh,
                               file_name=os.path.basename(r["csv_path"]),
                               use_container_width=True, key="fb_dl_csv")
    if r["calib_path"] and os.path.exists(r["calib_path"]):
        with open(r["calib_path"], "rb") as fh:
            d3.download_button("📐  Download calibration JSON", fh,
                               file_name=os.path.basename(r["calib_path"]),
                               use_container_width=True, key="fb_dl_calib")

    # ── heatmaps ─────────────────────────────────────────────────────────────
    heatmaps_dir = r.get("heatmaps_dir")
    if heatmaps_dir and os.path.isdir(heatmaps_dir):
        heat_files = sorted(os.listdir(heatmaps_dir))
        if heat_files:
            sec("🔥", "Player Heatmaps")
            st.caption(f"{len(heat_files)} heatmaps saved to `{heatmaps_dir}`")
            with st.expander("View heatmaps"):
                for fname in heat_files:
                    st.image(os.path.join(heatmaps_dir, fname), caption=fname,
                             use_container_width=True)


def _render_landing():
    sec("🏆", "What This App Does")
    feature_cards([
        ("🏃", "Player Tracking", "YOLO + ByteTrack detection and tracking of every player, referee and the ball."),
        ("👕", "Team Assignment", "KMeans jersey-colour clustering splits the squads automatically."),
        ("🤝", "Ball Possession", "Nearest-player assignment turns the ball track into per-team possession %."),
        ("🎥", "Camera Motion", "Optical-flow camera-movement estimation keeps positions stable on panning shots."),
        ("📐", "Perspective Transform", "Four pitch corners map image pixels to real-world metres."),
        ("⚡", "Speed & Distance", "Per-player speed (km/h) and cumulative distance (m) overlays."),
        ("🔥", "Heatmaps", "Optional per-player positional heatmaps across the match."),
        ("📊", "CSV Export", "Every frame's tracked data written out for your own analysis."),
    ])

    spacer()
    sec("🚀", "Getting Started")
    html("""
    <div class="steps">
        <b>1.</b> Upload a football video above<br>
        <b>2.</b> Point <b>Detection</b> at your trained weights (defaults to <code>models/best.pt</code>)<br>
        <b>3.</b> Optionally set the four pitch corners under <b>Calibration</b> for speed &amp; distance<br>
        <b>4.</b> Hit <span class="hl">Run Analysis</span> — the pipeline logs stream straight into the page<br>
        <b>5.</b> Watch the annotated video, then export the CSV and heatmaps
    </div>""")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def render(sport=None):
    """Render the whole Football Analysis page. Called by the router."""
    cfg = _sidebar()

    sec("📁", "Upload Video")
    uploaded = st.file_uploader(
        "Choose a football video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed",
        key="fb_upload",
    )

    if uploaded is None:
        html("""
        <div class="upload-zone">
            <div style="font-size:2.5rem;margin-bottom:0.6rem;">📹</div>
            <div style="color:#e6edf3;font-weight:600;font-size:1rem;">Drop your video here</div>
            <div style="color:#8b949e;font-size:0.8rem;margin-top:0.3rem;">MP4 · AVI · MOV · MKV · WEBM</div>
        </div>""")
        if STATE_KEY in st.session_state:
            _render_results(st.session_state[STATE_KEY])
        else:
            _render_landing()
        return

    input_path = save_upload(uploaded)
    video_name = os.path.splitext(os.path.basename(uploaded.name))[0]

    frame0, preview_scale = read_first_frame(input_path)
    st.image(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB),
             caption="First frame (calibration preview)",
             use_container_width=True)

    sec("📐", "Calibration — needed for speed & distance")
    calib_method = st.radio(
        "Calibration method",
        ["None (boxes only)", "Automatic (detect pitch)", "Manual corners"],
        index=1, horizontal=True, key="fb_calib_method",
        label_visibility="collapsed",
        help="Speed and distance come from mapping pitch pixels to real metres, "
             "which needs the four pitch corners. Automatic finds them from the "
             "grass and the painted lines; Manual lets you type them. Without "
             "either, boxes, team colours and the ball triangle still render.")

    calib_corners = None
    if calib_method == "Automatic (detect pitch)":
        calib_corners = _auto_calibration(input_path, preview_scale)
    elif calib_method == "Manual corners":
        calib_corners = _calibration_corners(frame0, preview_scale)
    else:
        st.caption("Pick **Automatic** above and the pitch corners are found "
                   "for you — no pixel coordinates to type.")

    spacer("0.5rem")
    left, right = st.columns([1, 3])
    with left:
        run_analysis = st.button("🚀  Run Analysis", type="primary",
                                 use_container_width=True, key="fb_run")
    with right:
        cap = cv2.VideoCapture(input_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if cfg["max_frames"]:
            n_frames = min(n_frames, cfg["max_frames"])
        est = estimate_runtime(n_frames, cfg["detect_stride"] or 1,
                               cfg["imgsz"], cfg["backend"])
        eta = f"{est/60:.1f} min" if est > 60 else f"{est:.0f} s"

        if not os.path.exists(cfg["model"]):
            st.error(f"⚠️ Detector not found at `{cfg['model']}` — set the path "
                     f"under **Detection** in the sidebar.")
        elif est > 240:
            st.warning(f"⏱️ ~{eta} of work for {n_frames:,} frames on "
                       f"**{accel.label_for(cfg['backend'])}**. Cap **Max "
                       f"frames**, raise **Detect every Nth frame** or lower "
                       f"**Detection resolution** to bring this down.")
        else:
            st.success(f"✅ {n_frames:,} frames on "
                       f"**{accel.label_for(cfg['backend'])}** — roughly {eta}."
                       + ("" if calib_corners is not None else
                          " No calibration, so speed & distance are skipped."))

    if run_analysis:
        results = _run_pipeline(input_path, video_name, cfg, calib_corners)
        if results is None:
            return
        st.session_state[STATE_KEY] = results
        st.success("Done! Annotated video below.")

    if STATE_KEY in st.session_state:
        _render_results(st.session_state[STATE_KEY])
    else:
        _render_landing()
