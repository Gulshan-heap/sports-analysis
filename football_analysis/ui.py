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
    sec, kpi, kpi_grid, chip, spacer, feature_cards,
)

STATE_KEY = "football_results"

UPLOAD_DIR = os.path.join(ROOT, "app_uploads")
OUTPUT_DIR = os.path.join(ROOT, "output_videos")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_MODEL = os.path.join(MODEL_DIR, "best.pt")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_upload(uploaded_file):
    dst = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dst


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
    cmd += ["--progress-every", "30"]
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


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        model_ok = os.path.exists(DEFAULT_MODEL)
        chip("✅ Detector found" if model_ok else "⚠️ models/best.pt missing",
             os.path.relpath(DEFAULT_MODEL, ROOT),
             "#3fb950" if model_ok else "#f85149",
             state="ok" if model_ok else "bad")

        with st.expander("🎯 Detection", expanded=True):
            model_path = st.text_input("YOLO model (.pt)", value=DEFAULT_MODEL,
                                       key="fb_model")
            use_cache = st.checkbox("Use cached tracking (same video re-runs)",
                                    value=False, key="fb_cache")
            max_frames = st.number_input("Max frames (0 = all)", min_value=0,
                                         value=0, step=50, key="fb_maxframes")
            scale = st.number_input("Frame scale (1.0 = original)", min_value=0.1,
                                    max_value=1.0, value=1.0, step=0.1,
                                    key="fb_scale")
            detect_stride = st.number_input(
                "Detect stride (1 = every frame; 3 ≈ 3x faster)",
                min_value=1, value=1, step=1, key="fb_stride")
            fps = st.number_input("FPS override (0 = auto)", min_value=0, value=0,
                                  step=1, key="fb_fps")
            do_heatmaps = st.checkbox("Generate player heatmaps", value=False,
                                      key="fb_heatmaps")

        with st.expander("📐 Calibration", expanded=False):
            calib_method = st.radio(
                "Calibration method",
                ["None (boxes only)", "Manual corners"],
                index=0,
                key="fb_calib_method",
                help="Speed/distance labels only appear for players inside the "
                     "calibrated region. Without calibration, boxes, team "
                     "colors and the ball triangle still render.")

        if STATE_KEY in st.session_state:
            if st.button("🗑️ Clear results", use_container_width=True,
                         key="fb_clear"):
                del st.session_state[STATE_KEY]
                st.rerun()

    return {
        "model": model_path,
        "use_cache": use_cache,
        "max_frames": int(max_frames) if max_frames else None,
        "scale": scale if scale and scale != 1.0 else None,
        "detect_stride": int(detect_stride) if detect_stride else None,
        "fps": fps if fps else None,
        "do_heatmaps": do_heatmaps,
        "calib_method": calib_method,
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

    args = dict(cfg)
    args["export_csv"] = csv_path
    args["heatmaps_dir"] = heatmaps_dir if cfg["do_heatmaps"] else None
    cmd = build_command(input_path, raw_out, args, calib_path)

    with st.status("Running analysis pipeline…", expanded=True) as status:
        lines = []
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for stdout_line in iter(proc.stdout.readline, ""):
            lines.append(stdout_line.rstrip("\n"))
            status.write("\n".join(lines[-40:]))
        proc.wait()
        if proc.returncode != 0:
            status.update(label="Pipeline failed", state="error")
            st.error("Pipeline exited with an error:\n\n" + "\n".join(lines[-50:]))
            return None
        status.update(label="Pipeline finished", state="complete")

    with st.spinner("Encoding H.264 for browser playback…"):
        try:
            from utils import convert_to_h264
            convert_to_h264(raw_out, h264_out, fps=cfg.get("fps"))
        except Exception as exc:            # fall back to showing the raw file
            st.warning(f"Could not create H.264 preview ({exc}). Showing raw instead.")
            h264_out = raw_out

    return {
        "raw_out": raw_out,
        "h264_out": h264_out,
        "csv_path": csv_path,
        "calib_path": calib_path,
        "heatmaps_dir": heatmaps_dir if cfg["do_heatmaps"] else None,
        "stats": parse_result_json(lines),
        "log": lines,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def _render_results(r):
    stats = r.get("stats") or {}

    sec("📹", "Analysed Video")
    if os.path.exists(r["h264_out"]):
        st.video(r["h264_out"])

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
            st.markdown(f"""
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
            </div>""", unsafe_allow_html=True)

        sec("⚙️", "Pipeline Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Processing speed", f"{stats.get('processing_fps', 0)} fps",
                  help="Frames analysed per second (whole pipeline)")
        m2.metric("Detection speed", f"{stats.get('detection_fps', 0)} fps",
                  help=f"YOLO frames/sec at stride {stats.get('detect_stride', 1)}")
        m3.metric("Run time", f"{stats.get('total_time_s', 0)} s")
        m4.metric("Video fps", stats.get("video_fps", 0))

        with st.expander("Full run stats"):
            st.json(stats)

    # ── per-frame data ───────────────────────────────────────────────────────
    if os.path.exists(r["csv_path"]):
        sec("📈", "Tracked Data")
        try:
            df = pd.read_csv(r["csv_path"])
            st.dataframe(df.head(500), use_container_width=True, hide_index=True)
            st.caption(f"Showing the first {min(len(df), 500):,} of {len(df):,} rows "
                       f"— download the full CSV below.")
        except Exception as exc:
            st.warning(f"Could not preview the CSV ({exc}).")

    # ── downloads ────────────────────────────────────────────────────────────
    sec("⬇️", "Export")
    d1, d2, d3 = st.columns(3)
    if os.path.exists(r["raw_out"]):
        with open(r["raw_out"], "rb") as fh:
            d1.download_button("🎬  Download raw video", fh.read(),
                               file_name=os.path.basename(r["raw_out"]),
                               use_container_width=True, key="fb_dl_raw")
    if os.path.exists(r["csv_path"]):
        with open(r["csv_path"], "rb") as fh:
            d2.download_button("📄  Download CSV", fh.read(),
                               file_name=os.path.basename(r["csv_path"]),
                               use_container_width=True, key="fb_dl_csv")
    if r["calib_path"] and os.path.exists(r["calib_path"]):
        with open(r["calib_path"], "rb") as fh:
            d3.download_button("📐  Download calibration JSON", fh.read(),
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
    st.markdown("""
    <div class="steps">
        <b>1.</b> Upload a football video above<br>
        <b>2.</b> Point <b>Detection</b> at your trained weights (defaults to <code>models/best.pt</code>)<br>
        <b>3.</b> Optionally set the four pitch corners under <b>Calibration</b> for speed &amp; distance<br>
        <b>4.</b> Hit <span class="hl">Run Analysis</span> — the pipeline logs stream straight into the page<br>
        <b>5.</b> Watch the annotated video, then export the CSV and heatmaps
    </div>""", unsafe_allow_html=True)


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
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:2.5rem;margin-bottom:0.6rem;">📹</div>
            <div style="color:#e6edf3;font-weight:600;font-size:1rem;">Drop your video here</div>
            <div style="color:#8b949e;font-size:0.8rem;margin-top:0.3rem;">MP4 · AVI · MOV · MKV · WEBM</div>
        </div>""", unsafe_allow_html=True)
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

    calib_corners = None
    if cfg["calib_method"] == "Manual corners":
        sec("📐", "Calibration")
        calib_corners = _calibration_corners(frame0, preview_scale)

    spacer("0.5rem")
    left, right = st.columns([1, 3])
    with left:
        run_analysis = st.button("🚀  Run Analysis", type="primary",
                                 use_container_width=True, key="fb_run")
    with right:
        if calib_corners is None:
            st.info("ℹ️ No calibration — boxes, team colours and the ball "
                    "triangle still render, but speed & distance are skipped.")
        else:
            st.success("✅ Pitch calibrated — speed & distance will be computed.")

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
