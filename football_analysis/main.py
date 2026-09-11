import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np

from utils import pick_corners, get_center_of_bbox, get_foot_position
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from analytics import export_frame_data_csv, generate_player_heatmap, team_possession_summary


def _draw_hud(frame, frame_idx, total_frames, t_loop):
    """Overlay live processing metrics (fps / progress) onto the output frame."""
    h, w = frame.shape[:2]
    s = max(0.6, min(1.6, min(h, w) / 720.0))
    elapsed = time.perf_counter() - t_loop
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    pct = (frame_idx / total_frames * 100) if total_frames else 0.0
    text = f"{proc_fps:.1f} fps | frame {frame_idx}/{total_frames or '?'} ({pct:.0f}%)"
    font_scale = max(0.45, min(0.8, 0.55 * s))
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    org = (w - tw - int(12 * s), int(30 * s))
    # dark outline + white fill so it reads on any background
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), max(1, int(3 * s)), cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), max(1, int(round(1.4 * s))), cv2.LINE_AA)


def load_calibration(calib_path, input_video):
    """Return (pixel_vertices, region_length, region_width).

    calib_path : path to JSON with keys pixel_vertices / region_length /
                 region_width, OR the string "interactive" to click 4 corners.
    None       : fall back to the built-in demo calibration.
    """
    if calib_path and os.path.exists(calib_path):
        with open(calib_path) as f:
            calib = json.load(f)
        return (
            calib.get("pixel_vertices"),
            calib.get("region_length"),
            calib.get("region_width"),
        )
    if calib_path == "interactive":
        print("Interactive calibration: click 4 pitch corners (TL, TR, BR, BL), press 'q'.")
        pts = pick_corners(input_video)
        region_length = float(input("Region length in meters (real-world x span): ") or 105)
        region_width = float(input("Region width in meters (real-world y span): ") or 68)
        return (pts, region_length, region_width)
    return (None, None, None)


def build_parser():
    parser = argparse.ArgumentParser(description="Football analysis pipeline for any video")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default=None, help="Path to output video")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--calib", default=None,
                        help="Path to calibration JSON, or 'interactive' to click corners")
    parser.add_argument("--use_cache", action="store_true",
                        help="Reuse cached tracking stubs (only for re-runs on the SAME video)")
    parser.add_argument("--fps", type=float, default=None, help="Override video fps")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Only process the first N frames")
    parser.add_argument("--scale", type=float, default=None,
                        help="Downscale frames by this factor (e.g. 0.5)")
    parser.add_argument("--export-csv", default="output_videos/match_data.csv",
                        help="Path for frame CSV (pass '' to disable)")
    parser.add_argument("--heatmaps-dir", default=None,
                        help="Directory to save heatmaps for ALL players (None = skip)")
    parser.add_argument("--detect-stride", type=int, default=1,
                        help="Run YOLO every Nth frame; skipped frames reuse the "
                             "last known boxes (big latency win, default 1 = off)")
    parser.add_argument("--progress-every", type=int, default=30,
                        help="Print a progress line every N frames (0 = silent)")
    parser.add_argument("--no-hud", dest="show_hud", action="store_false",
                        help="Do not burn the live fps/progress overlay into the output video")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO input size. Lower is quadratically faster but "
                             "misses small/distant objects (the ball especially)")
    parser.add_argument("--device", default=None,
                        help="Inference device: cpu, cuda, or intel:gpu for an "
                             "OpenVINO IR model")
    parser.add_argument("--task", default=None,
                        help="Model task (detect/pose/...). Required when --model "
                             "points at an exported OpenVINO directory")
    return parser


def emit(kind, **fields):
    """Machine-readable progress line for the Streamlit front-end.

    The UI shows these as named stages with a progress bar instead of dumping
    raw stdout (which is full of library warnings nobody can act on). Humans
    running the CLI still get the plain-text lines below.
    """
    print(f"@{kind} " + json.dumps(fields), flush=True)


def run_pipeline(args):
    """Run the full analysis pipeline and return a dict of produced artifacts.

    Streams frame-by-frame (constant memory — no whole-video RAM buffering)
    and prints live progress. Kept separate from main() so the CLI and the
    Streamlit web app share the exact same code path.
    """
    t_start = time.perf_counter()

    video_name = os.path.splitext(os.path.basename(args.input))[0]
    output_path = args.output or f"output_videos/{video_name}_output.mp4"
    stub_dir = f"stubs/{video_name}"
    os.makedirs(stub_dir, exist_ok=True)
    stub_path = os.path.join(stub_dir, "track_stubs.pkl")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    emit("STAGE", pct=3, msg="Opening video")
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 24
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Output size honours --scale so high-resolution videos process faster.
    out_w, out_h = src_w, src_h
    if args.scale and args.scale != 1.0:
        out_w = max(2, int(src_w * args.scale))
        out_h = max(2, int(src_h * args.scale))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))

    emit("STAGE", pct=8, msg="Loading detector")
    tracker = Tracker(args.model, detect_stride=args.detect_stride,
                      imgsz=args.imgsz, device=args.device, task=args.task)

    # Cached raw tracks from a previous run (bbox only). Everything derived
    # (positions, teams, speed, possession) is recomputed deterministically
    # during replay, so the cache only needs what detection produced.
    cached_tracks = None
    if args.use_cache and os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            cached_tracks = pickle.load(f)
        print(f"Loaded cached tracks from {stub_path}; replaying for annotation.")

    tracks = {"players": [], "referees": [], "ball": []}

    pixel_vertices, region_length, region_width = load_calibration(args.calib, args.input)

    # Calibration is marked on full-resolution frames, but --scale resizes every
    # frame before tracking, so the vertices have to move with them. Without
    # this, --scale 0.5 silently halves every distance and speed.
    if pixel_vertices is not None and (out_w, out_h) != (src_w, src_h):
        sx, sy = out_w / float(src_w), out_h / float(src_h)
        pixel_vertices = [[x * sx, y * sy] for x, y in pixel_vertices]

    view_transformer = ViewTransformer(pixel_vertices, region_length, region_width)

    camera_estimator = None
    speed_estimator = SpeedAndDistance_Estimator(fps=fps)
    team_assigner = TeamAssigner()
    player_assigner = None

    possession_counts = {1: 0, 2: 0}
    last_team = 1
    team_ball_control = []

    emit("STAGE", pct=12, msg="Analysing frames", total=total_frames)
    frame_num = 0
    detect_seconds = 0.0
    detect_count = 0
    t_loop = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_num >= args.max_frames:
            break
        if (out_w, out_h) != (src_w, src_h):
            frame = cv2.resize(frame, (out_w, out_h))

        if camera_estimator is None:
            tracker.set_frame_size(frame)
            camera_estimator = CameraMovementEstimator(frame)
            player_assigner = PlayerBallAssigner(frame_height=frame.shape[0])

        # ---- 1. detection / tracking (or cached replay) ----
        if cached_tracks is not None and frame_num < len(cached_tracks["players"]):
            entry = {
                "players": cached_tracks["players"][frame_num],
                "referees": cached_tracks["referees"][frame_num],
                "ball": cached_tracks["ball"][frame_num],
            }
        else:
            run_detection = (frame_num % max(1, tracker.detect_stride) == 0)
            t0 = time.perf_counter()
            entry = tracker.step(frame, run_detection=run_detection)
            detect_seconds += time.perf_counter() - t0
            if run_detection:
                detect_count += 1
        tracks["players"].append(entry["players"])
        tracks["referees"].append(entry["referees"])
        tracks["ball"].append(entry["ball"])
        player_track = entry["players"]

        # ---- 2. positions: pixels -> camera-adjusted -> real-world meters ----
        movement = camera_estimator.process_frame(frame)
        for role_name, role in (("players", player_track),
                                ("referees", entry["referees"]),
                                ("ball", entry["ball"])):
            for info in role.values():
                bbox = info["bbox"]
                pos = (get_center_of_bbox(bbox) if role_name == "ball"
                       else get_foot_position(bbox))
                adjusted = (pos[0] - movement[0], pos[1] - movement[1])
                info["position"] = pos
                info["position_adjusted"] = adjusted
                transformed = view_transformer.transform_point(adjusted)
                info["position_transformed"] = (
                    transformed.squeeze().tolist() if transformed is not None else None)

        # ---- 3. teams (assigned once per track id, cached afterwards) ----
        if not team_assigner.team_colors and player_track:
            team_assigner.assign_team_color(frame, player_track)
        for pid, info in player_track.items():
            team = team_assigner.get_player_team(frame, info["bbox"], pid)
            info["team"] = team
            info["team_color"] = team_assigner.team_colors.get(team, (0, 0, 255))

        # ---- 4. ball possession (carries last team through detection gaps) ----
        ball_dict = entry["ball"]
        ball_bbox = next(iter(ball_dict.values()), {}).get("bbox") if ball_dict else None
        assigned = (player_assigner.assign_ball_to_player(player_track, ball_bbox)
                    if ball_bbox else -1)
        if assigned != -1:
            player_track[assigned]["has_ball"] = True
            last_team = player_track[assigned]["team"]
        team_ball_control.append(last_team)
        possession_counts[last_team] = possession_counts.get(last_team, 0) + 1

        # ---- 5. speed & distance (rolling windows) ----
        speed_estimator.update(tracks)

        # ---- 6. annotate + write ----
        out_frame = tracker.annotate_frame(
            frame, player_track, entry["referees"], entry["ball"],
            control_counts=dict(possession_counts))
        camera_estimator.annotate_frame(out_frame, movement)
        speed_estimator.annotate_frame(out_frame, player_track)
        if args.show_hud:
            _draw_hud(out_frame, frame_num + 1, total_frames, t_loop)
        writer.write(out_frame)

        frame_num += 1
        if args.progress_every and frame_num % args.progress_every == 0:
            elapsed = time.perf_counter() - t_loop
            proc_fps = frame_num / elapsed if elapsed else 0.0
            eta = ((total_frames - frame_num) / proc_fps
                   if proc_fps and total_frames else 0.0)
            emit("PROGRESS", frame=frame_num, total=total_frames,
                 fps=round(proc_fps, 1), eta=round(eta),
                 t1=possession_counts.get(1, 0), t2=possession_counts.get(2, 0))
            print(f"frame {frame_num}/{total_frames or '?'} | {proc_fps:.1f} fps | "
                  f"eta {eta:.0f}s | possession T1 {possession_counts.get(1, 0)} "
                  f"T2 {possession_counts.get(2, 0)}")

    cap.release()
    writer.release()
    emit("STAGE", pct=88, msg="Calculating speed & distance")
    speed_estimator.finalize(tracks)

    if frame_num == 0:
        raise RuntimeError("No frames could be read from the input video.")

    # Persist tracks so a re-run with --use_cache skips detection entirely.
    if cached_tracks is None:
        with open(stub_path, "wb") as f:
            pickle.dump(tracks, f)

    elapsed = time.perf_counter() - t_start
    proc_fps = frame_num / elapsed if elapsed else 0.0
    det_fps = detect_count / detect_seconds if detect_seconds else 0.0

    team_ball_control = np.array(team_ball_control)

    if args.export_csv:
        emit("STAGE", pct=92, msg="Exporting per-frame CSV")
        export_frame_data_csv(tracks, team_ball_control, output_path=args.export_csv)
    team_possession_summary(team_ball_control)

    heatmap_paths = []
    if args.heatmaps_dir:
        emit("STAGE", pct=95, msg="Generating player heatmaps")
        os.makedirs(args.heatmaps_dir, exist_ok=True)
        player_ids = set()
        for frame in tracks['players']:
            player_ids.update(frame.keys())
        for player_id in sorted(player_ids):
            heatmap_path = os.path.join(args.heatmaps_dir, f"player_{player_id}_heatmap.png")
            generate_player_heatmap(tracks, player_id=player_id, output_path=heatmap_path)
            if os.path.exists(heatmap_path):
                heatmap_paths.append(heatmap_path)

    total = max(1, possession_counts.get(1, 0) + possession_counts.get(2, 0))
    stats = {
        "frames_processed": frame_num,
        "total_time_s": round(elapsed, 2),
        "processing_fps": round(proc_fps, 2),
        "detection_fps": round(det_fps, 2),
        "detect_stride": tracker.detect_stride,
        "video_fps": round(float(fps), 2),
        "players_tracked": len({pid for fr in tracks["players"] for pid in fr}),
        "possession_team1_pct": round(possession_counts.get(1, 0) / total * 100, 1),
        "possession_team2_pct": round(possession_counts.get(2, 0) / total * 100, 1),
        "output_path": output_path,
        "csv_path": args.export_csv or None,
        "heatmap_paths": heatmap_paths,
        "video_name": video_name,
    }
    # Machine-readable summary line, parsed by the Streamlit app for metrics.
    emit("STAGE", pct=99, msg="Finishing up")
    print("RESULT_JSON: " + json.dumps(stats))

    return stats


def main():
    if len(sys.argv) == 1:
        # No arguments at all — almost certainly a `streamlit run main.py`
        # mistake (the web UI lives in app.py). Explain instead of crashing.
        print(
            "\nNo --input video given - nothing to analyse.\n"
            "\n  This is the command-line tool. Use it like:\n"
            "      python main.py --input path/to/video.mp4\n"
            "\n  For the interactive web app, run:\n"
            "      streamlit run app.py\n"
        )
        return
    args = build_parser().parse_args()
    results = run_pipeline(args)
    print(f"Done. Output saved to {results['output_path']}")
    if results.get("csv_path"):
        print(f"CSV saved to {results['csv_path']}")
    if results.get("heatmap_paths"):
        print(f"Heatmaps saved ({len(results['heatmap_paths'])} files) in {args.heatmaps_dir}")
    return results


if __name__ == '__main__':
    main()