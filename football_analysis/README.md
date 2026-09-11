# ⚽ Football Video Analysis (generic, low-latency)

Detect, track and analyse players, referees and the ball in **any football
video** — then get an annotated output video, per-frame CSV data, player
heatmaps and live performance/possession metrics. Works from the command line
or a Streamlit web app.

## Features

- **Any video** — resolution, fps and length are read from the file; overlays
  (ellipses, IDs, panels, HUD) scale with the frame size.
- **Streaming pipeline** — frames are read → analysed → annotated → written one
  at a time, so memory stays constant no matter how long the video is.
- **Detect stride** — run YOLO every Nth frame (`--detect-stride`); skipped
  frames reuse the last known boxes for a near-linear speedup.
- **Frame scale** — `--scale 0.5` halves the working resolution for fast
  previews or long videos.
- **Track cache** — `--use_cache` replays saved detections instantly
  (re-annotation runs at several× real-time, no YOLO involved).
- **Metrics** — live fps/ETA progress in the console, an fps HUD burned into
  the output video, and a final machine-readable `RESULT_JSON` summary
  (parsing into metric cards in the Streamlit app).
- **Analysis** — team colors (KMeans on jersey crops), ball possession with
  carry-over, camera-motion compensation, homography to real-world meters,
  per-player speed (km/h) and distance (m), player heatmaps.

## Setup

```bash
pip install -r requirements.txt
```

Place your YOLO model at `models/best.pt` (or pass `--model path/to/model.pt`).
The model must have classes matching (or remappable via) the tracker's
`class_map`: `player`, `referee`, `ball`, optional `goalkeeper`.

## Command line

```bash
# full analysis of any video
python main.py --input path/to/video.mp4 --output output_videos/out.mp4

# fast pass: half resolution, YOLO every 2nd frame, plus CSV + heatmaps
python main.py --input video.mp4 --scale 0.5 --detect-stride 2 \
    --export-csv out.csv --heatmaps-dir heatmaps

# instant re-render from cached detections (same video)
python main.py --input video.mp4 --use_cache --export-csv out.csv
```

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Path to any input video |
| `--output` | `output_videos/<name>_output.mp4` | Output video path |
| `--model` | `models/best.pt` | YOLO model weights |
| `--calib` | demo calibration | Calibration JSON, or `interactive` to click 4 pitch corners |
| `--use_cache` | off | Reuse cached detections for the same video |
| `--fps` | auto | Override output video fps |
| `--max-frames N` | all | Only process the first N frames |
| `--scale F` | 1.0 | Downscale frames (e.g. `0.5`) — big speedup |
| `--export-csv PATH` | `output_videos/match_data.csv` | Per-frame player data CSV |
| `--heatmaps-dir DIR` | off | Save one heatmap PNG per tracked player |
| `--detect-stride N` | 1 | Run YOLO every Nth frame (1 = every frame) |
| `--progress-every N` | 30 | Print a progress line every N frames (0 = silent) |
| `--no-hud` | off | Don't burn the fps/progress overlay into the video |

### Calibration JSON

Speed/distance metrics need the pixel region of the visible pitch mapped to
real-world meters:

```json
{
  "pixel_vertices": [[x_tl, y_tl], [x_tr, y_tr], [x_br, y_br], [x_bl, y_bl]],
  "region_length": 23.32,
  "region_width": 68.0
}
```

`region_length`/`region_width` are the real-world meters spanned by the
quadrilateral you picked (a single camera sees only part of the pitch —
calibrate just the visible region). Without a calibration, boxes, team colors,
possession and the HUD still work; only speed/distance/heatmaps need it.

## Web app

```bash
streamlit run app.py
```

Upload a video, pick options (scale, detect stride, cache, calibration
corners), run, and get the annotated video plus metric cards (processing speed,
detection speed, possession, frames, run time, players tracked).

## Outputs

- **Annotated video** — tracked boxes/IDs, team colors, ball triangle, speed
  and distance labels, camera-movement panel, live possession panel, fps HUD.
- **CSV** — columns: `frame, player_id, team, x, y, speed, distance,
  has_ball, team_in_control` (x/y in real-world meters when calibrated).
- **Heatmaps** — one KDE PNG per player from their calibrated positions.
- **`RESULT_JSON:` console line** — frames processed, processing/detection fps,
  detect stride, players tracked, possession %, artifact paths.

## Notes

- The first run per video caches detections under `stubs/<video_name>/`;
  re-runs with `--use_cache` skip YOLO entirely.
- The `ByteTrack` deprecation FutureWarning from `supervision` is harmless.
- Team "1"/"2" labels follow KMeans cluster order (seeded for reproducibility);
  which cluster is called team 1 may differ between videos.
