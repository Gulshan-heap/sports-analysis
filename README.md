# 🏅 Sports Analysis

Two computer-vision match-analysis pipelines — **Basketball** and **Football** —
behind one Streamlit app with a sport switcher.

```
streamlit run app.py
```

![switch sports from the sidebar](https://img.shields.io/badge/sports-basketball%20%7C%20football-blue)

---

## Layout

```
sports_analysis/
├── app.py                    ← unified entry point (the router)
├── requirements.txt          ← merged dependencies for both sports
├── packages.txt              ← apt packages for Streamlit Cloud deploys
│
├── sports_core/              ← shared infrastructure
│   ├── registry.py           ← the list of sports + their metadata
│   ├── isolation.py          ← keeps the two package trees from colliding
│   └── theme.py              ← one dark design system, re-skinned per sport
│
├── basketball_analysis/      ← unchanged pipeline + ui.py page module
│   ├── ui.py                 ← render(sport) — the page
│   ├── app.py                ← standalone entry point (optional)
│   ├── trackers/  utils/  team_assigner/  drawers/  …
│   └── main.py               ← original CLI, untouched
│
└── football_analysis/        ← unchanged pipeline + ui.py page module
    ├── ui.py                 ← render(sport) — the page
    ├── app.py                ← standalone entry point (optional)
    ├── trackers/  utils/  team_assigner/  view_transformer/  …
    └── main.py               ← original CLI, untouched
```

Each sport's project folder is **self-contained and unmodified** — its
pipeline packages, models, stubs and `main.py` CLI all work exactly as before.
The only new file inside each is `ui.py`.

---

## Running

| Command | What you get |
| --- | --- |
| `streamlit run app.py` | Both sports, switchable from the sidebar |
| `streamlit run app.py` then `?sport=football` | Deep-link straight to a sport |
| `streamlit run basketball_analysis/app.py` | Basketball only |
| `streamlit run football_analysis/app.py` | Football only |
| `python basketball_analysis/main.py <video>` | Basketball CLI (unchanged) |
| `python football_analysis/main.py --input <video> --output <out>` | Football CLI (unchanged) |

### Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

`ffmpeg` on your PATH is optional but recommended — both sports use it to
re-encode output to browser-playable H.264.

---

## How the two projects coexist

This is the one genuinely tricky part, so it's worth spelling out.

Both projects independently define **top-level packages with the same names**:

| Package | basketball | football |
| --- | :---: | :---: |
| `utils` | ✅ | ✅ |
| `trackers` | ✅ | ✅ |
| `team_assigner` | ✅ | ✅ |

In one Python process, whichever imports first wins `sys.modules` — so a naive
merge would silently feed football's `Tracker` to the basketball pipeline.
Renaming the packages would mean rewriting every import in both codebases.

Instead, `sports_core/isolation.py` sandboxes them. Streamlit re-runs the whole
script on every interaction, which gives a clean hook. Before rendering a sport,
`activate()`:

1. **evicts** every cached module whose file lives under a *different* sport's
   folder, and
2. **rewrites `sys.path`** so only the active sport's folder is importable.

Site-packages (torch, ultralytics, cv2, …) live outside the sport folders and
are never touched, so switching sports costs milliseconds rather than a reload.
Each sport's `ui.py` is then imported *by file path* under a unique module name
(`_sports_ui_basketball`), so the page modules can't collide either.

```python
isolation.activate(sport, ALL_ROOTS)   # sandbox
page = isolation.load_ui(sport)        # import ui.py by path
page.render(sport)                     # draw
```

---

## Performance notes (basketball, CPU)

Measured on a 2-core i3-1115G4, 720p footage, `imgsz=640`:

| Model | Params | CPU cost |
| --- | ---: | ---: |
| `player_detector.pt` | 86 M | ~0.9 s/frame |
| `ball_detector_model.pt` | 86 M | ~1.0 s/frame |
| `court_keypoint_detector.pt` | 70 M | ~1.7–2.3 s/frame |

That is **~3.6 s of compute per frame**, so a 10-second 30 fps clip is about
**18 minutes** of detection if every model runs on every frame. Model loading
is ~1 s and the Google Drive fetch happens once and is cached — neither is the
bottleneck. The cost is three large YOLO models on a CPU.

The **Performance** panel in the sidebar exposes the levers:

| Lever | Default | Effect |
| --- | --- | --- |
| Court key-points every Nth frame | **30** | ~2x overall. The court is static, so this is nearly free accuracy-wise — the single best setting. |
| Analyse first N seconds | off | Linear. The quickest way to get *a* result out of a long clip. |
| Detect every Nth frame | **1** (off) | ~Nx faster, but ByteTrack splits players into more IDs (13 → 18 on a 117-frame clip at stride 3), which skews per-player distance. Opt in for a quick look, not for final numbers. |
| Detection resolution | 640 | Quadratic. 416 is ~2.4x faster but misses the ball more often. |

Detections are cached per *(video content, settings)* under `stubs/<hash>_s…/`,
so re-running the same clip is instant. The cache used to be keyed by filename
alone, which meant a different video with the same frame count would silently
reuse the previous video's tracks.

There is **no NVIDIA GPU** on this machine, so CUDA is not an option. The
remaining hardware-level lever is an **OpenVINO export** (`model.export(
format="openvino")`, then load the exported directory) — it targets Intel CPUs
and iGPUs and is typically 2–3x faster than PyTorch CPU for the same weights.

---

## Adding a third sport

1. Drop the project folder in next to the others.
2. Add a `ui.py` to it exposing `render(sport)`.
3. Append one `Sport(...)` entry to `sports_core/registry.py`.

Nothing else changes — the switcher, theming and isolation pick it up
automatically.

---

## What each sport does

### 🏀 Basketball
Runs **in-process**. YOLO player + ball detection with ByteTrack, Fashion-CLIP
(or automatic colour-clustering) team assignment, court-keypoint homography to
a top-down tactical view, per-player speed & distance, pass/interception
detection, plus a frame explorer and CSV/JSON export.

Model weights are fetched automatically from Google Drive on first run and
cached; local `models/*.pt` files take priority if present.

### ⚽ Football
Runs the pipeline as a **subprocess** (`main.py`), streaming its logs live into
the page. YOLO + ByteTrack tracking, KMeans jersey-colour team assignment,
nearest-player ball possession, optical-flow camera-movement compensation,
perspective transform from four pitch corners, speed & distance, optional
per-player heatmaps, and per-frame CSV export.

Point the **Detection** panel at your trained weights (defaults to
`football_analysis/models/best.pt`).
