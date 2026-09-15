"""
Slimming YOLO checkpoints before they are loaded.

The problem
-----------
The court detector shipped with this project is a *training* checkpoint, not
an inference one. On disk:

    player_detector.pt          165 MB
    ball_detector_model.pt      165 MB
    court_keypoint_detector.pt  398 MB   <- carries optimizer state + EMA

The court file is that size because it still holds the Adam optimizer state
(two moment tensors per parameter, for a 69.5M-parameter pose model), the EMA
copy of the weights and the full training history. Inference reads none of it,
and all of it is paged in when ultralytics loads the file.

Measured peak RSS for the three models (86M-parameter detectors, so the
weights alone are not small):

    all three held at once, as shipped      1404 MB
    all three held at once, stripped        1113 MB
    one at a time, as shipped                704 MB
    one at a time, stripped                  464 MB

On a container with under 3 GB total, the first of those is most of the budget
spent before a single video frame is decoded. Combined with holding one
detector at a time (see basketball_analysis/ui.py), stripping turns 1.4 GB
into 464 MB.

The fix
-------
`ultralytics.utils.torch_utils.strip_optimizer` drops the optimizer, the EMA
and the training metadata and re-saves the weights at half precision. The
result is loaded back as a normal fp32 model on CPU, so inference is unchanged
— the same tensors, without the training baggage.

Stripping happens once per file into a cache outside the repo, keyed by the
source file's size and mtime, so a redeploy pays for it once and every later
run loads the slim copy directly. Checkpoints that turn out to be lean already
(the player and ball detectors here) are marked so they are not re-tested on
every boot.
"""

import os
import shutil
import tempfile
from pathlib import Path

CACHE_DIR = os.path.join(tempfile.gettempdir(), "sports_analysis_slim_weights")

# Below this there is nothing worth stripping: a checkpoint this small is
# already inference-only, and rewriting it just burns deploy time.
MIN_STRIP_MB = 60

# Marks a checkpoint we have already tested and found nothing to strip from.
NOOP_SUFFIX = ".lean"


def _cache_path(src):
    """Cache name that changes whenever the source file does."""
    stat = os.stat(src)
    stem = Path(src).stem
    return os.path.join(CACHE_DIR,
                        f"{stem}-{stat.st_size}-{int(stat.st_mtime)}.pt")


def size_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def worth_stripping(path):
    return bool(path) and os.path.exists(path) and size_mb(path) >= MIN_STRIP_MB


def strip(path, force=False):
    """Return a path to an inference-only copy of `path`.

    Returns the original path unchanged when the checkpoint is already small,
    or when stripping fails for any reason — a fat model that loads beats a
    slim one that does not exist.
    """
    if not worth_stripping(path):
        return path

    os.makedirs(CACHE_DIR, exist_ok=True)
    dest = _cache_path(path)
    if os.path.exists(dest) and not force:
        return dest
    # Previously found to be lean already — skip the (slow) re-test.
    if os.path.exists(dest + NOOP_SUFFIX) and not force:
        return path

    try:
        from ultralytics.utils.torch_utils import strip_optimizer
    except Exception:
        return path

    tmp = dest + ".part"
    try:
        shutil.copyfile(path, tmp)
        # strip_optimizer rewrites the file it is given, in place.
        strip_optimizer(tmp)
        os.replace(tmp, dest)
    except Exception:
        for leftover in (tmp,):
            try:
                os.remove(leftover)
            except OSError:
                pass
        return path

    # Some checkpoints are already inference-only — the player and ball
    # detectors here are. Stripping them is a no-op, so drop the copy rather
    # than leaving a second 165 MB file on a disk that has to hold the
    # originals, the uploads and the rendered videos too.
    if size_mb(dest) >= size_mb(path):
        try:
            os.remove(dest)
            with open(dest + NOOP_SUFFIX, "w") as marker:
                marker.write("already inference-only")
        except OSError:
            pass
        return path
    return dest


def strip_all(paths, on_progress=None):
    """Strip several checkpoints, reporting progress by role.

    `paths` is a {role: path} mapping; the return value has the same shape.
    """
    out = {}
    for role, path in paths.items():
        if on_progress is not None and worth_stripping(path) \
                and not os.path.exists(_cache_path(path)):
            on_progress(role, size_mb(path))
        out[role] = strip(path)
    return out


def savings(paths):
    """(original_mb, slim_mb) across a {role: path} mapping, for reporting."""
    original = slim = 0.0
    for path in paths.values():
        if not path or not os.path.exists(path):
            continue
        original += size_mb(path)
        cached = _cache_path(path)
        slim += size_mb(cached) if os.path.exists(cached) else size_mb(path)
    return original, slim
