"""
Memory budgeting for the hosted app.

Why this exists
---------------
The pipelines here decode video into raw BGR frames. One 720p frame is
1280*720*3 = 2.76 MB; one 1080p frame is 6.2 MB. A 30-second 1080p clip is
therefore 5.6 GB of pixels if it is held in a list — and the hosted container
(Streamlit Community Cloud) has under 3 GB *including* the ~700 MB that torch,
ultralytics and OpenCV occupy just by being imported. That is the whole reason
the app came back up after a reboot and then died again on the first upload:
nothing was wrong with the boot, the first video simply exhausted the container.

The pipelines now stream, so the frame buffer is no longer the binding
constraint. What is left is bounding the *inputs* — resolution and duration —
so that a single upload cannot blow through the container's disk or its
remaining RAM headroom. The helpers below report what is actually available
and turn that into concrete caps the UI can apply and explain.
"""

import os

# What the interpreter costs before a single frame is decoded: torch +
# ultralytics + OpenCV + streamlit, measured on the deployed container.
BASELINE_RSS_MB = 750

# Leave this much of the container untouched. Python's allocator does not
# return everything promptly, and the OOM killer is not negotiable.
SAFETY_MARGIN_MB = 400

# Assumed container size when nothing can be detected (Streamlit Community
# Cloud's documented limit).
DEFAULT_LIMIT_MB = 2700


def _cgroup_limit_mb():
    """Container memory limit from cgroup v2, then v1. None if not in one."""
    candidates = [
        "/sys/fs/cgroup/memory.max",                     # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",   # cgroup v1
    ]
    for path in candidates:
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except (OSError, ValueError):
            continue
        if raw in ("max", ""):
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        # An unset v1 limit shows up as a huge sentinel, not as "max".
        if 0 < value < (1 << 60):
            return value / (1024 * 1024)
    return None


def total_mb():
    """Total memory this process may use, in MB."""
    limit = _cgroup_limit_mb()
    if limit:
        return limit
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 * 1024)
    except Exception:
        pass
    try:                                    # POSIX fallback
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                / (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return DEFAULT_LIMIT_MB


def process_rss_mb():
    """Resident set size of this process in MB, or None if unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def headroom_mb():
    """Roughly how much more this process can allocate before it is killed."""
    used = process_rss_mb()
    if used is None:
        used = BASELINE_RSS_MB
    return max(0.0, total_mb() - used - SAFETY_MARGIN_MB)


def is_constrained():
    """True on a small container, where the input caps genuinely matter."""
    return total_mb() < 4096


def frame_mb(width, height):
    """Megabytes one decoded BGR frame occupies."""
    return width * height * 3 / (1024 * 1024)


def fit_scale(width, height, max_long_side):
    """Scale factor that brings the long side down to `max_long_side`.

    Returns 1.0 when the video is already small enough — upscaling a video to
    fill a cap would only make everything slower.
    """
    long_side = max(width, height)
    if not max_long_side or long_side <= max_long_side:
        return 1.0
    return max_long_side / float(long_side)


def describe():
    """One line for the sidebar: what the container has and what is left."""
    total = total_mb()
    used = process_rss_mb()
    if used is None:
        return f"{total / 1024:.1f} GB total"
    return f"{used:.0f} MB used of {total / 1024:.1f} GB"
