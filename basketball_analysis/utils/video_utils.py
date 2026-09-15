"""
Video I/O for the basketball pipeline.

The functions here are deliberately *streaming*: a decoded 720p frame is
~2.8 MB, so a 10-second clip is ~830 MB and a 30-second 1080p clip is ~5.6 GB.
Holding a whole video in a Python list is what made the hosted app get killed
by the OOM reaper the moment anyone uploaded something. Everything below reads
one frame at a time and hands it straight to the caller.

`read_video` (eager, whole-video-in-RAM) is kept only so the legacy
`main.py` CLI keeps working, and it now refuses to load more than
`MAX_EAGER_FRAMES` frames rather than dying silently.
"""

import os
import warnings

import cv2

# A hard ceiling for the eager reader: 600 frames of 720p is ~1.7 GB, which is
# already past what a 1-vCPU cloud container can spare. The streaming paths
# have no such limit because they never hold more than a frame or two.
MAX_EAGER_FRAMES = 600


def get_video_info(video_path):
    """Return (width, height, fps, frame_count) from the container metadata.

    frame_count is None when the container does not report one (some streams
    and partially written files); callers must handle that.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    fps = fps if fps and fps > 0 else 24.0
    frame_count = frame_count if frame_count and frame_count > 0 else None
    return width, height, fps, frame_count


def scaled_size(width, height, scale):
    """Even-numbered output dimensions for a scale factor (H.264 needs even)."""
    if not scale or scale == 1.0:
        return int(width), int(height)
    w = max(2, int(round(width * scale)))
    h = max(2, int(round(height * scale)))
    return w & ~1, h & ~1


def frame_reader(video_path, max_frames=None, scale=None, stride=1, start=0):
    """Yield (source_index, frame) lazily, at constant memory.

    Args:
        max_frames: stop after this many frames have been *yielded*.
        scale: resize factor applied to every yielded frame.
        stride: only yield every Nth source frame.
        start: skip this many source frames before yielding anything.

    The frame handed to the caller is a fresh buffer each iteration, so it is
    safe to keep — but keeping all of them is exactly the thing this function
    exists to avoid.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    stride = max(1, int(stride))
    yielded = 0
    src_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx >= start and (src_idx - start) % stride == 0:
                if scale is not None and scale != 1.0:
                    w, h = scaled_size(frame.shape[1], frame.shape[0], scale)
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                yield src_idx, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            src_idx += 1
    finally:
        cap.release()


def read_video(video_path, max_frames=None, scale=None, stride=1):
    """Eagerly read frames into a list. Prefer `frame_reader`.

    Kept for the CLI in `main.py`, where the whole-video-in-RAM cost is the
    caller's own machine to spend. It is capped at `MAX_EAGER_FRAMES` by
    default because an uncapped call here is the single biggest memory
    consumer in the project, and it says so out loud rather than quietly
    handing back a truncated video.

    Pass `max_frames=0` to read everything regardless.
    """
    if max_frames == 0:
        limit = None
    elif max_frames is None:
        limit = MAX_EAGER_FRAMES
    else:
        limit = min(max_frames, MAX_EAGER_FRAMES)

    frames = [frame for _, frame in frame_reader(video_path, limit, scale, stride)]

    if limit is not None and len(frames) == limit:
        _, _, _, total = get_video_info(video_path)
        if total and total > limit:
            warnings.warn(
                f"read_video() stopped at {limit} of {total} frames "
                f"(~{limit * (frames[0].nbytes / 1e6):.0f} MB already). Pass "
                f"max_frames=0 to read the whole file, or use frame_reader() "
                f"to stream it at constant memory.",
                stacklevel=2)
    return frames


def fourcc_for_path(path):
    ext = str(path).lower().rsplit('.', 1)[-1] if '.' in str(path) else ''
    return {
        'mp4': 'mp4v',
        'mkv': 'X264',
        'avi': 'XVID',
        'mov': 'mp4v',
    }.get(ext, 'XVID')


def open_video_writer(output_video_path, width, height, fps=24):
    """A writer that frames can be streamed into one at a time."""
    out_dir = os.path.dirname(str(output_video_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_for_path(output_video_path))
    return cv2.VideoWriter(str(output_video_path), fourcc, float(fps),
                           (int(width), int(height)))


def save_video(output_video_frames, output_video_path, fps=24):
    """Write an in-memory frame list. Prefer `open_video_writer` + a loop."""
    if not output_video_frames:
        raise ValueError("No frames to save")
    h, w = output_video_frames[0].shape[:2]
    writer = open_video_writer(output_video_path, w, h, fps)
    try:
        for frame in output_video_frames:
            writer.write(frame)
    finally:
        writer.release()


def batched(iterable, size):
    """Yield lists of at most `size` items from `iterable`.

    Detection runs batch-by-batch so that a batch of YOLO results (each one
    holding a full copy of its input frame in `.orig_img`) is converted to
    plain track dicts and released before the next batch is decoded.
    """
    size = max(1, int(size))
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
