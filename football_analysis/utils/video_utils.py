import cv2


def get_video_info(video_path):
    """Return (width, height, fps, frame_count) from the container metadata.

    frame_count can be 0 / -1 for some streams (webcams, partial files);
    callers must handle an unknown count gracefully.
    """
    cap = cv2.VideoCapture(video_path)
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


def frame_reader(video_path, max_frames=None, scale=None, stride=1):
    """Yield (source_index, frame) lazily without loading the whole video.

    stride > 1 analyses every Nth source frame (the caller must divide the
    fps by the stride when computing real-world speeds). max_frames caps the
    number of frames *yielded* (i.e. analysed), not raw frames read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    yielded = 0
    src_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if src_idx % stride == 0:
                if scale is not None and scale != 1.0:
                    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                yield src_idx, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            src_idx += 1
    finally:
        cap.release()


def read_video(video_path, max_frames=None, scale=None, stride=1):
    """Eagerly read frames into a list (kept for backwards compatibility)."""
    return [frame for _, frame in frame_reader(video_path, max_frames, scale, stride)]


def open_video_writer(output_video_path, width, height, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*_fourcc_for_path(output_video_path))
    return cv2.VideoWriter(output_video_path, fourcc, fps, (int(width), int(height)))


def save_video(output_video_frames, output_video_path, fps=24):
    if not output_video_frames:
        raise ValueError("No frames to save")
    h, w = output_video_frames[0].shape[:2]
    out = open_video_writer(output_video_path, w, h, fps)
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def _fourcc_for_path(path):
    ext = str(path).lower().rsplit('.', 1)[-1] if '.' in str(path) else ''
    return {
        'mp4': 'mp4v',
        'mkv': 'X264',
        'avi': 'XVID',
        'mov': 'mp4v',
    }.get(ext, 'XVID')


# Codecs a browser will actually decode. OpenCV writes the pipeline's raw
# output with the "mp4v" fourcc, which is MPEG-4 Part 2 — a perfectly valid
# .mp4 that Chrome, Firefox and Safari all refuse to play. Handing one of
# those to st.video() produces a player that loads and then shows nothing,
# which is indistinguishable from "the video is broken".
BROWSER_CODECS = ("h264", "avc1", "vp8", "vp9", "av1")


def probe_codec(path):
    """Best-effort video codec name for `path`, or None if it cannot be read.

    Uses ffprobe when it is on PATH, and otherwise sniffs the sample-description
    box in the file header, which is enough to tell avc1 from mp4v.
    """
    import shutil
    import subprocess

    if shutil.which("ffprobe"):
        try:
            done = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                check=True, capture_output=True, text=True, timeout=30)
            name = done.stdout.strip().splitlines()
            if name:
                return name[0].strip().lower()
        except Exception:
            pass

    try:
        with open(path, "rb") as fh:
            head = fh.read(1 << 20)
    except OSError:
        return None
    for tag in (b"avc1", b"hvc1", b"hev1", b"vp09", b"av01", b"mp4v"):
        if tag in head:
            return tag.decode()
    return None


def is_browser_playable(path):
    """True when a browser can be expected to decode this file."""
    import os
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    codec = probe_codec(path)
    return codec is not None and codec in BROWSER_CODECS


def convert_to_h264(input_path, output_path, fps=None, overwrite=True,
                    max_long_side=None):
    """Transcode a video to a browser-playable H.264 .mp4.

    Uses the system ffmpeg first (handles odd dimensions, pixel formats and
    container subtleties correctly), falling back to PyAV if ffmpeg is missing.

    `max_long_side` caps the preview's long edge. That matters because
    st.video() hands the whole file to the browser through Streamlit's media
    server, which holds it in memory: a 30-second 1080p annotation is ~22 MB,
    and on a small container that is both slow and risky. The full-resolution
    render is still written separately and offered for download.
    """
    import subprocess
    import shutil

    # Prefer ffmpeg on PATH for robustness.
    if shutil.which("ffmpeg"):
        # yuv420p requires even dimensions, so any scaling — including none —
        # is rounded down to an even number of pixels. Without this an
        # odd-sized source fails the encode outright, and the caller is left
        # falling back to a file the browser cannot play.
        if max_long_side:
            scale = (f"scale='if(gt(iw,ih),min({max_long_side},iw),-2)':"
                     f"'if(gt(iw,ih),-2,min({max_long_side},ih))'")
            vf = f"{scale},scale=trunc(iw/2)*2:trunc(ih/2)*2"
        else:
            vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

        cmd = [
            "ffmpeg", "-y" if overwrite else "-n",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-vf", vf,
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "23",
            "-movflags", "+faststart",
        ]
        if fps:
            cmd += ["-r", str(fps)]
        cmd.append(str(output_path))
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    # Fallback: write H.264 directly with PyAV.
    import av
    import numpy as np

    if fps is None:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        cap.release()

    container_in = av.open(input_path)
    container_out = av.open(output_path, "w")
    stream_out = container_out.add_stream("libx264", rate=float(fps))
    stream_out.pix_fmt = "yuv420p"
    stream_out.options = {"preset": "veryfast", "crf": "23"}

    first = True
    width = height = None
    for frame in container_in.decode(video=0):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        if first:
            # yuv420p requires even dimensions; pad an odd frame by 1px.
            width, height = (w + 1) & ~1, (h + 1) & ~1
            stream_out.width, stream_out.height = width, height
            first = False
        if (w, h) != (width, height):
            img = np.pad(img, ((0, height - h), (0, width - w), (0, 0)), constant_values=0)
        vf = av.VideoFrame.from_ndarray(img, format="bgr24")
        for packet in stream_out.encode(vf):
            container_out.mux(packet)
    for packet in stream_out.encode(None):
        container_out.mux(packet)
    container_out.close()
    container_in.close()
    return output_path