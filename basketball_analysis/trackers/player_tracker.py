from ultralytics import YOLO
import supervision as sv
import torch
import sys # helps in going back to directory
sys.path.append('../')
from utils import read_stub, save_stub, batched

class PlayerTracker:
    """
    A class that handles player detection and tracking using YOLO and ByteTrack.

    This class combines YOLO object detection with ByteTrack tracking to maintain consistent
    player identities across frames while processing detections in batches.

    Memory note: every ultralytics ``Results`` object keeps its input image
    alive in ``.orig_img``. Accumulating one per frame therefore costs a second
    full copy of the video (~2.8 MB per 720p frame). Detection here consumes
    one batch at a time and drops the results as soon as they have been reduced
    to bbox dictionaries, so peak usage is bounded by ``batch_size``, not by
    the length of the video.
    """
    def __init__(self, model_path, device=None, imgsz=None,
                 task=None, batch_size=20):
        """
        Initialize the PlayerTracker with YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
            device (str, optional): 'cuda' or 'cpu'. Auto-detected if not given.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path, task=task) if task else YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            # Non-PyTorch backend (e.g. an OpenVINO IR directory) — there is no
            # tensor to move; the device is selected per predict() call instead.
            pass
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.tracker = sv.ByteTrack()

    def _predict(self, frames):
        """Run YOLO over one batch of frames."""
        return self.model.predict(
            frames, conf=0.5, device=self.device, verbose=False,
            **({"imgsz": self.imgsz} if self.imgsz else {}))

    def detect_frames(self, frames):
        """
        Detect players in a sequence of frames using batch processing.

        Retained for backwards compatibility. It materialises every ``Results``
        object at once — prefer :meth:`get_object_tracks`, which streams.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        detections = []
        for chunk in batched(frames, self.batch_size):
            detections += self._predict(chunk)
        return detections

    def _tracks_from_detection(self, detection):
        """Reduce one YOLO result to ``{track_id: {'bbox': [...]}}``."""
        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}

        # Convert to supervision Detection format
        detection_supervision = sv.Detections.from_ultralytics(detection)

        # Track Objects
        detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

        frame_tracks = {}
        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]

            if cls_id == cls_names_inv['Player']:
                frame_tracks[track_id] = {"bbox": bbox}
        return frame_tracks

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None,
                          expected_count=None, on_progress=None):
        """
        Get player tracking results for a sequence of frames with optional caching.

        Args:
            frames: Iterable of video frames. A generator is fine and is the
                memory-cheap option — pass ``expected_count`` alongside it so
                the cache can still be length-checked.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.
            expected_count (int, optional): Number of frames the iterable will
                yield. Defaults to ``len(frames)`` when that is available.
            on_progress (callable, optional): Called with (done, expected)
                after each batch.

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        """
        if expected_count is None:
            expected_count = len(frames) if hasattr(frames, "__len__") else None

        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if expected_count is None or len(tracks) == expected_count:
                return tracks

        tracks = []
        for chunk in batched(frames, self.batch_size):
            for detection in self._predict(chunk):
                tracks.append(self._tracks_from_detection(detection))
            # `chunk` and this batch's results go out of scope here: the frames
            # they pinned are reclaimed before the next batch is decoded.
            if on_progress is not None:
                on_progress(len(tracks), expected_count)

        save_stub(stub_path, tracks)
        return tracks
