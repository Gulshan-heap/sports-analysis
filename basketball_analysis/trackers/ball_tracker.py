from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import torch
import sys 
sys.path.append('../')
from utils import read_stub, save_stub, batched


class BallTracker:
    """
    A class that handles basketball detection and tracking using YOLO.

    This class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.

    Detection is consumed batch-by-batch: an ultralytics ``Results`` object
    pins its input frame in ``.orig_img``, so keeping one per frame would cost
    a second full copy of the video. See :class:`PlayerTracker` for the same
    note.
    """
    def __init__(self, model_path, device=None, imgsz=None,
                 task=None, batch_size=20):
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

    def _predict(self, frames):
        """Run YOLO over one batch of frames."""
        return self.model.predict(
            frames, conf=0.5, device=self.device, verbose=False,
            **({"imgsz": self.imgsz} if self.imgsz else {}))

    def detect_frames(self, frames):
        """
        Detect the ball in a sequence of frames using batch processing.

        Retained for backwards compatibility; it materialises every ``Results``
        object at once. Prefer :meth:`get_object_tracks`, which streams.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        detections = []
        for chunk in batched(frames, self.batch_size):
            detections += self._predict(chunk)
        return detections

    def _track_from_detection(self, detection):
        """Reduce one YOLO result to the single highest-confidence ball box."""
        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}

        detection_supervision = sv.Detections.from_ultralytics(detection)

        chosen_bbox = None
        max_confidence = 0

        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            confidence = frame_detection[2]

            if cls_id == cls_names_inv['Ball']:
                if max_confidence < confidence:
                    chosen_bbox = bbox
                    max_confidence = confidence

        return {1: {"bbox": chosen_bbox}} if chosen_bbox is not None else {}

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None,
                          expected_count=None, on_progress=None):
        """
        Get ball tracking results for a sequence of frames with optional caching.

        Args:
            frames: Iterable of video frames (a generator is fine — pass
                ``expected_count`` with it so the cache stays length-checked).
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.
            expected_count (int, optional): Number of frames the iterable yields.
            on_progress (callable, optional): Called with (done, expected).

        Returns:
            list: List of dictionaries containing ball tracking information for each frame.
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
                tracks.append(self._track_from_detection(detection))
            if on_progress is not None:
                on_progress(len(tracks), expected_count)

        save_stub(stub_path, tracks)

        return tracks

    @staticmethod
    def remove_wrong_detections(ball_positions):
        """
        Filter out incorrect ball detections based on maximum allowed movement distance.

        Args:
            ball_positions (list): List of detected ball positions across frames.

        Returns:
            list: Filtered ball positions with incorrect detections removed.
        """
        
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                # First valid detection
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    @staticmethod
    def interpolate_ball_positions(ball_positions):
        """
        Interpolate missing ball positions to create smooth tracking results.

        Args:
            ball_positions (list): List of ball positions with potential gaps.

        Returns:
            list: List of ball positions with interpolated values filling the gaps.
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions