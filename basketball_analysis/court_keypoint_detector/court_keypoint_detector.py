import numpy as np
from ultralytics import YOLO
import torch
import sys
sys.path.append('../')
from utils import read_stub, save_stub, batched


class CourtKeypoints:
    """A plain-numpy stand-in for ultralytics' ``Keypoints``.

    The detector used to hand the ultralytics object straight through. That
    object keeps the whole torch result graph alive, is expensive to pickle
    into the stub cache, and gets ``deepcopy``-ed for every frame by
    ``TacticalViewConverter.validate_keypoints`` — three good reasons to
    detach it to numpy at the point of detection instead.

    Exposes the ``.xy`` / ``.xyn`` attributes (shape ``(1, K, 2)``) that the
    tactical-view converter and the keypoint drawer read.
    """

    __slots__ = ("xy", "xyn")

    def __init__(self, xy, xyn=None):
        self.xy = np.asarray(xy, dtype=np.float32)
        if self.xy.ndim == 2:                       # (K, 2) -> (1, K, 2)
            self.xy = self.xy[None, ...]
        if xyn is None:
            xyn = np.zeros_like(self.xy)
        self.xyn = np.asarray(xyn, dtype=np.float32)
        if self.xyn.ndim == 2:
            self.xyn = self.xyn[None, ...]

    @classmethod
    def from_ultralytics(cls, keypoints):
        """Detach an ultralytics ``Keypoints`` (or an empty result) to numpy."""
        if keypoints is None:
            return cls(np.zeros((1, 0, 2), dtype=np.float32))

        def _np(attr):
            value = getattr(keypoints, attr, None)
            if value is None:
                return None
            if hasattr(value, "cpu"):               # torch tensor
                value = value.cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        xy = _np("xy")
        if xy is None or xy.size == 0:
            return cls(np.zeros((1, 0, 2), dtype=np.float32))
        return cls(xy, _np("xyn"))

    def __deepcopy__(self, memo):
        return CourtKeypoints(self.xy.copy(), self.xyn.copy())


class CourtKeypointDetector:
    """
    The CourtKeypointDetector class uses a YOLO model to detect court keypoints in image frames.
    It also provides functionality to draw these detected keypoints on the frames.

    Detection is consumed one batch at a time and reduced to :class:`CourtKeypoints`
    immediately, so the YOLO results (each pinning a full copy of its input
    frame in ``.orig_img``) never accumulate across the video.
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
        return self.model.predict(
            frames, conf=0.5, device=self.device, verbose=False,
            **({"imgsz": self.imgsz} if self.imgsz else {}))

    def get_court_keypoints(self, frames, read_from_stub=False, stub_path=None,
                            expected_count=None, on_progress=None):
        """
        Detect court keypoints for a sequence of frames using the YOLO model. If requested,
        attempts to read previously detected keypoints from a stub file before running the model.

        Args:
            frames: Iterable of frames (a generator is fine — pass
                ``expected_count`` with it so the cache stays length-checked).
            read_from_stub (bool, optional): Read keypoints from a stub file
                instead of running the detection model. Defaults to False.
            stub_path (str, optional): The file path for the stub file.
            expected_count (int, optional): Number of frames the iterable yields.
            on_progress (callable, optional): Called with (done, expected).

        Returns:
            list: A :class:`CourtKeypoints` per input frame.
        """
        if expected_count is None:
            expected_count = len(frames) if hasattr(frames, "__len__") else None

        court_keypoints = read_stub(read_from_stub, stub_path)
        if court_keypoints is not None:
            if expected_count is None or len(court_keypoints) == expected_count:
                return court_keypoints

        court_keypoints = []
        for chunk in batched(frames, self.batch_size):
            for detection in self._predict(chunk):
                court_keypoints.append(
                    CourtKeypoints.from_ultralytics(detection.keypoints))
            if on_progress is not None:
                on_progress(len(court_keypoints), expected_count)

        save_stub(stub_path, court_keypoints)

        return court_keypoints
