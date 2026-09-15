import numpy as np
import supervision as sv


def _to_sv_keypoints(keypoints):
    """
    Convert whatever the court detector returned into `sv.KeyPoints`.

    Handing ultralytics' own `Keypoints` object straight to supervision only
    works on some version pairs — newer ultralytics dropped the `.visible`
    attribute that older supervision reaches for. Going through the raw xy
    array is stable across both, and is also what
    `CourtKeypointDetector.CourtKeypoints` stores.
    """
    if isinstance(keypoints, sv.KeyPoints):
        return keypoints

    xy = getattr(keypoints, "xy", keypoints)
    if hasattr(xy, "cpu"):                      # torch tensor
        xy = xy.cpu().numpy()
    xy = np.asarray(xy, dtype=np.float32)

    if xy.ndim == 2:                            # (K, 2) -> (1, K, 2)
        xy = xy[None, ...]
    return sv.KeyPoints(xy=xy)


class CourtKeypointDrawer:
    """
    A drawer class responsible for drawing court keypoints on a sequence of frames.

    Attributes:
        keypoint_color (str): Hex color value for the keypoints.
    """
    def __init__(self):
        self.keypoint_color = '#ff2c2c'
        # Built once and reused. These used to be constructed inside draw(),
        # which was fine for a whole-list call but wasteful once frames are
        # streamed through draw_frame() one at a time.
        self._vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=8)
        self._label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1)

    def draw_frame(self, frame, frame_keypoints):
        """
        Draw one frame's court keypoints, in place.

        Args:
            frame (numpy.ndarray): The frame to annotate. Modified in place.
            frame_keypoints: Keypoints for this frame.

        Returns:
            numpy.ndarray: The same frame, annotated.
        """
        keypoints = _to_sv_keypoints(frame_keypoints)
        if keypoints.xy.size == 0:
            return frame
        frame = self._vertex_annotator.annotate(scene=frame, key_points=keypoints)
        frame = self._label_annotator.annotate(scene=frame, key_points=keypoints)
        return frame

    def draw(self, frames, court_keypoints):
        """
        Draws court keypoints on a given list of frames.

        Args:
            frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            court_keypoints (list): A corresponding list of lists where each sub-list contains
                the (x, y) coordinates of court keypoints for that frame.

        Returns:
            list: A list of frames with keypoints drawn on them.
        """
        return [self.draw_frame(frame.copy(), court_keypoints[index])
                for index, frame in enumerate(frames)]
