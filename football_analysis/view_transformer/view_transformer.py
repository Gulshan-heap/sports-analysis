import numpy as np
import cv2


class ViewTransformer:
    def __init__(self, pixel_vertices=None, region_length=None, region_width=None):
        """
        Maps pixels inside a calibrated region of the frame onto real-world (x, y)
        meters via a homography (perspective transform).

        region_length / region_width : the real-world meters spanned by the
            quadrilateral you calibrated on the frame (NOT necessarily the full
            105x68 pitch - a single camera only sees part of the pitch, so calibrate
            only the visible region and give THOSE region's meters here).

        A per-video calibration can also be supplied from JSON via from_calibration().
        """
        if pixel_vertices is None:
            # Fallback: the original demo video's calibration strip.
            # Prefer providing your own via --calib / pick_corners.
            pixel_vertices = [[110, 1035], [265, 275], [910, 260], [1640, 915]]
            region_length = 23.32 if region_length is None else region_length
            region_width = 68 if region_width is None else region_width

        if region_length is None or region_width is None:
            raise ValueError("region_length and region_width (real-world meters of the "
                             "calibrated region) are required")

        self.region_length = region_length
        self.region_width = region_width
        self.pixel_vertices = np.array(pixel_vertices, dtype=np.float32)

        self.target_vertices = np.array([
            [0, region_width],
            [0, 0],
            [region_length, 0],
            [region_length, region_width]
        ], dtype=np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    @classmethod
    def from_calibration(cls, calibration):
        """Build from a per-video dict, e.g. loaded from a calibration JSON."""
        return cls(
            pixel_vertices=calibration.get("pixel_vertices"),
            region_length=calibration.get("region_length"),
            region_width=calibration.get("region_width"),
        )

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get('position_adjusted')
                    if position is None:
                        # No adjusted pixel position for this track (e.g. a ball
                        # entry rebuilt by interpolation) -> skip, leave as None.
                        tracks[object_name][frame_num][track_id]['position_transformed'] = None
                        continue
                    transformed_position = self.transform_point(position)
                    if transformed_position is not None:
                        transformed_position = transformed_position.squeeze().tolist()
                    tracks[object_name][frame_num][track_id]['position_transformed'] = transformed_position