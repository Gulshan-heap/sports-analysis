import pickle
import cv2
import numpy as np
import os
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5
        h, w = frame.shape[:2]
        self.frame_size = (w, h)

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        # Track vertical strips near the left / right edges, proportional to frame width.
        left_strip = max(1, int(w * 0.02))
        right_start = int(w * 0.88)
        mask_features[:, 0:left_strip] = 1
        mask_features[:, right_start:] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )

        # Streaming state (used by process_frame)
        self._prev_gray = None
        self._prev_features = None

    def reset(self):
        """Forget streaming state so a new video can be processed."""
        self._prev_gray = None
        self._prev_features = None

    def process_frame(self, frame):
        """Estimate camera movement for ONE frame relative to the previous one.

        Returns [dx, dy] in pixels (the first frame always returns [0, 0]).
        Keeps only per-frame grayscale + feature state, so long videos run in
        constant memory.
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = frame_gray
            self._prev_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            return [0, 0]

        movement = self._flow(self._prev_gray, frame_gray, self._prev_features)
        # _flow re-detects features on the new frame when it accepts a movement
        self._prev_gray = frame_gray
        return movement

    def _flow(self, old_gray, frame_gray, old_features):
        """Core optical-flow step; returns [dx, dy] for this transition."""
        if old_features is None or len(old_features) == 0:
            self._prev_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            return [0, 0]

        new_features, _, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, old_features, None, **self.lk_params)
        if new_features is None or len(new_features) == 0:
            self._prev_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            return [0, 0]

        # Vectorised equivalent of the per-feature distance loop.
        new_pts = new_features.reshape(-1, 2)
        old_pts = old_features.reshape(-1, 2)
        distances = np.linalg.norm(new_pts - old_pts, axis=1)
        idx = int(np.argmax(distances))
        max_distance = float(distances[idx])
        camera_movement_x = float(old_pts[idx, 0] - new_pts[idx, 0])
        camera_movement_y = float(old_pts[idx, 1] - new_pts[idx, 1])

        if max_distance > self.minimum_distance:
            self._prev_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            return [camera_movement_x, camera_movement_y]

        # Small movement: keep tracking the same features (original behaviour).
        return [0, 0]

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    


    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)
        if not frames:
            return camera_movement

        self.reset()
        self.process_frame(frames[0])  # primes state; movement is [0, 0]
        for frame_num in range(1, len(frames)):
            camera_movement[frame_num] = self.process_frame(frames[frame_num])
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def annotate_frame(self, frame, camera_movement):
        """Draw the camera movement panel for ONE frame (scale aware)."""
        h, w = frame.shape[:2]
        s = max(0.6, min(1.6, min(h, w) / 720.0))

        panel_w, panel_h = int(w * 0.25), int(h * 0.09)

        overlay = frame.copy()
        cv2.rectangle(overlay,(0,0),(panel_w,panel_h),(255,255,255),-1)
        alpha =0.6
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

        x_movement, y_movement = camera_movement
        line_y = max(int(28 * s), int(panel_h * 0.42))
        font_scale = max(0.5, min(0.9, s))
        thickness = max(1, int(2 * s))
        cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}",
                    (int(10 * s), line_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)
        cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}",
                    (int(10 * s), int(line_y + panel_h*0.35)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)

        return frame

    def draw_camera_movement(self,frames, camera_movement_per_frame):
        return [
            self.annotate_frame(frame, camera_movement_per_frame[frame_num])
            for frame_num, frame in enumerate(frames)
        ]