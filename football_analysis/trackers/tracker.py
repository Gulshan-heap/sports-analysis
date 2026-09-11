from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path, class_map=None, conf=0.1, imgsz=640,
                 device=None, half=None, batch_size=20, verbose=False,
                 detect_stride=1):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        # Run YOLO every Nth frame; skipped frames reuse the last known boxes.
        self.detect_stride = max(1, int(detect_stride))
        # half precision is only valid on CUDA
        self.half = half if half is not None else (
            device is not None and str(device).startswith("cuda"))
        # Map each analysis role -> YOLO class name. Lets you swap in a model
        # trained with different class names without touching the pipeline.
        self.class_map = class_map or {
            "player": "player",
            "referee": "referee",
            "ball": "ball",
            "goalkeeper": "goalkeeper",  # merged into "player" below
        }
        # Streaming state: last known boxes per role, reused on skipped frames
        # (detect-stride) so every analysed frame still has boxes to draw.
        self._last_boxes = {"players": {}, "referees": {}, "ball": {}}
        # Drawing scale: 1.0 for a 720p-ish frame, grows with resolution so
        # markers/labels stay readable (and don't dwarf small videos).
        self._draw_scale = 1.0
        # O(1) running possession counters for draw_team_ball_control
        self._pbc_counts = [0, 0]
        self._pbc_next = 0

    # --------------------------- streaming API --------------------------- #
    def set_frame_size(self, frame):
        """Record frame size once so annotations scale with the video."""
        h, w = frame.shape[:2]
        self._draw_scale = max(0.6, min(1.6, min(h, w) / 720.0))

    def _predict(self, frame):
        kwargs = dict(conf=self.conf, imgsz=self.imgsz, device=self.device,
                      verbose=self.verbose)
        # Only pass half when actually enabled (ultralytics warns otherwise).
        if self.half:
            kwargs["half"] = True
        return self.model.predict(frame, **kwargs)[0]

    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        # Collect the ball bbox from whatever track id it has in each frame.
        ball_bboxes = []
        for frame in ball_positions:
            bbox = next(iter(frame.values()), {}).get('bbox', []) if frame else []
            ball_bboxes.append(bbox)

        df_ball_positions = pd.DataFrame(ball_bboxes, columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # Re-key under a stable id (1) for the rest of the pipeline.
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def _detection_to_track_entry(self, detection):
        """Convert one YOLO Result into (players, referees, ball) dicts."""
        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}

        # Covert to supervision Detection format
        detection_supervision = sv.Detections.from_ultralytics(detection)

        player_id = cls_names_inv.get(self.class_map["player"])
        referee_id = cls_names_inv.get(self.class_map["referee"])
        ball_id = cls_names_inv.get(self.class_map["ball"])
        goalkeeper_id = cls_names_inv.get(self.class_map["goalkeeper"])

        # Convert GoalKeeper to player object (if the model has that class)
        if goalkeeper_id is not None and player_id is not None:
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_id == goalkeeper_id:
                    detection_supervision.class_id[object_ind] = player_id

        # Track Objects
        detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

        players, referees, ball = {}, {}, {}
        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]

            if player_id is not None and cls_id == player_id:
                players[track_id] = {"bbox": bbox}
            if referee_id is not None and cls_id == referee_id:
                referees[track_id] = {"bbox": bbox}
            if ball_id is not None and cls_id == ball_id:
                ball[track_id] = {"bbox": bbox}
        return players, referees, ball

    def step(self, frame, run_detection=True):
        """Process ONE frame -> {"players":..., "referees":..., "ball":...}.

        run_detection=False carries the last known boxes forward (used with a
        detection stride > 1 to skip expensive YOLO frames while keeping the
        output complete).
        """
        if run_detection:
            players, referees, ball = self._detection_to_track_entry(self._predict(frame))
            # Only overwrite a role when the model actually saw something, so a
            # single empty frame doesn't wipe tracked objects on skipped frames.
            if players:
                self._last_boxes["players"] = players
            if referees:
                self._last_boxes["referees"] = referees
            if ball:
                self._last_boxes["ball"] = ball
        else:
            players = dict(self._last_boxes["players"])
            referees = dict(self._last_boxes["referees"])
            ball = dict(self._last_boxes["ball"])
        return {"players": players, "referees": referees, "ball": ball}

    def detect_frames(self, frames, conf=None, imgsz=None, device=None):
        batch_size = self.batch_size
        conf = self.conf if conf is None else conf
        imgsz = self.imgsz if imgsz is None else imgsz
        device = self.device if device is None else device
        detections = []
        for i in range(0, len(frames), batch_size):
            kwargs = dict(conf=conf, imgsz=imgsz, device=device,
                          verbose=self.verbose)
            if self.half:
                kwargs["half"] = True
            detections_batch = self.model.predict(
                frames[i:i + batch_size], **kwargs)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        if frames:
            self.set_frame_size(frames[0])

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        for frame_num, frame in enumerate(frames):
            # Reuse the last known boxes on skipped frames (detect stride).
            entry = self.step(
                frame, run_detection=(frame_num % self.detect_stride == 0))
            tracks["players"].append(entry["players"])
            tracks["referees"].append(entry["referees"])
            tracks["ball"].append(entry["ball"])

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        s = self._draw_scale
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=max(1, int(2 * s)),
            lineType=cv2.LINE_4
        )

        rectangle_width = int(40 * s)
        rectangle_height = int(20 * s)
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + int(15 * s)
        y2_rect = (y2 + rectangle_height // 2) + int(15 * s)

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + int(12 * s)
            if track_id > 99:
                x1_text -= int(10 * s)

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15 * s)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * s,
                (0, 0, 0),
                max(1, int(2 * s))
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        s = self._draw_scale
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - int(10 * s), y - int(20 * s)],
            [x + int(10 * s), y - int(20 * s)],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), max(1, int(2 * s)))

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """Draw the possession panel using O(1) running counters.

        Frames are expected to be drawn in order (both the streaming loop and
        the batch wrapper do this); out-of-order calls fall back to a recompute.
        """
        tbc = np.asarray(team_ball_control)
        if frame_num == self._pbc_next:
            if frame_num < len(tbc):
                team = int(tbc[frame_num])
                if team in (1, 2):
                    self._pbc_counts[team - 1] += 1
            self._pbc_next = frame_num + 1
        else:
            prefix = tbc[:frame_num + 1]
            counts = np.bincount(prefix, minlength=3)
            self._pbc_counts = [int(counts[1]), int(counts[2])]
            self._pbc_next = frame_num + 1

        counts = {1: self._pbc_counts[0], 2: self._pbc_counts[1]}
        return self.draw_team_ball_control_from_counts(frame, counts)

    def draw_team_ball_control_from_counts(self, frame, counts, panel_lines=None):
        """Draw the possession panel from pre-computed running counts {team: frames}."""
        s = self._draw_scale
        h, w = frame.shape[:2]

        # Semi-transparent panel sized proportionally to the frame.
        panel_left = int(w * 0.66)
        panel_top = h - int(h * 0.16)
        panel_right = w - 10
        panel_bottom = h - int(h * 0.03)

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_left, panel_top), (panel_right, panel_bottom),
                      (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_1_frames = counts.get(1, 0)
        team_2_frames = counts.get(2, 0)
        total = team_1_frames + team_2_frames
        if total > 0:
            team_1 = team_1_frames / total
            team_2 = team_2_frames / total
        else:
            team_1 = team_2 = 0.0

        text_x = panel_left + int(30 * s)
        font_scale = max(0.5, min(1.0, s))
        thickness = max(1, int(2 * s))
        line_h = int(50 * s)
        y = panel_top + line_h
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",
                    (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        y += line_h
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",
                    (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        for line in (panel_lines or []):
            y += line_h
            cv2.putText(frame, line, (text_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        return frame

    def annotate_frame(self, frame, players, referees, ball,
                       control_counts=None, panel_lines=None):
        """Draw all annotations for ONE frame in place (streaming path).

        control_counts: running {team: frame_count} dict of possession so far.
        panel_lines: extra text lines appended under the possession panel.
        """
        # Draw Players
        for track_id, player in players.items():
            color = player.get("team_color", (0, 0, 255))
            frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
            if player.get('has_ball', False):
                frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

        # Draw Referees
        for _, referee in referees.items():
            frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

        # Draw ball
        for _, ball_obj in ball.items():
            frame = self.draw_traingle(frame, ball_obj["bbox"], (0, 255, 0))

        # Draw Team Ball Control (panel)
        if control_counts is not None or panel_lines:
            frame = self.draw_team_ball_control_from_counts(
                frame, control_counts or {}, panel_lines=panel_lines)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """Batch wrapper (backwards compatible).

        Uses cumulative counts so the whole video is annotated in O(N)
        instead of the previous O(N^2) prefix scan per frame.
        """
        output_video_frames = []
        tbc = np.asarray(team_ball_control)
        # cumulative counts of team 1 / team 2 possession up to each frame
        cum1 = np.cumsum(tbc == 1)
        cum2 = np.cumsum(tbc == 2)
        for frame_num, frame in enumerate(video_frames):
            control_counts = {
                1: int(cum1[frame_num]) if frame_num < len(cum1) else 0,
                2: int(cum2[frame_num]) if frame_num < len(cum2) else 0,
            }
            annotated = self.annotate_frame(
                frame.copy(),
                tracks["players"][frame_num],
                tracks["referees"][frame_num],
                tracks["ball"][frame_num],
                control_counts=control_counts,
            )
            output_video_frames.append(annotated)

        return output_video_frames