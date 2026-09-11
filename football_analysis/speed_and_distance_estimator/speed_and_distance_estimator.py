import cv2
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self,fps=24,frame_window=5):
        self.frame_window= frame_window
        self.frame_rate= fps
        # Streaming state: index of the next window start to process.
        self._next_window = 0
        self._total_distance = {}

    def reset(self):
        """Forget streaming state so a new video can be processed."""
        self._next_window = 0
        self._total_distance = {}
    
    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance= {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position,end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def update(self, tracks):
        """Streaming version of add_speed_and_distance_to_tracks.

        Call after every new frame is appended to tracks. Processes every
        window that has become complete; identical maths to the batch method.
        Returns the dict of total distances so far.
        """
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue
            number_of_frames = len(object_tracks)
            # A full window needs last_frame = start + window to be a valid
            # index (<= number_of_frames - 1); the trailing partial window is
            # handled by finalize().
            while self._next_window + self.frame_window < number_of_frames:
                frame_num = self._next_window
                last_frame = frame_num + self.frame_window
                self._process_window(tracks, object, object_tracks,
                                     frame_num, last_frame)
                self._next_window += self.frame_window
        return self._total_distance

    def finalize(self, tracks):
        """Process the trailing partial window at the end of the video."""
        number_of_frames = 0
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue
            number_of_frames = len(object_tracks)
            if self._next_window < number_of_frames:
                frame_num = self._next_window
                last_frame = number_of_frames - 1
                self._process_window(tracks, object, object_tracks,
                                     frame_num, last_frame)
        self._next_window = number_of_frames
        return self._total_distance

    def _process_window(self, tracks, object, object_tracks,
                        frame_num, last_frame):
        for track_id, _ in object_tracks[frame_num].items():
            if track_id not in object_tracks[last_frame]:
                continue

            start_position = object_tracks[frame_num][track_id]['position_transformed']
            end_position = object_tracks[last_frame][track_id]['position_transformed']

            if start_position is None or end_position is None:
                continue

            distance_covered = measure_distance(start_position, end_position)
            time_elapsed = (last_frame - frame_num) / self.frame_rate
            if time_elapsed <= 0:
                continue
            speed_meteres_per_second = distance_covered / time_elapsed
            speed_km_per_hour = speed_meteres_per_second * 3.6

            if object not in self._total_distance:
                self._total_distance[object] = {}

            if track_id not in self._total_distance[object]:
                self._total_distance[object][track_id] = 0

            self._total_distance[object][track_id] += distance_covered

            for frame_num_batch in range(frame_num, last_frame):
                if track_id not in tracks[object][frame_num_batch]:
                    continue
                tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                tracks[object][frame_num_batch][track_id]['distance'] = self._total_distance[object][track_id]

    def annotate_frame(self, frame, players):
        """Draw speed/distance labels for ONE frame (scale aware)."""
        h, w = frame.shape[:2]
        s = max(0.6, min(1.6, min(h, w) / 720.0))
        font_scale = max(0.4, min(0.7, 0.5 * s))
        thickness = max(1, int(round(s)))

        for _, track_info in players.items():
            if "speed" not in track_info:
                continue
            speed = track_info.get('speed')
            distance = track_info.get('distance')
            if speed is None or distance is None:
                continue

            bbox = track_info['bbox']
            position = list(get_foot_position(bbox))
            position[1] += int(40 * s)
            position = tuple(map(int, position))
            cv2.putText(frame, f"{speed:.2f} km/h", position,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + int(20 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        return frame

    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                frame = self.annotate_frame(frame, object_tracks[frame_num])
            output_frames.append(frame)
        
        return output_frames