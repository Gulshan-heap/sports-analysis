import cv2


class SpeedAndDistanceDrawer():
    """Draws per-player speed and cumulative distance.

    Cumulative distance is inherently stateful: each frame adds to a running
    total. When frames are streamed one at a time the accumulator has to live
    on the instance rather than inside draw(), so `reset()` is what starts a
    fresh video.
    """

    def __init__(self):
        self.total_distances = {}

    def reset(self):
        """Clear the running per-player distance totals."""
        self.total_distances = {}

    def draw_frame(self, frame, player_tracks, player_distance, player_speed):
        """
        Draw one frame's speed/distance labels, in place, advancing the totals.

        Args:
            frame (numpy.ndarray): The frame to annotate. Modified in place.
            player_tracks (dict): {track_id: {'bbox': [...]}} for this frame.
            player_distance (dict): Distance covered this frame, per player.
            player_speed (dict): Current speed (km/h), per player.

        Returns:
            numpy.ndarray: The same frame, annotated.
        """
        # Get Total Distance
        for player_id, distance in player_distance.items():
            if player_id not in self.total_distances:
                self.total_distances[player_id] = 0
            self.total_distances[player_id] += distance

        for player_id, bbox in player_tracks.items():
            x1, y1, x2, y2 = bbox['bbox']
            position = [int((x1 + x2) / 2), int(y2)]
            position[1] += 40

            distance = self.total_distances.get(player_id, None)
            speed = player_speed.get(player_id, None)
            if speed is not None:
                cv2.putText(frame, f"{speed:.2f} km/h", position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            if distance is not None:
                cv2.putText(frame, f"{distance:.2f} m",
                            (position[0], position[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame

    def draw(self, video_frames, player_tracks, player_distances_per_frame,
             player_speed_per_frame):
        self.reset()
        output_video_frames = []

        for frame, tracks, player_distance, player_speed in zip(
                video_frames, player_tracks, player_distances_per_frame,
                player_speed_per_frame):
            output_video_frames.append(self.draw_frame(
                frame.copy(), tracks, player_distance, player_speed))

        return output_video_frames
