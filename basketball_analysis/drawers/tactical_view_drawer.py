import cv2


class TacticalViewDrawer:
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):
        self.start_x = 20
        self.start_y = 40
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color
        # The court image is decoded and resized once per (path, size) and
        # reused for every frame — it used to be re-read inside draw().
        self._court_cache = {}

    def court_image(self, court_image_path, width, height):
        key = (str(court_image_path), int(width), int(height))
        if key not in self._court_cache:
            image = cv2.imread(str(court_image_path))
            if image is None:
                raise FileNotFoundError(
                    f"Court image not found: {court_image_path}")
            self._court_cache = {key: cv2.resize(image, (int(width), int(height)))}
        return self._court_cache[key]

    def draw_frame(self,
                   frame,
                   court_image,
                   width,
                   height,
                   tactical_court_keypoints,
                   frame_positions=None,
                   frame_assignments=None,
                   player_with_ball=-1):
        """
        Draw the tactical overlay onto one frame, in place.

        Args:
            frame (numpy.ndarray): The frame to annotate. Modified in place.
            court_image (numpy.ndarray): Pre-decoded, pre-resized court image
                (see :meth:`court_image`).
            width (int): Width of the tactical view.
            height (int): Height of the tactical view.
            tactical_court_keypoints (list): Court keypoints in tactical view.
            frame_positions (dict, optional): {player_id: (x, y)} this frame.
            frame_assignments (dict, optional): {player_id: team_id} this frame.
            player_with_ball: Player ID holding the ball, or -1.

        Returns:
            numpy.ndarray: The same frame, annotated.
        """
        y1 = self.start_y
        y2 = self.start_y + height
        x1 = self.start_x
        x2 = self.start_x + width

        alpha = 0.6  # Transparency factor
        overlay = frame[y1:y2, x1:x2].copy()
        cv2.addWeighted(court_image, alpha, overlay, 1 - alpha, 0, frame[y1:y2, x1:x2])

        # Draw court keypoints
        for keypoint_index, keypoint in enumerate(tactical_court_keypoints):
            x, y = keypoint
            x += self.start_x
            y += self.start_y
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(keypoint_index), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw player positions in tactical view if available
        if frame_positions:
            frame_assignments = frame_assignments or {}
            for player_id, position in frame_positions.items():
                # Get player's team (default to team 1 if not assigned)
                team_id = frame_assignments.get(player_id, 1)

                # Set color based on team
                color = self.team_1_color if team_id == 1 else self.team_2_color

                # Adjust position to overlay coordinates
                x, y = int(position[0]) + self.start_x, int(position[1]) + self.start_y

                # Draw player circle
                player_radius = 8
                cv2.circle(frame, (x, y), player_radius, color, -1)

                # Highlight player with ball
                if player_id == player_with_ball:
                    cv2.circle(frame, (x, y), player_radius + 3, (0, 0, 255), 2)

        return frame

    def draw(self,
             video_frames,
             court_image_path,
             width,
             height,
             tactical_court_keypoints,
             tactical_player_positions=None,
             player_assignment=None,
             ball_acquisition=None):
        """
        Draw tactical view with court keypoints and player positions.

        Args:
            video_frames (list): List of video frames to draw on.
            court_image_path (str): Path to the court image.
            width (int): Width of the tactical view.
            height (int): Height of the tactical view.
            tactical_court_keypoints (list): List of court keypoints in tactical view.
            tactical_player_positions (list, optional): List of dictionaries mapping player IDs to
                their positions in tactical view coordinates.
            player_assignment (list, optional): List of dictionaries mapping player IDs to team assignments.
            ball_acquisition (list, optional): List indicating which player has the ball in each frame.

        Returns:
            list: List of frames with tactical view drawn on them.
        """
        court_image = self.court_image(court_image_path, width, height)

        output_video_frames = []
        for frame_idx, frame in enumerate(video_frames):
            positions = None
            assignments = None
            with_ball = -1
            if tactical_player_positions and player_assignment \
                    and frame_idx < len(tactical_player_positions):
                positions = tactical_player_positions[frame_idx]
                assignments = (player_assignment[frame_idx]
                               if frame_idx < len(player_assignment) else {})
                with_ball = (ball_acquisition[frame_idx]
                             if ball_acquisition and frame_idx < len(ball_acquisition)
                             else -1)

            output_video_frames.append(self.draw_frame(
                frame.copy(), court_image, width, height,
                tactical_court_keypoints, positions, assignments, with_ball))

        return output_video_frames
