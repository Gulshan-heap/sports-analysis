from .utils import draw_traingle

class BallTracksDrawer:
    """
    A drawer class responsible for drawing ball tracks on video frames.

    Attributes:
        ball_pointer_color (tuple): The color used to draw the ball pointers (in BGR format).
    """

    def __init__(self):
        """
        Initialize the BallTracksDrawer instance with default settings.
        """
        self.ball_pointer_color = (0, 255, 0)

    def draw_frame(self, frame, ball_dict):
        """
        Draw one frame's ball pointer, in place.

        Args:
            frame (numpy.ndarray): The frame to annotate. Modified in place.
            ball_dict (dict): Ball detections for this frame.

        Returns:
            numpy.ndarray: The same frame, annotated.
        """
        for _, ball in ball_dict.items():
            if ball.get("bbox") is None:
                continue
            frame = draw_traingle(frame, ball["bbox"], self.ball_pointer_color)
        return frame

    def draw(self, video_frames, tracks):
        """
        Draws ball pointers on each video frame based on provided tracking information.

        Args:
            video_frames (list): A list of video frames (as NumPy arrays or image objects).
            tracks (list): A list of dictionaries where each dictionary contains ball information
                for the corresponding frame.

        Returns:
            list: A list of processed video frames with drawn ball pointers.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            output_video_frames.append(
                self.draw_frame(frame.copy(), tracks[frame_num]))

        return output_video_frames
