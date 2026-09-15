import cv2


class FrameNumberDrawer:
    def __init__(self):
        pass

    def draw_frame(self, frame, frame_num):
        """Write `frame_num` on the top-left corner of `frame`, in place."""
        cv2.putText(frame, str(frame_num), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        return frame

    def draw(self, frames):
        # Write the frame number on the top left corner of the frame
        return [self.draw_frame(frames[i].copy(), i) for i in range(len(frames))]
