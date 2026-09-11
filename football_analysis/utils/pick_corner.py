import cv2


def _gui_available():
    """Return False when OpenCV was built without GUI support (headless)."""
    try:
        import numpy as np
        cv2.imshow("_probe", np.zeros((2, 2, 3), np.uint8))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except Exception:
        return False


def pick_corners(video_path, num_points=4):
    if not _gui_available():
        raise ValueError(
            "Interactive calibration is not available: this OpenCV build has no GUI "
            "support (headless). Use the Streamlit web app (app.py -> Manual corners) "
            "or supply a calibration JSON via '--calib <file.json>'."
        )
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read first frame")

    points = []
    clone = frame.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(clone, str(len(points)), (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Pick 4 pitch corners (TL, TR, BR, BL order)", clone)

    cv2.imshow("Pick 4 pitch corners (TL, TR, BR, BL order)", clone)
    cv2.setMouseCallback("Pick 4 pitch corners (TL, TR, BR, BL order)", click_event)
    print("Click the 4 pitch corners in order: top-left, top-right, bottom-right, bottom-left.")
    print("Press 'q' once done.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(points) >= num_points:
            break

    cv2.destroyAllWindows()
    print("Picked pixel coordinates:")
    for p in points:
        print(p)
    return points


if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else "input_videos/08fd33_4.mp4"
    pick_corners(video_path)