"""
Automatic pitch calibration
===========================
Estimate the four pitch corners from a single frame so speed and distance can
be computed without the user typing pixel coordinates.

How it works
------------
1. Segment the grass. The pitch hue is learned from the frame itself (the modal
   hue of the lower-middle region, which is almost always turf) rather than a
   fixed green range, so it survives floodlights, shadow and broadcast colour
   grading.
2. Take the largest grass component, close gaps (players and lines punch holes
   in the mask), and reduce its convex hull to a quadrilateral.
3. Where the white pitch lines are visible, refine the quad by intersecting the
   outermost near-horizontal and near-vertical lines — lines are a much more
   precise boundary than the grass edge, which bleeds into the crowd.
4. Score the result, so the UI can tell the user whether to trust it.

What this cannot do
-------------------
Corner *geometry* is recoverable from one frame. The real-world *scale* is not:
the same trapezoid could be a full pitch (105x68 m) or one half of it, and
speed and distance scale linearly with that choice. `guess_region_size()`
returns the full-pitch assumption plus a flag, and the caller is expected to
show it as an assumption rather than a measurement.
"""

import cv2
import numpy as np

FULL_PITCH_LENGTH_M = 105.0
FULL_PITCH_WIDTH_M = 68.0


# ─────────────────────────────────────────────────────────────────────────────
# GRASS SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def grass_mask(frame):
    """
    Binary mask of the playing surface.

    The reference hue is sampled from the frame rather than hard-coded: a fixed
    "green" range fails on evening matches and heavily graded broadcast feeds.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    height, width = h.shape

    # Sample a central band that is turf in essentially any pitch-side camera.
    band = h[int(height * 0.45):int(height * 0.85),
             int(width * 0.25):int(width * 0.75)]
    band_s = s[int(height * 0.45):int(height * 0.85),
               int(width * 0.25):int(width * 0.75)]
    turf = band[band_s > 30]
    if turf.size < 100:
        return np.zeros((height, width), np.uint8), None

    hist = np.bincount(turf.ravel(), minlength=180)
    ref_hue = int(np.argmax(hist))

    # Hue is circular, so compare with wraparound.
    diff = np.minimum(np.abs(h.astype(np.int16) - ref_hue),
                      180 - np.abs(h.astype(np.int16) - ref_hue))
    mask = ((diff <= 18) & (s > 25) & (v > 25)).astype(np.uint8) * 255

    # Players, lines and shadows punch holes; close them before taking a hull.
    k = max(5, int(min(height, width) * 0.02) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask, ref_hue


def largest_component(mask):
    """Keep only the biggest blob — drops crowd greenery and advertising."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.uint8) * 255


# ─────────────────────────────────────────────────────────────────────────────
# QUAD FITTING
# ─────────────────────────────────────────────────────────────────────────────
def order_corners(points):
    """Return corners as TL, TR, BR, BL — the order the pipeline expects."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]                 # counter-clockwise from -pi
    start = int(np.argmin(pts.sum(axis=1)))       # top-left has the least x+y
    pts = np.roll(pts, -start, axis=0)
    if pts[1][0] < pts[3][0]:                     # ensure clockwise TL,TR,BR,BL
        pts = pts[[0, 3, 2, 1]]
    return [(float(x), float(y)) for x, y in pts]


def quad_from_mask(mask):
    """Reduce a grass mask to four corners via its convex hull."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))

    # Try a genuine 4-gon first; a pitch hull usually simplifies cleanly.
    peri = cv2.arcLength(hull, True)
    for frac in (0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(hull, frac * peri, True)
        if len(approx) == 4:
            return order_corners(approx)

    # Otherwise take the hull's extreme points in the four diagonal directions,
    # which is a good trapezoid fit for a perspective view of a pitch.
    pts = hull.reshape(-1, 2).astype(np.float32)
    s, d = pts.sum(axis=1), pts[:, 0] - pts[:, 1]
    corners = [pts[np.argmin(s)], pts[np.argmax(d)],
               pts[np.argmax(s)], pts[np.argmin(d)]]
    if len({tuple(c) for c in corners}) < 4:
        return None
    return order_corners(corners)


# ─────────────────────────────────────────────────────────────────────────────
# LINE REFINEMENT
# ─────────────────────────────────────────────────────────────────────────────
def _intersect(l1, l2):
    (x1, y1, x2, y2), (x3, y3, x4, y4) = l1, l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def refine_with_lines(frame, mask, quad):
    """
    Sharpen a grass-derived quad using the white pitch lines.

    Returns a new quad, or None when the lines are too sparse to be trusted.
    The grass boundary bleeds into the crowd and the technical area; the painted
    lines do not, so this is the more accurate boundary when it is visible.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 60) & (hsv[..., 2] > 160)).astype(np.uint8) * 255
    white = cv2.bitwise_and(white, mask)          # only lines *on* the pitch
    if cv2.countNonZero(white) < 200:
        return None

    height, width = white.shape
    segments = cv2.HoughLinesP(white, 1, np.pi / 180,
                               threshold=60,
                               minLineLength=int(min(height, width) * 0.25),
                               maxLineGap=25)
    if segments is None or len(segments) < 4:
        return None

    horizontals, verticals = [], []
    for x1, y1, x2, y2 in segments[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180 - angle)
        (horizontals if angle < 35 else verticals).append((x1, y1, x2, y2))

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    top = min(horizontals, key=lambda l: (l[1] + l[3]) / 2)
    bottom = max(horizontals, key=lambda l: (l[1] + l[3]) / 2)
    left = min(verticals, key=lambda l: (l[0] + l[2]) / 2)
    right = max(verticals, key=lambda l: (l[0] + l[2]) / 2)

    corners = [_intersect(top, left), _intersect(top, right),
               _intersect(bottom, right), _intersect(bottom, left)]
    if any(c is None for c in corners):
        return None

    # Reject a refinement that wandered far outside the frame or collapsed.
    pad = 0.35
    for x, y in corners:
        if not (-pad * width <= x <= (1 + pad) * width and
                -pad * height <= y <= (1 + pad) * height):
            return None
    refined = order_corners(corners)
    if quad_area(refined) < 0.25 * quad_area(quad):
        return None
    return refined


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────
def quad_area(quad):
    pts = np.asarray(quad, dtype=np.float32)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def clipped_edges(quad, shape, tol=0.01):
    """
    Which of the quad's four corners sit on the frame border.

    This matters more than it looks. A quad whose edges run off the frame is
    the *visible grass*, not the pitch boundary — the pitch continues out of
    shot. The geometry is still usable for a homography, but the real-world
    size of that region is then unknown, so 105x68 would be flatly wrong.
    """
    height, width = shape[:2]
    mx, my = tol * width, tol * height
    on_border = 0
    for x, y in quad:
        if x <= mx or x >= width - 1 - mx or y <= my or y >= height - 1 - my:
            on_border += 1
    return on_border


def score(frame, mask, quad):
    """
    0..1 confidence that `quad` really is the *pitch boundary*.

    Combines how much of the quad is actually grass, how much of the frame it
    covers, whether the shape is convex, and — decisively — whether its corners
    are pinned to the frame edge. Without that last term a zoomed-in view
    scores ~99% while describing a region whose dimensions we cannot know.
    """
    height, width = mask.shape
    filled = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(filled, np.asarray(quad, dtype=np.int32), 255)

    area = quad_area(quad)
    if area < 1:
        return 0.0

    inside = cv2.countNonZero(cv2.bitwise_and(filled, mask))
    purity = inside / max(1, cv2.countNonZero(filled))
    coverage = area / float(width * height)

    pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
    convex = cv2.isContourConvex(pts)

    # A pitch view fills a lot of the frame; a sliver or a tiny patch does not.
    coverage_score = min(1.0, coverage / 0.35)
    shape_score = 1.0 if convex else 0.4

    base = purity * 0.5 + coverage_score * 0.3 + shape_score * 0.2

    # Each corner stuck on the frame edge means one less pitch corner we have
    # actually seen. Four visible corners is the only case where the span is
    # safely inferable.
    penalty = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.35, 4: 0.25}
    return round(float(base * penalty[clipped_edges(quad, mask.shape)]), 3)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def detect_pitch_corners(frame):
    """
    Estimate the pitch corners in `frame`.

    Returns (corners, confidence, info) where corners is [TL, TR, BR, BL] in
    this frame's pixel coordinates, or (None, 0.0, info) on failure. `info`
    explains what happened so the UI can say something useful.
    """
    info = {"method": None, "reason": None, "ref_hue": None}

    mask, ref_hue = grass_mask(frame)
    info["ref_hue"] = ref_hue
    if ref_hue is None or cv2.countNonZero(mask) < 0.05 * mask.size:
        info["reason"] = "No pitch-like surface found in this frame."
        return None, 0.0, info

    biggest = largest_component(mask)
    if biggest is None:
        info["reason"] = "The pitch region could not be isolated."
        return None, 0.0, info

    quad = quad_from_mask(biggest)
    if quad is None:
        info["reason"] = "Could not reduce the pitch region to four corners."
        return None, 0.0, info
    info["method"] = "grass"

    refined = refine_with_lines(frame, biggest, quad)
    if refined is not None:
        refined_score, grass_score = (score(frame, biggest, refined),
                                      score(frame, biggest, quad))
        if refined_score >= grass_score:
            quad, info["method"] = refined, "grass+lines"

    return quad, score(frame, biggest, quad), info


def detect_from_video(video_path, samples=5):
    """
    Run detection on several frames and keep the most confident result.

    A single frame can be unlucky — a replay wipe, a close-up, a graphic
    overlay — so sampling across the clip is meaningfully more robust.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    best = (None, 0.0, {"method": None, "reason": "No frames could be read."})

    if total <= 0:
        indices = list(range(samples))
    else:
        step = max(1, total // (samples + 1))
        indices = [min(total - 1, step * (i + 1)) for i in range(samples)]

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        corners, conf, info = detect_pitch_corners(frame)
        if corners is not None and conf > best[1]:
            info = dict(info, frame_index=idx)
            best = (corners, conf, info)

    cap.release()
    return best


def guess_region_size(quad=None, shape=None):
    """
    Real-world span of the detected quad, in metres.

    Returns (length, width, note). The numbers are an *assumption*, never a
    measurement: a single frame cannot tell a full pitch from one half of it,
    and speed/distance scale linearly with the value. The note says how much
    to trust it so the UI can pass that on instead of implying precision.
    """
    if quad is None or shape is None:
        return FULL_PITCH_LENGTH_M, FULL_PITCH_WIDTH_M, "assumed-full-pitch"

    on_border = clipped_edges(quad, shape)
    if on_border == 0:
        return (FULL_PITCH_LENGTH_M, FULL_PITCH_WIDTH_M, "assumed-full-pitch")
    return (FULL_PITCH_LENGTH_M, FULL_PITCH_WIDTH_M, "region-clipped")


def draw_overlay(frame, corners, confidence=None):
    """Return a copy of `frame` with the detected quad drawn on it."""
    out = frame.copy()
    if not corners:
        return out
    pts = np.asarray(corners, dtype=np.int32)
    cv2.polylines(out, [pts], True, (0, 255, 0), 3)
    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), pts):
        cv2.circle(out, (int(x), int(y)), 6, (255, 0, 0), -1)
        cv2.putText(out, label, (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if confidence is not None:
        cv2.putText(out, f"confidence {confidence:.0%}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, f"confidence {confidence:.0%}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
