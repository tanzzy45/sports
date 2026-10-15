"""
Camera motion compensation for broadcast cricket footage.

The camera behind the bowler pans and zooms while the ball is in the air, usually
following it toward the batter. It turns about a fixed point, so consecutive frames
are related exactly by a homography. Chaining those homographies maps every frame of
a delivery into one reference frame, where the camera is effectively still.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

ANALYSIS_WIDTH = 640
STILL_THRESHOLD = 2  # median gray-level change between frames of a still camera
CUT_THRESHOLD = 0.7  # color histogram correlation below which frames are a cut
# Largest camera move between consecutive frames: zoom factor, shift as a fraction
# of the width, and perspective change across the width.
MAX_STEP_ZOOM = 1.25
MAX_STEP_SHIFT = 0.15
MAX_STEP_PERSPECTIVE = 0.4
MAX_TOTAL_ZOOM = 10.0  # broadcast zooms stay well below this over a delivery


def static_overlay_mask(frames: List[np.ndarray], std_threshold: float = 3.0,
                        dilation: int = 9) -> np.ndarray:
    """
    Mask of the pixels usable for motion estimation: 255 where the image changes over
    the clip, 0 on graphics that never move (score bars, logos, watermarks). Static
    graphics would otherwise make the camera look still. When most of the image is
    static, the camera itself is still and nothing is masked.
    """
    sample = frames[::max(1, len(frames) // 30)]
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in sample])
    static = gray.std(axis=0) < std_threshold
    if static.mean() > 0.5:
        return np.full(static.shape, 255, dtype=np.uint8)
    edges = cv2.Canny(gray[0], 50, 150) > 0
    overlay = cv2.dilate((static & edges).astype(np.uint8),
                         np.ones((dilation, dilation), np.uint8))
    return np.where(overlay > 0, 0, 255).astype(np.uint8)


def _pair_homography(previous: np.ndarray, current: np.ndarray, mask: np.ndarray,
                     scale: float) -> Optional[np.ndarray]:
    """Homography mapping `previous` frame pixels to `current` frame pixels."""
    size = (int(previous.shape[1] * scale), int(previous.shape[0] * scale))
    a = cv2.cvtColor(cv2.resize(previous, size), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(current, size), cv2.COLOR_BGR2GRAY)
    small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    if _is_cut(previous, current):
        return None
    homography = _track(a, b, small_mask)
    if homography is None:
        # Too little texture to track, e.g. a plain field: if the frames barely
        # differ, the camera is still. The median ignores players and the ball.
        if np.median(cv2.absdiff(a, b)[small_mask > 0]) < STILL_THRESHOLD:
            return np.eye(3)
        return None
    to_small = np.diag([scale, scale, 1.0])
    return np.linalg.inv(to_small) @ homography @ to_small


def _is_cut(previous: np.ndarray, current: np.ndarray) -> bool:
    """
    Whether two frames belong to different shots. Even during a fast zoom,
    consecutive frames have nearly the same colors; across a cut they do not, and
    feature tracking can still find a homography that looks valid.
    """
    histograms = []
    for frame in (previous, current):
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        histograms.append(cv2.normalize(histogram, histogram))
    return cv2.compareHist(histograms[0], histograms[1], cv2.HISTCMP_CORREL) < CUT_THRESHOLD


def _track(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    points = cv2.goodFeaturesToTrack(a, maxCorners=600, qualityLevel=0.01,
                                     minDistance=8, mask=mask)
    if points is None or len(points) < 20:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, winSize=(21, 21),
                                               maxLevel=3)
    good = status.ravel() == 1
    if good.sum() < 20:
        return None
    homography, inliers = cv2.findHomography(points[good], moved[good], cv2.RANSAC, 1.0)
    if homography is None or inliers.sum() < 15 or not _plausible(homography, a.shape[1]):
        return None
    return homography


def _plausible(homography: np.ndarray, width: int) -> bool:
    """
    Whether a frame-to-frame homography is a camera move rather than a cut matched by
    accident: between consecutive frames the zoom changes by a few percent and the
    image shifts a little.
    """
    homography = homography / homography[2, 2]
    scale = np.sqrt(abs(np.linalg.det(homography[:2, :2])))
    shift = np.linalg.norm(homography[:2, 2]) / width
    perspective = np.linalg.norm(homography[2, :2]) * width
    return 1 / MAX_STEP_ZOOM < scale < MAX_STEP_ZOOM and shift < MAX_STEP_SHIFT \
        and perspective < MAX_STEP_PERSPECTIVE


def _runs(steps: List[Optional[np.ndarray]]) -> List[Tuple[int, int]]:
    """First and last frame of every stretch the camera can be followed over."""
    runs, start = [], 0
    for t, step in enumerate(steps + [None]):
        if step is None:
            runs.append((start, t))
            start = t + 1
    return runs


def frame_to_reference(frames: List[np.ndarray], mask: Optional[np.ndarray] = None,
                       anchor: Optional[int] = None) -> List[Optional[np.ndarray]]:
    """
    Homographies mapping each frame's pixels into a reference frame's pixels. The
    reference is the first frame of the stretch the camera can be followed over that
    contains frame `anchor`, or of the longest stretch without an anchor; frames
    outside that stretch, e.g. across a cut, are `None`.
    """
    if mask is None:
        mask = static_overlay_mask(frames)
    scale = ANALYSIS_WIDTH / frames[0].shape[1]
    steps = [_pair_homography(previous, current, mask, scale)
             for previous, current in zip(frames, frames[1:])]  # frame t -> t + 1
    runs = _runs(steps)
    if anchor is None:
        first, last = max(runs, key=lambda run: run[1] - run[0])
    else:
        first, last = next(run for run in runs if run[0] <= anchor <= run[1])

    result: List[Optional[np.ndarray]] = [None] * len(frames)
    result[first] = np.eye(3)
    for t in range(first + 1, last + 1):
        result[t] = result[t - 1] @ np.linalg.inv(steps[t - 1])
        zoom = 1 / np.sqrt(abs(np.linalg.det(result[t][:2, :2])))
        if not 1 / MAX_TOTAL_ZOOM < zoom < MAX_TOTAL_ZOOM:  # tracking went astray
            result[t] = None
            break
    return result


def transform(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) == 0:
        return points.reshape(-1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)
