"""
Build a cricket pitch map from behind-the-bowler video.

1. Camera motion - broadcast cameras zoom toward the batter while the ball is in
   the air. Every frame is mapped into a reference frame of the delivery, where the
   camera is effectively still, see camera_motion.py.
2. Calibration - the homography between the reference frame and the pitch comes
   either from four crease points clicked once with `calibrate`, or from a pitch
   keypoint model (`--pitch_model`, trained with
   notebooks/train_pitch_keypoint_detector.ipynb) for every delivery.
3. Ball detection - three-frame differencing inside the pitch area needs no model; a
   ball detector hosted on Roboflow (`--ball_model`) is more reliable on broadcasts.
4. `run` links the detections into a trajectory, finds the bounce, projects it onto
   the pitch, and renders the video next to a live pitch map.
"""
import argparse
import base64
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from camera_motion import frame_to_reference, static_overlay_mask, transform
from prepare_pitch_dataset import CONFIG_VERTICES
from sports.common.view import ViewTransformer
from sports.cricket import (
    DEFAULT_LENGTH_ZONES,
    DEFAULT_ZONE_COLORS,
    BatterHand,
    Bounce,
    CricketPitchConfiguration,
    classify_length,
    classify_line,
    detect_bounce,
    distance_from_stumps,
    draw_pitch_map,
    lateral_offset,
)

CONFIG = CricketPitchConfiguration()

# Where the return creases meet the popping creases at both ends: clearly painted
# and spread over the whole pitch, which keeps the homography well conditioned.
CALIBRATION_LABELS = ["02", "03", "13", "14"]
CALIBRATION_PROMPTS = [
    "NEAR end (bowler): popping crease x LEFT return crease",
    "NEAR end (bowler): popping crease x RIGHT return crease",
    "FAR end (batter): popping crease x LEFT return crease",
    "FAR end (batter): popping crease x RIGHT return crease",
]

# Keypoints of the pitch model at each end, see prepare_pitch_dataset.py.
FAR_KEYPOINTS = [0, 1, 4, 5, 8, 9, 10]
NEAR_KEYPOINTS = [2, 3, 6, 7, 11, 12, 13]

PITCH_MAP_VISIBLE_LENGTH = 1500  # [cm] in front of the batter's stumps
PITCH_MAP_PADDING = 30
PANEL_TEXT_WIDTH = 340
TRAIL_COLOR = sv.Color.from_hex("#FFD700")
BOUNCE_COLOR = sv.Color.from_hex("#FF1493")
CREASE_COLOR = sv.Color.from_hex("#00BFFF")


@dataclass
class Calibration:
    """Homographies between the pitch and the reference frame of a delivery."""
    image_to_pitch: ViewTransformer
    pitch_to_image: ViewTransformer
    batting_end: int


@dataclass
class Delivery:
    frames: List[np.ndarray]
    # per frame: homography from the frame to the reference frame, None if lost
    to_reference: List[Optional[np.ndarray]]
    calibration: Calibration
    trajectory: np.ndarray
    bounce: Optional[Bounce]
    bounce_pitch_xy: Optional[np.ndarray]


def read_frames(source_video_path: str) -> List[np.ndarray]:
    return list(sv.get_video_frames_generator(source_path=source_video_path))


def calibrate(source_video_path: str, calibration_path: str, frame_index: int,
              batting_end: int) -> None:
    """Click the four calibration points on one frame and save them."""
    frames = read_frames(source_video_path)
    frame = frames[min(frame_index, len(frames) - 1)]
    points: List[Tuple[int, int]] = []
    window = "calibrate: click points, [r] reset, [enter] save, [esc] quit"

    def on_click(event: int, x: int, y: int, *_) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(CALIBRATION_LABELS):
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_click)
    while True:
        canvas = frame.copy()
        for point in points:
            cv2.circle(canvas, point, 6, BOUNCE_COLOR.as_bgr(), -1)
        prompt = (CALIBRATION_PROMPTS[len(points)] if len(points) < len(CALIBRATION_PROMPTS)
                  else "press [enter] to save")
        cv2.putText(canvas, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == 13 and len(points) == len(CALIBRATION_LABELS):
            break
        elif key == 27:
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()

    with open(calibration_path, "w") as f:
        json.dump({
            "labels": CALIBRATION_LABELS,
            "image_points": points,
            "batting_end": batting_end,
        }, f, indent=2)
    print(f"saved calibration to {calibration_path}")


def calibration_from_points(image_points: np.ndarray, pitch_points: np.ndarray,
                            batting_end: int = 2,
                            ransac_threshold: float = 30.0) -> Calibration:
    """
    Homographies between the image and the pitch from matching points. With more than
    four points, outliers more than `ransac_threshold` [cm] off are dropped first.
    """
    image_points = np.asarray(image_points, dtype=np.float32)
    pitch_points = np.asarray(pitch_points, dtype=np.float32)
    if len(image_points) > 4:
        cv2.setRNGSeed(0)  # RANSAC samples randomly: keep calibrations repeatable
        _, inliers = cv2.findHomography(image_points, pitch_points, cv2.RANSAC,
                                        ransac_threshold)
        if inliers is not None and inliers.sum() >= 4:
            keep = inliers.ravel().astype(bool)
            image_points, pitch_points = image_points[keep], pitch_points[keep]
    return Calibration(
        image_to_pitch=ViewTransformer(source=image_points, target=pitch_points),
        pitch_to_image=ViewTransformer(source=pitch_points, target=image_points),
        batting_end=batting_end,
    )


def load_calibration(calibration_path: str) -> Calibration:
    with open(calibration_path) as f:
        data = json.load(f)
    pitch_points = [CONFIG.vertices[CONFIG.labels.index(label)] for label in data["labels"]]
    return calibration_from_points(data["image_points"], pitch_points,
                                   batting_end=data.get("batting_end", 2))


def calibrate_with_model(frames: List[np.ndarray],
                         to_reference: List[Optional[np.ndarray]], model,
                         min_confidence: float = 0.5, samples: int = 8,
                         imgsz: int = 1280) -> Calibration:
    """
    Calibrate the reference frame of a delivery with the pitch keypoint model. The
    keypoints found in several frames are mapped into the reference frame and
    combined with a median, which also recovers points hidden by the bowler or the
    umpire in some of the frames.
    """
    followed = [t for t, h in enumerate(to_reference) if h is not None]
    picks = np.linspace(0, len(followed) - 1, min(samples, len(followed))).astype(int)
    indices = np.unique([followed[i] for i in picks])
    found = np.full((len(indices), len(CONFIG_VERTICES), 2), np.nan)
    for row, index in enumerate(indices):
        result = model(frames[index], imgsz=imgsz, verbose=False)[0]
        if len(result.boxes) == 0 or result.keypoints is None:
            continue
        best = int(result.boxes.conf.argmax())
        xy = result.keypoints.xy[best].cpu().numpy()
        confidence = (result.keypoints.conf[best].cpu().numpy()
                      if result.keypoints.conf is not None else np.ones(len(xy)))
        keep = confidence >= min_confidence
        found[row, keep] = transform(to_reference[index], xy[keep])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        image_points = np.nanmedian(found, axis=0)
    visible = ~np.isnan(image_points).any(axis=1)
    far, near = visible[FAR_KEYPOINTS].sum(), visible[NEAR_KEYPOINTS].sum()
    if far < 2 or near < 2:
        raise ValueError(
            f"the pitch model found {far} keypoints at the far end and {near} at the "
            "near end; at least 2 at each end are needed. Calibrate this camera "
            "manually with `calibrate` and pass --calibration_path instead.")
    pitch_points = np.array([CONFIG.vertices[v - 1] for v in CONFIG_VERTICES])
    return calibration_from_points(image_points[visible], pitch_points[visible])


def pitch_region_mask(
    shape: Tuple[int, int],
    calibration: Calibration,
    margin: float = 150.0
) -> np.ndarray:
    """
    Image mask of the pitch strip, widened by `margin` [cm] on each side. It is
    dilated upward, since a ball in the air appears above its ground position.
    """
    half_width = CONFIG.pitch_width / 2 + margin
    corners = np.array([
        [0, CONFIG.middle_stump_y - half_width],
        [CONFIG.length, CONFIG.middle_stump_y - half_width],
        [CONFIG.length, CONFIG.middle_stump_y + half_width],
        [0, CONFIG.middle_stump_y + half_width],
    ], dtype=np.float32)
    polygon = calibration.pitch_to_image.transform_points(corners).astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    lift = max(3, shape[0] // 4)
    kernel = np.zeros((2 * lift + 1, 1), dtype=np.uint8)
    kernel[lift:] = 1
    return cv2.dilate(mask, kernel)


def inside_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    inside = ((points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) &
              (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0]))
    points = points[inside]
    keep = mask[points[:, 1].astype(int), points[:, 0].astype(int)] > 0
    return points[keep]


class MotionBallDetector:
    """
    Finds small moving blobs with three-frame differencing: a pixel counts as moving
    at frame t only if it differs from both frame t-1 and frame t+1, which keeps the
    ball's current position and drops where it was or will be.
    """

    def __init__(self, threshold: int = 25, min_area: int = 3,
                 max_area: int = 400) -> None:
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    def detect(self, previous: np.ndarray, current: np.ndarray,
               following: np.ndarray) -> np.ndarray:
        """Moving blobs in `current`; the neighbours must be aligned to it already."""
        previous, current, following = map(self._gray, (previous, current, following))
        motion = cv2.min(cv2.absdiff(current, previous), cv2.absdiff(following, current))
        _, binary = cv2.threshold(motion, self.threshold, 255, cv2.THRESH_BINARY)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
        areas = stats[1:count, cv2.CC_STAT_AREA]
        keep = (areas >= self.min_area) & (areas <= self.max_area)
        return centroids[1:count][keep]

    def detect_frames(self, frames: List[np.ndarray], mask: np.ndarray,
                      to_reference: List[Optional[np.ndarray]]) -> List[np.ndarray]:
        """Candidates per frame, in reference frame pixels inside `mask`."""
        height, width = frames[0].shape[:2]
        # Graphics stay put on screen, so they move once the camera is compensated.
        overlay = static_overlay_mask(frames)
        candidates = [np.empty((0, 2)) for _ in frames]
        for t in range(1, len(frames) - 1):
            if any(to_reference[i] is None for i in (t - 1, t, t + 1)):
                continue
            # Warp the neighbours onto frame t to cancel the camera motion.
            to_current = np.linalg.inv(to_reference[t])
            neighbours = [cv2.warpPerspective(frames[i], to_current @ to_reference[i],
                                              (width, height), borderValue=0)
                          for i in (t - 1, t + 1)]
            points = inside_mask(self.detect(neighbours[0], frames[t], neighbours[1]),
                                 overlay)
            candidates[t] = inside_mask(transform(to_reference[t], points), mask)
        return candidates


class FrameBallDetector:
    """A ball detector that looks at one frame at a time."""

    def detect(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def detect_frames(self, frames: List[np.ndarray], mask: np.ndarray,
                      to_reference: List[Optional[np.ndarray]]) -> List[np.ndarray]:
        """Candidates per frame, in reference frame pixels inside `mask`."""
        candidates = []
        for frame, homography in tqdm(list(zip(frames, to_reference)),
                                      desc="ball detection", leave=False):
            if homography is None:
                candidates.append(np.empty((0, 2)))
                continue
            candidates.append(inside_mask(transform(homography, self.detect(frame)), mask))
        return candidates


class YoloBallDetector(FrameBallDetector):
    """Ball detection with a local Ultralytics model, e.g. one trained with
    notebooks/train_ball_detector.ipynb."""

    def __init__(self, model_path: str, confidence: float = 0.3,
                 imgsz: int = 1280) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> np.ndarray:
        result = self.model(frame, imgsz=self.imgsz, conf=self.confidence,
                            verbose=False)[0]
        names = result.names
        return np.array([
            [(x1 + x2) / 2, (y1 + y2) / 2]
            for (x1, y1, x2, y2), cls in zip(result.boxes.xyxy.cpu().numpy(),
                                             result.boxes.cls.cpu().numpy().astype(int))
            if "ball" in names[cls].lower()
        ]).reshape(-1, 2)


class RoboflowBallDetector(FrameBallDetector):
    """
    Ball detection with a model hosted on Roboflow, e.g. `cricket-balls-l1us5/3`.
    Uses `ROBOFLOW_API_KEY`, or the key saved by `roboflow login`.
    """

    def __init__(self, model_id: str, confidence: float = 0.3) -> None:
        from roboflow.config import load_roboflow_api_key

        self.api_key = load_roboflow_api_key()
        if not self.api_key:
            raise ValueError("no Roboflow API key: set ROBOFLOW_API_KEY or run "
                             "`roboflow login`.")
        self.url = f"https://detect.roboflow.com/{model_id}"
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> np.ndarray:
        import requests

        _, buffer = cv2.imencode(".jpg", frame)
        data = None
        for attempt in range(3):
            try:
                response = requests.post(
                    self.url,
                    params={"api_key": self.api_key,
                            "confidence": int(self.confidence * 100)},
                    data=base64.b64encode(buffer).decode("ascii"),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException as error:
                print(f"ball detection request failed ({error.__class__.__name__}), "
                      f"attempt {attempt + 1}/3")
        if data is None:  # treat the frame as one where the ball was not found
            return np.empty((0, 2))
        # Predictions are in the coordinates of the image the model saw.
        scale_x = frame.shape[1] / float(data["image"]["width"])
        scale_y = frame.shape[0] / float(data["image"]["height"])
        return np.array([
            [prediction["x"] * scale_x, prediction["y"] * scale_y]
            for prediction in data["predictions"]
            if "ball" in prediction["class"].lower()
        ]).reshape(-1, 2)


def link_trajectory(candidates: List[np.ndarray], max_jump: float,
                    max_missed: int = 2, min_length: int = 6) -> np.ndarray:
    """
    Link per-frame candidates into tracks with a constant-velocity prediction and
    return the longest moving track as a `(T, 2)` array with `NaN` gaps.
    """
    tracks: List[Dict] = []
    for t, points in enumerate(candidates):
        unused = list(range(len(points)))
        active = [track for track in tracks if t - track["last_t"] <= max_missed + 1]
        for track in sorted(active, key=lambda track: -len(track["points"])):
            if not unused:
                break
            elapsed = t - track["last_t"]
            predicted = track["last"] + track["velocity"] * elapsed
            distances = [np.linalg.norm(points[i] - predicted) for i in unused]
            best = int(np.argmin(distances))
            if distances[best] <= max_jump:
                point = points[unused.pop(best)]
                track["velocity"] = (point - track["last"]) / elapsed
                track["last"], track["last_t"] = point, t
                track["points"][t] = point
        for i in unused:
            tracks.append({"points": {t: points[i]}, "last": points[i], "last_t": t,
                           "velocity": np.zeros(2)})

    trajectory = np.full((len(candidates), 2), np.nan)
    moving = [
        track for track in tracks
        if len(track["points"]) >= min_length
        and np.linalg.norm(np.ptp(np.array(list(track["points"].values())), axis=0))
        > max_jump
    ]
    if not moving:
        return trajectory
    best = max(moving, key=lambda track: len(track["points"]))
    for t, point in best["points"].items():
        trajectory[t] = point
    return trajectory


def select_bounce_manually(frames: List[np.ndarray],
                           to_reference: List[Optional[np.ndarray]],
                           calibration: Calibration,
                           trajectory: np.ndarray) -> Optional[Bounce]:
    """
    Step through frames and click where the ball bounced. When the bounce falls
    between two frames, click the ball in the last frame before it and the first
    frame after it: the bounce is placed halfway between the two. Returns the bounce
    in reference frame pixels, so a click on a zoomed frame lands in the right place.
    """
    index, selected = 0, []
    window = ("bounce: [a]/[d] prev/next frame, click the ball (1 or 2 frames), "
              "[r] reset, [enter] ok, [esc] skip")

    def on_click(event: int, x: int, y: int, *_) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or to_reference[index] is None:
            return
        others = [click for click in selected if click[0] != index]
        selected[:] = (others if len(others) < 2 else [])[-1:] + [(index, x, y)]

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_click)
    while True:
        canvas = frames[index].copy()
        if to_reference[index] is not None:
            from_reference = np.linalg.inv(to_reference[index])
            canvas = draw_creases(canvas, calibration, from_reference)
            trail = trajectory[:index + 1].copy()
            visible = ~np.isnan(trail).any(axis=1)
            trail[visible] = transform(from_reference, trail[visible])
            canvas = draw_trail(canvas, trail)
        for frame_index, x, y in selected:
            if frame_index == index:
                cv2.circle(canvas, (x, y), 8, BOUNCE_COLOR.as_bgr(), 2)
        clicked = ", ".join(str(click[0] + 1) for click in sorted(selected))
        cv2.putText(canvas, f"frame {index + 1}/{len(frames)}   clicked: {clicked or '-'}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("a"):
            index = max(0, index - 1)
        elif key == ord("d"):
            index = min(len(frames) - 1, index + 1)
        elif key == ord("r"):
            selected.clear()
        elif key == 13 and selected:
            break
        elif key == 27:
            selected.clear()
            break
    cv2.destroyWindow(window)
    if not selected:
        return None
    points = np.array([
        transform(to_reference[frame_index], np.array([[x, y]], dtype=np.float32))[0]
        for frame_index, x, y in selected
    ])
    frame_index = float(np.mean([click[0] for click in selected]))
    return Bounce(frame_index=frame_index, xy=points.mean(axis=0).astype(float))


def process_delivery(source_video_path: str, calibration: Optional[Calibration],
                     pitch_model, ball_detector, manual_fallback: bool,
                     manual_bounce: bool = False,
                     saved: Optional[dict] = None) -> Delivery:
    frames = read_frames(source_video_path)
    height, width = frames[0].shape[:2]
    to_reference = frame_to_reference(frames)
    if calibration is None:
        calibration = calibrate_with_model(frames, to_reference, pitch_model)
    mask = pitch_region_mask((height, width), calibration)

    if saved is not None and saved.get("bounce_reference_xy") is not None:
        # Reuse a bounce from an earlier run, e.g. one selected by hand.
        trajectory = np.full((len(frames), 2), np.nan)
        bounce = Bounce(frame_index=float(saved["bounce_frame"]),
                        xy=np.array(saved["bounce_reference_xy"], dtype=float))
    elif manual_bounce:
        trajectory = np.full((len(frames), 2), np.nan)
        bounce = select_bounce_manually(frames, to_reference, calibration, trajectory)
    else:
        candidates = ball_detector.detect_frames(frames, mask, to_reference)
        trajectory = link_trajectory(candidates, max_jump=0.08 * width)
        bounce = detect_bounce(trajectory)
        if bounce is None and manual_fallback:
            print(f"no bounce found in {source_video_path}, select it manually")
            bounce = select_bounce_manually(frames, to_reference, calibration,
                                            trajectory)

    bounce_pitch_xy = None
    if bounce is not None:
        bounce_pitch_xy = calibration.image_to_pitch.transform_points(
            bounce.xy[np.newaxis].astype(np.float32))[0]
    return Delivery(frames, to_reference, calibration, trajectory, bounce,
                    bounce_pitch_xy)


def draw_trail(frame: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    points = [tuple(int(v) for v in p) for p in trajectory if not np.isnan(p).any()]
    for start, end in zip(points, points[1:]):
        cv2.line(frame, start, end, TRAIL_COLOR.as_bgr(), 2, cv2.LINE_AA)
    for point in points[-1:]:
        cv2.circle(frame, point, 6, TRAIL_COLOR.as_bgr(), 2, cv2.LINE_AA)
    return frame


def draw_creases(frame: np.ndarray, calibration: Calibration,
                 from_reference: np.ndarray) -> np.ndarray:
    """Project the crease lines onto the frame to show the calibration."""
    vertices = np.array(CONFIG.vertices, dtype=np.float32)
    projected = transform(from_reference,
                          calibration.pitch_to_image.transform_points(vertices)).astype(int)
    for start, end in CONFIG.edges:
        cv2.line(frame, tuple(projected[start - 1]), tuple(projected[end - 1]),
                 CREASE_COLOR.as_bgr(), 1, cv2.LINE_AA)
    return frame


def describe(bounce_xy: np.ndarray, batting_end: int, batter_hand: BatterHand) -> str:
    distance = distance_from_stumps(CONFIG, bounce_xy, batting_end)[0]
    offset = lateral_offset(CONFIG, bounce_xy, batting_end, batter_hand)[0]
    zone = classify_length(distance)
    line = classify_line(offset, CONFIG)
    return f"{zone.name if zone else 'n/a'} | {distance / 100:.1f} m | {line.value}"


def render_panel(bounces: List[np.ndarray], descriptions: List[str], height: int,
                 batting_end: int) -> np.ndarray:
    """
    Vertical pitch map with the batter at the top, as seen from the bowler, cropped
    to the part of the pitch where deliveries bounce, next to a legend.
    """
    visible_length = CONFIG.end_margin + PITCH_MAP_VISIBLE_LENGTH
    scale = (height - 2 * PITCH_MAP_PADDING) / visible_length
    bounce_xy = np.array(bounces).reshape(-1, 2)
    pitch_map = draw_pitch_map(CONFIG, bounce_xy, batting_end=batting_end,
                               radius=8, padding=PITCH_MAP_PADDING, scale=scale)
    # Rotate so the batting end is at the top and the bowler's right stays right,
    # then keep the top of the pitch.
    pitch_map = np.rot90(pitch_map, k=1 if batting_end == 2 else -1)[:height].copy()
    map_width = pitch_map.shape[1]

    panel = np.full((height, map_width + PANEL_TEXT_WIDTH, 3), 24, dtype=np.uint8)
    panel[:pitch_map.shape[0], :map_width] = pitch_map

    x, y = map_width + 16, 40
    for zone, color in zip(DEFAULT_LENGTH_ZONES, DEFAULT_ZONE_COLORS):
        cv2.circle(panel, (x + 8, y - 6), 7, color.as_bgr(), -1)
        cv2.putText(panel, zone.name, (x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)
        y += 26
    y += 20
    max_lines = max((height - y) // 26, 1)
    first = max(len(descriptions) - max_lines, 0)
    for index in range(first, len(descriptions)):
        cv2.putText(panel, f"{index + 1}. {descriptions[index]}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return panel


def run(source_video_paths: List[str], calibration_path: Optional[str],
        pitch_model_path: Optional[str], ball_model_id: Optional[str],
        ball_confidence: float, target_video_path: str, batter_hand: BatterHand,
        manual_fallback: bool, results_path: Optional[str], display: bool,
        manual_bounce: bool = False, bounces_path: Optional[str] = None) -> None:
    calibration = load_calibration(calibration_path) if calibration_path else None
    saved = {}
    if bounces_path:
        with open(bounces_path) as f:
            saved = {os.path.basename(r["clip"]): r for r in json.load(f)}
    pitch_model = None
    if pitch_model_path:
        from ultralytics import YOLO

        pitch_model = YOLO(pitch_model_path)
    if not ball_model_id:
        ball_detector = MotionBallDetector()
    elif ball_model_id.endswith(".pt"):
        ball_detector = YoloBallDetector(ball_model_id, ball_confidence)
    else:
        ball_detector = RoboflowBallDetector(ball_model_id, ball_confidence)

    video_info = sv.VideoInfo.from_video_path(source_video_paths[0])
    width, height = video_info.width, video_info.height

    deliveries, paths = [], []
    for path in tqdm(source_video_paths, desc="processing deliveries"):
        try:
            deliveries.append(process_delivery(
                path, calibration, pitch_model, ball_detector, manual_fallback,
                manual_bounce, saved.get(os.path.basename(path))))
            paths.append(path)
        except ValueError as error:
            print(f"skipping {path}: {error}")
    if not deliveries:
        raise SystemExit("no delivery could be processed.")

    batting_end = deliveries[0].calibration.batting_end
    panel_width = render_panel([], [], height, batting_end).shape[1]
    output_info = sv.VideoInfo(width=width + panel_width, height=height,
                               fps=video_info.fps)
    bounces, descriptions, results = [], [], []
    with sv.VideoSink(target_video_path, output_info) as sink:
        for path, delivery in zip(paths, deliveries):
            description = None
            if delivery.bounce_pitch_xy is not None:
                description = describe(delivery.bounce_pitch_xy,
                                       delivery.calibration.batting_end, batter_hand)
            for t, frame in enumerate(delivery.frames):
                bounced = delivery.bounce is not None and t >= delivery.bounce.frame_index
                if delivery.to_reference[t] is not None:
                    # Overlays live in reference frame pixels: map them onto frame t.
                    from_reference = np.linalg.inv(delivery.to_reference[t])
                    frame = draw_creases(frame.copy(), delivery.calibration, from_reference)
                    trail = delivery.trajectory[:t + 1].copy()
                    visible = ~np.isnan(trail).any(axis=1)
                    trail[visible] = transform(from_reference, trail[visible])
                    frame = draw_trail(frame, trail)
                    if bounced:
                        center = transform(from_reference, delivery.bounce.xy)[0]
                        cv2.circle(frame, tuple(int(v) for v in center), 10,
                                   BOUNCE_COLOR.as_bgr(), 2, cv2.LINE_AA)
                frame = cv2.resize(frame, (width, height))
                shown_bounces = bounces + ([delivery.bounce_pitch_xy] if bounced else [])
                shown = descriptions + ([description] if bounced else [])
                panel = render_panel(shown_bounces, shown, height, batting_end)
                output = np.hstack([frame, panel])
                sink.write_frame(output)
                if display:
                    cv2.imshow("pitch map", output)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        display = False
                        cv2.destroyAllWindows()

            if delivery.bounce_pitch_xy is not None:
                bounces.append(delivery.bounce_pitch_xy)
                descriptions.append(description)
            results.append({
                "clip": path,
                "detections": int((~np.isnan(delivery.trajectory).any(axis=1)).sum()),
                "bounce_frame": None if delivery.bounce is None
                else delivery.bounce.frame_index,
                "bounce_reference_xy": None if delivery.bounce is None
                else delivery.bounce.xy.tolist(),
                "bounce_pitch_xy": None if delivery.bounce_pitch_xy is None
                else delivery.bounce_pitch_xy.tolist(),
                "description": description,
            })
            print(f"{path}: {description or 'no bounce found'}")

    if display:
        cv2.destroyAllWindows()
    if results_path:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument('--source_video_path', type=str, required=True)
    calibrate_parser.add_argument('--calibration_path', type=str, required=True)
    calibrate_parser.add_argument('--frame_index', type=int, default=0)
    calibrate_parser.add_argument('--batting_end', type=int, default=2,
                                  help='2 when the camera is behind the bowler')

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument('--source_video_paths', type=str, nargs='+', required=True)
    calibration_group = run_parser.add_mutually_exclusive_group(required=True)
    calibration_group.add_argument('--calibration_path', type=str,
                                   help='calibration saved by `calibrate`')
    calibration_group.add_argument('--pitch_model', type=str,
                                   help='pitch keypoint model (best.pt) that calibrates '
                                        'every delivery automatically')
    run_parser.add_argument('--ball_model', type=str, default=None,
                            help='local model (.pt) or Roboflow model id, e.g. '
                                 'cricket-balls-l1us5/3; frame differencing is used '
                                 'when omitted')
    run_parser.add_argument('--ball_confidence', type=float, default=0.1,
                            help='low by default: linking detections into a trajectory '
                                 'filters out most false detections')
    run_parser.add_argument('--target_video_path', type=str, required=True)
    run_parser.add_argument('--batter_hand', type=BatterHand, default=BatterHand.RIGHT)
    run_parser.add_argument('--manual_fallback', action='store_true',
                            help='click the bounce when it is not found automatically')
    run_parser.add_argument('--manual_bounce', action='store_true',
                            help='click the bounce of every delivery instead of '
                                 'detecting the ball')
    run_parser.add_argument('--results_path', type=str, default=None)
    run_parser.add_argument('--bounces_path', type=str, default=None,
                            help='results of an earlier run: reuse its bounces, e.g. '
                                 'to re-render without selecting them again')
    run_parser.add_argument('--display', action='store_true')

    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(
            source_video_path=args.source_video_path,
            calibration_path=args.calibration_path,
            frame_index=args.frame_index,
            batting_end=args.batting_end
        )
    else:
        run(
            source_video_paths=args.source_video_paths,
            calibration_path=args.calibration_path,
            pitch_model_path=args.pitch_model,
            ball_model_id=args.ball_model,
            ball_confidence=args.ball_confidence,
            target_video_path=args.target_video_path,
            batter_hand=args.batter_hand,
            manual_fallback=args.manual_fallback,
            manual_bounce=args.manual_bounce,
            bounces_path=args.bounces_path,
            results_path=args.results_path,
            display=args.display
        )
