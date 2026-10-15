"""
Build a cricket pitch map from behind-the-bowler video, without trained models.

1. `calibrate` - click four crease points once to compute the homography between
   the image and the pitch. The camera barely moves during a delivery, so one
   calibration per camera position is enough.
2. `run` - find the ball by three-frame differencing inside the pitch area, link the
   detections into a trajectory, find the bounce, project it onto the pitch, and
   render the video next to a live pitch map.
"""
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

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

PITCH_MAP_VISIBLE_LENGTH = 1500  # [cm] in front of the batter's stumps
PITCH_MAP_PADDING = 30
PANEL_TEXT_WIDTH = 340
TRAIL_COLOR = sv.Color.from_hex("#FFD700")
BOUNCE_COLOR = sv.Color.from_hex("#FF1493")
CREASE_COLOR = sv.Color.from_hex("#00BFFF")


@dataclass
class Calibration:
    image_to_pitch: ViewTransformer
    pitch_to_image: ViewTransformer
    batting_end: int


@dataclass
class Delivery:
    frames: List[np.ndarray]
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


def load_calibration(calibration_path: str) -> Calibration:
    with open(calibration_path) as f:
        data = json.load(f)
    image_points = np.array(data["image_points"], dtype=np.float32)
    pitch_points = np.array(
        [CONFIG.vertices[CONFIG.labels.index(label)] for label in data["labels"]],
        dtype=np.float32)
    return Calibration(
        image_to_pitch=ViewTransformer(source=image_points, target=pitch_points),
        pitch_to_image=ViewTransformer(source=pitch_points, target=image_points),
        batting_end=data.get("batting_end", 2),
    )


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


class MotionBallDetector:
    """
    Finds small moving blobs with three-frame differencing: a pixel counts as moving
    at frame t only if it differs from both frame t-1 and frame t+1, which keeps the
    ball's current position and drops where it was or will be.
    """

    def __init__(self, mask: np.ndarray, threshold: int = 25, min_area: int = 3,
                 max_area: int = 400) -> None:
        self.mask = mask
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    def detect(self, previous: np.ndarray, current: np.ndarray,
               following: np.ndarray) -> np.ndarray:
        previous, current, following = map(self._gray, (previous, current, following))
        motion = cv2.min(cv2.absdiff(current, previous), cv2.absdiff(following, current))
        _, binary = cv2.threshold(motion, self.threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_and(binary, self.mask)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
        areas = stats[1:count, cv2.CC_STAT_AREA]
        keep = (areas >= self.min_area) & (areas <= self.max_area)
        return centroids[1:count][keep]


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
                           trajectory: np.ndarray) -> Optional[Bounce]:
    """Step through frames and click where the ball bounced."""
    index, selected = 0, []
    window = "bounce: [a]/[d] prev/next frame, click bounce, [enter] ok, [esc] skip"

    def on_click(event: int, x: int, y: int, *_) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected[:] = [(index, x, y)]

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_click)
    while True:
        canvas = draw_trail(frames[index].copy(), trajectory[:index + 1])
        if selected:
            cv2.circle(canvas, selected[0][1:], 8, BOUNCE_COLOR.as_bgr(), 2)
        cv2.putText(canvas, f"frame {index + 1}/{len(frames)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("a"):
            index = max(0, index - 1)
        elif key == ord("d"):
            index = min(len(frames) - 1, index + 1)
        elif key == 13 and selected:
            break
        elif key == 27:
            selected.clear()
            break
    cv2.destroyWindow(window)
    if not selected:
        return None
    frame_index, x, y = selected[0]
    return Bounce(frame_index=float(frame_index), xy=np.array([x, y], dtype=float))


def process_delivery(source_video_path: str, calibration: Calibration,
                     manual_fallback: bool) -> Delivery:
    frames = read_frames(source_video_path)
    height, width = frames[0].shape[:2]
    detector = MotionBallDetector(
        mask=pitch_region_mask((height, width), calibration))

    candidates = [np.empty((0, 2))]
    for previous, current, following in zip(frames, frames[1:], frames[2:]):
        candidates.append(detector.detect(previous, current, following))
    candidates.append(np.empty((0, 2)))

    trajectory = link_trajectory(candidates, max_jump=0.08 * width)
    bounce = detect_bounce(trajectory)
    if bounce is None and manual_fallback:
        print(f"no bounce found in {source_video_path}, select it manually")
        bounce = select_bounce_manually(frames, trajectory)

    bounce_pitch_xy = None
    if bounce is not None:
        bounce_pitch_xy = calibration.image_to_pitch.transform_points(
            bounce.xy[np.newaxis].astype(np.float32))[0]
    return Delivery(frames, trajectory, bounce, bounce_pitch_xy)


def draw_trail(frame: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    points = [tuple(int(v) for v in p) for p in trajectory if not np.isnan(p).any()]
    for start, end in zip(points, points[1:]):
        cv2.line(frame, start, end, TRAIL_COLOR.as_bgr(), 2, cv2.LINE_AA)
    for point in points[-1:]:
        cv2.circle(frame, point, 6, TRAIL_COLOR.as_bgr(), 2, cv2.LINE_AA)
    return frame


def draw_creases(frame: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Project the crease lines back onto the frame to show the calibration."""
    vertices = np.array(CONFIG.vertices, dtype=np.float32)
    projected = calibration.pitch_to_image.transform_points(vertices).astype(int)
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


def run(source_video_paths: List[str], calibration_path: str, target_video_path: str,
        batter_hand: BatterHand, manual_fallback: bool, results_path: Optional[str],
        display: bool) -> None:
    calibration = load_calibration(calibration_path)
    video_info = sv.VideoInfo.from_video_path(source_video_paths[0])
    width, height = video_info.width, video_info.height

    deliveries = []
    for path in tqdm(source_video_paths, desc="processing deliveries"):
        deliveries.append(process_delivery(path, calibration, manual_fallback))

    panel_width = render_panel([], [], height, calibration.batting_end).shape[1]
    output_info = sv.VideoInfo(width=width + panel_width, height=height,
                               fps=video_info.fps)
    bounces, descriptions, results = [], [], []
    with sv.VideoSink(target_video_path, output_info) as sink:
        for path, delivery in zip(source_video_paths, deliveries):
            description = None
            if delivery.bounce_pitch_xy is not None:
                description = describe(delivery.bounce_pitch_xy,
                                       calibration.batting_end, batter_hand)
            for t, frame in enumerate(delivery.frames):
                frame = cv2.resize(frame, (width, height))
                frame = draw_creases(frame, calibration)
                frame = draw_trail(frame, delivery.trajectory[:t + 1])
                bounced = delivery.bounce is not None and t >= delivery.bounce.frame_index
                if bounced:
                    center = tuple(int(v) for v in delivery.bounce.xy)
                    cv2.circle(frame, center, 10, BOUNCE_COLOR.as_bgr(), 2, cv2.LINE_AA)
                shown_bounces = bounces + ([delivery.bounce_pitch_xy] if bounced else [])
                shown = descriptions + ([description] if bounced else [])
                panel = render_panel(shown_bounces, shown, height,
                                     calibration.batting_end)
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
                "bounce_frame": None if delivery.bounce is None
                else delivery.bounce.frame_index,
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
    run_parser.add_argument('--calibration_path', type=str, required=True)
    run_parser.add_argument('--target_video_path', type=str, required=True)
    run_parser.add_argument('--batter_hand', type=BatterHand, default=BatterHand.RIGHT)
    run_parser.add_argument('--manual_fallback', action='store_true')
    run_parser.add_argument('--results_path', type=str, default=None)
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
            target_video_path=args.target_video_path,
            batter_hand=args.batter_hand,
            manual_fallback=args.manual_fallback,
            results_path=args.results_path,
            display=args.display
        )
