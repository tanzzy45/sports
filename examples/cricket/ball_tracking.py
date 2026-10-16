"""
Follow the ball from the bowler's hand.

Finding a ball of a few pixels anywhere in a broadcast frame is hard. Knowing where
it starts makes it much easier: a pose model finds the bowler and the release, the
moment the bowling wrist is highest, and from there the ball is searched for only in a
small window along its predicted path. The window is cropped and enlarged before
running the ball detector, so the ball is bigger and little else in the window can be
mistaken for it, which lets weak detections be accepted.

Positions are in pixels of the delivery's reference frame (see camera_motion.py).
"""
from typing import Callable, List, Optional, Tuple

import numpy as np

from camera_motion import transform
from sports.cricket import CricketPitchConfiguration

CONFIG = CricketPitchConfiguration()
LEFT_WRIST, RIGHT_WRIST = 9, 10  # COCO keypoints


class ReleaseBallTracker:
    """
    Args:
        ball_model_path (str): Ball detector, e.g. trained with train_ball_detector.ipynb.
        pose_model_path (str): Person pose model with COCO keypoints.
        search_confidence (float): Lowest ball confidence accepted inside the window.
        zoom (float): How much the window is enlarged before detection.
        max_misses (int): Frames without the ball before the search stops.
        min_flight (float): Shortest flight from release to the batter [s].
        max_flight (float): Longest flight from release to the batter [s].
        max_releases (int): Release candidates to follow the ball from.
        release_gap (float): Shortest time between two release candidates [s].
        min_extent (float): Shortest track kept, as a fraction of the pitch on screen;
            shorter tracks are usually the bowler's hand.
    """

    def __init__(self, ball_model_path: str, pose_model_path: str = "yolov8s-pose.pt",
                 search_confidence: float = 0.02, zoom: float = 2.0, max_misses: int = 4,
                 min_flight: float = 0.35, max_flight: float = 1.2,
                 max_releases: int = 4, release_gap: float = 0.3,
                 min_extent: float = 0.15) -> None:
        from ultralytics import YOLO

        self.ball = YOLO(ball_model_path)
        self.pose = YOLO(pose_model_path)
        self.search_confidence = search_confidence
        self.zoom = zoom
        self.max_misses = max_misses
        self.min_flight = min_flight
        self.max_flight = max_flight
        self.max_releases = max_releases
        self.release_gap = release_gap
        self.min_extent = min_extent

    def find_releases(self, frames: List[np.ndarray],
                      to_reference: List[Optional[np.ndarray]], calibration,
                      fps: float) -> List[Tuple[int, np.ndarray]]:
        """
        Candidate releases, highest first: frames where a wrist of a person standing near
        the bowler's popping crease peaks. The highest wrist alone is not enough, since
        the front arm is raised as high as the bowling arm in the gather, so every peak
        at least `release_gap` seconds from a higher one is a candidate.
        """
        popping_x = CONFIG.popping_crease_x(3 - calibration.batting_end)
        direction = 1 if calibration.batting_end == 2 else -1
        highest = {}
        for t, (frame, homography) in enumerate(zip(frames, to_reference)):
            if homography is None:
                continue
            result = self.pose(frame, imgsz=640, verbose=False)[0]
            if result.keypoints is None or len(result.boxes) == 0 or result.keypoints.conf is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy()
            confidences = result.keypoints.conf.cpu().numpy()
            for box, points, confidence in zip(boxes, keypoints, confidences):
                feet = transform(homography, np.array([[(box[0] + box[2]) / 2, box[3]]]))
                x, y = calibration.image_to_pitch.transform_points(feet.astype(np.float32))[0]
                along = direction * (x - popping_x)  # positive toward the batter
                if not (-400 <= along <= 250 and abs(y - CONFIG.middle_stump_y) <= 300):
                    continue  # not where the bowler delivers from
                wrists = [points[i] for i in (LEFT_WRIST, RIGHT_WRIST) if confidence[i] >= 0.3]
                if not wrists:
                    continue
                wrist = transform(homography, np.array([min(wrists, key=lambda p: p[1])]))[0]
                if t not in highest or wrist[1] < highest[t][1]:
                    highest[t] = wrist
        gap = self.release_gap * fps
        releases = []
        for t, wrist in sorted(highest.items(), key=lambda item: item[1][1]):
            if all(abs(t - other) >= gap for other, _ in releases):
                releases.append((t, wrist))
        return releases[:self.max_releases]

    def _search(self, frame: np.ndarray, homography: np.ndarray, center: np.ndarray,
                radius: float) -> List[Tuple[np.ndarray, float]]:
        """Ball candidates as (reference position, confidence) within `radius`."""
        from_reference = np.linalg.inv(homography)
        center_frame = transform(from_reference, center[np.newaxis])[0]
        scale = 1 / np.sqrt(abs(np.linalg.det(homography[:2, :2])))  # frame px per reference px
        half = int(np.clip(radius * scale + 16, 32, 320))
        height, width = frame.shape[:2]
        x0, y0 = int(max(center_frame[0] - half, 0)), int(max(center_frame[1] - half, 0))
        x1, y1 = int(min(center_frame[0] + half, width)), int(min(center_frame[1] + half, height))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []
        crop = frame[y0:y1, x0:x1]
        imgsz = int(np.ceil(max(crop.shape[:2]) * self.zoom / 32) * 32)
        result = self.ball(crop, imgsz=imgsz, conf=self.search_confidence, verbose=False)[0]
        candidates = []
        for (bx0, by0, bx1, by1), confidence in zip(result.boxes.xyxy.cpu().numpy(),
                                                    result.boxes.conf.cpu().numpy()):
            point = np.array([[x0 + (bx0 + bx1) / 2, y0 + (by0 + by1) / 2]])
            reference = transform(homography, point)[0]
            if np.linalg.norm(reference - center) <= radius:
                candidates.append((reference, float(confidence)))
        return candidates

    def track(self, frames: List[np.ndarray], to_reference: List[Optional[np.ndarray]],
              calibration, fps: float,
              accept: Optional[Callable[[np.ndarray], bool]] = None) -> np.ndarray:
        """
        Ball trajectory in reference pixels, shape `(T, 2)`, `NaN` where not found.

        The ball is followed from every release candidate. Tracks that barely move are
        dropped; of the others, the longest one that `accept` accepts (e.g. one with a
        bounce on the pitch) is returned, or the longest one if none is accepted.
        """
        ends = np.array([[CONFIG.popping_crease_x(3 - calibration.batting_end),
                          CONFIG.middle_stump_y],
                         [CONFIG.bowling_crease_x(calibration.batting_end),
                          CONFIG.middle_stump_y]], dtype=np.float32)
        popping, target = calibration.pitch_to_image.transform_points(ends)
        pitch_length = np.linalg.norm(target - popping)
        tracks = []
        for start, origin in self.find_releases(frames, to_reference, calibration, fps):
            trajectory = self._follow(frames, to_reference, start, origin, target,
                                      pitch_length, fps)
            points = trajectory[~np.isnan(trajectory).any(axis=1)]
            extent = 0.0 if len(points) < 2 else float(np.linalg.norm(points.max(0) - points.min(0)))
            if extent >= self.min_extent * pitch_length:
                tracks.append((extent, trajectory))
        tracks.sort(key=lambda track: -track[0])
        for _, trajectory in tracks:
            if accept is None or accept(trajectory):
                return trajectory
        return tracks[0][1] if tracks else np.full((len(frames), 2), np.nan)

    def _follow(self, frames: List[np.ndarray], to_reference: List[Optional[np.ndarray]],
                start: int, origin: np.ndarray, target: np.ndarray, pitch_length: float,
                fps: float) -> np.ndarray:
        """Follow the ball from a release at frame `start` and reference position `origin`."""
        trajectory = np.full((len(frames), 2), np.nan)
        # The ball is in the air, so its image does not head straight for the stumps:
        # it drops toward its bounce first. Only its speed on screen is bounded, by
        # the length of the pitch on screen over the shortest flight.
        fastest = pitch_length / (self.min_flight * fps)  # reference px per frame
        margin = 0.04 * pitch_length

        position, velocity, last, hits, misses = origin, None, start, 0, 0
        for t in range(start + 1, min(len(frames), start + int(self.max_flight * fps) + 1)):
            homography = to_reference[t]
            if homography is None:
                break
            elapsed = t - last
            if hits == 0:
                # Close to the hand, anywhere the ball can have reached.
                near, far = origin, origin
                radius = 0.1 * pitch_length + 0.5 * fastest * elapsed
            elif velocity is None:
                near, far = position, position
                radius = fastest * elapsed + margin
            else:
                # Along the current direction, or toward the batter after a bounce.
                near = position + velocity * elapsed
                toward = (target - position) / max(np.linalg.norm(target - position), 1e-6)
                far = position + toward * max(np.linalg.norm(velocity), fastest / 2) * elapsed
                radius = max(margin, 0.6 * np.linalg.norm(velocity) * elapsed)
            center = (near + far) / 2
            candidates = self._search(frames[t], homography, center,
                                      radius + np.linalg.norm(far - near) / 2)
            if candidates:
                point = max(candidates, key=lambda c: c[1] - _segment_distance(
                    c[0], near, far) / radius)[0]
                if hits > 0:
                    measured = (point - position) / elapsed
                    velocity = measured if velocity is None else 0.5 * velocity + 0.5 * measured
                position, last = point, t
                trajectory[t] = point
                hits, misses = hits + 1, 0
                points = trajectory[~np.isnan(trajectory).any(axis=1)]
                if hits >= 3 and np.ptp(points, axis=0).max() < margin:
                    # Every hit so far in one spot: the bowler's hand, not the ball.
                    # Start over and look for the ball leaving it.
                    trajectory[:] = np.nan
                    position, velocity, hits = origin, None, 0
            else:
                misses += 1
                if misses > self.max_misses:
                    break
        return trajectory


def _segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    s = 0.0 if not ab.any() else float(np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(point - (a + s * ab)))
