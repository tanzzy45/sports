"""
Generate synthetic behind-the-bowler cricket clips with known bounce positions.

Each clip shows one delivery on a pitch rendered in perspective, with a moving
bowler and sensor noise. The script also writes the matching calibration file and
the ground-truth bounce of every delivery, so `pitch_map.py` can be tested and its
error measured without real footage.
"""
import argparse
import json
import os
from typing import Tuple

import cv2
import numpy as np

from sports.cricket import CricketPitchConfiguration, draw_pitch

CONFIG = CricketPitchConfiguration()
CALIBRATION_LABELS = ["02", "03", "13", "14"]
GRAVITY = 981.0  # [cm/s^2]
RELEASE_HEIGHT = 220.0  # [cm]
BALL_SPEED = 3600.0  # [cm/s], about 130 km/h
BOUNCE_RESTITUTION = 0.55
BOUNCE_SPEED_RETENTION = 0.85
PITCH_SCALE = 0.5
PITCH_PADDING = 400


def camera_homography(width: int, height: int) -> np.ndarray:
    """Homography from pitch coordinates [cm] to image pixels."""
    labels = CONFIG.labels
    source = np.array(
        [CONFIG.vertices[labels.index(label)] for label in CALIBRATION_LABELS],
        dtype=np.float32)
    target = np.array([
        [0.40 * width, 0.86 * height],
        [0.60 * width, 0.86 * height],
        [0.475 * width, 0.30 * height],
        [0.525 * width, 0.30 * height],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(source, target)


def project(m: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(xy, m).reshape(-1, 2)


def render_background(m: np.ndarray, width: int, height: int) -> np.ndarray:
    grass = (34, 139, 34)
    pitch = draw_pitch(CONFIG, padding=PITCH_PADDING, scale=PITCH_SCALE)
    to_pitch_image = np.array([
        [PITCH_SCALE, 0, PITCH_PADDING],
        [0, PITCH_SCALE, PITCH_PADDING],
        [0, 0, 1],
    ])
    warp = m @ np.linalg.inv(to_pitch_image)
    background = cv2.warpPerspective(
        pitch, warp, (width, height), borderValue=grass)
    texture = np.random.default_rng(0).normal(0, 6, background.shape)
    return np.clip(background + texture, 0, 255).astype(np.uint8)


def ball_state(
    t: float,
    release_xy: np.ndarray,
    bounce_xy: np.ndarray,
    bounce_time: float
) -> Tuple[np.ndarray, float]:
    """Ground position [cm] and height [cm] of the ball `t` seconds after release."""
    direction = (bounce_xy - release_xy) / np.linalg.norm(bounce_xy - release_xy)
    vertical_speed = (0.5 * GRAVITY * bounce_time ** 2 - RELEASE_HEIGHT) / bounce_time
    if t <= bounce_time:
        ground = release_xy + direction * BALL_SPEED * t
        height = RELEASE_HEIGHT + vertical_speed * t - 0.5 * GRAVITY * t ** 2
        return ground, max(height, 0.0)

    impact_speed = vertical_speed - GRAVITY * bounce_time
    rebound_speed = -BOUNCE_RESTITUTION * impact_speed
    dt = t - bounce_time
    ground = bounce_xy + direction * BALL_SPEED * BOUNCE_SPEED_RETENTION * dt
    height = rebound_speed * dt - 0.5 * GRAVITY * dt ** 2
    return ground, max(height, 0.0)


def make_clip(
    path: str,
    m: np.ndarray,
    background: np.ndarray,
    bounce_xy: np.ndarray,
    fps: int,
    rng: np.random.Generator
) -> None:
    height, width = background.shape[:2]
    release_xy = np.array([
        CONFIG.popping_crease_x(1) + 30,
        CONFIG.middle_stump_y + rng.uniform(-40, -15)
    ])
    bounce_time = np.linalg.norm(bounce_xy - release_xy) / BALL_SPEED
    stumps_x = CONFIG.bowling_crease_x(2)
    lead_frames, total_frames = 12, 40
    phase = rng.uniform(0, 1)

    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(total_frames):
        frame = background.copy()

        bowler_progress = min(i, lead_frames) / lead_frames
        bowler_xy = (0.43 * width, (1.05 - 0.17 * bowler_progress) * height)
        cv2.ellipse(frame, (int(bowler_xy[0]), int(bowler_xy[1])), (28, 70), 0, 0,
                    360, (40, 40, 160), -1)

        t = (i - lead_frames + phase) / fps
        if t >= 0:
            ground, ball_height = ball_state(t, release_xy, bounce_xy, bounce_time)
            if ground[0] < stumps_x:
                ground_px = project(m, ground)[0]
                side_px = project(m, ground + np.array([0, 10]))[0]
                px_per_cm = np.linalg.norm(side_px - ground_px) / 10
                center = ground_px - np.array([0, ball_height * px_per_cm])
                radius = max(2, int(round(3.6 * px_per_cm * 1.5)))
                cv2.circle(frame, tuple(int(v) for v in center), radius,
                           (235, 235, 245), -1, lineType=cv2.LINE_AA)

        noise = rng.normal(0, 2, frame.shape)
        writer.write(np.clip(frame + noise, 0, 255).astype(np.uint8))
    writer.release()


def main(target_dir: str, deliveries: int, fps: int, width: int, height: int,
         seed: int) -> None:
    os.makedirs(target_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    m = camera_homography(width, height)
    background = render_background(m, width, height)

    labels = CONFIG.labels
    calibration = {
        "labels": CALIBRATION_LABELS,
        "image_points": project(m, np.array([
            CONFIG.vertices[labels.index(label)] for label in CALIBRATION_LABELS
        ])).tolist(),
        "batting_end": 2,
    }
    with open(os.path.join(target_dir, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)

    ground_truth = []
    stumps_x = CONFIG.bowling_crease_x(2)
    for index in range(deliveries):
        distance = rng.uniform(80, 1300)
        offset = rng.normal(-20, 25)
        bounce_xy = np.array([stumps_x - distance, CONFIG.middle_stump_y + offset])
        name = f"delivery_{index + 1:02d}.mp4"
        make_clip(os.path.join(target_dir, name), m, background, bounce_xy, fps, rng)
        ground_truth.append({"clip": name, "bounce_xy": bounce_xy.tolist()})

    with open(os.path.join(target_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"wrote {deliveries} clips, calibration.json and ground_truth.json "
          f"to {target_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target_dir', type=str, default='data/synthetic')
    parser.add_argument('--deliveries', type=int, default=6)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(
        target_dir=args.target_dir,
        deliveries=args.deliveries,
        fps=args.fps,
        width=args.width,
        height=args.height,
        seed=args.seed
    )
