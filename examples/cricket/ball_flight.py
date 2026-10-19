"""
Fit a physical flight to the ball's detections.

Pitch calibration maps the ground to the image. A broadcast camera turns and zooms
about a fixed point, so with square pixels and the principal point at the centre of
the image, the ground homography also gives the full camera: its focal length,
orientation and position. Points above the ground can then be projected too.

Positions are in pitch coordinates [cm] (see `sports.cricket.CricketPitchConfiguration`)
with the height above the ground; image points are in pixels of the delivery's
reference frame.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from sports.cricket import CricketPitchConfiguration

CONFIG = CricketPitchConfiguration()
GRAVITY = 981.0  # [cm/s^2]


@dataclass
class Camera:
    focal: float  # [px]
    rotation: np.ndarray  # 3x3, world to camera
    center: np.ndarray  # camera position in world coordinates [cm]
    principal_point: np.ndarray  # [px]
    up: float  # +1 or -1: the sign of the world z axis that points up

    @property
    def position(self) -> np.ndarray:
        """Camera position: pitch coordinates and height [cm]."""
        return np.array([self.center[0], self.center[1], self.up * self.center[2]])

    def project(self, points: np.ndarray) -> np.ndarray:
        """Image pixels of `(N, 3)` points: pitch coordinates and height [cm]."""
        world = np.column_stack([points[:, 0], points[:, 1], self.up * points[:, 2]])
        camera = (world - self.center) @ self.rotation.T
        return self.focal * camera[:, :2] / camera[:, 2:] + self.principal_point


def _project(params: np.ndarray, world: np.ndarray, principal_point: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(params[1:4])
    camera = (world - params[4:7]) @ rotation.T
    return params[0] * camera[:, :2] / camera[:, 2:] + principal_point


def camera_from_homography(pitch_to_image: np.ndarray, image_size: tuple) -> Camera:
    """
    The camera whose view of the ground is `pitch_to_image`, a homography from pitch
    coordinates [cm] to pixels, for an image of `(width, height)` pixels. Fitted to the
    pitch as the homography sees it: a closed-form estimate is too sensitive here, as
    a long, narrow pitch seen from far away constrains the camera poorly.
    """
    width, height = image_size
    principal_point = np.array([width / 2, height / 2])
    # Points over the whole pitch, as the calibration maps them.
    xs = np.linspace(CONFIG.bowling_crease_x(1), CONFIG.bowling_crease_x(2), 9)
    ys = np.linspace(CONFIG.middle_stump_y - 150, CONFIG.middle_stump_y + 150, 5)
    ground = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    target = cv2.perspectiveTransform(ground[np.newaxis], pitch_to_image)[0]
    world = np.column_stack([ground, np.zeros(len(ground))])

    def residuals(params: np.ndarray) -> np.ndarray:
        return (_project(params, world, principal_point) - target).ravel()

    # Start behind the bowler's end, looking down the pitch, then let the fit decide.
    start_x = CONFIG.bowling_crease_x(1) - 8000
    best = None
    for focal in (3000.0, 10000.0):
        for up in (1.0, -1.0):
            center = np.array([start_x, CONFIG.middle_stump_y, up * 1300.0])
            forward = np.array([CONFIG.bowling_crease_x(2), CONFIG.middle_stump_y, 0]) - center
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0, 0, up])
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            rotation = np.vstack([right, down, forward])
            if np.linalg.det(rotation) < 0:
                rotation[0] = -rotation[0]
            rotation_vector, _ = cv2.Rodrigues(rotation)
            fit = least_squares(residuals, np.r_[focal, rotation_vector.ravel(), center])
            if best is None or fit.cost < best.cost:
                best = fit
    focal, rotation_vector, center = best.x[0], best.x[1:4], best.x[4:7]
    rotation, _ = cv2.Rodrigues(rotation_vector)
    if focal < 0:  # the same camera, turned half a turn about its axis
        focal, rotation = -focal, np.diag([-1, -1, 1]) @ rotation
    return Camera(focal=focal, rotation=rotation, center=center,
                  principal_point=principal_point, up=float(np.sign(center[2])))


@dataclass
class Flight:
    bounce: np.ndarray  # pitch coordinates of the bounce [cm]
    bounce_time: float  # [s]
    inliers: np.ndarray  # frames whose detection the flight explains
    residual: float  # median distance of those detections from the flight [px]
    params: np.ndarray

    @property
    def bounce_point(self) -> np.ndarray:
        return np.array([self.bounce[0], self.bounce[1], 0.0])

    def positions(self, times: np.ndarray) -> np.ndarray:
        """Ball positions (pitch coordinates and height [cm]) at `times` [s]."""
        return _positions(self.params, np.asarray(times, dtype=float))


def _positions(params: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Ball positions (pitch coordinates and height [cm]) at `times` [s]."""
    xb, yb, tb, vx, vy, vz_in, retention, deviation, vz_out = params
    dt = times - tb
    after = dt >= 0
    cos, sin = np.cos(deviation), np.sin(deviation)
    vx_out = retention * (cos * vx - sin * vy)
    vy_out = retention * (sin * vx + cos * vy)
    x = xb + np.where(after, vx_out, vx) * dt
    y = yb + np.where(after, vy_out, vy) * dt
    z = np.where(after, vz_out, vz_in) * dt - 0.5 * GRAVITY * dt ** 2
    return np.column_stack([x, y, z])


def _release_height(params: np.ndarray, release_x: float) -> float:
    """Height [cm] of the incoming flight above the bowler's popping crease."""
    xb, _, _, vx, _, vz_in = params[:6]
    dt = (release_x - xb) / vx
    return vz_in * dt - 0.5 * GRAVITY * dt ** 2


def fit_flight(trajectory: np.ndarray, times: np.ndarray, camera: Camera, batting_end: int,
               image_to_pitch: np.ndarray, tolerance: float = 3.0, min_before: int = 3,
               min_after: int = 2, release_height: float = 210.0,
               release_spread: float = 30.0) -> Optional[Flight]:
    """
    The flight from the bowler to the batter that best explains the detections in
    `trajectory` (`(T, 2)` reference-frame pixels, `NaN` where not found), shown at
    `times` (`(T,)` seconds).

    Before the bounce the ball moves in a straight line seen from above and falls
    under gravity, from about `release_height` [cm] above the bowler's popping crease;
    at the bounce, on the ground, it keeps its line within a few degrees, loses some
    pace and rebounds. Only the flight up to the batter's popping crease is modelled,
    so what happens after the shot cannot be taken for the bounce. Detections more
    than about `tolerance` pixels off the flight count little; `None` when fewer than
    `min_before` detections before the bounce or `min_after` after it are explained.
    """
    frames = np.flatnonzero(~np.isnan(trajectory).any(axis=1))
    if len(frames) < min_before + min_after:
        return None
    observed = trajectory[frames]
    seen = np.asarray(times, dtype=float)[frames]
    toward = 1.0 if batting_end == 2 else -1.0
    release_x = CONFIG.popping_crease_x(3 - batting_end)
    start_x = release_x - toward * 150
    end_x = CONFIG.popping_crease_x(batting_end)
    on_ground = cv2.perspectiveTransform(observed[np.newaxis].astype(np.float64),
                                         image_to_pitch)[0]

    def in_flight(params: np.ndarray) -> np.ndarray:
        x = _positions(params, seen)[:, 0]
        return (toward * (x - start_x) >= 0) & (toward * (end_x - x) >= 0)

    def residuals(params: np.ndarray) -> np.ndarray:
        error = camera.project(_positions(params, seen)) - observed
        # Detections outside the modelled flight neither help nor hurt.
        error = np.where(in_flight(params)[:, None], error, 0.0).ravel()
        release = (_release_height(params, release_x) - release_height) / release_spread
        return np.append(error, tolerance * release)

    # Bounce at least 4 m from the bowler's popping crease and within 3 m of the middle
    # stump line, 36-180 km/h along the pitch, a deviation of up to about 9 degrees.
    my = CONFIG.middle_stump_y
    near_x = release_x + toward * 400
    slowest, fastest = (1000, 5000) if toward > 0 else (-5000, -1000)
    lower = [min(near_x, end_x), my - 300, seen[0], slowest, -600, -2500, 0.5, -0.15, 30]
    upper = [max(near_x, end_x), my + 300, seen[-1], fastest, 600, -30, 1.0, 0.15, 2500]
    best, best_key = None, None
    for i in range(1, len(frames) - 1):
        for speed in (2200.0, 3600.0):
            guess = np.array([*np.clip(on_ground[i], [lower[0], lower[1]], [upper[0], upper[1]]),
                              seen[i], toward * speed, 0.0, -400.0, 0.8, 0.0, 300.0])
            guess = np.clip(guess, np.array(lower) + 1e-6, np.array(upper) - 1e-6)
            fit = least_squares(residuals, guess, bounds=(lower, upper), loss="soft_l1",
                                f_scale=tolerance)
            error = np.linalg.norm(camera.project(_positions(fit.x, seen)) - observed, axis=1)
            inlier = (error <= 3 * tolerance) & in_flight(fit.x)
            before = int((inlier & (seen < fit.x[2])).sum())
            after = int((inlier & (seen >= fit.x[2])).sum())
            if before < min_before or after < min_after:
                continue
            key = (before + after, -float(np.median(error[inlier])))
            if best_key is None or key > best_key:
                best, best_key = (fit.x, inlier, error), key
    if best is None:
        return None
    params, inlier, error = best
    return Flight(bounce=params[:2].copy(), bounce_time=float(params[2]),
                  inliers=frames[inlier], residual=float(np.median(error[inlier])),
                  params=params)
