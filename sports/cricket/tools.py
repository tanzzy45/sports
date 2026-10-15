from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from sports.cricket.config import CricketPitchConfiguration

BALL_RADIUS = 3.6  # [cm]


class BatterHand(Enum):
    RIGHT = "right"
    LEFT = "left"


@dataclass(frozen=True)
class LengthZone:
    """
    A band of the pitch measured from the batter's stumps toward the bowler.

    Attributes:
        name (str): Zone name, e.g. `"good"`.
        start (float): Distance from the batter's stumps where the zone starts [cm].
        end (float): Distance from the batter's stumps where the zone ends [cm].
    """
    name: str
    start: float
    end: float


# Length zones are a broadcast/analytics convention, not part of the Laws, and vary
# by provider and bowling speed. These defaults suit pace bowling.
DEFAULT_LENGTH_ZONES: List[LengthZone] = [
    LengthZone("yorker", 0, 200),
    LengthZone("full", 200, 600),
    LengthZone("good", 600, 800),
    LengthZone("back of length", 800, 1000),
    LengthZone("short", 1000, 2012),
]


class Line(Enum):
    OUTSIDE_LEG = "outside leg"
    STUMPS = "stumps"
    OUTSIDE_OFF = "outside off"


@dataclass(frozen=True)
class Bounce:
    """
    Bounce of the ball found in an image-space trajectory.

    Attributes:
        frame_index (float): Fractional frame index of the bounce.
        xy (np.ndarray): Image coordinates of the bounce, shape `(2,)`.
    """
    frame_index: float
    xy: np.ndarray


def distance_from_stumps(
    config: CricketPitchConfiguration,
    xy: np.ndarray,
    batting_end: int = 2
) -> np.ndarray:
    """
    Distance of pitch points from the batter's stumps, measured along the pitch.

    Args:
        config (CricketPitchConfiguration): Pitch configuration.
        xy (np.ndarray): Pitch coordinates of shape `(N, 2)` [cm].
        batting_end (int): End of the pitch where the batter stands (1 or 2).

    Returns:
        np.ndarray: Distances of shape `(N,)` [cm], positive toward the bowler.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    stumps_x = config.bowling_crease_x(batting_end)
    direction = 1 if batting_end == 1 else -1
    return direction * (xy[:, 0] - stumps_x)


def lateral_offset(
    config: CricketPitchConfiguration,
    xy: np.ndarray,
    batting_end: int = 2,
    batter_hand: BatterHand = BatterHand.RIGHT
) -> np.ndarray:
    """
    Signed distance of pitch points from the middle stump line.

    The batter's off side is the bowler's left for a right-handed batter and the
    bowler's right for a left-handed batter.

    Args:
        config (CricketPitchConfiguration): Pitch configuration.
        xy (np.ndarray): Pitch coordinates of shape `(N, 2)` [cm].
        batting_end (int): End of the pitch where the batter stands (1 or 2).
        batter_hand (BatterHand): Batting hand of the batter.

    Returns:
        np.ndarray: Offsets of shape `(N,)` [cm], positive toward the off side.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    # +y is the right of whoever looks from end 1 toward end 2, so a bowler at
    # end 1 (batting_end == 2) has their left at -y.
    bowler_left = -1 if batting_end == 2 else 1
    off_side = bowler_left if batter_hand == BatterHand.RIGHT else -bowler_left
    return off_side * (xy[:, 1] - config.middle_stump_y)


def classify_length(
    distance: float,
    zones: List[LengthZone] = DEFAULT_LENGTH_ZONES
) -> Optional[LengthZone]:
    """
    Find the length zone containing a distance from the batter's stumps.

    Args:
        distance (float): Distance from the batter's stumps [cm].
        zones (List[LengthZone]): Length zones to search.

    Returns:
        Optional[LengthZone]: The matching zone, or `None` if no zone contains it.
    """
    for zone in zones:
        if zone.start <= distance < zone.end:
            return zone
    return None


def classify_line(
    offset: float,
    config: CricketPitchConfiguration,
    ball_radius: float = BALL_RADIUS
) -> Line:
    """
    Classify the line of a delivery from its offset from the middle stump line.

    Args:
        offset (float): Offset from the middle stump line, positive toward the off
            side [cm], as returned by `lateral_offset`.
        config (CricketPitchConfiguration): Pitch configuration.
        ball_radius (float): Radius of the ball [cm]. A ball overlapping the
            stumps' width counts as on the stumps.

    Returns:
        Line: Line of the delivery.
    """
    limit = config.wicket_width / 2 + ball_radius
    if offset > limit:
        return Line.OUTSIDE_OFF
    if offset < -limit:
        return Line.OUTSIDE_LEG
    return Line.STUMPS


def _fit_line(t: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    slope, intercept = np.polyfit(t, v, 1)
    return float(slope), float(intercept)


def detect_bounce(
    xy: np.ndarray,
    window: int = 3,
    min_points_per_side: int = 2,
    min_slope_change: float = 3.0
) -> Optional[Bounce]:
    """
    Find where the ball bounces in its image-space trajectory.

    Assumes a camera behind the bowler, where the ball moves up the image toward
    the batter. Perspective makes it slow down smoothly on screen as it moves away,
    while the bounce makes its vertical image velocity jump upward. For every split
    between consecutive detections, a line is fitted to the nearest `window`
    detections on each side; the bounce is at the split with the largest upward
    velocity jump, where the two lines intersect (sub-frame accuracy).

    Args:
        xy (np.ndarray): Ball image coordinates of shape `(T, 2)`, one row per frame,
            with `NaN` rows for frames where the ball was not found.
        window (int): Number of detections used on each side of a split.
        min_points_per_side (int): Minimum detections before and after the bounce.
        min_slope_change (float): Minimum upward jump in vertical image velocity
            [px/frame] at the bounce.

    Returns:
        Optional[Bounce]: The bounce, or `None` if no bounce was found.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    valid = ~np.isnan(xy).any(axis=1)
    t = np.flatnonzero(valid).astype(float)
    points = xy[valid]
    n = len(t)
    if n < 2 * min_points_per_side:
        return None

    best = None
    for k in range(min_points_per_side, n - min_points_per_side + 1):
        before = slice(max(0, k - window), k)
        after = slice(k, min(n, k + window))
        slope_before, _ = _fit_line(t[before], points[before, 1])
        slope_after, _ = _fit_line(t[after], points[after, 1])
        slope_change = slope_before - slope_after
        if slope_change >= min_slope_change and (best is None or slope_change > best[0]):
            best = (slope_change, before, after)

    if best is None:
        return None

    _, before, after = best
    slope_before, intercept_before = _fit_line(t[before], points[before, 1])
    slope_after, intercept_after = _fit_line(t[after], points[after, 1])
    frame_index = (intercept_after - intercept_before) / (slope_before - slope_after)
    frame_index = float(np.clip(frame_index, t[before][-1], t[after][0]))

    x_slope_before, x_intercept_before = _fit_line(t[before], points[before, 0])
    x_slope_after, x_intercept_after = _fit_line(t[after], points[after, 0])
    x = np.mean([
        x_slope_before * frame_index + x_intercept_before,
        x_slope_after * frame_index + x_intercept_after,
    ])
    y = slope_before * frame_index + intercept_before
    return Bounce(frame_index=frame_index, xy=np.array([x, y]))
