from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from sports.cricket.config import CricketPitchConfiguration
from sports.cricket.tools import (
    DEFAULT_LENGTH_ZONES,
    LengthZone,
    classify_length,
    distance_from_stumps,
)

DEFAULT_ZONE_COLORS: List[sv.Color] = [
    sv.Color.from_hex("#8E44AD"),
    sv.Color.from_hex("#3498DB"),
    sv.Color.from_hex("#2ECC71"),
    sv.Color.from_hex("#F39C12"),
    sv.Color.from_hex("#E74C3C"),
]


def _to_pixel(
    point: Tuple[float, float],
    scale: float,
    padding: int
) -> Tuple[int, int]:
    return int(point[0] * scale) + padding, int(point[1] * scale) + padding


def _draw_markings(
    config: CricketPitchConfiguration,
    image: np.ndarray,
    line_color: sv.Color,
    stump_color: sv.Color,
    line_thickness: int,
    stump_radius: int,
    padding: int,
    scale: float
) -> np.ndarray:
    vertices = config.vertices
    for start, end in config.edges:
        cv2.line(
            img=image,
            pt1=_to_pixel(vertices[start - 1], scale, padding),
            pt2=_to_pixel(vertices[end - 1], scale, padding),
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    stump_indices = [8, 9, 10, 19, 20, 21]
    for index in stump_indices:
        cv2.circle(
            img=image,
            center=_to_pixel(vertices[index], scale, padding),
            radius=stump_radius,
            color=stump_color.as_bgr(),
            thickness=-1
        )

    return image


def draw_pitch(
    config: CricketPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    pitch_color: sv.Color = sv.Color(196, 168, 120),
    line_color: sv.Color = sv.Color.WHITE,
    stump_color: sv.Color = sv.Color.from_hex("#FFD700"),
    padding: int = 50,
    line_thickness: int = 2,
    stump_radius: int = 3,
    scale: float = 0.5
) -> np.ndarray:
    """
    Draws a top-down view of a cricket pitch with its creases and stumps.

    Args:
        config (CricketPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        background_color (sv.Color, optional): Color of the grass around the pitch.
            Defaults to sv.Color(34, 139, 34).
        pitch_color (sv.Color, optional): Color of the pitch strip.
            Defaults to sv.Color(196, 168, 120).
        line_color (sv.Color, optional): Color of the crease lines.
            Defaults to sv.Color.WHITE.
        stump_color (sv.Color, optional): Color of the stumps.
            Defaults to `#FFD700`.
        padding (int, optional): Padding around the markings in pixels.
            Defaults to 50.
        line_thickness (int, optional): Thickness of the crease lines in pixels.
            Defaults to 2.
        stump_radius (int, optional): Radius of the stumps in pixels.
            Defaults to 3.
        scale (float, optional): Pixels per centimeter. Defaults to 0.5.

    Returns:
        np.ndarray: Image of the cricket pitch.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # The prepared strip extends past the creases, so draw it over the full length
    # of the markings rather than only between the bowling creases.
    half_pitch_width = config.pitch_width / 2
    cv2.rectangle(
        img=pitch_image,
        pt1=_to_pixel((0, config.middle_stump_y - half_pitch_width), scale, padding),
        pt2=_to_pixel(
            (config.length, config.middle_stump_y + half_pitch_width), scale, padding),
        color=pitch_color.as_bgr(),
        thickness=-1
    )

    return _draw_markings(
        config=config,
        image=pitch_image,
        line_color=line_color,
        stump_color=stump_color,
        line_thickness=line_thickness,
        stump_radius=stump_radius,
        padding=padding,
        scale=scale
    )


def draw_points_on_pitch(
    config: CricketPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 6,
    thickness: int = 1,
    padding: int = 50,
    scale: float = 0.5,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a cricket pitch.

    Args:
        config (CricketPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        xy (np.ndarray): Array of points to be drawn, with each point represented by
            its (x, y) coordinates in centimeters.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels. Defaults to 6.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 1.
        padding (int, optional): Padding around the markings in pixels.
            Defaults to 50.
        scale (float, optional): Pixels per centimeter. Defaults to 0.5.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw points
            on. If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the cricket pitch with points drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    for point in xy:
        center = _to_pixel(point, scale, padding)
        cv2.circle(
            img=pitch,
            center=center,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=pitch,
            center=center,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch


def draw_paths_on_pitch(
    config: CricketPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.5,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws paths on a cricket pitch.

    Args:
        config (CricketPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        paths (List[np.ndarray]): List of paths, where each path is an array of
            (x, y) coordinates in centimeters.
        color (sv.Color, optional): Color of the paths. Defaults to sv.Color.WHITE.
        thickness (int, optional): Thickness of the paths in pixels. Defaults to 2.
        padding (int, optional): Padding around the markings in pixels.
            Defaults to 50.
        scale (float, optional): Pixels per centimeter. Defaults to 0.5.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw paths
            on. If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the cricket pitch with paths drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    for path in paths:
        scaled_path = [
            _to_pixel(point, scale, padding)
            for point in path if point.size > 0
        ]
        for start, end in zip(scaled_path, scaled_path[1:]):
            cv2.line(
                img=pitch,
                pt1=start,
                pt2=end,
                color=color.as_bgr(),
                thickness=thickness
            )

    return pitch


def draw_length_zones(
    config: CricketPitchConfiguration,
    batting_end: int = 2,
    zones: List[LengthZone] = DEFAULT_LENGTH_ZONES,
    colors: List[sv.Color] = DEFAULT_ZONE_COLORS,
    opacity: float = 0.35,
    line_color: sv.Color = sv.Color.WHITE,
    stump_color: sv.Color = sv.Color.from_hex("#FFD700"),
    line_thickness: int = 2,
    stump_radius: int = 3,
    padding: int = 50,
    scale: float = 0.5,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Shades the length zones on the pitch strip, measured from the batter's stumps.
    Crease lines and stumps are redrawn on top of the shading.

    Args:
        config (CricketPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        batting_end (int, optional): End of the pitch where the batter stands
            (1 or 2). Defaults to 2.
        zones (List[LengthZone], optional): Length zones to shade.
            Defaults to DEFAULT_LENGTH_ZONES.
        colors (List[sv.Color], optional): One color per zone.
            Defaults to DEFAULT_ZONE_COLORS.
        opacity (float, optional): Opacity of the shading. Defaults to 0.35.
        line_color (sv.Color, optional): Color of the redrawn crease lines.
            Defaults to sv.Color.WHITE.
        stump_color (sv.Color, optional): Color of the redrawn stumps.
            Defaults to `#FFD700`.
        line_thickness (int, optional): Thickness of the crease lines in pixels.
            Defaults to 2.
        stump_radius (int, optional): Radius of the stumps in pixels.
            Defaults to 3.
        padding (int, optional): Padding around the markings in pixels.
            Defaults to 50.
        scale (float, optional): Pixels per centimeter. Defaults to 0.5.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw the
            zones on. If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the cricket pitch with the length zones shaded.
    """
    if len(colors) < len(zones):
        raise ValueError("colors must contain one color per zone.")
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    stumps_x = config.bowling_crease_x(batting_end)
    direction = 1 if batting_end == 1 else -1
    y1 = config.middle_stump_y - config.pitch_width / 2
    y2 = config.middle_stump_y + config.pitch_width / 2

    overlay = pitch.copy()
    for zone, color in zip(zones, colors):
        x1 = stumps_x + direction * zone.start
        x2 = stumps_x + direction * min(zone.end, config.pitch_length)
        cv2.rectangle(
            img=overlay,
            pt1=_to_pixel((x1, y1), scale, padding),
            pt2=_to_pixel((x2, y2), scale, padding),
            color=color.as_bgr(),
            thickness=-1
        )

    shaded = cv2.addWeighted(overlay, opacity, pitch, 1 - opacity, 0)
    return _draw_markings(
        config=config,
        image=shaded,
        line_color=line_color,
        stump_color=stump_color,
        line_thickness=line_thickness,
        stump_radius=stump_radius,
        padding=padding,
        scale=scale
    )


def draw_pitch_map(
    config: CricketPitchConfiguration,
    bounce_xy: np.ndarray,
    batting_end: int = 2,
    zones: List[LengthZone] = DEFAULT_LENGTH_ZONES,
    colors: List[sv.Color] = DEFAULT_ZONE_COLORS,
    radius: int = 6,
    padding: int = 50,
    scale: float = 0.5,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws a pitch map: bounce positions colored by the length zone they landed in.

    Args:
        config (CricketPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        bounce_xy (np.ndarray): Bounce positions of shape `(N, 2)` in centimeters.
        batting_end (int, optional): End of the pitch where the batter stands
            (1 or 2). Defaults to 2.
        zones (List[LengthZone], optional): Length zones used for coloring.
            Defaults to DEFAULT_LENGTH_ZONES.
        colors (List[sv.Color], optional): One color per zone.
            Defaults to DEFAULT_ZONE_COLORS.
        radius (int, optional): Radius of the bounce markers in pixels.
            Defaults to 6.
        padding (int, optional): Padding around the markings in pixels.
            Defaults to 50.
        scale (float, optional): Pixels per centimeter. Defaults to 0.5.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw the
            pitch map on. If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the pitch map.
    """
    pitch = draw_length_zones(
        config=config,
        batting_end=batting_end,
        zones=zones,
        colors=colors,
        padding=padding,
        scale=scale,
        pitch=pitch
    )

    bounce_xy = np.asarray(bounce_xy, dtype=float).reshape(-1, 2)
    distances = distance_from_stumps(config, bounce_xy, batting_end=batting_end)
    for point, distance in zip(bounce_xy, distances):
        zone = classify_length(distance, zones)
        face_color = colors[zones.index(zone)] if zone else sv.Color.WHITE
        pitch = draw_points_on_pitch(
            config=config,
            xy=point[np.newaxis],
            face_color=face_color,
            edge_color=sv.Color.WHITE,
            radius=radius,
            thickness=2,
            padding=padding,
            scale=scale,
            pitch=pitch
        )

    return pitch
