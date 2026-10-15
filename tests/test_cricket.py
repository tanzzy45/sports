import numpy as np
import pytest
import supervision as sv

from sports.cricket import (
    BOUNCER_LENGTH_ZONES,
    DEFAULT_LENGTH_ZONES,
    DEFAULT_ZONE_COLORS,
    BatterHand,
    CricketPitchConfiguration,
    Line,
    classify_length,
    classify_line,
    detect_bounce,
    distance_from_stumps,
    draw_paths_on_pitch,
    draw_pitch,
    draw_pitch_map,
    lateral_offset,
)

CONFIG = CricketPitchConfiguration()
SCALE = 0.5
PADDING = 50


def _vertex(number: int) -> np.ndarray:
    return np.array(CONFIG.vertices[number - 1])


def _to_pixel(x: float, y: float) -> tuple:
    return int(x * SCALE) + PADDING, int(y * SCALE) + PADDING


def test_metadata_has_one_entry_per_vertex() -> None:
    assert len(CONFIG.vertices) == 22
    assert len(CONFIG.labels) == 22
    assert len(CONFIG.colors) == 22
    assert len(set(CONFIG.labels)) == 22
    for start, end in CONFIG.edges:
        assert 1 <= start <= 22 and 1 <= end <= 22


def test_vertices_lie_within_extents() -> None:
    vertices = np.array(CONFIG.vertices)
    assert vertices[:, 0].min() == 0
    assert vertices[:, 0].max() == pytest.approx(CONFIG.length)
    assert vertices[:, 1].min() == 0
    assert vertices[:, 1].max() == pytest.approx(CONFIG.width)


def test_distances_match_laws_of_cricket() -> None:
    # Law 6.1: 20.12 m between the bowling creases (middle stumps).
    assert np.linalg.norm(_vertex(21) - _vertex(10)) == pytest.approx(2012)
    # Law 7.3: popping crease 1.22 m in front of the bowling crease.
    assert _vertex(2)[0] - _vertex(5)[0] == pytest.approx(122)
    assert _vertex(16)[0] - _vertex(13)[0] == pytest.approx(122)
    # Law 7.2: bowling crease 2.64 m long, between the return creases.
    assert np.linalg.norm(_vertex(6) - _vertex(5)) == pytest.approx(264)
    # Law 7.4: return creases marked 2.44 m behind the popping crease.
    assert np.linalg.norm(_vertex(2) - _vertex(7)) == pytest.approx(244)
    # Law 8.2: wicket 22.86 cm wide.
    assert np.linalg.norm(_vertex(11) - _vertex(9)) == pytest.approx(22.86)


def test_ends_mirror_each_other() -> None:
    vertices = np.array(CONFIG.vertices)
    end_1, end_2 = vertices[:11], vertices[11:]
    np.testing.assert_allclose(end_2[:, 0], CONFIG.length - end_1[:, 0])
    np.testing.assert_allclose(end_2[:, 1], end_1[:, 1])


def test_invalid_end_raises() -> None:
    with pytest.raises(ValueError):
        CONFIG.bowling_crease_x(3)


def test_distance_from_stumps_is_positive_toward_bowler() -> None:
    stumps_2 = CONFIG.bowling_crease_x(2)
    stumps_1 = CONFIG.bowling_crease_x(1)
    xy = np.array([[stumps_2 - 700, 100.0], [stumps_1 + 300, 100.0]])

    assert distance_from_stumps(CONFIG, xy[:1], batting_end=2)[0] == pytest.approx(700)
    assert distance_from_stumps(CONFIG, xy[1:], batting_end=1)[0] == pytest.approx(300)


@pytest.mark.parametrize("batting_end, hand, expected_sign", [
    (2, BatterHand.RIGHT, 1),
    (2, BatterHand.LEFT, -1),
    (1, BatterHand.RIGHT, -1),
    (1, BatterHand.LEFT, 1),
])
def test_lateral_offset_off_side_convention(batting_end, hand, expected_sign) -> None:
    # A point on the bowler's-left of a bowler at end 1 (smaller y).
    xy = np.array([[1000.0, CONFIG.middle_stump_y - 50]])

    offset = lateral_offset(CONFIG, xy, batting_end=batting_end, batter_hand=hand)

    assert offset[0] == pytest.approx(expected_sign * 50)


@pytest.mark.parametrize("distance, expected", [
    (0, "yorker"),
    (199.9, "yorker"),
    (200, "full"),
    (700, "good"),
    (900, "back of length"),
    (1500, "short"),
])
def test_classify_length(distance, expected) -> None:
    assert classify_length(distance).name == expected


def test_length_zones_property_matches_default_zones() -> None:
    assert CONFIG.length_zones == {
        zone.name: (zone.start, zone.end) for zone in DEFAULT_LENGTH_ZONES
    }


@pytest.mark.parametrize("zones", [DEFAULT_LENGTH_ZONES, BOUNCER_LENGTH_ZONES])
def test_length_zone_presets_cover_pitch_without_gaps(zones) -> None:
    assert zones[0].start == 0
    assert zones[-1].end == CONFIG.pitch_length
    for previous, current in zip(zones, zones[1:]):
        assert previous.end == current.start


@pytest.mark.parametrize("distance, expected", [
    (300, "full"),
    (500, "good"),
    (800, "short"),
    (1200, "bouncer"),
])
def test_classify_length_with_bouncer_zones(distance, expected) -> None:
    assert classify_length(distance, BOUNCER_LENGTH_ZONES).name == expected


def test_classify_length_outside_zones() -> None:
    assert classify_length(-10) is None
    assert classify_length(5000) is None


def test_classify_line() -> None:
    assert classify_line(0, CONFIG) == Line.STUMPS
    assert classify_line(14, CONFIG) == Line.STUMPS
    assert classify_line(30, CONFIG) == Line.OUTSIDE_OFF
    assert classify_line(-30, CONFIG) == Line.OUTSIDE_LEG


def _bounce_trajectory(bounce_t: float, n: int = 16) -> np.ndarray:
    t = np.arange(n, dtype=float)
    bounce_y = 400.0
    y = np.where(t < bounce_t, bounce_y + 6 * (bounce_t - t), bounce_y - 14 * (t - bounce_t))
    x = 600 + 2.0 * t
    return np.stack([x, y], axis=1)


def test_detect_bounce_recovers_sub_frame_position() -> None:
    xy = _bounce_trajectory(bounce_t=8.4)

    bounce = detect_bounce(xy)

    assert bounce is not None
    assert bounce.frame_index == pytest.approx(8.4, abs=1e-6)
    np.testing.assert_allclose(bounce.xy, [600 + 2.0 * 8.4, 400.0], atol=1e-6)


def test_detect_bounce_with_perspective_slowdown() -> None:
    # Seen from behind the bowler, the ball slows down smoothly on screen as it
    # moves away, then speeds up abruptly when it bounces.
    t = np.arange(14, dtype=float)
    bounce_t = 7.5
    y = np.where(
        t < bounce_t,
        280 + 1.2 * (bounce_t - t) ** 2 + 4 * (bounce_t - t),
        280 - 18 * (t - bounce_t) + 0.8 * (t - bounce_t) ** 2,
    )
    xy = np.stack([600 + 2.0 * t, y], axis=1)

    bounce = detect_bounce(xy)

    assert bounce is not None
    assert bounce.frame_index == pytest.approx(bounce_t, abs=0.2)
    np.testing.assert_allclose(bounce.xy, [600 + 2.0 * bounce_t, 280], atol=3)


def test_detect_bounce_handles_missing_frames() -> None:
    xy = _bounce_trajectory(bounce_t=8.4)
    xy[[2, 5, 11]] = np.nan

    bounce = detect_bounce(xy)

    assert bounce is not None
    assert bounce.frame_index == pytest.approx(8.4, abs=1e-6)


def test_detect_bounce_returns_none_without_bounce() -> None:
    t = np.arange(16, dtype=float)
    xy = np.stack([600 + 2 * t, 500 - 10 * t], axis=1)

    assert detect_bounce(xy) is None


def test_detect_bounce_returns_none_with_too_few_points() -> None:
    assert detect_bounce(_bounce_trajectory(bounce_t=1.5, n=3)) is None


def test_draw_pitch_shape_and_markings() -> None:
    pitch = draw_pitch(CONFIG, scale=SCALE, padding=PADDING)

    assert pitch.shape == (
        int(CONFIG.width * SCALE) + 2 * PADDING,
        int(CONFIG.length * SCALE) + 2 * PADDING,
        3,
    )
    # A point in the middle of the end-1 popping crease line is white.
    x, y = _to_pixel(CONFIG.popping_crease_x(1), CONFIG.middle_stump_y)
    assert np.array_equal(pitch[y, x], sv.Color.WHITE.as_bgr())


def test_draw_paths_on_pitch_draws_every_path() -> None:
    blank = np.zeros_like(draw_pitch(CONFIG, scale=SCALE, padding=PADDING))
    paths = [
        np.array([[400.0, 50.0], [800.0, 50.0]]),
        np.array([[1200.0, 300.0], [1600.0, 300.0]]),
    ]

    result = draw_paths_on_pitch(
        CONFIG, paths, color=sv.Color.RED, scale=SCALE, padding=PADDING, pitch=blank)

    for path in paths:
        x, y = _to_pixel(*path.mean(axis=0))
        assert np.array_equal(result[y, x], sv.Color.RED.as_bgr())


def test_draw_pitch_map_colors_bounce_by_length_zone() -> None:
    stumps_x = CONFIG.bowling_crease_x(2)
    good_length = np.array([[stumps_x - 700, CONFIG.middle_stump_y]])

    pitch_map = draw_pitch_map(CONFIG, good_length, scale=SCALE, padding=PADDING)

    x, y = _to_pixel(*good_length[0])
    good_index = [zone.name for zone in DEFAULT_LENGTH_ZONES].index("good")
    assert np.array_equal(pitch_map[y, x], DEFAULT_ZONE_COLORS[good_index].as_bgr())


def test_draw_pitch_map_with_bouncer_zones() -> None:
    stumps_x = CONFIG.bowling_crease_x(2)
    bouncer = np.array([[stumps_x - 1200, CONFIG.middle_stump_y]])

    pitch_map = draw_pitch_map(
        CONFIG, bouncer, zones=BOUNCER_LENGTH_ZONES, scale=SCALE, padding=PADDING)

    x, y = _to_pixel(*bouncer[0])
    bouncer_index = [zone.name for zone in BOUNCER_LENGTH_ZONES].index("bouncer")
    assert np.array_equal(pitch_map[y, x], DEFAULT_ZONE_COLORS[bouncer_index].as_bgr())
