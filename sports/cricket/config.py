from dataclasses import dataclass, field
from typing import List, Tuple

VERTICES_PER_END = 11


@dataclass
class CricketPitchConfiguration:
    """
    Dimensions and markings of a cricket pitch, following the MCC Laws of Cricket
    (Law 6 - The pitch, Law 7 - The creases, Law 8 - The wickets). All values are in
    centimeters.

    Coordinate system:
        - `x` runs along the pitch, from end 1 to end 2.
        - `y` runs across the pitch and increases toward the right of a person
          standing at end 1 and looking toward end 2 (e.g. a bowler bowling from
          end 1, or a camera behind them).
        - The origin is at the back of the end-1 return crease markings, on the
          left edge of the popping crease, so every marking has non-negative
          coordinates.

    Vertices are numbered 1-11 at end 1 and 12-22 at end 2, in the same order:

        1. popping crease, left end
        2. popping crease x left return crease
        3. popping crease x right return crease
        4. popping crease, right end
        5. bowling crease x left return crease
        6. bowling crease x right return crease
        7. left return crease, back end
        8. right return crease, back end
        9. left stump, base
        10. middle stump, base
        11. right stump, base
    """
    pitch_length: float = 2012  # [cm] between the bowling creases
    pitch_width: float = 305  # [cm]
    popping_crease_distance: float = 122  # [cm] bowling crease -> popping crease
    popping_crease_half_length: float = 183  # [cm] marked each side of middle stump
    return_crease_distance: float = 132  # [cm] middle stump line -> return crease
    return_crease_length: float = 244  # [cm] marked behind the popping crease
    wicket_width: float = 22.86  # [cm]
    stump_height: float = 71.12  # [cm]

    @property
    def end_margin(self) -> float:
        """Distance from the back of the return crease markings to the stumps."""
        return self.return_crease_length - self.popping_crease_distance

    @property
    def length(self) -> float:
        """Extent of all markings along the pitch (x)."""
        return self.pitch_length + 2 * self.end_margin

    @property
    def width(self) -> float:
        """Extent of all markings across the pitch (y)."""
        return 2 * self.popping_crease_half_length

    @property
    def middle_stump_y(self) -> float:
        return self.popping_crease_half_length

    def bowling_crease_x(self, end: int) -> float:
        """x coordinate of the bowling crease (line of the stumps) at `end`."""
        self._validate_end(end)
        return self.end_margin if end == 1 else self.end_margin + self.pitch_length

    def popping_crease_x(self, end: int) -> float:
        """x coordinate of the popping crease at `end`."""
        direction = 1 if end == 1 else -1
        return self.bowling_crease_x(end) + direction * self.popping_crease_distance

    @property
    def pitch_corners(self) -> List[Tuple[float, float]]:
        """
        Corners of the pitch rectangle (bowling crease to bowling crease), clockwise
        starting at end 1, left side.
        """
        x1, x2 = self.bowling_crease_x(1), self.bowling_crease_x(2)
        y1 = self.middle_stump_y - self.pitch_width / 2
        y2 = self.middle_stump_y + self.pitch_width / 2
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def _end_vertices(self, end: int) -> List[Tuple[float, float]]:
        bowling_x = self.bowling_crease_x(end)
        popping_x = self.popping_crease_x(end)
        back_x = 0.0 if end == 1 else self.length
        middle_y = self.middle_stump_y
        left_y = middle_y - self.return_crease_distance
        right_y = middle_y + self.return_crease_distance
        half_wicket = self.wicket_width / 2
        return [
            (popping_x, 0.0),  # 1
            (popping_x, left_y),  # 2
            (popping_x, right_y),  # 3
            (popping_x, self.width),  # 4
            (bowling_x, left_y),  # 5
            (bowling_x, right_y),  # 6
            (back_x, left_y),  # 7
            (back_x, right_y),  # 8
            (bowling_x, middle_y - half_wicket),  # 9
            (bowling_x, middle_y),  # 10
            (bowling_x, middle_y + half_wicket),  # 11
        ]

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        return self._end_vertices(1) + self._end_vertices(2)

    @staticmethod
    def _validate_end(end: int) -> None:
        if end not in (1, 2):
            raise ValueError(f"end must be 1 or 2, got {end}.")

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (5, 6), (2, 5), (5, 7), (3, 6), (6, 8),
        (12, 13), (13, 14), (14, 15), (16, 17), (13, 16), (16, 18), (14, 17), (17, 19),
    ])

    labels: List[str] = field(default_factory=lambda: [
        f"{i:02d}" for i in range(1, 2 * VERTICES_PER_END + 1)
    ])

    colors: List[str] = field(default_factory=lambda: (
        ["#FF1493"] * 8 + ["#FFD700"] * 3 + ["#00BFFF"] * 8 + ["#FFD700"] * 3
    ))
