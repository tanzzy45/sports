"""
Compare automatic bounces with reference bounces, e.g. ones selected by hand.

References are results files of `pitch_map.py run`. Every delivery with a bounce in
them is run again with automatic ball detection, and the automatic bounce is compared
with the reference on the pitch. Calibration is repeatable, so both bounces are in the
same pitch coordinates.
"""
import argparse
import json
import os
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

import pitch_map as pm


def load_references(paths: List[str], exclude: List[str]) -> List[dict]:
    excluded = {os.path.normpath(path) for path in exclude}
    references = []
    for path in paths:
        with open(path) as f:
            for result in json.load(f):
                clip = os.path.normpath(result["clip"])
                if result.get("bounce_pitch_xy") is not None and clip not in excluded:
                    image_xy = result.get("bounce_reference_xy")
                    references.append(dict(
                        clip=clip, xy=np.array(result["bounce_pitch_xy"]),
                        image_xy=None if image_xy is None else np.array(image_xy)))
    return references


def reference_on_pitch(reference: dict, calibration) -> np.ndarray:
    """
    The reference bounce on the pitch. A bounce selected by hand is stored in pixels of
    the delivery's reference frame, so it is projected with the same calibration as the
    automatic bounce: the comparison measures the ball tracking, whatever changes in
    calibration since the bounce was selected.
    """
    if reference["image_xy"] is None:
        return reference["xy"]
    return calibration.image_to_pitch.transform_points(
        reference["image_xy"][np.newaxis].astype(np.float32))[0]


def save_debug(path: str, delivery, reference_xy: np.ndarray) -> None:
    """Reference frame with the trajectory, the bounce found (red) and the reference (green)."""
    index = int(np.argmin([np.inf if h is None else np.abs(h - np.eye(3)).sum()
                           for h in delivery.to_reference]))
    image = pm.draw_creases(delivery.frames[index].copy(), delivery.calibration, np.eye(3))
    image = pm.draw_trail(image, delivery.trajectory)
    expected = delivery.calibration.pitch_to_image.transform_points(
        reference_xy[np.newaxis].astype(np.float32))[0]
    cv2.circle(image, tuple(int(v) for v in expected), 9, (0, 255, 0), 2, cv2.LINE_AA)
    if delivery.bounce is not None:
        cv2.circle(image, tuple(int(v) for v in delivery.bounce.xy), 9, (0, 0, 255), 2,
                   cv2.LINE_AA)
    cv2.imwrite(path, image)


def main(references: List[str], exclude: List[str], pitch_model: str, ball_model: str,
         tracking: str, ball_confidence: float, close: float,
         debug_dir: Optional[str] = None, use_flight: bool = True) -> None:
    pitch = YOLO(pitch_model)
    detector = pm.YoloBallDetector(ball_model, ball_confidence)
    tracker = pm.ReleaseBallTracker(ball_model) if tracking == "release" else None
    rows = []
    for reference in load_references(references, exclude):
        if not os.path.exists(reference["clip"]):
            print(f"skipping {reference['clip']}: file not found")
            continue
        delivery = pm.process_delivery(reference["clip"], None, pitch, detector,
                                       manual_fallback=False, tracker=tracker,
                                       use_flight=use_flight)
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            name = reference["clip"].replace(os.sep, "_").replace(".mp4", ".jpg")
            save_debug(os.path.join(debug_dir, name), delivery,
                       reference_on_pitch(reference, delivery.calibration))
        error = None
        if delivery.bounce_pitch_xy is not None:
            error = float(np.linalg.norm(delivery.bounce_pitch_xy
                                         - reference_on_pitch(reference, delivery.calibration)))
        rows.append((reference["clip"], error))
        print(f"{reference['clip']}: {'no bounce' if error is None else f'{error:.0f} cm'}",
              flush=True)

    print(f"\n{'delivery':45} {'error':>10}")
    for clip, error in rows:
        print(f"{clip:45} {'no bounce' if error is None else f'{error:7.0f} cm':>10}")
    errors = [error for _, error in rows if error is not None]
    found = sum(error <= close for error in errors)
    print(f"\n{tracking}: {found}/{len(rows)} bounces within {close:.0f} cm of the reference, "
          f"{len(errors) - found} further away, {len(rows) - len(errors)} not found")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--references', type=str, nargs='+', required=True)
    parser.add_argument('--exclude', type=str, nargs='*', default=[],
                        help='clips whose reference is known to be wrong')
    parser.add_argument('--pitch_model', type=str, required=True)
    parser.add_argument('--ball_model', type=str, required=True)
    parser.add_argument('--tracking', choices=["link", "release"], default="link",
                        help='release: also follow the ball from the bowler\'s release, '
                             'and keep the track with the most detections')
    parser.add_argument('--ball_confidence', type=float, default=0.1)
    parser.add_argument('--close', type=float, default=100.0,
                        help='largest error [cm] that counts as the same bounce')
    parser.add_argument('--debug_dir', type=str, default=None,
                        help='save each reference frame with the trajectory and bounces')
    parser.add_argument('--no_flight', action='store_true',
                        help='score the bounce at the change of direction only')
    args = parser.parse_args()
    main(args.references, args.exclude, args.pitch_model, args.ball_model, args.tracking,
         args.ball_confidence, args.close, args.debug_dir, not args.no_flight)
