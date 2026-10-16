"""
Extract frames for labelling the ball, from clips cut with find_deliveries.py.

The broadcast camera starts zooming in on the batter just after release, so the ball
is in the air from shortly before the zoom starts until about a second after. Frames
are taken from that window; for clips without a zoom they are spread over the clip.
File names start with the clip name, so a training split can hold out whole
deliveries instead of neighbouring frames of the same delivery.
"""
import argparse
import glob
import os
from typing import List

import cv2
import numpy as np

from camera_motion import frame_to_reference
from pitch_map import read_frames


def zoom_onset(to_reference: List[np.ndarray], threshold: float = 1.05) -> int:
    """First frame zoomed in by `threshold` compared with the reference frame."""
    for t, homography in enumerate(to_reference):
        if homography is not None and 1 / np.sqrt(abs(np.linalg.det(homography[:2, :2]))) > threshold:
            return t
    return -1


def pick_frames(frames: List[np.ndarray], fps: float, per_clip: int, before: float,
                after: float) -> List[int]:
    to_reference = frame_to_reference(frames)
    followed = [t for t, h in enumerate(to_reference) if h is not None]
    onset = zoom_onset(to_reference)
    if onset >= 0:
        first = max(followed[0], onset - int(before * fps))
        last = min(followed[-1], onset + int(after * fps))
    else:
        first = followed[0] + len(followed) // 5
        last = followed[-1] - len(followed) // 5
    count = min(per_clip, last - first + 1)
    return sorted(set(np.linspace(first, last, count).astype(int).tolist()))


def main(source_dirs: List[str], target_dir: str, per_clip: int, before: float,
         after: float) -> None:
    os.makedirs(target_dir, exist_ok=True)
    total = 0
    for source_dir in source_dirs:
        # Only the clips: rendered outputs in the same folder have overlays drawn.
        for path in sorted(glob.glob(os.path.join(source_dir, "delivery_*.mp4"))):
            fps = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS)
            frames = read_frames(path)
            picks = pick_frames(frames, fps, per_clip, before, after)
            clip = f"{os.path.basename(os.path.normpath(source_dir))}_" \
                   f"{os.path.splitext(os.path.basename(path))[0]}"
            for t in picks:
                cv2.imwrite(os.path.join(target_dir, f"{clip}_f{t:04d}.jpg"), frames[t],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            total += len(picks)
            print(f"{clip}: {len(picks)} frames ({picks[0]}-{picks[-1]})")
    print(f"{total} frames written to {target_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source_dirs', type=str, nargs='+', required=True,
                        help='folders of delivery clips from find_deliveries.py')
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--per_clip', type=int, default=20)
    parser.add_argument('--before', type=float, default=0.4,
                        help='seconds before the zoom starts')
    parser.add_argument('--after', type=float, default=1.2,
                        help='seconds after the zoom starts')
    args = parser.parse_args()
    main(args.source_dirs, args.target_dir, args.per_clip, args.before, args.after)
