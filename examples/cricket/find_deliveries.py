"""
Cut a match video into one clip per delivery.

Deliveries are filmed with the wide camera behind the bowler, where both ends of the
pitch are in view. Right after the ball is released, the camera zooms in on the
batter. The pitch keypoint model finds the wide shots on sampled frames, and a wide
shot counts as a delivery when it ends with that zoom rather than with a cut, which
skips the wide shots between deliveries. Each clip is centred on the start of the
zoom. Replays filmed from the same camera can still appear as extra clips.
"""
import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from camera_motion import frame_to_reference

FAR_KEYPOINTS = [0, 1, 4, 5]  # corners and popping crease points, see prepare_pitch_dataset
NEAR_KEYPOINTS = [2, 3, 6, 7]


def wide_shot(model, frame: np.ndarray, imgsz: int, min_confidence: float) -> bool:
    """Whether both ends of the pitch are visible."""
    result = model(frame, imgsz=imgsz, verbose=False)[0]
    if len(result.boxes) == 0 or result.keypoints is None or result.keypoints.conf is None:
        return False
    best = int(result.boxes.conf.argmax())
    confident = result.keypoints.conf[best].cpu().numpy() >= min_confidence
    return confident[FAR_KEYPOINTS].sum() >= 2 and confident[NEAR_KEYPOINTS].sum() >= 2


def find_wide_shots(source_video_path: str, model, stride: int, imgsz: int,
                    min_confidence: float, limit: int) -> List[Tuple[int, int]]:
    """Frame ranges of the wide shots, stopping after `limit` of them."""
    capture = cv2.VideoCapture(source_video_path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    shots, start, last_wide = [], None, None
    for index in tqdm(range(0, total, stride), desc="scanning"):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            break
        if wide_shot(model, frame, imgsz, min_confidence):
            start = index if start is None else start
            last_wide = index
        elif start is not None:
            shots.append((start, last_wide))
            start = None
            if len(shots) >= limit:
                break
    if start is not None and len(shots) < limit:
        shots.append((start, last_wide))
    return shots


def read(source_video_path: str, first: int, count: int) -> List[np.ndarray]:
    capture = cv2.VideoCapture(source_video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def zooms_in(source_video_path: str, last_wide: int, fps: float,
             min_zoom: float = 1.4) -> bool:
    """Whether the camera keeps rolling and zooms in after the wide shot ends."""
    frames = read(source_video_path, last_wide, int(fps))
    to_reference = frame_to_reference(frames)
    if to_reference[0] is None or to_reference[-1] is None:  # a cut
        return False
    scale = np.sqrt(abs(np.linalg.det(to_reference[-1][:2, :2])))
    return 1 / scale >= min_zoom


def write_clip(source_video_path: str, target_path: str, first: int, last: int) -> None:
    capture = cv2.VideoCapture(source_video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(3)), int(capture.get(4)))
    writer = cv2.VideoWriter(target_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    for _ in range(first, last + 1):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()


def main(source_video_path: str, pitch_model: str, target_dir: str, limit: int,
         stride: int, before: float, after: float, min_wide: float, imgsz: int) -> None:
    from ultralytics import YOLO

    model = YOLO(pitch_model)
    fps = cv2.VideoCapture(source_video_path).get(cv2.CAP_PROP_FPS)
    shots = find_wide_shots(source_video_path, model, stride, imgsz, 0.5, limit * 10)
    os.makedirs(target_dir, exist_ok=True)
    number = 0
    for first, last in shots:
        if (last - first) / fps < min_wide:
            continue
        # The wide shot is only sampled every `stride` frames: start the zoom check
        # at the last sampled wide frame.
        if not zooms_in(source_video_path, last, fps):
            continue
        number += 1
        # Keep only the continuous shot around the zoom: cuts break the camera
        # tracking in pitch_map.py.
        start = max(0, last - int(before * fps))
        window = read(source_video_path, start, int((before + after) * fps))
        followed = [t for t, h in enumerate(frame_to_reference(window, anchor=last - start))
                    if h is not None]
        start, end = start + followed[0], start + followed[-1]
        path = os.path.join(target_dir, f"delivery_{number:02d}.mp4")
        write_clip(source_video_path, path, start, end)
        print(f"{path}: {start / fps:7.1f}s - {end / fps:7.1f}s")
        if number >= limit:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--pitch_model', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=6, help='number of deliveries')
    parser.add_argument('--stride', type=int, default=10, help='frames between checks')
    parser.add_argument('--before', type=float, default=2.5,
                        help='seconds kept before the zoom starts')
    parser.add_argument('--after', type=float, default=2.5,
                        help='seconds kept after the zoom starts')
    parser.add_argument('--min_wide', type=float, default=1.0,
                        help='shortest wide shot that counts as a delivery [s]')
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    main(args.source_video_path, args.pitch_model, args.target_dir, args.limit,
         args.stride, args.before, args.after, args.min_wide, args.imgsz)
