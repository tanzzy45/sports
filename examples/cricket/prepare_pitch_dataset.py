"""
Build a pitch keypoint dataset for cricket by merging public Roboflow Universe
datasets that label the same real-world points with different names.

Every source is downloaded with the Roboflow API, its keypoints are renamed to one
shared schema, and the result is written in YOLO pose format. Nothing from the
sources is stored in this repository; run the script to recreate the dataset.

Sources (all CC BY 4.0, credit the authors if you publish models trained on them):

- nishith-n6waq/cricket-pitch-keypoints
- maree/cricket-pitch-detection-wdtth
- adrian-charan/cricket-pitch-keypoint-kbktj

Keypoints, for a camera behind the bowler (far = batter's end, near = bowler's end).
`corner` is the bowling crease x return crease, `pop` is the popping crease x return
crease, and `stump` is the base of a stump:

    0 far_left_corner   1 far_right_corner   2 near_right_corner   3 near_left_corner
    4 far_left_pop      5 far_right_pop      6 near_right_pop      7 near_left_pop
    8-10  far stumps (left, middle, right)   11-13  near stumps (left, middle, right)

Fitting a homography from the real pitch to each image with these points gives a
median reprojection error of 0.3-0.6% of the pitch height, which checks the mapping.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil
from typing import Dict, List, Optional

import cv2
import numpy as np

from sports.cricket import CricketPitchConfiguration

KEYPOINT_NAMES = [
    "far_left_corner", "far_right_corner", "near_right_corner", "near_left_corner",
    "far_left_pop", "far_right_pop", "near_right_pop", "near_left_pop",
    "far_stump_left", "far_stump_mid", "far_stump_right",
    "near_stump_left", "near_stump_mid", "near_stump_right",
]
# Vertex number in CricketPitchConfiguration of each keypoint.
CONFIG_VERTICES = [16, 17, 6, 5, 13, 14, 3, 2, 20, 21, 22, 9, 10, 11]
# Swapping left and right under a horizontal flip.
FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 10, 9, 8, 13, 12, 11]

# project -> (version, {source keypoint name or index: shared keypoint index})
SOURCES = {
    "nishith-n6waq/cricket-pitch-keypoints": (5, {
        "batting_left_corner": 0, "batting_right_corner": 1,
        "bowling_right_corner": 2, "bowling_left_corner": 3,
        "batting_left_pop": 4, "batting_right_pop": 5,
        "bowling-right_pop": 6, "bowling_left_pop": 7,
    }),
    "maree/cricket-pitch-detection-wdtth": (2, {
        "1": 2, "2": 13, "3": 12, "4": 11, "5": 3, "6": 0, "7": 8, "8": 9, "9": 10,
        "10": 1,
    }),
    "adrian-charan/cricket-pitch-keypoint-kbktj": (1, {
        "bowl-2": 0, "bowl-1": 1, "bowl-3": 2, "bowl-4": 3,
        "pop-7": 4, "pop-8": 5, "pop-3": 6, "pop-4": 7,
        "stump-3": 8, "stump-2": 9, "stump-1": 10,
        "stump-6": 11, "stump-5": 12, "stump-4": 13,
    }),
}


def download(project: str, version: int, location: str, api_key: Optional[str]) -> str:
    from roboflow import Roboflow

    workspace, name = project.split("/")
    Roboflow(api_key=api_key).workspace(workspace).project(name).version(
        version).download("coco", location=location, overwrite=True)
    return location


def video_group(file_name: str) -> str:
    """Name of the video a frame was cut from, to keep its frames in one split."""
    source = re.split(r"\.rf\.", file_name)[0]
    source = re.sub(
        r"(?i)[-_]?(mp4|jpg|jpeg|png)[-_].*$|[-_]frame[-_]?\d+.*$", "", source)
    return re.sub(r"[-_ ]?\d+$", "", source) or source


def frame_number(file_name: str) -> int:
    numbers = re.findall(r"\d+", re.split(r"\.rf\.", os.path.basename(file_name))[0])
    return int(numbers[-1]) if numbers else 0


def load_records(project: str, root: str, mapping: Dict[str, int]) -> List[dict]:
    """Read a downloaded COCO export as records with the shared keypoint schema."""
    records, seen = [], set()
    for split in sorted(os.listdir(root)):
        annotation_path = os.path.join(root, split, "_annotations.coco.json")
        if not os.path.exists(annotation_path):
            continue
        with open(annotation_path) as f:
            data = json.load(f)
        images = {image["id"]: image for image in data["images"]}
        names = next(c["keypoints"] for c in data["categories"] if c.get("keypoints"))
        index_map = {names.index(name): index for name, index in mapping.items()}
        for annotation in data["annotations"]:
            image = images[annotation["image_id"]]
            source = re.split(r"\.rf\.", image["file_name"])[0]
            if source in seen:  # skip the augmented copies that Roboflow exports
                continue
            seen.add(source)
            raw = np.array(annotation["keypoints"], dtype=float).reshape(-1, 3)
            keypoints = np.zeros((len(KEYPOINT_NAMES), 3))
            for source_index, index in index_map.items():
                x, y, visibility = raw[source_index]
                if visibility > 0 and (x, y) != (0, 0):
                    keypoints[index] = [x, y, 2]
            if (keypoints[:, 2] > 0).sum() < 4:
                continue
            records.append(dict(
                project=project,
                path=os.path.join(root, split, image["file_name"]),
                size=(image["width"], image["height"]),
                keypoints=keypoints,
                group=f"{project}:{video_group(image['file_name'])}",
                frame=frame_number(image["file_name"]),
            ))
    return records


def assign_splits(records: List[dict], chunk: int = 6) -> Dict[int, str]:
    """
    Split 80/10/10 within each project. Frames cut from one video are near-duplicates
    when adjacent, so runs of `chunk` consecutive frames always stay in one split.
    """
    splits = {}
    for project in {r["project"] for r in records}:
        own = [i for i, r in enumerate(records) if r["project"] == project]
        chunks = collections.defaultdict(list)
        by_group = collections.defaultdict(list)
        for i in own:
            by_group[records[i]["group"]].append(i)
        for group, indices in by_group.items():
            for rank, i in enumerate(sorted(indices, key=lambda i: records[i]["frame"])):
                chunks[f"{group}#{rank // chunk}"].append(i)
        total, accumulated = len(own), 0
        for key in sorted(chunks, key=lambda k: hashlib.md5(k.encode()).hexdigest()):
            split = ("train" if accumulated < 0.8 * total
                     else "val" if accumulated < 0.9 * total else "test")
            for i in chunks[key]:
                splits[i] = split
            accumulated += len(chunks[key])
    return splits


def reprojection_errors(records: List[dict]) -> Dict[str, float]:
    """Median error [fraction of pitch height] of a homography fitted per image."""
    config = CricketPitchConfiguration()
    world = np.array([config.vertices[v - 1] for v in CONFIG_VERTICES], np.float64)
    errors = collections.defaultdict(list)
    for record in records:
        visible = record["keypoints"][:, 2] > 0
        points = record["keypoints"][visible, :2]
        homography, _ = cv2.findHomography(world[visible], points, 0)
        if homography is None:
            continue
        projected = cv2.perspectiveTransform(world[visible][None], homography)[0]
        errors[record["project"]].append(
            np.linalg.norm(projected - points, axis=1).mean() / max(np.ptp(points[:, 1]), 1))
    return {project: float(np.median(values)) for project, values in errors.items()}


def write_dataset(records: List[dict], splits: Dict[int, str], target_dir: str) -> None:
    shutil.rmtree(target_dir, ignore_errors=True)
    for i, record in enumerate(records):
        split = splits[i]
        image = cv2.imread(record["path"])
        height, width = image.shape[:2]
        keypoints = record["keypoints"].copy()
        keypoints[:, 0] *= width / record["size"][0]
        keypoints[:, 1] *= height / record["size"][1]
        visible = keypoints[:, 2] > 0
        xs, ys = keypoints[visible, 0], keypoints[visible, 1]
        pad = 0.06 * max(np.ptp(xs), np.ptp(ys), 20)
        x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, width)
        y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, height)

        name = hashlib.md5(record["path"].encode()).hexdigest()[:10]
        name = f"{record['project'].split('/')[0]}_{name}"
        os.makedirs(os.path.join(target_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "labels", split), exist_ok=True)
        cv2.imwrite(os.path.join(target_dir, "images", split, f"{name}.jpg"), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        fields = ["0", f"{(x1 + x2) / 2 / width:.6f}", f"{(y1 + y2) / 2 / height:.6f}",
                  f"{(x2 - x1) / width:.6f}", f"{(y2 - y1) / height:.6f}"]
        for x, y, v in keypoints:
            fields += [f"{x / width:.6f}", f"{y / height:.6f}", str(int(v))] if v > 0 \
                else ["0", "0", "0"]
        with open(os.path.join(target_dir, "labels", split, f"{name}.txt"), "w") as f:
            f.write(" ".join(fields) + "\n")

    with open(os.path.join(target_dir, "data.yaml"), "w") as f:
        f.write(
            f"path: {os.path.abspath(target_dir)}\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            f"kpt_shape: [{len(KEYPOINT_NAMES)}, 3]\nflip_idx: {FLIP_IDX}\n"
            "names:\n  0: pitch\n")
    with open(os.path.join(target_dir, "schema.json"), "w") as f:
        json.dump(dict(keypoints=KEYPOINT_NAMES, config_vertices=CONFIG_VERTICES,
                       sources=list(SOURCES), license="CC BY 4.0"), f, indent=2)


def main(target_dir: str, download_dir: str, api_key: Optional[str]) -> None:
    # Without a key, the Roboflow SDK uses the one saved by `roboflow login`.
    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")

    records = []
    for project, (version, mapping) in SOURCES.items():
        root = download(project, version, os.path.join(
            download_dir, project.split("/")[0]), api_key)
        loaded = load_records(project, root, mapping)
        print(f"{project}: kept {len(loaded)} distinct images")
        records += loaded

    for project, error in reprojection_errors(records).items():
        print(f"  mapping check {project}: median reprojection error "
              f"{error * 100:.2f}% of pitch height")

    splits = assign_splits(records)
    write_dataset(records, splits, target_dir)
    print({s: sum(1 for v in splits.values() if v == s) for s in ("train", "val", "test")})
    print(f"dataset written to {target_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target_dir', type=str, default='data/cricket-pitch')
    parser.add_argument('--download_dir', type=str, default='data/downloads',
                        help='keep this path short on Windows: some source file names '
                             'are very long')
    parser.add_argument('--api_key', type=str, default=None)
    args = parser.parse_args()
    main(args.target_dir, args.download_dir, args.api_key)
