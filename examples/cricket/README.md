# Cricket AI 🏏

Build a pitch map (where each delivery bounced, by length and line) from video
filmed behind the bowler, without any trained models.

## 💻 install

```bash
pip install git+https://github.com/roboflow/sports.git
cd examples/cricket
```

No extra dependencies are needed.

## 🧭 how it works

- **Pitch geometry** - `sports.cricket.CricketPitchConfiguration` describes the
  creases and stumps in centimeters, following the MCC Laws of Cricket (Laws 6-8).
- **Calibration** - you click the four points where the return creases meet the
  popping creases. They give the homography between the image and the pitch. The
  camera behind the bowler barely moves during a delivery, so one calibration per
  camera position is enough.
- **Ball detection** - three-frame differencing finds small moving blobs inside the
  pitch area, and a constant-velocity tracker links them into a trajectory.
- **Bounce detection** - perspective makes the ball slow down smoothly on screen as
  it moves away, while the bounce makes it jump upward. `detect_bounce` finds that
  jump and intersects the trajectory on both sides of it for sub-frame accuracy.
- **Pitch map** - a homography is only valid for points on the ground, and the
  bounce is the one moment the ball is guaranteed to be on the ground. Only the
  bounce is projected onto the pitch, then classified by length and line.

## 🛠️ usage

Calibrate once per camera position. Click the points in the order shown on screen,
then press `enter`:

```bash
python pitch_map.py calibrate \
  --source_video_path data/delivery_01.mp4 \
  --calibration_path data/calibration.json
```

Build the pitch map from one or more deliveries (one clip per delivery):

```bash
python pitch_map.py run \
  --source_video_paths data/delivery_*.mp4 \
  --calibration_path data/calibration.json \
  --target_video_path data/pitch_map.mp4 \
  --results_path data/results.json \
  --batter_hand right --manual_fallback
```

`--manual_fallback` lets you step through the frames (`a`/`d`) and click the bounce
when it is not found automatically. `--display` shows the output while rendering.

## 🧪 synthetic clips

`make_synthetic_clip.py` renders behind-the-bowler deliveries with known bounce
positions, with the matching calibration and ground truth, so the pipeline can be
tested without footage:

```bash
python make_synthetic_clip.py --target_dir data/synthetic
python pitch_map.py run \
  --source_video_paths data/synthetic/delivery_*.mp4 \
  --calibration_path data/synthetic/calibration.json \
  --target_video_path data/synthetic/pitch_map.mp4
```

On the default 6 synthetic deliveries at 25 fps, every bounce is found, within 3 cm
across the pitch and 23-39 cm along it, about a quarter of the distance the ball
travels between frames. Higher frame rates reduce this error.

## ⚠️ limitations

- Synthetic clips have a static background and a clean ball. Real footage adds
  camera shake, compression and players moving near the ball. The detector will miss
  or mistake the ball more often, so use `--manual_fallback`.
- Expects one delivery per clip, filmed from behind the bowler with both popping
  creases visible.
- Yorkers leave few frames after the bounce and may need manual selection.
- Length zones are a broadcast convention and vary by bowling speed. Pass your own
  `LengthZone` list to adapt them.

## 🗺️ roadmap

- [ ] Pitch keypoint detector to replace manual calibration.
- [ ] Trained ball detector for real broadcast footage.
- [ ] Beehive view: where the ball passed the stumps, using the vertical plane of
  the batter's stumps.
