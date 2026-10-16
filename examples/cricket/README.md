# Cricket AI 🏏

Build a pitch map (where each delivery bounced, by length and line) from broadcast
video filmed behind the bowler.

## 💻 install

```bash
pip install git+https://github.com/roboflow/sports.git
cd examples/cricket
pip install -r requirements.txt
```

## 🧭 how it works

- **Deliveries** - `find_deliveries.py` cuts a match video into one clip per delivery:
  a wide shot behind the bowler, found by the pitch keypoint model, followed by the
  zoom onto the batter that comes right after release.
- **Pitch calibration** - a pitch keypoint model finds the creases and stump bases.
  Their positions on a real pitch, from the MCC Laws of Cricket (Laws 6-8, see
  `sports.cricket.CricketPitchConfiguration`), give the homography between the image
  and the pitch.
- **Camera motion** - broadcast cameras zoom 2-4x toward the batter while the ball is
  in the air. `camera_motion.py` follows the camera with optical flow and maps every
  frame into one reference frame of the delivery, masking static graphics and
  stopping at cuts. Everything below happens in that reference frame.
- **Ball** - a YOLOv8 detector trained on frames of the same kind of footage finds the
  ball, and a constant-velocity tracker links the detections into a trajectory. The
  threshold is low (0.1): linking filters out most false detections.
- **Bounce** - perspective makes the ball slow down smoothly on screen as it moves
  away, while the bounce makes it jump upward. `detect_bounce` finds that jump and
  intersects the trajectory on both sides of it for sub-frame accuracy.
- **Pitch map** - a homography is only valid for points on the ground, and the bounce
  is the one moment the ball is guaranteed to be on the ground. Only the bounce is
  projected onto the pitch, then classified by length and line.

## 🛠️ usage

Cut a match video into deliveries:

```bash
python find_deliveries.py \
  --source_video_path data/match.mp4 \
  --pitch_model data/best.pt \
  --target_dir data/match
```

Build the pitch map, all automatic:

```bash
python pitch_map.py run \
  --source_video_paths data/match/delivery_*.mp4 \
  --pitch_model data/best.pt \
  --ball_model data/ball.pt \
  --target_video_path data/match/pitch_map.mp4 \
  --results_path data/match/results.json
```

When the ball detector does not work on some footage, select the bounces by hand;
calibration and everything else stay automatic:

- `--manual_bounce` asks for the bounce of every delivery: step through the frames
  with `a`/`d` and click the ball where it lands. When the bounce falls between two
  frames, click the ball in the frame before and the frame after it, then press
  `enter`.
- `--manual_fallback` asks only for deliveries where no bounce was found.
- `--bounces_path` reuses the bounces of an earlier run's results, so a video can be
  rendered again without selecting them again.

`--ball_model` also accepts a model hosted on Roboflow, e.g. `cricket-balls-l1us5/3`.
Without `--ball_model`, frame differencing looks for the ball, which only works with
a still camera.

For a still camera, e.g. a phone on a tripod, calibration can also be done by
clicking four crease points once, instead of `--pitch_model`:

```bash
python pitch_map.py calibrate \
  --source_video_path data/delivery_01.mp4 \
  --calibration_path data/calibration.json
python pitch_map.py run \
  --source_video_paths data/delivery_*.mp4 \
  --calibration_path data/calibration.json \
  --target_video_path data/pitch_map.mp4
```

## 🤖 models

- **Pitch keypoints** - `prepare_pitch_dataset.py` merges three public datasets that
  label the same real-world points with different names into one 14-point dataset (670
  images, split by video): the 4 corners (bowling crease x return crease), the 4
  popping crease points and the base of the 3 stumps at each end. A homography fitted
  with these points gives a median reprojection error of 0.3-0.6% of the pitch height
  on every source, which checks the mapping. Train with
  [train_pitch_keypoint_detector.ipynb](notebooks/train_pitch_keypoint_detector.ipynb).
- **Ball** - public ball detectors did not transfer to the broadcasts tested here.
  `extract_ball_frames.py` picks frames around the ball's flight from delivery clips,
  named after their delivery; label the ball in them, then train with
  [train_ball_detector.ipynb](notebooks/train_ball_detector.ipynb). It splits by
  delivery, so the test score comes from deliveries the model never saw, and can add
  public ball datasets to the training data.

## 📊 results on broadcast footage

Pitch calibration found all 8 crease points with confidence 0.94-1.00 on every
broadcast tried: a Test at The Oval (2019), an ODI at the SCG (2019), a day-night Test
at Adelaide (2024, 498x282 pixels) and an ODI in New Zealand (a 596x350 screen
recording).

A ball detector trained on 320 frames from 20 deliveries of the first two broadcasts
finds 59% of the balls on 4 deliveries it never saw (0.58 false detections per frame).
Run through the whole pipeline on 3 held-out deliveries, every bounce was found
automatically:

| delivery | ball | result |
|:---------|:-----|:-------|
| The Oval, pace | red | full, 5.1 m, outside off: 5 cm from the hand-selected bounce |
| SCG, spin | white | good, 6.5 m, outside off |
| SCG, spin | white | full, 5.5 m, outside off |

Each trajectory was checked frame by frame. Three deliveries is a small test.

The detector does not generalize to broadcasts it was not trained on yet. On 12
deliveries from other broadcasts (a compilation of Test catches from several
broadcasters, an IPL match at 480p, a pink-ball Test at Adelaide and a screen
recording played at double speed), it did not find a single bounce: it missed the
ball in flight, and two tracks followed the ball off the edge instead, which
`pitch_map.py` now discards. Pitch calibration worked on all of them.

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

## 📚 datasets

| dataset | author | license | used for |
|:--------|:-------|:--------|:---------|
| [cricket-pitch-keypoints](https://universe.roboflow.com/nishith-n6waq/cricket-pitch-keypoints) | nishith-n6waq | CC BY 4.0 | pitch keypoints |
| [cricket-pitch-detection](https://universe.roboflow.com/maree/cricket-pitch-detection-wdtth) | maree | CC BY 4.0 | pitch keypoints |
| [cricket-pitch-keypoint](https://universe.roboflow.com/adrian-charan/cricket-pitch-keypoint-kbktj) | adrian-charan | CC BY 4.0 | pitch keypoints |
| [cricket-balls](https://universe.roboflow.com/cricket-rfyd8/cricket-balls-l1us5) | cricket-rfyd8 | CC BY 4.0 | optional ball training data |
| [stumps](https://universe.roboflow.com/cardiff-university-moxse/stumps) | cardiff-university-moxse | MIT | optional ball training data |

The datasets are broadcast frames. They are downloaded with your own API key and are
not stored in this repository.

## ⚠️ limitations

- The ball detector only works on footage like the footage it was trained on: 720p or
  more, at normal speed, with a red or white ball. Label frames of new footage to
  extend it.
- When one end of the pitch is out of view, the pitch keypoint model can mistake one
  end for the other. Check that the crease lines drawn on the output sit on the
  painted creases.
- Expects one delivery per clip, filmed from behind the bowler.
- Yorkers leave few frames after the bounce and may need `--manual_fallback`.
- Length zones are a broadcast convention and vary by bowling speed. Pass your own
  `LengthZone` list to adapt them.

## 🗺️ roadmap

- [x] Pitch keypoint detector, evaluated on four broadcasts.
- [x] Camera motion compensation for broadcast zooms.
- [x] Ball detector trained on broadcast footage, evaluated on held-out deliveries.
- [ ] A larger held-out test set, across more broadcasts.
- [ ] Track the ball along its predicted path, accepting weaker detections near it.
- [ ] Beehive view: where the ball passed the stumps, using the vertical plane of
  the batter's stumps. It needs stump tops, which the datasets above do not have; the
  [stumps](https://universe.roboflow.com/sarahs-workspace-7m1bb/stumps-q76nc) and
  [wicket](https://universe.roboflow.com/ibo-6phpm/wicket) keypoint datasets do.
