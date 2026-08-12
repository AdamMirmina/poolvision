# poolvision

Automatic scoring for backyard pool basketball from a single fixed camera. One
iPhone watches the pool, and the pipeline finds each shot, calls it a make or a
miss, and tries to attribute it to a player. Results are meant to land in
[poolean](https://github.com/AdamMirmina/poolean)'s existing database, so the app
stays the place everyone actually looks.

Kept separate from the poolean repo on purpose. That repo uploads to EAS on every
iOS build, and a Python project with torch, model weights and thousands of frames
has no business inside it.

## Where it stands

| | |
|---|---|
| Footage | 118 minutes of 4K across 7 recordings, hand-labeled |
| Shot detection | 20 of 20 on a hand-marked six-minute drill segment. The rules were tuned against those same 20, so treat it as a sanity check rather than a measured result |
| Make / miss | **82.2%** out-of-fold on 146 hand-judged shots, holding at **76.5%** on a recording held out entirely |
| Shooter attribution | **Does not work.** 41% against a 33% chance baseline |
| Live | Nothing runs live yet, and nothing writes into poolean's real stats |

Scope was fixed at the start and has not moved: detect the shot, call the
outcome, name the shooter. Assists, rebounds, contested shots and defensive stats
are deliberately excluded, because nothing in the literature supports them at this
camera distance.

## How it works

**The probe came before the pipeline.** The first commit is `src/probe.py`, which
measures four things against one real recording: whether people are detected,
whether the ball is detected, whether tracks survive, and whether the cap colors
separate. Nothing downstream was written until those numbers existed, because
writing a tracker first means writing it against an assumed camera angle, assumed
ball visibility, and assumed lighting.

**Two detection passes at two scales.** Stock COCO YOLO runs on the frame as read.
Where it finds nothing, a native-resolution crop is taken around the position a
fitted parabola predicts, and a locally fine-tuned ball model is asked instead.
That fine-tune scores badly by conventional metrics and does the one job it has.

**Rim crops rather than whole frames**, so the model sees the hoop at a useful
resolution instead of a handful of pixels in a 4K image.

**Identity comes from colored swim caps**, not faces or body appearance. People who
are shirtless, wet, and half-submerged thirty feet from a camera do not carry the
cues a re-identification model depends on. This is also the part that does not work
yet.

Everything runs on CPU. There is no NVIDIA GPU on the machine this was built on,
which shaped several of the choices above.

## Stack

Ultralytics YOLO11 for detection and pose plus a local fine-tune, OpenCV, ffmpeg
for seeking, scikit-learn gradient boosting for make/miss, and a frozen ResNet18
with a linear head in PyTorch as a second opinion. Labeling and review run through
poolean's web app against a `vision_shots` collection.

## Layout

```
src/          pipeline: probe, detection, tracking, rim watching, classification
labels/       hand-called ground truth, kept in git deliberately
tools/        one-off scripts for cutting clips, uploading, and review
docs/         design notes
```

Footage, frames and model weights are not in the repository and never will be.
`labels/` holds the hand-called outcomes, which are small, expensive to recreate,
and worth more than the video they came from.
