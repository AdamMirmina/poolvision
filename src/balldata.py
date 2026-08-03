"""Build a ball-detection training set from the flights we already fitted.

Every ceiling hit today traces to one thing: the detector loses the ball exactly
where it matters. the through-the-hoop rule is only observable on 38% of
makes, his overlap veto lands below chance, arcs stop before the ball lands. One
cause, three symptoms.

The obvious fix is a ball detector fine-tuned on this footage, and the obvious
obstacle is labels. This produces them without anyone drawing a box:

  a parabola fitted to the sightings that DID happen predicts where the ball was
  in the frames that missed it.

Those predicted positions are exactly the hard cases -- the ball blurred, against
skin, inside the net, half-occluded by the rim -- which is precisely the data a
fine-tune needs and precisely what hand-labeling would find most tedious. The
easy frames come free alongside them, from the real detections.

Written as YOLO format: one image and one .txt per sample, class 0 = ball.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc
import hoops
import shooter
from frames import FFMPEG

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out/balldata"

# A crop big enough to hold context but small enough that the ball is a real
# fraction of the image. 960 at native resolution puts a 35px ball at ~4% of the
# frame, against 0.9% in a 3840-wide frame downscaled to 1280.
CROP = 960
# Only trust the fit where it is actually good. A loose fit predicts positions
# that are confidently wrong, and a wrong label is worse than a missing one.
MAX_RMS = 14.0
MIN_PTS = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="IMG_2481,IMG_2482,IMG_2528")
    p.add_argument("--per-shot", type=int, default=7)
    p.add_argument("--val-frac", type=float, default=0.15)
    return p.parse_args()


def sample_frames(video, times, box):
    """Pull a set of native-resolution crops in one ffmpeg pass each."""
    import cv2
    got = {}
    x1, y1, x2, y2 = box
    for t in times:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "f.png"
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
                            "-i", str(video), "-frames:v", "1",
                            "-vf", f"crop={x2-x1}:{y2-y1}:{x1}:{y1}", str(f)],
                           capture_output=True)
            if f.exists():
                im = cv2.imread(str(f))
                if im is not None:
                    got[t] = im
    return got


def main():
    args = parse_args()
    import numpy as np

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    kept = {"train": 0, "val": 0}
    hard = 0

    for vid in args.videos.split(","):
        scans = sorted((ROOT / "out/scans").glob(f"{vid}_*.json"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if not scans:
            print(f"{vid}: no cached scans, skipping")
            continue
        video = ROOT / "footage" / f"{vid}.MOV"
        if not video.exists():
            print(f"{vid}: footage missing, skipping")
            continue
        rig = hoops.rig_for(f"{vid}.MOV")
        n_v = 0
        for sp in scans:
            n = int(sp.stem.rsplit("_", 1)[1])
            d = json.loads(sp.read_text(encoding="utf-8"))
            track = d["ball_track"]
            if len(track) < MIN_PTS:
                continue
            fl = shooter.fit_arc(track)
            if not fl or fl["rms"] > MAX_RMS or fl["n"] < MIN_PTS:
                continue

            seen = {round(b["t"], 3): b for b in track}
            sizes = [b.get("w", 34) for b in track if b.get("w")]
            size = float(np.median(sizes)) if sizes else 34.0

            t0, t1 = fl["t"], fl["t"] + fl["span"]
            fps = 60.0 if vid in ("IMG_2528", "IMG_2529") else 30.0
            steps = [t0 + (t1 - t0) * i / (args.per_shot - 1) for i in range(args.per_shot)]
            # Prefer the moments the detector MISSED: those are the hard ones and
            # the whole reason for doing this.
            steps.sort(key=lambda t: round(t, 3) in seen)

            wants = {}
            for t in steps:
                bx, by = arc.at(fl["fit"], t)
                if not (0 < bx < 3840 and 0 < by < 2160):
                    continue
                cx = int(min(3840 - CROP, max(0, bx - CROP / 2)))
                cy = int(min(2160 - CROP, max(0, by - CROP / 2)))
                wants[t] = (bx, by, (cx, cy, cx + CROP, cy + CROP))

            for t, (bx, by, box) in wants.items():
                ims = sample_frames(video, [t], box)
                im = ims.get(t)
                if im is None or im.shape[0] < 10:
                    continue
                lx = (bx - box[0]) / CROP
                ly = (by - box[1]) / CROP
                lw = lh = size / CROP
                if not (0.02 < lx < 0.98 and 0.02 < ly < 0.98):
                    continue
                split = "val" if rng.random() < args.val_frac else "train"
                stem = f"{vid}_{n}_{int(t*1000)}"
                import cv2
                cv2.imwrite(str(OUT / f"images/{split}/{stem}.jpg"), im,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (OUT / f"labels/{split}/{stem}.txt").write_text(
                    f"0 {lx:.6f} {ly:.6f} {lw:.6f} {lh:.6f}\n", encoding="utf-8")
                kept[split] += 1
                if round(t, 3) not in seen:
                    hard += 1
            n_v += 1
        print(f"{vid}: {n_v} shots contributed")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: 1\nnames: [ball]\n", encoding="utf-8")
    tot = kept["train"] + kept["val"]
    print(f"\n{tot} samples ({kept['train']} train, {kept['val']} val)")
    print(f"{hard} of them are frames the detector MISSED -- the point of the exercise")
    print(f"wrote {OUT/'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
