"""Does the fine-tune fill the holes in the arcs? The measurement that matters.

Three separate ceilings on this project trace back to one cause: the detector
loses the ball at the rim. Through-the-hoop is observable on 38% of makes, the
overlap veto lands below chance, and arcs stop before the ball lands. So the
useful question about a new detector is not its mAP, it is whether it recovers
the frames the current one drops.

A hole is a frame inside a fitted flight where the track has no sighting. The
fitted parabola says where the ball should have been. Both detectors are asked,
and scored against that position.

Two things keep this honest:

  Frames used in training are excluded by name. balldata sampled 7 frames per
  shot and encoded the shot and millisecond in every stem, so the exact set is
  known and can be removed rather than estimated.

  The parabola was fitted WITHOUT these frames, since a hole is by definition a
  frame with no detection to fit. So the reference position is a prediction at
  that time, not a memory of it.

Each detector is run the way it would actually be used: stock on the downscaled
frame as the pipeline does today, the fine-tune on a native-resolution crop
around the expected position, which is the scale it was trained at.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc
import shooter
from frames import FFMPEG

ROOT = Path(__file__).resolve().parent.parent
TUNED = ROOT / "out/balltrain/ball/weights/best.pt"
TRAINED_ON = ROOT / "out/balldata"
CROP = 960
BALL = 32
# A recovery has to land near the predicted position to count. Same standard for
# both detectors.
TOL_PX = 55.0


def trained_times():
    """(video, shot, ms) the model actually saw, read off the sample filenames."""
    seen = set()
    for split in ("train", "val"):
        for f in (TRAINED_ON / f"images/{split}").glob("*.jpg"):
            parts = f.stem.rsplit("_", 2)
            if len(parts) == 3:
                seen.add((parts[0], parts[1], parts[2]))
    return seen


def grab(video, t, box=None):
    import cv2
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "f.png"
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
               "-i", str(video), "-frames:v", "1"]
        if box:
            x1, y1, x2, y2 = box
            cmd += ["-vf", f"crop={x2-x1}:{y2-y1}:{x1}:{y1}"]
        cmd.append(str(f))
        subprocess.run(cmd, capture_output=True)
        return cv2.imread(str(f)) if f.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="IMG_2482,IMG_2528")
    p.add_argument("--holes", type=int, default=60)
    # Shift the crop so the ball is NOT at its center. Without this the test is
    # confounded: the crop is centered on the predicted position and a hit is
    # scored within a radius of that same point, so a model that always draws a
    # box in the middle scores 100% without looking at anything. Offsetting by
    # more than the tolerance means only a detector that actually finds the ball
    # can pass.
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()

    import cv2
    from ultralytics import YOLO

    if not TUNED.exists():
        print("no fine-tuned weights")
        return 1
    skip = trained_times()
    stock = YOLO(str(ROOT / "yolo11s.pt"))
    tuned = YOLO(str(TUNED))

    holes = []
    for vid in args.videos.split(","):
        video = ROOT / "footage" / f"{vid}.MOV"
        if not video.exists():
            continue
        fps = 60.0 if vid in ("IMG_2528", "IMG_2529") else 30.0
        for sp in sorted((ROOT / "out/scans").glob(f"{vid}_*.json")):
            n = sp.stem.rsplit("_", 1)[1]
            d = json.loads(sp.read_text(encoding="utf-8"))
            track = d.get("ball_track") or []
            if len(track) < 8:
                continue
            fl = shooter.fit_arc(track)
            if not fl or fl["rms"] > 14.0:
                continue
            seen = {round(b["t"], 2) for b in track}
            t0, t1 = fl["t"], fl["t"] + fl["span"]
            k = 0
            while t0 + k / fps <= t1:
                t = t0 + k / fps
                k += 1
                if round(t, 2) in seen:
                    continue
                if (vid, n, str(int(t * 1000))) in skip:
                    continue
                bx, by = arc.at(fl["fit"], t)
                if not (0 < bx < 3840 and 0 < by < 2160):
                    continue
                holes.append((video, vid, t, bx, by))
    if not holes:
        print("no holes found")
        return 1

    step = max(1, len(holes) // args.holes)
    holes = holes[:step][:args.holes]
    print(f"{len(holes)} holes, none of them frames the model trained on\n")

    s_hit = t_hit = 0
    for video, vid, t, bx, by in holes:
        fr = grab(video, t)
        if fr is not None:
            r = stock.predict(fr, conf=0.10, classes=[BALL], imgsz=1280,
                              device="cpu", verbose=False)[0]
            H, W = fr.shape[:2]
            for b in r.boxes:
                x, y = b.xywh[0].tolist()[:2]
                if ((x * 3840 / W - bx) ** 2 + (y * 2160 / H - by) ** 2) ** 0.5 < TOL_PX:
                    s_hit += 1
                    break

        off = args.offset
        cx = int(min(3840 - CROP, max(0, bx - CROP / 2 + off)))
        cy = int(min(2160 - CROP, max(0, by - CROP / 2 - off)))
        cr = grab(video, t, (cx, cy, cx + CROP, cy + CROP))
        if cr is not None:
            r = tuned.predict(cr, conf=0.05, imgsz=640, device="cpu", verbose=False)[0]
            for b in sorted(r.boxes, key=lambda b: -float(b.conf.item()))[:3]:
                x, y = b.xywh[0].tolist()[:2]
                if ((x + cx - bx) ** 2 + (y + cy - by) ** 2) ** 0.5 < TOL_PX:
                    t_hit += 1
                    break

    n = len(holes)
    print(f"stock, on the frame as the pipeline reads it today")
    print(f"  recovered {s_hit}/{n}  ({100*s_hit/n:.0f}%)")
    print(f"fine-tuned, on a native-resolution crop at the expected position")
    print(f"  recovered {t_hit}/{n}  ({100*t_hit/n:.0f}%)")
    d = t_hit - s_hit
    print()
    if d > 0:
        print(f"{d} holes filled that are empty today, {100*d/n:.0f}% of them.")
        print("Those are the frames the through-the-hoop rule and the overlap")
        print("veto were missing, so both get re-measured on the fuller arcs.")
    else:
        print("No gain on the frames that actually matter. The mAP number was")
        print("measuring the easy cases; do not adopt it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
