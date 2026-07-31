"""Who took the shot? A pilot, on twenty shots, before spending hours on it.

Scoring works by watching two small crops around the rims, which is cheap. The
shooter is thirty feet away in the pool and never appears in those crops, so
attribution needs the opposite thing: the full frame.

The route is one phase 3 already measured. The ball tracks WELL in open water --
99.5% of samples correct, unbroken runs to 9.2 seconds -- and it was the rim
region that failed. So follow the ball backwards from the hoop to where it was
released, find the person nearest that point, and read their cap.

Every step of that can fail, and the point of a pilot is to find out WHICH
before committing to a full pass:

  1. is the ball detected in open water on the way to the hoop?
  2. does a release point fall out of the trajectory?
  3. is anyone detected near it?
  4. does that person carry a readable cap color?

Deliberately reports a count for each stage rather than a single success rate.
"A quarter of shots attributed" says nothing about what to fix; "the ball is
found 80% of the time but a person is only near it 30% of the time" says
exactly what to fix.

    python src/attribute.py --video IMG_2482.MOV --shots 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops

ROOT = Path(__file__).resolve().parent.parent
SPORTS_BALL, PERSON = 32, 0

# The pool, in frame coordinates. Outside it is deck, grass and foliage: all
# strongly colored, none of it a swimmer.
POOL = (1150, 250, 3500, 1750)

# Measured off this footage in caps_probe: the water sits near 192 degrees, the
# ball at 320-354, skin around 0-20. A cap has to be saturated and away from
# all of those to count.
WATER_HUE_CV = 96      # OpenCV's 0-179 scale
WATER_MARGIN = 18
MIN_SAT = 110


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482.MOV")
    p.add_argument("--shots", type=int, default=20)
    p.add_argument("--lookback", type=float, default=2.0,
                   help="seconds before the descent to search for the release")
    return p.parse_args()


def cap_hue(frame, box, np, cv2):
    """The dominant cap-like hue in the top of a person's box.

    The top, not above it: someone with their arms up has a detection box that
    already contains whatever they are holding, so sampling above the box lands
    on the water behind them. That mistake cost a whole stage in phase 3.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    h = max(1, y2 - y1)
    crop = frame[max(0, y1):max(1, y1 + int(h * 0.35)), max(0, x1):max(1, x2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    dh = np.minimum(np.abs(H - WATER_HUE_CV), 180 - np.abs(H - WATER_HUE_CV))
    m = (S > MIN_SAT) & (V > 70) & (dh > WATER_MARGIN)
    if m.sum() < 60:
        return None
    return float(np.median(H[m])) * 2, int(m.sum())


def main():
    args = parse_args()
    import cv2
    import numpy as np
    from ultralytics import YOLO

    import makemiss as M

    judged = [j for j in json.loads((ROOT / "labels/judged.json").read_text(encoding="utf-8"))
              if j["video"] == args.video and j["label"] != "notshot"]
    judged.sort(key=lambda j: j["t"])
    shots = judged[:args.shots]
    print(f"{len(shots)} shots from {args.video}\n")

    rig = hoops.rig_for(args.video)
    cap = cv2.VideoCapture(str(ROOT / "footage" / args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO("yolo11s.pt")

    stage = Counter()
    rows = []
    for j in shots:
        stage["shots"] += 1
        rim = rig.rims[j["hoop"]]
        rimx, rimy = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2

        f_end = int(j["t"] * fps)
        f_start = max(0, int((j["t"] - args.lookback) * fps))
        # Every 3rd frame: the ball moves far enough between them to trace an
        # arc, and this is a pilot, not the production pass.
        frames = list(range(f_start, f_end + 1, 3))

        balls, people = [], []
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                continue
            r = model.predict(fr, conf=0.20, verbose=False, classes=[SPORTS_BALL, PERSON],
                              imgsz=1280)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cls = int(b.cls[0])
                if cls == SPORTS_BALL and POOL[0] <= cx <= POOL[2] and POOL[1] <= cy <= POOL[3]:
                    balls.append({"f": f, "x": cx, "y": cy, "conf": float(b.conf[0])})
                elif cls == PERSON and POOL[0] <= cx <= POOL[2] and POOL[1] <= cy <= POOL[3]:
                    people.append({"f": f, "box": (x1, y1, x2, y2), "x": cx, "y": cy, "frame": fr})

        if not balls:
            rows.append({"clock": j["clock"], "stage": "no ball in open water"})
            continue
        stage["ball seen in flight"] += 1

        # The release is the earliest sighting far enough from the rim to be a
        # throw rather than the shot arriving.
        far = [b for b in balls if ((b["x"] - rimx) ** 2 + (b["y"] - rimy) ** 2) ** 0.5 > 400]
        if not far:
            rows.append({"clock": j["clock"], "stage": "ball never far from the rim"})
            continue
        rel = min(far, key=lambda b: b["f"])
        stage["release point found"] += 1

        near = [p for p in people if abs(p["f"] - rel["f"]) <= 6]
        if not near:
            rows.append({"clock": j["clock"], "stage": "nobody detected near the release"})
            continue
        shooter = min(near, key=lambda p: ((p["x"] - rel["x"]) ** 2 + (p["y"] - rel["y"]) ** 2) ** 0.5)
        dist = ((shooter["x"] - rel["x"]) ** 2 + (shooter["y"] - rel["y"]) ** 2) ** 0.5
        stage["person found near release"] += 1

        hue = cap_hue(shooter["frame"], shooter["box"], np, cv2)
        if not hue:
            rows.append({"clock": j["clock"], "stage": "shooter found, no cap color", "dist": round(dist)})
            continue
        stage["cap color read"] += 1
        rows.append({"clock": j["clock"], "stage": "attributed", "dist": round(dist),
                     "hue": round(hue[0]), "px": hue[1]})
    cap.release()

    print(f"{'stage':<34} {'shots':>6}")
    for k in ("shots", "ball seen in flight", "release point found",
              "person found near release", "cap color read"):
        print(f"{k:<34} {stage[k]:>6}")

    print("\nper shot:")
    for r in rows:
        extra = ""
        if "hue" in r:
            extra = f"  hue {r['hue']}deg from {r['px']}px, shooter {r['dist']}px from the ball"
        elif "dist" in r:
            extra = f"  shooter {r['dist']}px from the ball"
        print(f"  {r['clock']:>6}  {r['stage']}{extra}")

    hues = [r["hue"] for r in rows if "hue" in r]
    if len(hues) >= 3:
        bins = Counter(int(h // 30) * 30 for h in hues)
        print(f"\ncap hues found: {sorted(bins.items())}")
        print(f"distinct 30-degree bands: {len(bins)} across {len(hues)} attributed shots")

    (ROOT / "out/attribute_pilot.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
