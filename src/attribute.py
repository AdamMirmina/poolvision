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
    p.add_argument("--skip", type=int, default=0,
                   help="skip this many shots first. The caps go on partway through "
                        "IMG_2481, and the first run took the earliest 20 shots, which "
                        "are all bare-headed -- a clean pipeline reporting a real zero.")
    p.add_argument("--lookback", type=float, default=2.0,
                   help="seconds before the descent to search for the release")
    return p.parse_args()


# Skin and the ball both live at the warm end. Measured on this footage: skin
# 0-20 degrees, the ball 320-354. The caps that showed up in the caps-only
# videos sit at 40-100. So a cap reading has to EXCLUDE the warm end, not just
# exclude the water -- the first version of this did only the latter and
# duly reported 12 of 13 shooters as hue 6-10, which is a suntan.
SKIN_BALL_LO, SKIN_BALL_HI = 300, 30      # degrees, wrapping through 0
CAP_LO, CAP_HI = 32, 300                  # everything else


def _is_cap_hue(deg):
    return CAP_LO <= deg <= CAP_HI


def cap_hue(frame, box, np, cv2):
    """The cap color on a person, or nothing.

    Samples the top FIFTH of the detection box, not the top third. A swimmer
    with their arms raised has a box whose upper third is mostly arm, and arm
    reads as a confident, saturated, completely useless hue.

    Returns None rather than a warm-end hue on purpose. "No cap seen" is a
    usable answer; "this player's cap is the color of skin" is not, and
    silently accepting it is how the phase 3 color stage came to report the
    water as everyone's identity.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    h = max(1, y2 - y1)
    crop = frame[max(0, y1):max(1, y1 + max(6, int(h * 0.20))), max(0, x1):max(1, x2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    dh = np.minimum(np.abs(H - WATER_HUE_CV), 180 - np.abs(H - WATER_HUE_CV))
    deg = H * 2
    colored = (S > MIN_SAT) & (V > 70) & (dh > WATER_MARGIN) & (deg >= CAP_LO) & (deg <= CAP_HI)
    # A white cap is in play (seen in the contact sheet) and has almost no
    # saturation, so a hue-based test throws it away. It is still perfectly
    # distinguishable: bright and colorless, which nothing else in the frame is
    # -- the water is saturated turquoise and skin is saturated warm.
    white = (S < 55) & (V > 165)
    if white.sum() > max(120, colored.sum() * 1.4):
        return -1.0, int(white.sum())          # -1 stands for white
    if colored.sum() < 40:
        return None
    return float(np.median(deg[colored])), int(colored.sum())


def main():
    args = parse_args()
    import cv2
    import numpy as np
    from ultralytics import YOLO

    import makemiss as M

    judged = [j for j in json.loads((ROOT / "labels/judged.json").read_text(encoding="utf-8"))
              if j["video"] == args.video and j["label"] != "notshot"]
    judged.sort(key=lambda j: j["t"])
    shots = judged[args.skip:args.skip + args.shots]
    # The label export doesn't carry the clock string, so derive it here rather
    # than re-exporting for one cosmetic field.
    for j in shots:
        j.setdefault("clock", f"{int(j['t']) // 60}:{int(j['t']) % 60:02d}")
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
