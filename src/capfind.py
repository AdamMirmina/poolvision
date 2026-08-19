"""Find the colored swim caps in a frame, as people.

Five of the ten wrong attributions review commented on are the same failure: the
shooter was never detected as a person, so no rule could pick him. Twice review says
why in the same breath -- "partially submerged but CAP and body and arm still
visible", "green was behind white and not recognized as a person despite CAP
visible".

The cap is the better person-detector here, and it is not a workaround. A pose
model needs a body, and these bodies are half underwater, behind each other, or
sixty feet away at the deep end. A cap is a saturated blob of a color nothing
else in the scene has, sitting on the one part of a swimmer that is always above
water. It survives exactly the conditions that defeat pose.

The caps were always in the plan as IDENTITY. This makes them detection as well,
which is what the notes have been pointing at all along.

    python src/capfind.py --video IMG_2482.MOV --t 462
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent

# A cap is small, round-ish and strongly colored. Sized in pixels at native 4K:
# measured off real frames, caps run about 30-70px across.
MIN_AREA = 220
MAX_AREA = 6000
MIN_FILL = 0.45        # blob area over its bounding box, so long streaks are out
MAX_ASPECT = 2.4
# The pool's own blue, excluded: it is the one saturated color that is not a cap.
WATER_H = (86, 112)
MIN_SAT = 60           # pink tops out near 87, so this cannot go much higher
MIN_VAL = 70
# How much of the patch below a blob must be water for it to be a head.
WATER_BELOW = 0.25


def find(frame, pool=None):
    """[{x, y, w, hue, sat, val}] for each cap-like blob, brightest first.

    `pool` limits the search to where people can be, which keeps it off the
    umbrella, the pool floats and the neighbour's flowers.
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = (s >= MIN_SAT) & (v >= MIN_VAL)
    m &= ~((h >= WATER_H[0]) & (h <= WATER_H[1]))       # not the water
    mask = m.astype(np.uint8)
    if pool:
        keep = np.zeros_like(mask)
        x1, y1, x2, y2 = [int(z) for z in pool]
        keep[max(0, y1):y2, max(0, x1):x2] = 1
        mask *= keep
    # Close pinholes from the cap's own highlight, then drop speckle.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # A cap is on a swimmer, and a swimmer is IN the water. Everything this
    # found outside the pool -- foliage along the fence, deck stains, a pool
    # float, the backboard -- is the right color and the right size and is not a
    # person. Requiring water underneath removes them without touching the caps,
    # because a head always has water below it and a leaf never does.
    water = ((h >= WATER_H[0]) & (h <= WATER_H[1]) & (s >= 50)).astype(np.uint8)
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        if bw == 0 or bh == 0:
            continue
        asp = max(bw / bh, bh / bw)
        if asp > MAX_ASPECT or area / float(bw * bh) < MIN_FILL:
            continue
        cx, cy = cent[i]
        # Look just below the blob, a couple of cap-heights down, for water.
        span = int(max(bw, bh))
        y0 = min(frame.shape[0] - 1, y + bh + span // 2)
        y1b = min(frame.shape[0], y0 + span * 3)
        x0 = max(0, int(cx) - span)
        x1b = min(frame.shape[1], int(cx) + span)
        below = water[y0:y1b, x0:x1b]
        if below.size == 0 or below.mean() < WATER_BELOW:
            continue
        sub = lab[y:y + bh, x:x + bw] == i
        out.append({
            "x": float(cx), "y": float(cy), "w": float(max(bw, bh)),
            "hue": float(np.median(h[y:y + bh, x:x + bw][sub])),
            "sat": float(np.median(s[y:y + bh, x:x + bw][sub])),
            "val": float(np.median(v[y:y + bh, x:x + bw][sub])),
            "area": int(area),
        })
    out.sort(key=lambda c: -c["area"])
    return out


def as_people(caps, boxes, frame_shape):
    """Caps with no person box on them, as candidate people.

    A cap already inside somebody's box is that person and adds nothing. One with
    no box is the case that costs attribution: a real player the pose model never
    saw. The body is estimated from the cap -- roughly two cap-widths wide and
    four tall, below it -- which is enough for the distance-to-ball comparison
    that decides the pick. It carries no keypoints, because there are none.
    """
    out = []
    for c in caps:
        covered = False
        for (bx1, by1, bx2, by2) in boxes:
            if bx1 <= c["x"] <= bx2 and by1 <= c["y"] <= by2:
                covered = True
                break
        if covered:
            continue
        w = max(24.0, c["w"])
        H = frame_shape[0]
        out.append({
            "box": (c["x"] - w * 1.2, c["y"] - w * 0.6,
                    c["x"] + w * 1.2, min(H - 1, c["y"] + w * 4.0)),
            "cap": c,
            "kp": None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="IMG_2482.MOV")
    ap.add_argument("--t", type=float, default=462.0)
    args = ap.parse_args()

    import cv2
    import hoops
    import rigcheck

    rig = hoops.rig_for(args.video)
    fr = rigcheck.grab(ROOT / "footage" / args.video, args.t)
    if fr is None:
        print("no frame")
        return 1
    caps = find(fr, rig.pool)
    print(f"{len(caps)} cap-like blobs at t={args.t:g}")
    for c in caps[:12]:
        print(f"  ({c['x']:.0f},{c['y']:.0f}) {c['w']:.0f}px  "
              f"hue {c['hue']:.0f} sat {c['sat']:.0f} val {c['val']:.0f}  area {c['area']}")
    for c in caps:
        cv2.circle(fr, (int(c["x"]), int(c["y"])), int(c["w"]), (0, 255, 255), 3)
    out = ROOT / f"out/capfind_{Path(args.video).stem}_{int(args.t)}.jpg"
    cv2.imwrite(str(out), cv2.resize(fr, None, fx=0.5, fy=0.5),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
