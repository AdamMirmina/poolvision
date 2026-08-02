"""Measure this session's rim boxes, and prove them by drawing.

The camera moves between sessions. hoops.py raises on an unknown recording
rather than falling back to another session's numbers, because a wrong rim box
produces a confident-looking overnight run that finds nothing, and "zero shots
detected" reads as a detector problem rather than a coordinates problem. That is
the expensive kind of wrong, and it has already cost this project one night.

Finds the hoops by their color. Both rims here are orange-red steel against
water, deck and foliage, and the backboard is a bright rectangle above each.
Every candidate is drawn onto a real frame so the boxes can be checked by eye
before hours of compute are spent on them.

    python src/newrig.py --video IMG_2528.MOV
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2528.MOV")
    p.add_argument("--samples", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(ROOT / "footage" / args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print("video not readable yet")
        return 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Average several frames so swimmers, splashes and the ball average away and
    # the fixed furniture stays. A hoop does not move; everything else does.
    acc = None
    n = 0
    for i in range(args.samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (0.2 + 0.6 * i / max(1, args.samples - 1))))
        ok, fr = cap.read()
        if not ok:
            continue
        acc = fr.astype(np.float32) if acc is None else acc + fr.astype(np.float32)
        n += 1
    cap.release()
    if not n:
        print("could not read frames")
        return 1
    bg = (acc / n).astype(np.uint8)
    cv2.imwrite(str(ROOT / f"out/rig_{args.video.replace('.MOV','')}_bg.png"), bg)

    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[:, :, 0].astype(int) * 2, hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    # Orange-red steel: warm hue, well saturated, not the pool's turquoise.
    rim = (((Hh <= 25) | (Hh >= 340)) & (S > 90) & (V > 70)).astype(np.uint8)
    rim = cv2.morphologyEx(rim, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnt, lab, stats, cent = cv2.connectedComponentsWithStats(rim, 8)

    cands = []
    for i in range(1, cnt):
        x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                            stats[i, cv2.CC_STAT_AREA])
        if area < 600 or area > 120000:
            continue
        if w < 40 or h < 20 or w > 900:
            continue
        if h > w * 2.2:                       # a rim is wider than it is tall
            continue
        if y > H * 0.82:                      # hoops are not on the floor of the frame
            continue
        cands.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})
    cands.sort(key=lambda c: -c["area"])

    vis = bg.copy()
    for k, c in enumerate(cands[:12]):
        cv2.rectangle(vis, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), (0, 255, 0), 4)
        cv2.putText(vis, f"{k}: {c['w']}x{c['h']} area {c['area']}", (c["x"], max(28, c["y"] - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    out = ROOT / f"out/rig_{args.video.replace('.MOV','')}_candidates.png"
    cv2.imwrite(str(out), cv2.resize(vis, (1800, int(1800 * H / W))))
    print(f"{len(cands)} rim-colored candidates in {args.video} ({W}x{H})")
    for k, c in enumerate(cands[:12]):
        side = "left" if c["x"] + c["w"] / 2 < W / 2 else "right"
        print(f"  {k}: ({c['x']},{c['y']})-({c['x']+c['w']},{c['y']+c['h']})  "
              f"{c['w']}x{c['h']}  area {c['area']}  {side} half")
    (ROOT / f"out/rig_{args.video.replace('.MOV','')}.json").write_text(
        json.dumps(cands[:12], indent=1), encoding="utf-8")
    print(f"wrote {out.name} -- LOOK at it before trusting any of these")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
