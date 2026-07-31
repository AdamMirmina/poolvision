"""Do the swim caps actually separate by color, on real footage?

Everything measured so far was a proxy: pool noodles and small balls held
deliberately still, in one frame, by people who were standing. The 2026-07-29
session is the first with caps on heads, on people moving, over ten minutes of
changing light. That is the question the whole identity plan rests on and it has
never been tested.

Deliberately model-free. Person detection is the expensive part of the pipeline
and the CPU is busy; a cap is a small, strongly saturated blob whose color is
nothing like the water, so it can be found by color alone. That also isolates
the measurement: if the caps don't separate here, no amount of better person
detection rescues them.

Method: sample frames across the video, mask pixels inside the pool that are
saturated AND far from the water's own hue, keep blobs of roughly cap size, and
report where their hues cluster. Distinct caps should show as distinct peaks.

Run: python src/caps_probe.py footage/IMG_2481.MOV --samples 40
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

WATER_HUE = 96          # OpenCV 0-179 scale, measured off this pool
WATER_MARGIN = 18       # hue distance before a pixel counts as "not water"
MIN_SAT = 110
MIN_VAL = 70
CAP_MIN_PX = 250        # a cap is small but not noise
CAP_MAX_PX = 6000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--out", type=Path, default=None)
    # The pool, in frame coordinates. Everything outside is grass, deck, foliage
    # and toys -- all strongly colored, none of it a cap.
    p.add_argument("--roi", default="1150,250,3500,1750", help="x1,y1,x2,y2")
    return p.parse_args()


def main():
    args = parse_args()
    import cv2
    import numpy as np

    x1, y1, x2, y2 = (int(v) for v in args.roi.split(","))
    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, total // args.samples)

    blobs = []
    for i in range(args.samples):
        f = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        roi = fr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
        dh = np.minimum(np.abs(H - WATER_HUE), 180 - np.abs(H - WATER_HUE))
        mask = ((S > MIN_SAT) & (V > MIN_VAL) & (dh > WATER_MARGIN)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        for j in range(1, n):
            area = stats[j, cv2.CC_STAT_AREA]
            if not (CAP_MIN_PX <= area <= CAP_MAX_PX):
                continue
            w, h = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
            if max(w, h) > 3 * min(w, h):     # a cap is roughly round, not a streak
                continue
            m = lab == j
            blobs.append({
                "t": round(f / fps, 1),
                "hue": float(np.median(H[m])) * 2,
                "sat": float(np.median(S[m])),
                "area": int(area), "w": int(w), "h": int(h),
                "x": int(cent[j][0]) + x1, "y": int(cent[j][1]) + y1,
            })
    cap.release()

    print(f"{len(blobs)} cap-sized colored blobs across {args.samples} frames\n")
    if not blobs:
        print("none found -- widen the ROI or loosen the thresholds")
        return

    # 20-degree hue bins: distinct caps should land in distinct bins.
    bins = Counter(int(b["hue"] // 20) * 20 for b in blobs)
    print(f"{'hue band':>12}  {'count':>5}  {'median size':>12}")
    for band in sorted(bins):
        sizes = [b["w"] for b in blobs if int(b["hue"] // 20) * 20 == band]
        bar = "#" * min(40, bins[band])
        print(f"{band:>4}-{band + 19:<7} {bins[band]:>5}  {sorted(sizes)[len(sizes) // 2]:>9} px  {bar}")

    strong = [band for band, c in bins.items() if c >= max(3, len(blobs) * 0.05)]
    print(f"\ndistinct color groups seen: {len(strong)} -> {sorted(strong)}")
    ws = sorted(b["w"] for b in blobs)
    print(f"cap width px: min {ws[0]}  median {ws[len(ws) // 2]}  max {ws[-1]}")

    if args.out:
        args.out.write_text(json.dumps(blobs, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
