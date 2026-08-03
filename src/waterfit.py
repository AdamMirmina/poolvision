"""Measure the waterline and the wall's direction under each hoop.

RIG-SETUP.md step 5. Walk down several columns under the rim until the water mask
starts, fit a line through those entry points: the intercept is the waterline and
the slope is the wall. The drop zone has to start at the waterline and shear to
the wall, so both numbers feed it directly.

Skipping this does not fail loudly. IMG_2528 had no water measured at all, so its
drop zone silently fell back to an axis-aligned rectangle, and IMG_2482's right
hoop carried an intercept ~80px above the real waterline, which put the top of
the zone up on the coping where a ball can never be.

Robust to swimmers and splashes by fitting on the MEDIAN of several frames and
rejecting columns whose entry point sits far off the line, since a person
standing at the wall produces an entry point metres from it.

    python src/waterfit.py --video IMG_2482.MOV
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops
from frames import FFMPEG

ROOT = Path(__file__).resolve().parent.parent

# The pool's own blue. Wide enough to hold across sun and shade, narrow enough to
# exclude the sky and the tile.
H_LO, H_HI, S_MIN, V_MIN = 82, 110, 60, 90
SPAN = 1.4     # how far either side of the rim, in rim widths, to sample
STEP = 12      # column spacing
RUN = 14       # consecutive water pixels needed to call it the waterline


def frames(video, times):
    import cv2
    out = []
    for t in times:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "f.png"
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
                            "-i", str(video), "-frames:v", "1", str(f)],
                           capture_output=True)
            im = cv2.imread(str(f)) if f.exists() else None
            if im is not None:
                out.append(im)
    return out


def water_mask(im):
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((h >= H_LO) & (h <= H_HI) & (s >= S_MIN) & (v >= V_MIN)).astype(np.uint8)
    # Close over swimmers and ripples so a column does not enter the water, exit
    # through someone's shoulder, and re-enter lower down.
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))


def entries(mask, rim):
    """First row of a sustained run of water, per column, under the rim."""
    import numpy as np
    x1, y1, x2, y2 = rim
    w = x2 - x1
    cx = (x1 + x2) / 2
    lo = max(0, int(cx - SPAN * w))
    hi = min(mask.shape[1] - 1, int(cx + SPAN * w))
    y_start = int(y2)
    pts = []
    for x in range(lo, hi, STEP):
        col = mask[y_start:, x]
        if col.size < RUN:
            continue
        run = 0
        for i, val in enumerate(col):
            run = run + 1 if val else 0
            if run >= RUN:
                pts.append((x, y_start + i - RUN + 1))
                break
    return pts


def fit(pts, cx):
    """Least squares, then one rejection pass, then refit."""
    import numpy as np
    if len(pts) < 8:
        return None
    a = np.array(pts, dtype=float)
    for _ in range(2):
        m, b = np.polyfit(a[:, 0], a[:, 1], 1)
        resid = np.abs(a[:, 1] - (m * a[:, 0] + b))
        keep = resid <= max(18.0, 2.0 * np.median(resid))
        if keep.sum() < 8:
            break
        a = a[keep]
    m, b = np.polyfit(a[:, 0], a[:, 1], 1)
    return {"water_y_at_center": float(m * cx + b), "slope": float(m), "n": int(len(a))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482.MOV")
    p.add_argument("--times", default="")
    # Draw the entry points the fit was made from. A number that disagrees with
    # the eye is not settled by arguing about it: the points show whether the
    # mask found the waterline or something else.
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    import numpy as np
    rig = hoops.rig_for(args.video)
    video = ROOT / "footage" / args.video
    times = ([float(s) for s in args.times.split(",")] if args.times
             else [20.0, 60.0, 120.0, 200.0, 300.0])
    ims = frames(video, times)
    if not ims:
        print("no frames read")
        return 1
    masks = [water_mask(im) for im in ims]

    print(f"{args.video}: {len(ims)} frames\n")
    result = {}
    for hoop, rim in rig.rims.items():
        cx = (rim[0] + rim[2]) / 2
        per = []
        for m in masks:
            f = fit(entries(m, rim), cx)
            if f:
                per.append(f)
        if not per:
            print(f"  {hoop}: no waterline found")
            continue
        # Median across frames, so one frame full of swimmers cannot carry it.
        wy = float(np.median([f["water_y_at_center"] for f in per]))
        sl = float(np.median([f["slope"] for f in per]))
        result[hoop] = {"water_y_at_center": wy, "slope": sl}
        was = (rig.water or {}).get(hoop)
        print(f"  {hoop}: water_y_at_center={wy:.1f}  slope={sl:+.4f}   "
              f"(from {len(per)} frames)")
        if was:
            print(f"      rig currently says {was['water_y_at_center']:.1f} / "
                  f"{was['slope']:+.4f}   -> off by "
                  f"{wy - was['water_y_at_center']:+.0f}px, "
                  f"{sl - was['slope']:+.3f} slope")
        else:
            print("      rig currently has NOTHING for this hoop")

    if args.show:
        import cv2
        im = ims[0].copy()
        for hoop, rim in rig.rims.items():
            cx = (rim[0] + rim[2]) / 2
            pts = entries(masks[0], rim)
            for (x, y) in pts:
                cv2.circle(im, (int(x), int(y)), 5, (0, 0, 255), -1)
            f = result.get(hoop)
            if f:
                m, wy = f["slope"], f["water_y_at_center"]
                x1 = int(cx - 400)
                x2 = int(cx + 400)
                cv2.line(im, (x1, int(wy + m * (x1 - cx))), (x2, int(wy + m * (x2 - cx))),
                         (0, 255, 255), 3)
            cv2.rectangle(im, (int(rim[0]), int(rim[1])), (int(rim[2]), int(rim[3])),
                          (255, 255, 255), 2)
        out = ROOT / f"out/waterfit_{Path(args.video).stem}.jpg"
        cv2.imwrite(str(out), cv2.resize(im, None, fx=0.5, fy=0.5),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\nwrote {out}  (red dots = where a column first hit water)")

    print("\npaste into hoops.py:")
    print("        water=" + repr({k: {"water_y_at_center": round(v["water_y_at_center"], 1),
                                       "slope": round(v["slope"], 4)}
                                   for k, v in result.items()}) + ",")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
