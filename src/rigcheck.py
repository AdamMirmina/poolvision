"""Draw the rig's geometry on a real frame, so it can be checked by looking.

RIG-SETUP.md's first rule is the one that keeps getting skipped: draw it on a
real frame and look at it before trusting a single number that comes out of it.
Every geometric constant here has been wrong at least once, and every one was
caught by drawing rather than by reasoning.

This draws all of it at once -- rim box and tilted ellipse, net box, drop zone,
waterline, wall direction, three-point posts -- and says in the corner which
pieces the rig is actually carrying, because a missing measurement is invisible
otherwise. IMG_2528 had no water measured at all, so its drop zone silently fell
back to an axis-aligned rectangle nowhere near the wall.

    python src/rigcheck.py --video IMG_2482.MOV --t 30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dropzone
import hoops
from frames import FFMPEG

ROOT = Path(__file__).resolve().parent.parent

RIM = (60, 235, 245)     # yellow
DROP = (70, 70, 70)      # dark gray
WATERC = (255, 190, 60)  # the measured waterline
WALLC = (120, 200, 255)  # the wall direction it was fitted to
POSTC = (200, 120, 255)  # three-point posts
WHITE = (255, 255, 255)


def grab(video, t):
    import cv2
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "f.png"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
                        "-i", str(video), "-frames:v", "1", str(f)],
                       capture_output=True)
        return cv2.imread(str(f)) if f.exists() else None


def label(im, xy, text, color):
    import cv2
    x, y = int(xy[0]), int(xy[1])
    cv2.rectangle(im, (x - 2, y - 20), (x + 11 * len(text), y + 5), (0, 0, 0), -1)
    cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def draw(im, rig, hoop):
    import cv2
    import numpy as np
    rim = rig.rims[hoop]
    x1, y1, x2, y2 = [int(v) for v in rim]
    w = x2 - x1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    cv2.rectangle(im, (x1, y1), (x2, y2), RIM, 3)
    tilt = (rig.tilt or {}).get(hoop, 0.0)
    cv2.ellipse(im, (cx, cy), (w // 2, max(6, (y2 - y1) // 2)), tilt, 0, 360, RIM, 3)
    label(im, (x1, y1 - 10), f"rim {hoop} tilt {tilt:g}", RIM)

    b = dropzone.box(rim)
    cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (200, 200, 200), 2)
    label(im, (int(b[0]), int(b[1]) - 8), "net box", (200, 200, 200))

    water = (rig.water or {}).get(hoop)
    dropax = (rig.drop or {}).get(hoop)
    q = dropzone.zone(rim, water, dropax)
    pts = np.array([[int(a), int(bb)] for a, bb in q], dtype=np.int32)
    cv2.polylines(im, [pts], True, DROP, 4)
    label(im, (pts[0][0], pts[0][1] - 8),
          "drop zone" + ("" if water else "  NO WATER MEASURED -- axis aligned"),
          DROP if water else (60, 60, 255))

    if water:
        # The waterline it was told about, and the wall it was told to follow,
        # drawn across the zone's own width so a wrong slope is obvious.
        wy, m = water["water_y_at_center"], water["slope"]
        half = int(dropzone.HALF_W * w) + 40
        p0 = (cx - half, int(wy - m * half))
        p1 = (cx + half, int(wy + m * half))
        cv2.line(im, p0, p1, WATERC, 3)
        label(im, (p0[0], p0[1] - 8), f"waterline y={wy:.0f} slope={m:+.2f}", WATERC)
        # The wall direction as an arrow, so its sign can be read at a glance.
        cv2.arrowedLine(im, (cx, int(wy)), (cx + 220, int(wy + m * 220)), WALLC, 3, tipLength=0.18)

    # {side: (anchor, direction)}. The anchor is meant to be the BASE of the
    # white pole where it enters the float, not the detection box's center and
    # not the blue base beside it -- that distinction has been got wrong before.
    for side, tl in (rig.three_lines or {}).items():
        (ax, ay), (dx, dy) = tl
        ax, ay = int(ax), int(ay)
        cv2.drawMarker(im, (ax, ay), POSTC, cv2.MARKER_CROSS, 70, 4)
        cv2.circle(im, (ax, ay), 18, POSTC, 3)
        n = max(abs(dx), abs(dy)) or 1
        ux, uy = dx / n, dy / n
        cv2.line(im, (int(ax - ux * 3000), int(ay - uy * 3000)),
                 (int(ax + ux * 3000), int(ay + uy * 3000)), POSTC, 2, cv2.LINE_AA)
        label(im, (ax + 24, ay - 24), f"{side} post base ({ax},{ay})", POSTC)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482.MOV")
    p.add_argument("--t", type=float, default=30.0)
    p.add_argument("--hoop", default="both")
    args = p.parse_args()

    import cv2
    rig = hoops.rig_for(args.video)
    im = grab(ROOT / "footage" / args.video, args.t)
    if im is None:
        print("could not read a frame")
        return 1

    hoopsel = list(rig.rims) if args.hoop == "both" else [args.hoop]
    for h in hoopsel:
        draw(im, rig, h)

    have = []
    have.append(f"water: {'yes' if rig.water else 'MISSING'}")
    have.append(f"tilt: {'yes' if rig.tilt else 'missing'}")
    have.append(f"three_lines: {'yes' if rig.three_lines else 'missing'}")
    label(im, (40, 60), f"{args.video} t={args.t:g}   " + "   ".join(have), WHITE)

    out = ROOT / f"out/rigcheck_{Path(args.video).stem}_{args.t:g}.jpg"
    # Half size: the whole 4K frame is unreadable in a viewer, and the point is
    # to see whether the shapes sit on the pool, not to inspect pixels.
    cv2.imwrite(str(out), cv2.resize(im, None, fx=0.5, fy=0.5),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out}")
    print("  water:", rig.water)
    print("  three_lines:", rig.three_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
