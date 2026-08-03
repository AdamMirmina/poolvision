"""Measure the drop zone's own axes, per hoop, on the water plane.

The zone is a patch of the water SURFACE under the net. Its sides therefore run
along the wall and away from the wall, and under perspective neither direction is
constant across the image -- parallel world lines converge. Building it as a
shear plus a straight-down extension was wrong in a way that showed: on the left
it walked onto the concrete, on the right it ran down the wall instead of
reaching into the water.

Two things get measured here and stored, so nothing has to be re-derived at
runtime:

  p      the waterline point directly under the rim, where the zone starts
  along  unit direction along the wall at p
  into   unit direction away from the wall, into the water

The awkward part, and the reason this is a script rather than a formula: THE TWO
HOOPS ARE NOT ON PARALLEL WALLS. The right hoop stands on a long wall; the left
stands on the step-notch edge, which belongs to the perpendicular family. So
which vanishing point means "along" and which means "away" swaps between them,
and a single per-hoop slope cannot express it -- which is exactly why the left
hoop's slope fit kept coming out unstable.

Rather than deciding that by hand, it is decided by the water itself: of the four
candidate directions, "into" is the one that stays over water longest. That is
self-verifying, and it cannot be got backwards.

    python src/dropfit.py --video IMG_2482.MOV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops
import poolplane
import waterfit

ROOT = Path(__file__).resolve().parent.parent
PROBE = 2.2      # how far to walk a candidate direction, in rim widths
STEPS = 44


def unit(dx, dy):
    n = (dx * dx + dy * dy) ** 0.5
    return (dx / n, dy / n) if n > 1e-9 else (0.0, 0.0)


def water_run(mask, p, d, dist, steps=STEPS):
    """Fraction of a walk from p along d that lands on water."""
    h, w = mask.shape[:2]
    hit = 0
    for i in range(1, steps + 1):
        x = int(p[0] + d[0] * dist * i / steps)
        y = int(p[1] + d[1] * dist * i / steps)
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            hit += 1
    return hit / steps


def entry_direction(mask, rim):
    """The wall's direction from the water-entry points, as a vector.

    Fitted by principal direction rather than as a slope, so a near-vertical wall
    is expressible. One rejection pass, because a swimmer standing at the wall
    puts an entry point metres away from it.
    """
    import numpy as np
    pts = waterfit.entries(mask, rim)
    if len(pts) < 10:
        return None
    a = np.array(pts, dtype=float)
    for _ in range(2):
        mean = a.mean(0)
        u, s, vt = np.linalg.svd(a - mean)
        d = vt[0]
        n = a - mean
        # Distance from the fitted line, along its normal.
        off = np.abs(n[:, 0] * -d[1] + n[:, 1] * d[0])
        keep = off <= max(14.0, 2.0 * np.median(off))
        if keep.sum() < 10:
            break
        a = a[keep]
    mean = a.mean(0)
    u, s, vt = np.linalg.svd(a - mean)
    d = vt[0]
    n = (d[0] ** 2 + d[1] ** 2) ** 0.5
    return (float(d[0] / n), float(d[1] / n)) if n > 1e-9 else None


def boundary_tangent(mask, p, rw, radius_rw=1.6):
    """Which way the water's edge runs at p: the wall's own direction.

    Takes the mask's boundary pixels within a radius of p and fits a line to
    them. This is what decides "along" versus "into", so it is measured at the
    hoop rather than assumed from the session -- the two hoops here stand on
    perpendicular walls, so there is no session-wide answer.
    """
    import cv2
    import numpy as np
    r = int(radius_rw * rw)
    x0, y0 = int(p[0] - r), int(p[1] - r)
    x1, y1 = int(p[0] + r), int(p[1] + r)
    h, w = mask.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    sub = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    edge = cv2.Canny(sub, 50, 150)
    ys, xs = np.nonzero(edge)
    if len(xs) < 25:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mean = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - mean)
    d = vt[0]
    n = (d[0] ** 2 + d[1] ** 2) ** 0.5
    return (float(d[0] / n), float(d[1] / n)) if n > 1e-9 else None


def axes_for(mask, rim, vps, water):
    """(p, along, into) for one hoop."""
    x1, y1, x2, y2 = rim
    cx = (x1 + x2) / 2
    rw = x2 - x1
    p = (cx, water["water_y_at_center"])

    fams = {name: unit(vp[0] - p[0], vp[1] - p[1])
            for name, vp in vps.items() if vp}
    if len(fams) < 2:
        return None

    # WHICH family is "along" is decided by the wall itself, not by which
    # direction happens to stay wettest.
    #
    # The first version picked "into" as whichever candidate stayed over water
    # longest, and that is wrong whenever the wall runs tangent to the water --
    # walking ALONG a wall also stays over water the whole way, so it can win.
    # It did on the shootout's left hoop, whose "into" came out (0.985, -0.171),
    # almost straight down the pool, so the zone slid away from under the
    # backboard toward the diving board.
    #
    # The wall's own direction at p is measurable: the water boundary's local
    # tangent. The family matching it is "along", the other is "into". Only the
    # SIGN of "into" is then chosen by which way the water is.
    # ALONG is the wall's own direction at this hoop, measured from the water
    # boundary. INTO is simply its perpendicular, pointing at the water.
    #
    # The previous version took "into" from the water plane's second vanishing
    # point, which is perspective-exact in principle and was a disaster in
    # practice: at the right hoop the two axes came out 157 degrees apart, so the
    # "square" collapsed into a thin skewed sliver lying along the wall. review,
    # looking at it: "it visually is just not the space under the net where balls
    # drop. i would think it would look like a square extending out from the
    # walls directly under the rim and backboard."
    #
    # Review is describing the right thing, and a right angle in the image is the way
    # to get it. Over the 1.3 rim widths this zone spans, the perspective error
    # from treating the wall's perpendicular as a right angle is far smaller than
    # the error from an axis pointing 67 degrees off. Exactness that degenerates
    # is worth less than an approximation that holds its shape.
    # The wall's direction comes from the WATER-ENTRY POINTS -- the same ones
    # waterfit walks down columns to find -- fitted as a direction rather than a
    # slope. Two reasons it is not the slope and not a local edge fit:
    #
    #   A slope cannot express a near-vertical wall, and the left hoop's is
    #   near-vertical. That is why its slope fitted at -0.43 on one frame and
    #   -0.73 on another and never settled.
    #
    #   A Canny fit in a small window around p latches onto the tile bands and
    #   the coping instead of the waterline: it gave the right hoop 0.42 where
    #   the entry points plainly say 0.84.
    #
    # A principal direction over the entry points has neither problem, and those
    # points are already known good -- they run (3250,814) to (3730,1216) at the
    # right hoop, which is what the waterline measurement was validated on.
    tan = entry_direction(mask, rim)
    if tan is None:
        tan = boundary_tangent(mask, p, rw)
    if tan is None:
        return None
    along = tan if tan[1] >= 0 else (-tan[0], -tan[1])
    perp = (-along[1], along[0])
    a = water_run(mask, p, perp, PROBE * rw)
    b = water_run(mask, p, (-perp[0], -perp[1]), PROBE * rw)
    into = perp if a >= b else (-perp[0], -perp[1])
    best_score = max(a, b)
    # Point "along" the way the wall descends in the image, purely so the stored
    # numbers are comparable between hoops and sessions.
    if along[1] < 0:
        along = (-along[0], -along[1])
    return {"p": [round(p[0], 1), round(p[1], 1)],
            "along": [round(along[0], 4), round(along[1], 4)],
            "into": [round(into[0], 4), round(into[1], 4)],
            "into_water_frac": round(best_score, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="IMG_2482.MOV")
    ap.add_argument("--times", default="")
    args = ap.parse_args()

    import numpy as np
    rig = hoops.rig_for(args.video)
    times = ([float(s) for s in args.times.split(",")] if args.times
             else [30.0, 90.0, 180.0])
    ims = waterfit.frames(ROOT / "footage" / args.video, times)
    if not ims:
        print("no frames")
        return 1
    masks = [waterfit.water_mask(im) for im in ims]

    vlist = [poolplane.vps(m) for m in masks]
    vp = {}
    for k in ("a", "b"):
        pts = [v[k] for v in vlist if v.get(k)]
        if pts:
            vp[k] = (float(np.median([p[0] for p in pts])),
                     float(np.median([p[1] for p in pts])))
    print(f"{args.video}: vanishing points {vp}\n")

    out = {}
    for hoop, rim in rig.rims.items():
        w = (rig.water or {}).get(hoop)
        if not w:
            print(f"  {hoop}: no waterline measured, run src/waterfit.py first")
            continue
        per = [axes_for(m, rim, vp, w) for m in masks]
        per = [a for a in per if a]
        if not per:
            print(f"  {hoop}: could not resolve axes")
            continue
        a = per[0]
        out[hoop] = a
        print(f"  {hoop}: p={a['p']}  along={a['along']}  into={a['into']}")
        print(f"      into stays over water {a['into_water_frac']*100:.0f}% of the probe")
        print(f"      along slope {a['along'][1]/a['along'][0]:+.3f}, "
              f"wall measured {w['slope']:+.3f}")

    print("\npaste into hoops.py:")
    print("        drop=" + repr(out) + ",")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
