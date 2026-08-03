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


def axes_for(mask, rim, vps, water):
    """(p, along, into) for one hoop."""
    x1, y1, x2, y2 = rim
    cx = (x1 + x2) / 2
    rw = x2 - x1
    p = (cx, water["water_y_at_center"])

    cands = []
    for name, vp in vps.items():
        if not vp:
            continue
        u = unit(vp[0] - p[0], vp[1] - p[1])
        cands.append((name, u))
        cands.append((name, (-u[0], -u[1])))
    if len(cands) < 4:
        return None

    # "into" is whichever candidate stays over water longest. A direction running
    # along the wall skims the boundary and a direction pointing at the deck
    # leaves it immediately, so the winner is unambiguous.
    scored = [(water_run(mask, p, d, PROBE * rw), name, d) for name, d in cands]
    scored.sort(reverse=True)
    best_score, into_fam, into = scored[0]
    along = None
    for score, name, d in scored:
        if name != into_fam:
            along = d
            break
    if along is None:
        return None
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
