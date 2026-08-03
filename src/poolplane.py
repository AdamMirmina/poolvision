"""The two vanishing points of the water plane.

The drop zone is a patch of the WATER SURFACE under the net, so its sides run
along the wall and away from the wall. It was being built as a parallelogram
sheared to the wall and then extended straight DOWN the image, which is only the
same thing if the camera looks straight down. It does not, so the zone ran along
the pool's length instead of out from the wall -- on the left it walked onto the
concrete, and on the right it leaned down the wall rather than reaching into the
water.

Right. Two directions are needed at each hoop, and under perspective neither is
constant across the image: parallel lines in the world converge. So measure where
they converge.

  vp_long  -- where the two long walls meet. The direction ALONG a wall.
  vp_cross -- where the two end walls meet. The direction ACROSS the pool,
              which is the one that is perpendicular to the wall in the world.

At any point P on the water, "along the wall" is toward vp_long and "away from
the wall" is toward vp_cross. That is exact under perspective, not an
approximation, which is the whole reason to do it this way rather than fitting a
second slope by hand per hoop.

    python src/poolplane.py --video IMG_2482.MOV --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops
import waterfit

ROOT = Path(__file__).resolve().parent.parent


def meet(l1, l2):
    """Intersection of two lines, each given as two points."""
    (x1, y1), (x2, y2) = l1
    (x3, y3), (x4, y4) = l2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None          # parallel in the image: the world lines are, too
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return ((a * (x3 - x4) - (x1 - x2) * b) / d,
            (a * (y3 - y4) - (y1 - y2) * b) / d)


def edges(mask, min_len=260):
    """The pool's boundary as a small set of straight edges.

    The pool is not a clean rectangle in these frames -- there is a step notch at
    one corner -- so a four-corner fit is wrong. Approximating the contour and
    keeping the long edges gives the real walls and drops the notch, which is
    short.
    """
    import cv2
    import numpy as np
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, 0.012 * peri, True).reshape(-1, 2)
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        L = float(np.hypot(*(q - p)))
        if L >= min_len:
            out.append((tuple(map(float, p)), tuple(map(float, q)), L))
    return sorted(out, key=lambda e: -e[2])


def pair_by_direction(es):
    """Split the long edges into the two families of parallel world lines.

    Grouped by image direction: the two long walls point roughly the same way as
    each other and roughly opposite to the two end walls.
    """
    import numpy as np
    if len(es) < 4:
        return None, None
    ang = []
    for (p, q, L) in es:
        a = np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])) % 180.0
        ang.append(a)
    base = ang[0]
    fam_a = [es[i] for i in range(len(es)) if min(abs(ang[i] - base), 180 - abs(ang[i] - base)) < 35]
    fam_b = [es[i] for i in range(len(es)) if min(abs(ang[i] - base), 180 - abs(ang[i] - base)) >= 35]
    return fam_a, fam_b


def vps(mask):
    fam_a, fam_b = pair_by_direction(edges(mask))
    out = {}
    for name, fam in (("a", fam_a), ("b", fam_b)):
        if not fam or len(fam) < 2:
            out[name] = None
            continue
        best = meet((fam[0][0], fam[0][1]), (fam[1][0], fam[1][1]))
        out[name] = best
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482.MOV")
    p.add_argument("--times", default="")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    import cv2
    import numpy as np

    rig = hoops.rig_for(args.video)
    times = ([float(s) for s in args.times.split(",")] if args.times
             else [30.0, 90.0, 180.0])
    ims = waterfit.frames(ROOT / "footage" / args.video, times)
    if not ims:
        print("no frames")
        return 1

    got = {"a": [], "b": []}
    for im in ims:
        m = waterfit.water_mask(im)
        v = vps(m)
        for k in got:
            if v.get(k):
                got[k].append(v[k])
    res = {}
    for k, lst in got.items():
        if lst:
            res[k] = (float(np.median([p[0] for p in lst])),
                      float(np.median([p[1] for p in lst])))
    print(f"{args.video}: vanishing points from {len(ims)} frames")
    for k, v in res.items():
        print(f"  family {k}: ({v[0]:.0f}, {v[1]:.0f})")

    # Which family is which is decided by the hoops, not by guessing: the wall a
    # hoop stands on runs toward vp_long, so the family whose direction best
    # matches the measured wall slope at each rim IS the long one.
    if len(res) == 2 and rig.water:
        for k, v in res.items():
            err = 0.0
            for hoop, rim in rig.rims.items():
                w = (rig.water or {}).get(hoop)
                if not w:
                    continue
                cx = (rim[0] + rim[2]) / 2
                cy = w["water_y_at_center"]
                dx, dy = v[0] - cx, v[1] - cy
                slope = dy / dx if abs(dx) > 1e-6 else 1e6
                err += abs(slope - w["slope"])
            res[k] = (v, err)
            print(f"  family {k} slope mismatch against the measured walls: {err:.3f}")
        a, b = res["a"], res["b"]
        long_k = "a" if a[1] < b[1] else "b"
        cross_k = "b" if long_k == "a" else "a"
        print(f"\n  vp_long  = {tuple(round(c) for c in res[long_k][0])}   (along the wall)")
        print(f"  vp_cross = {tuple(round(c) for c in res[cross_k][0])}   (away from it)")
        print("\npaste into hoops.py:")
        print(f"        plane={{'vp_long': {tuple(round(c) for c in res[long_k][0])}, "
              f"'vp_cross': {tuple(round(c) for c in res[cross_k][0])}}},")

    if args.show:
        im = ims[0].copy()
        for (p0, p1, L) in edges(waterfit.water_mask(ims[0])):
            cv2.line(im, tuple(int(v) for v in p0), tuple(int(v) for v in p1),
                     (0, 0, 255), 4)
        out = ROOT / f"out/poolplane_{Path(args.video).stem}.jpg"
        cv2.imwrite(str(out), cv2.resize(im, None, fx=0.5, fy=0.5),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\nwrote {out}  (red = the long edges the vanishing points came from)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
