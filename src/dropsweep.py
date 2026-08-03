"""How small can the drop zone be before it stops working?

 Reasonable, and it needs measuring rather than shrinking, because
this rule's whole value is recall: a shot that never appears in the zone is
called a miss, so every make the zone fails to contain becomes a wrong answer.

Two things make the old justification for the current size invalid, so the sweep
has to be redone rather than reused:

1. The 34-of-37 was measured against dropzone.box() -- an axis-aligned rectangle
   -- because that is what dropped() actually tested, while quad()/zone() was
   only ever what got DRAWN. The picture showed one region and the rule tested
   another, which is precisely the drift this project has been bitten by before.
2. The zone's shape is now correct (a patch of the water plane running out from
   the wall), so a narrower one may hold recall that a narrower WRONGLY-ORIENTED
   one could not.

Recall is over judged makes, precision over judged misses (off the iron, air
balls, air net). Both from real ball sightings near the descent.

    python src/dropsweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dropzone
import hoops

ROOT = Path(__file__).resolve().parent.parent
# "inout" -- rattled in and out -- is a MISS. It existed in the labels
# and in nothing here, so those shots were silently dropped from every
# measurement rather than counted.
MISS = {"offiron", "airball", "airnet", "behind", "inout"}


def inside(poly, x, y):
    """Point in polygon, so the test matches the shape that gets drawn."""
    n = len(poly)
    hit = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xx = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xx:
                hit = not hit
    return hit


def hits_for(video):
    p = ROOT / f"out/rimwatch_{Path(video).stem}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("hits") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", default="0.35,0.45,0.55,0.65,0.75,0.9")
    ap.add_argument("--depth", default="0.8,1.0,1.3,1.6")
    args = ap.parse_args()

    judged = json.loads((ROOT / "labels/allshots.json").read_text(encoding="utf-8"))
    shots = [j for j in judged if j.get("label") in ({"make"} | MISS)]
    byvid = {}
    for j in shots:
        byvid.setdefault(j["video"], []).append(j)

    cache = {}
    for v in byvid:
        cache[v] = hits_for(v)

    print(f"{len(shots)} judged shots "
          f"({sum(1 for j in shots if j['label'] == 'make')} makes)\n")
    print(f"{'half':>5} {'depth':>6} | {'makes in zone':>15} | {'misses in zone':>15} | "
          f"{'zone w x d':>12}")
    print("-" * 74)

    rows = []
    for half in [float(s) for s in args.half.split(",")]:
        for depth in [float(s) for s in args.depth.split(",")]:
            mk_in = mk_tot = ms_in = ms_tot = 0
            for v, js in byvid.items():
                hits = cache.get(v)
                if not hits:
                    continue
                try:
                    rig = hoops.rig_for(v)
                except Exception:
                    continue
                for j in js:
                    hoop = j.get("hoop")
                    rim = rig.rims.get(hoop)
                    dax = (rig.drop or {}).get(hoop)
                    if not rim or not dax:
                        continue
                    poly = zone_sized(rim, dax, half, depth)
                    hs = hits.get(hoop) or []
                    t = float(j["t"])
                    got = any(inside(poly, h["x"], h["y"])
                              for h in hs
                              if t - 0.3 <= h["t"] <= t + dropzone.WINDOW_S)
                    if j["label"] == "make":
                        mk_tot += 1
                        mk_in += got
                    else:
                        ms_tot += 1
                        ms_in += got
            if not mk_tot:
                continue
            rec = mk_in / mk_tot
            fpr = ms_in / max(ms_tot, 1)
            rows.append((half, depth, rec, fpr, mk_in, mk_tot, ms_in, ms_tot))
            print(f"{half:>5.2f} {depth:>6.2f} | {mk_in:>4}/{mk_tot:<4} {100*rec:>6.0f}% | "
                  f"{ms_in:>4}/{ms_tot:<4} {100*fpr:>6.0f}% | "
                  f"{2*half:>4.2f} x {depth:<4.2f}")

    print("\nThe rule says a shot that NEVER appears here is a miss, so recall is")
    print("what protects it. A make outside the zone is a wrong answer; a miss")
    print("inside it only costs the veto a chance to fire.")
    best = [r for r in rows if r[2] >= 0.90]
    if best:
        # Smallest area that still holds 90% recall, and closest to square.
        best.sort(key=lambda r: (2 * r[0] * r[1], abs(2 * r[0] - r[1])))
        h, d, rec, fpr = best[0][:4]
        print(f"\nsmallest that keeps 90% recall: half={h}, depth={d} "
              f"({2*h:.2f} x {d:.2f} rim widths, recall {100*rec:.0f}%, "
              f"misses caught {100*fpr:.0f}%)")
    return 0


def zone_sized(rim, dax, half, depth):
    x1, _, x2, _ = rim
    w = x2 - x1
    px, py = dax["p"]
    ax, ay = dax["along"]
    ix, iy = dax["into"]
    c = []
    hd = depth / 2.0
    ca, ci = dax.get('center', (0.0, 0.8))   # measured landing point
    for s in (ca - half, ca + half):
        for d in (ci - hd, ci + hd):
            c.append((px + ax * s * w + ix * d * w, py + ay * s * w + iy * d * w))
    return [c[0], c[1], c[3], c[2]]


if __name__ == "__main__":
    raise SystemExit(main())
