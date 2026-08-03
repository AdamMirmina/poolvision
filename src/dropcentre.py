"""Where do balls from MADE shots actually land? Put the zone there.

The zone's position has now been wrong twice from two different directions, both
times by reasoning about it instead of measuring it:

  Derived from the wall alone, it ran down the image and walked onto concrete.
  Derived from the water plane and offset outward, it drifted north on the
  shootout's left hoop --

And when I centered it on the point under the rim, which seemed the more faithful
model of a ball falling straight down, recall collapsed from 64% to 36%. That is
the answer to the modeling question: the ball does NOT land under the rim. It
carries through the net and out into the pool, so a zone centered under the net
is centered on the wrong place.

So stop deriving the center. The judged makes say where balls land. This takes
their sightings near the descent and reports the median offset from the point
under the rim, per hoop, in that hoop's own along/into axes -- which is both the
right answer and a check on the axes themselves, since a nonsense "into"
direction shows up immediately as an offset pointing the wrong way.

    python src/dropcentre.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dropzone
import hoops

ROOT = Path(__file__).resolve().parent.parent


def main():
    import numpy as np
    judged = json.loads((ROOT / "labels/allshots.json").read_text(encoding="utf-8"))
    makes = [j for j in judged if j.get("label") == "make"]

    per = {}
    for j in makes:
        v, hoop = j["video"], j.get("hoop")
        rw = ROOT / f"out/rimwatch_{Path(v).stem}.json"
        if not rw.exists():
            continue
        try:
            rig = hoops.rig_for(v)
        except Exception:
            continue
        dax = (rig.drop or {}).get(hoop)
        rim = rig.rims.get(hoop)
        if not dax or not rim:
            continue
        hits = (json.loads(rw.read_text(encoding="utf-8")).get("hits") or {}).get(hoop) or []
        t = float(j["t"])
        near = [h for h in hits if t - 0.3 <= h["t"] <= t + dropzone.WINDOW_S]
        if not near:
            continue
        # The LAST sighting in the window is the one nearest the water.
        h = max(near, key=lambda z: z["t"])
        px, py = dax["p"]
        ax, ay = dax["along"]
        ix, iy = dax["into"]
        w = rim[2] - rim[0]
        dx, dy = h["x"] - px, h["y"] - py
        # Project into the hoop's own axes, in rim widths.
        s = (dx * ax + dy * ay) / w
        d = (dx * ix + dy * iy) / w
        per.setdefault((v, hoop), []).append((s, d))

    print(f"{len(makes)} judged makes; landing offsets in rim widths, "
          f"in each hoop's own axes\n")
    print(f"{'video / hoop':>26} | {'n':>3} | {'along':>14} | {'into':>14}")
    print("-" * 68)
    agg = {}
    for (v, hoop), pts in sorted(per.items()):
        a = np.array(pts)
        ms, md = float(np.median(a[:, 0])), float(np.median(a[:, 1]))
        print(f"{v + ' ' + hoop:>26} | {len(pts):>3} | "
              f"{ms:>+7.2f} (sd {a[:,0].std():.2f}) | {md:>+7.2f} (sd {a[:,1].std():.2f})")
        agg.setdefault(hoop, []).extend(pts)

    print("\nper hoop, pooled across sessions:")
    for hoop, pts in agg.items():
        a = np.array(pts)
        print(f"  {hoop}: along {np.median(a[:,0]):+.2f}, into {np.median(a[:,1]):+.2f} "
              f"({len(pts)} makes)")
    print("\n'into' well above zero means the ball carries out into the pool, so a")
    print("zone centered under the net is centered on the wrong place. A NEGATIVE")
    print("value would mean the axis is pointing the wrong way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
