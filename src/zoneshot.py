"""Close-up of each drop zone, filled, for a human to approve.

The whole-pool render proves a zone is not on the concrete. It does not show
whether the zone is where the ball actually goes, because at that scale it is a
small quadrilateral among a lot of pool. This crops to each hoop and fills the
zone, so the question "is this the right patch of water" can actually be
answered.

Also draws the real landing points of judged makes at that hoop, when there are
any, because that is what the zone is FOR. A zone that looks sensible and does
not contain the dots is wrong however sensible it looks.

    python src/zoneshot.py --video IMG_2482.MOV --t 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dropzone
import hoops
import rigcheck

ROOT = Path(__file__).resolve().parent.parent

RIM = (60, 235, 245)
NET = (210, 210, 210)
ZONE = (90, 200, 90)
LAND = (60, 120, 255)


def landings(video, hoop):
    """Where the ball was last seen on judged MAKES at this hoop."""
    lab = ROOT / "labels/allshots.json"
    rw = ROOT / f"out/rimwatch_{Path(video).stem}.json"
    if not lab.exists() or not rw.exists():
        return []
    hits = (json.loads(rw.read_text(encoding="utf-8")).get("hits") or {}).get(hoop) or []
    out = []
    for j in json.loads(lab.read_text(encoding="utf-8")):
        if j.get("video") != video or j.get("hoop") != hoop or j.get("label") != "make":
            continue
        t = float(j["t"])
        near = [h for h in hits if t - 0.3 <= h["t"] <= t + dropzone.WINDOW_S]
        if near:
            h = max(near, key=lambda z: z["t"])
            out.append((h["x"], h["y"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="IMG_2482.MOV")
    ap.add_argument("--t", type=float, default=30.0)
    args = ap.parse_args()

    import cv2
    import numpy as np

    rig = hoops.rig_for(args.video)
    frame = rigcheck.grab(ROOT / "footage" / args.video, args.t)
    if frame is None:
        print("no frame")
        return 1

    for hoop, rim in rig.rims.items():
        im = frame.copy()
        q = dropzone.zone(rim, (rig.water or {}).get(hoop), (rig.drop or {}).get(hoop))
        pts = np.array([[int(a), int(b)] for a, b in q], np.int32)

        fill = im.copy()
        cv2.fillPoly(fill, [pts], ZONE)
        cv2.addWeighted(fill, 0.28, im, 0.72, 0, im)
        cv2.polylines(im, [pts], True, ZONE, 3, cv2.LINE_AA)

        x1, y1, x2, y2 = [int(v) for v in rim]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.ellipse(im, (cx, cy), ((x2 - x1) // 2, max(6, (y2 - y1) // 2)),
                    (rig.tilt or {}).get(hoop, 0.0), 0, 360, RIM, 3, cv2.LINE_AA)
        b = dropzone.box(rim)
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), NET, 1)

        lands = landings(args.video, hoop)
        for (lx, ly) in lands:
            cv2.circle(im, (int(lx), int(ly)), 9, LAND, -1, cv2.LINE_AA)
            cv2.circle(im, (int(lx), int(ly)), 9, (255, 255, 255), 2, cv2.LINE_AA)

        # Crop around the rim AND the zone, with room to see the wall.
        xs = [p[0] for p in pts] + [x1, x2] + [p[0] for p in lands]
        ys = [p[1] for p in pts] + [y1, y2] + [p[1] for p in lands]
        pad = 200
        cx0 = max(0, int(min(xs)) - pad)
        cy0 = max(0, int(min(ys)) - pad)
        cx1 = min(im.shape[1], int(max(xs)) + pad)
        cy1 = min(im.shape[0], int(max(ys)) + pad)
        crop = im[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        scale = min(2.4, 1500 / max(1, crop.shape[1]))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        cap = (f"{Path(args.video).stem}  {hoop} hoop   "
               f"zone {2*dropzone.HALF_W:.2f} x {2*dropzone.HALF_D:.2f} rim widths"
               + (f"   {len(lands)} judged makes shown" if lands else "   no judged makes here"))
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(crop, cap, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (255, 255, 255), 1, cv2.LINE_AA)

        out = ROOT / f"out/zone_{Path(args.video).stem}_{hoop}.jpg"
        cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote {out}   ({len(lands)} makes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
