"""Look only at the hoops, at full resolution.

The full-frame pass tracks the ball fine in open water and never once sees it
within 150px of a rim. Two explanations fit that: the detector loses the ball
in the rim region, or there simply weren't many shots in the window. This tells
them apart, and if it's the former it is also most of the fix.

Downscaling a 3840-wide frame to 1920 for inference halves the ball. Cropping a
box around a hoop and running the model on that crop instead keeps native
pixels, so the ball is twice the size to the detector and the background is a
fraction of the scene. It is also cheaper per frame than the full-frame pass,
not more expensive, because the crop is small.

Output is a candidate list of moments the ball was near a rim, which is what
turns two hours of footage into a short list to confirm by eye.

Run: python src/rimwatch.py footage/x.MOV --from 280 --to 520
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hoops import rig_for  # noqa: E402

SPORTS_BALL = 32



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--from", dest="t0", type=float, default=280.0)
    p.add_argument("--to", dest="t1", type=float, default=520.0)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--step", type=int, default=1, help="frame stride")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--out", type=Path, default=Path("out/rimwatch.json"))
    return p.parse_args()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    rig = rig_for(args.video)
    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, end = int(args.t0 * fps), int(args.t1 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    log(f"rimwatch {args.t0:.0f}s-{args.t1:.0f}s ({end-start} frames), conf {args.conf}, imgsz {args.imgsz}")

    hits = {k: [] for k in rig.crops}
    f = start
    t0 = time.time()
    while f < end:
        ok, fr = cap.read()
        if not ok:
            break
        if (f - start) % args.step == 0:
            for name, (x1, y1, x2, y2) in rig.crops.items():
                crop = fr[y1:y2, x1:x2]
                r = model.predict(crop, conf=args.conf, verbose=False,
                                  classes=[SPORTS_BALL], imgsz=args.imgsz)[0]
                if r.boxes is None or not len(r.boxes):
                    continue
                for bx in r.boxes:
                    bx1, by1, bx2, by2 = bx.xyxy[0].tolist()
                    # back to full-frame coordinates
                    hits[name].append({
                        "frame": f,
                        "t": round(f / fps, 2),
                        "x": round(x1 + (bx1 + bx2) / 2, 1),
                        "y": round(y1 + (by1 + by2) / 2, 1),
                        "conf": round(float(bx.conf.item()), 3),
                        "w": round(bx2 - bx1, 1),
                    })
        f += args.step
        if (f - start) % 900 == 0:
            done, tot = f - start, end - start
            n = sum(len(v) for v in hits.values())
            log(f"  {done}/{tot} frames, {time.time()-t0:.0f}s, {n} rim-region detections")
    cap.release()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "window_s": [args.t0, args.t1],
        "conf": args.conf,
        "boxes": rig.crops,
        "hits": hits,
    }, indent=1), encoding="utf-8")
    for k, v in hits.items():
        log(f"{k}: {len(v)} detections near the rim")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
