"""Can the detector see the ball more often at the rim, for free?

The failure analysis is unambiguous: shots where the ball is detected once or
twice near the hoop are called wrong 41% of the time, shots with six or more
detections only 12%. The classifier is not the weak part. The number of frames
the ball is actually seen in, during the half second that decides the outcome,
is the weak part.

Two knobs cost nothing to change and can be tested on footage already recorded:
inference resolution, and the confidence floor. A ball at the far hoop is a
handful of pixels in the source; running the crop at 640 shrinks it further. So
this measures detections-at-the-rim across settings, on the windows around the
27 shots of the only fully hand-labeled video.

It measures DENSITY, not accuracy, and deliberately so. Density is the thing the
failure analysis identified, it needs no model retraining to read, and it is not
confounded by which classifier happens to be fitted on top.

Anything gained here is gained on every shot, past and future, with no new
judgment from review.

    python src/denser.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops

ROOT = Path(__file__).resolve().parent.parent
VIDEO = "IMG_2403.MOV"
SETTINGS = [
    (640, 0.10),     # what produced everything measured so far
    (960, 0.10),
    (1280, 0.10),
    (1280, 0.05),    # and with the floor dropped, since tracking rejects noise anyway
]


def main():
    import cv2
    import numpy as np
    from ultralytics import YOLO

    labeled = [j for j in json.loads((ROOT / "labels/judged.json").read_text(encoding="utf-8"))
                if j["video"] == VIDEO and j["label"] != "notshot"]
    print(f"{len(labeled)} hand-judged shots in {VIDEO}\n")

    rig = hoops.rig_for(VIDEO)
    cap = cv2.VideoCapture(str(ROOT / "footage" / VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Only the frames that matter: half a second either side of each descent.
    wanted: dict[str, set[int]] = {"left": set(), "right": set()}
    for j in labeled:
        t1 = j["tEnd"] if j["tEnd"] > j["t"] else j["t"] + 1.0
        for f in range(int((j["t"] - 0.5) * fps), int((t1 + 0.5) * fps) + 1):
            wanted[j["hoop"]].add(f)
    frames_needed = sorted(set().union(*wanted.values()))
    print(f"{len(frames_needed)} frames to inspect per setting\n")

    model = YOLO("yolo11s.pt")
    SPORTS_BALL = 32

    # Read once, hold the crops, then run every setting over the same pixels.
    # Re-decoding per setting would make the comparison partly a measurement of
    # seek performance.
    store: dict[int, dict[str, np.ndarray]] = {}
    for f in frames_needed:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        store[f] = {}
        for hoop, (x1, y1, x2, y2) in rig.crops.items():
            if f in wanted[hoop]:
                store[f][hoop] = fr[y1:y2, x1:x2].copy()
    cap.release()
    print(f"{len(store)} frames decoded\n")

    print(f"{'imgsz':>6} {'conf':>6}  {'median at rim':>14} {'0-2 dets':>9} {'6+ dets':>8}  {'secs/frame':>10}")
    results = []
    for imgsz, conf in SETTINGS:
        t0 = time.time()
        hits = {"left": [], "right": []}
        for f, crops in store.items():
            for hoop, crop in crops.items():
                r = model.predict(crop, conf=conf, verbose=False, classes=[SPORTS_BALL], imgsz=imgsz)[0]
                ox, oy = rig.crops[hoop][0], rig.crops[hoop][1]
                for b in r.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    hits[hoop].append({"frame": f, "t": f / fps,
                                       "x": (bx1 + bx2) / 2 + ox, "y": (by1 + by2) / 2 + oy,
                                       "conf": float(b.conf[0]), "w": bx2 - bx1})
        secs = (time.time() - t0) / max(1, sum(len(c) for c in store.values()))

        # Same measure the failure analysis used: detections within a rim-width
        # of the hoop, per shot.
        per_shot = []
        for j in labeled:
            rim = rig.rims[j["hoop"]]
            cx, cy = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2
            rw = rim[2] - rim[0]
            t1 = j["tEnd"] if j["tEnd"] > j["t"] else j["t"] + 1.0
            n = sum(1 for h in hits[j["hoop"]]
                    if j["t"] - 0.5 <= h["t"] <= t1 + 0.5
                    and abs(h["y"] - cy) < rw and abs(h["x"] - cx) < 1.5 * rw)
            per_shot.append(n)
        arr = np.array(per_shot)
        print(f"{imgsz:>6} {conf:>6.2f}  {np.median(arr):>14.0f} "
              f"{(arr <= 2).sum():>9} {(arr >= 6).sum():>8}  {secs:>10.3f}")
        results.append({"imgsz": imgsz, "conf": conf, "per_shot": per_shot,
                        "median": float(np.median(arr)), "thin": int((arr <= 2).sum()),
                        "dense": int((arr >= 6).sum()), "secs_per_crop": secs})

    (ROOT / "out/denser.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    b = SETTINGS[0]
    base = results[0]
    best = max(results, key=lambda r: (r["dense"], -r["thin"]))
    print(f"\nbaseline {b[0]}/{b[1]}: {base['thin']} thin shots, {base['dense']} dense")
    print(f"best     {best['imgsz']}/{best['conf']}: {best['thin']} thin shots, {best['dense']} dense")
    if best["thin"] < base["thin"]:
        print(f"\n{base['thin'] - best['thin']} shots move out of the 41%-wrong bucket for the cost of "
              f"{best['secs_per_crop'] / base['secs_per_crop']:.1f}x the compute.")
    else:
        print("\nResolution and confidence do not fix it. The detector needs fine-tuning "
              "on the ball at the rim, not more pixels.")


if __name__ == "__main__":
    main()
