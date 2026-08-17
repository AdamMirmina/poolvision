"""Where does a recording get too dark to use?

IMG_2932 starts at 7pm and runs 45 minutes into dusk.
Somewhere in there the detector stops having a ball to find, and scanning past
that point costs GPU hours and puts junk in the answer.

Picking the cutoff by eye from one or two frames is guessing. Measuring mean
brightness across the whole recording is cheap and gives a curve with an obvious
knee. But brightness alone is not the thing that matters -- what matters is
whether the BALL is still distinguishable, and a frame can be dim overall while
the lit pool area is fine. So this reports brightness over the rim regions
specifically, not the whole frame, and writes sample frames either side of the
knee so a person makes the final call on something they can see.

    python src/daylight.py --video IMG_2932.MOV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--every", type=float, default=10.0, help="seconds between samples")
    ap.add_argument("--out", default="out/daylight")
    args = ap.parse_args()

    src = ROOT / "footage" / args.video
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # No rig measured for this recording yet, so sample the middle band where the
    # pool and both hoops sit in every rig so far, rather than the whole frame.
    # The sky occupies the top of these shots and stays bright long after the
    # deck has gone dark, which would drag a whole-frame average up and put the
    # knee later than it really is.
    y0, y1 = int(H * 0.25), int(H * 0.95)

    step = int(args.every * fps)
    rows = []
    for f in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        band = fr[y0:y1]
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        rows.append({
            "t": round(f / fps, 1),
            "mean": round(float(g.mean()), 1),
            # Contrast matters more than level: a ball is found by standing out
            # from what is behind it, and that collapses before the mean does.
            "sd": round(float(g.std()), 1),
            "p95": round(float(np.percentile(g, 95)), 1),
        })
        if len(rows) % 30 == 0:
            print(f"  {rows[-1]['t']/60:5.1f} min  mean {rows[-1]['mean']:5.1f} "
                  f"sd {rows[-1]['sd']:5.1f}", flush=True)
    cap.release()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "curve.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    first = rows[0]["mean"]
    print(f"\n{len(rows)} samples over {rows[-1]['t']/60:.1f} min")
    print(f"start mean {first:.1f} -> end mean {rows[-1]['mean']:.1f}")
    for frac in (0.9, 0.75, 0.6, 0.5, 0.4):
        hit = next((r for r in rows if r["mean"] < first * frac), None)
        if hit:
            print(f"  drops below {frac*100:.0f}% of opening brightness at "
                  f"{int(hit['t'])//60}:{int(hit['t'])%60:02d}  "
                  f"(mean {hit['mean']}, sd {hit['sd']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
