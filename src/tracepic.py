"""Draw the ball's whole path over a window, on a real frame.

review, repeatedly, and it keeps being the thing that finds the bug: "present me
with as much information like this as possible so i can draw my conclusions by
eye and help you."

Numbers in a terminal cannot show that a shot went four rim widths wide, or that
the path everyone assumed was one flight is actually two different balls. This
draws the sightings themselves, colored by time, over the frame they came from,
with the rim and the drop zone in the same picture and the two quantities the
gates actually test written on it.

    python src/tracepic.py --hits out/key_right.json --at 1091 --hoop right
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clips
import dropzone
import hoops

ROOT = Path(__file__).resolve().parent.parent


def hms(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", required=True)
    ap.add_argument("--video", default="IMG_2528.MOV")
    ap.add_argument("--at", type=float, required=True)
    ap.add_argument("--hoop", required=True)
    ap.add_argument("--before", type=float, default=2.5)
    ap.add_argument("--after", type=float, default=3.5)
    ap.add_argument("--label", default="")
    ap.add_argument("--reason", default="", help="what the gates actually did")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import cv2
    import numpy as np

    rig = hoops.rig_for(args.video)
    clips.RIG_RIMS = rig.rims
    clips.RIG_DROP = rig.drop or {}
    clips._ZONES.clear()
    rim = rig.rims[args.hoop]
    rcx, rcy = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2
    rw = max(1.0, rim[2] - rim[0])

    hits = json.loads(Path(args.hits).read_text(encoding="utf-8"))["hits"][args.hoop]
    v = sorted([h for h in hits
                if args.at - args.before <= h["t"] <= args.at + args.after],
               key=lambda z: z["t"])
    # One box per ball per frame, or the path is drawn several times over and
    # reads as a much noisier flight than it is.
    v = clips._one_box_per_ball(v)
    if not v:
        print("no sightings in that window")
        return 1

    cap = cv2.VideoCapture(str(ROOT / "footage" / args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.at * fps))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        print("could not read the frame")
        return 1
    fr = (fr * 0.55).astype("uint8")   # dim it so the path reads clearly

    zone = clips._dropzone(args.hoop, rim)
    if zone:
        cv2.polylines(fr, [np.array(zone, dtype=np.int32)], True, (90, 220, 255), 5)
    cv2.rectangle(fr, (rim[0], rim[1]), (rim[2], rim[3]), (60, 235, 245), 4)

    t0, t1 = v[0]["t"], v[-1]["t"]
    span = max(1e-6, t1 - t0)
    prev = None
    for h in v:
        f = (h["t"] - t0) / span               # blue at the start, red at the end
        col = (int(255 * (1 - f)), int(90 + 80 * f), int(255 * f))
        cv2.circle(fr, (int(h["x"]), int(h["y"])), 7, col, -1)
        if prev is not None:
            cv2.line(fr, prev, (int(h["x"]), int(h["y"])), col, 2)
        prev = (int(h["x"]), int(h["y"]))

    near = min((((h["x"] - rcx) ** 2 + (h["y"] - rcy) ** 2) ** 0.5) / rw for h in v)
    nearest = min(v, key=lambda h: ((h["x"] - rcx) ** 2 + (h["y"] - rcy) ** 2))
    cv2.circle(fr, (int(nearest["x"]), int(nearest["y"])), 26, (255, 255, 255), 3)
    fall = max(h["y"] for h in v) - min(h["y"] for h in v)
    inz = sum(1 for h in v if zone and dropzone.contains(zone, h["x"], h["y"]))

    lines = [
        f"{args.label or hms(args.at)}   {args.hoop} hoop   {len(v)} sightings",
        f"closest to the ring: {near:.2f} rim widths   (rule wants under {clips.RING_NEAR})",
        f"total fall: {fall:.0f}px   (rule wants at least {clips.RING_FALL_PX})",
        f"sightings inside the drop zone: {inz}",
        "blue = start of the window, red = end.  white circle = closest approach",
    ]
    if args.reason:
        lines.append("")
        lines.append(args.reason)
    # These two numbers are measured over the DISPLAYED window. The gates measure
    # them over the run, which is usually a fragment of it, so they will not
    # always agree -- the picture shows the flight, not the gate's view of it.
    for i, txt in enumerate(lines):
        cv2.putText(fr, txt, (40, 70 + i * 52), cv2.FONT_HERSHEY_SIMPLEX,
                    1.35, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(fr, txt, (40, 70 + i * 52), cv2.FONT_HERSHEY_SIMPLEX,
                    1.35, (255, 255, 255), 2, cv2.LINE_AA)

    dest = Path(args.out) if args.out else ROOT / f"out/trace_{int(args.at)}_{args.hoop}.jpg"
    cv2.imwrite(str(dest), cv2.resize(fr, (1900, int(1900 * fr.shape[0] / fr.shape[1]))))
    print(f"{dest}   near={near:.2f}rw  fall={fall:.0f}px  in-zone={inz}  n={len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
