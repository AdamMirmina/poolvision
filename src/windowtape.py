"""Show what the detector actually LOOKS AT, on the footage, frame by frame.

review, repeatedly: "can you visually show me the windows in question on the video
so i know what the model is seeing? i keep telling you this but you need to
present me with as much information like this as possible so i can draw my
conclusions by eye and help you."

That is right and it keeps being the thing that finds the bug. Today's crop
disaster -- a detector window half the size it should have been for the entire
2026-08-01 session -- was invisible in every number I produced and obvious the
moment the two rigs were compared side by side. A picture of the search region
would have shown it immediately.

So this draws, on the real footage:

  the DETECTOR WINDOW, the only part of the frame the ball detector is ever shown
  the RIM box inside it
  every BALL SIGHTING, as it happens, so a ball crossing the frame outside the
    window is visibly not looked at

Watching it, "the ball flew in from up there and the box never reached that high"
is a thing a person can see in one pass and no summary statistic will say.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops

ROOT = Path(__file__).resolve().parent.parent

WIN = (90, 220, 255)      # the detector's window: amber, unmissable
RIM = (60, 235, 245)
BALL = (60, 255, 60)
MISS = (60, 60, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="IMG_2529.MOV")
    ap.add_argument("--hits", default="")
    ap.add_argument("--start", type=float, default=478.0)
    ap.add_argument("--dur", type=float, default=70.0)
    ap.add_argument("--marks", default="", help="t:hoop,... real shots to flag")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import cv2
    import imageio_ffmpeg
    import numpy as np

    rig = hoops.rig_for(args.video)
    src = ROOT / "footage" / args.video
    hits = {}
    hp = Path(args.hits) if args.hits else ROOT / f"out/rimwatch_{Path(args.video).stem}.json"
    if hp.exists():
        hits = json.loads(hp.read_text(encoding="utf-8")).get("hits") or {}

    marks = []
    for m in [z for z in args.marks.split(",") if z.strip()]:
        t, _, ho = m.partition(":")
        marks.append((float(t), ho or "?"))

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUT_W = 1280
    OUT_H = int(round(OUT_W * H / W / 2) * 2)
    sx, sy = OUT_W / W, OUT_H / H

    dest = Path(args.out) if args.out else ROOT / f"out/windowtape_{Path(args.video).stem}_{int(args.start)}.mp4"
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{OUT_W}x{OUT_H}", "-r", f"{fps:.3f}", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
        stdin=subprocess.PIPE)

    f0 = int(args.start * fps)
    f1 = int((args.start + args.dur) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    dets = rig.det_boxes()
    wrote = 0
    for f in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        t = f / fps
        im = cv2.resize(fr, (OUT_W, OUT_H))

        for hoop, d in dets.items():
            cv2.rectangle(im, (int(d[0] * sx), int(d[1] * sy)),
                          (int(d[2] * sx), int(d[3] * sy)), WIN, 2)
            cv2.putText(im, f"detector window ({hoop})",
                        (int(d[0] * sx) + 4, int(d[1] * sy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, WIN, 1, cv2.LINE_AA)
            r = rig.rims[hoop]
            cv2.rectangle(im, (int(r[0] * sx), int(r[1] * sy)),
                          (int(r[2] * sx), int(r[3] * sy)), RIM, 1)

        # Every sighting within a frame's time of now, drawn where it was seen.
        seen = 0
        for hoop, lst in hits.items():
            for hh in lst:
                if abs(hh["t"] - t) <= 0.5 / fps:
                    cv2.circle(im, (int(hh["x"] * sx), int(hh["y"] * sy)), 9, BALL, 2)
                    seen += 1
        cv2.putText(im, f"{int(t)//60}:{int(t)%60:02d}.{int((t%1)*100):02d}",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im, "GREEN = ball detected this frame" if seen else "no ball detected",
                    (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    BALL if seen else (190, 190, 190), 1, cv2.LINE_AA)

        for mt, mh in marks:
            if 0 <= t - mt <= 2.0:
                # "YOU MARKED" rather than a name. The old wording read as the model
                # naming the shooter -- review saw it and reported "they all said
                # review although some were another player", and I spent a search looking for
                # a name the pipeline does not have. It was this caption.
                cv2.putText(im, f"YOU MARKED A REAL SHOT HERE ({mh} hoop)", (16, 92),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, MISS, 2, cv2.LINE_AA)
        proc.stdin.write(im.tobytes())
        wrote += 1

    proc.stdin.close()
    proc.wait()
    cap.release()
    print(f"wrote {dest} ({wrote} frames, {dest.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
