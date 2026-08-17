"""Cut a clip that shows WHICH flight the model called.

The first attempt at getting these judged handed over nine-second windows with a
time burned in the corner, and review could not answer them: "there are several
shots in some windows and i don't know which specific section the model is
referring to when it makes those calls." That is a defect in the question, not
in the answer. Asking anyway would have produced guesses and I would have
treated them as data.

The model already knows exactly which flight it means, because every call comes
from one run of ball detections. This draws that run on the footage: a ring on
the ball in every frame it was seen, a trail behind it, and the moment of the
call marked. Nothing else in the window is annotated, so a clip containing three
shots still asks about exactly one of them.

    python src/callclip.py --video IMG_2770.MOV --hits out/key_2770b.json \
        --at 767 --label "EXTRA 12:47 right" --out out/j.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT_W = 960
PAD = 260          # px of context around the ball path
LEAD = 2.0         # seconds of run-up before the first sighting
TAIL = 2.2         # and after the last


def find_call(hits, at, video):
    """The call whose run starts nearest `at`, with its detections."""
    import clips
    import hoops
    # cluster() reads the rig from clips' module state, which its own main()
    # normally fills in. Every other tool that calls cluster() from outside has
    # to prime it the same way, and without it every hoop raises KeyError.
    rig = hoops.rig_for(Path(video).name)
    clips.RIG_RIMS = rig.rims
    clips.RIG_DROP = rig.drop or {}
    clips._ZONES.clear()
    best, bd = None, 1e9
    for hoop, dets in clips.cluster(hits, 1.0, 3, video=video):
        d = abs(dets[0]["t"] - at)
        if d < bd:
            best, bd = (hoop, dets), d
    return best, bd


def draw(fr, dets, t, crop):
    """Ring the ball, trail where it has been, and say when the call was made."""
    x0, y0, x1, y1 = crop
    img = fr[y0:y1, x0:x1].copy()
    seen = [d for d in dets if d["t"] <= t + 1e-6]

    # The trail, oldest faint and newest bright, so direction of travel reads
    # without an arrowhead cluttering the ball itself.
    pts = [(int(d["x"] - x0), int(d["y"] - y0)) for d in seen]
    for i in range(1, len(pts)):
        a = 0.25 + 0.75 * (i / max(1, len(pts) - 1))
        cv2.line(img, pts[i - 1], pts[i], (0, int(200 * a), int(255 * a)), 2, cv2.LINE_AA)

    # The ball right now. Two rings so it stays visible against both the water
    # and the deck, which are at opposite ends of the brightness range.
    here = [d for d in dets if abs(d["t"] - t) < 0.02]
    if here:
        d = here[0]
        c = (int(d["x"] - x0), int(d["y"] - y0))
        r = max(14, int(d["w"] * 0.8))
        cv2.circle(img, c, r + 3, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, c, r, (60, 235, 245), 3, cv2.LINE_AA)
    elif seen and t > dets[-1]["t"]:
        # After the run ends, hold a marker where it ended.
        #
        # Some calls rest on very little: one of these is 4 sightings spanning
        # 0.1 seconds, which at 60fps is six frames of ring in a four-second
        # clip. Asking someone to judge a flash they can easily miss is the same
        # mistake as the windows this file was written to replace, one level
        # smaller. The held marker also makes the flimsiness itself visible,
        # which is information about the call rather than a distraction from it.
        d = dets[-1]
        c = (int(d["x"] - x0), int(d["y"] - y0))
        r = max(14, int(d["w"] * 0.8))
        cv2.circle(img, c, r + 3, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(img, c, r, (110, 190, 200), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--hits", required=True)
    ap.add_argument("--at", type=float, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import json
    import imageio_ffmpeg

    raw = json.loads(Path(args.hits).read_text(encoding="utf-8"))
    src = ROOT / "footage" / args.video
    call, off = find_call(raw["hits"], args.at, str(src))
    if call is None:
        print("no call found at all")
        return 1
    hoop, dets = call
    # A call more than a couple of seconds from where it was asked for is a
    # DIFFERENT call, and drawing it would answer a question nobody asked.
    if off > 2.5:
        print(f"nearest call is {off:.1f}s away at {dets[0]['t']:.1f} -- not drawing it")
        return 1

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    xs = [d["x"] for d in dets]
    ys = [d["y"] for d in dets]
    x0 = max(0, int(min(xs)) - PAD)
    x1 = min(W, int(max(xs)) + PAD)
    y0 = max(0, int(min(ys)) - PAD)
    y1 = min(H, int(max(ys)) + PAD)
    # Even width and height, or the encoder rejects the stream.
    x1 -= (x1 - x0) % 2
    y1 -= (y1 - y0) % 2
    crop = (x0, y0, x1, y1)

    t_start = max(0.0, dets[0]["t"] - LEAD)
    t_stop = dets[-1]["t"] + TAIL
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_start * fps))

    ow = OUT_W
    oh = int(round((y1 - y0) * ow / (x1 - x0)))
    oh -= oh % 2
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{ow}x{oh}", "-r", f"{fps:.3f}", "-i", "-",
         "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)

    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = t_start + n / fps
        if t > t_stop:
            break
        img = cv2.resize(draw(fr, dets, t, crop), (ow, oh))
        if args.label:
            cv2.rectangle(img, (0, 0), (ow, 34), (0, 0, 0), -1)
            cv2.putText(img, args.label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 2, cv2.LINE_AA)
        # Before the ball is first seen there is no ring and no trail, which
        # would read as "the model saw nothing here" rather than "not yet".
        if t < dets[0]["t"]:
            cv2.putText(img, "watch the ring", (10, oh - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 235, 245), 2, cv2.LINE_AA)
        proc.stdin.write(img.tobytes())
        n += 1
    proc.stdin.close()
    proc.wait()
    cap.release()
    print(f"{Path(args.out).name}  {hoop}  {len(dets)} sightings, "
          f"{dets[0]['t']:.1f}-{dets[-1]['t']:.1f}s, {n} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
