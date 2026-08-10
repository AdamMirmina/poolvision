"""Blur every face before a frame leaves this laptop.



The answer is that faces are free to give up. **Nothing in this pipeline has ever
used one.** Shot detection uses the ball and the geometry of the hoop. Attribution
uses cap color and wrist position from pose keypoints. A face contributes
nothing to either, so blurring it costs no accuracy at all -- which makes this a
pure win rather than a tradeoff to argue about.

Where a frame can leave the machine, and therefore where this must run:

  a still opened in a chat -- goes to Anthropic as an image
  a clip or tape uploaded to poolean -- lands on the VPS, reachable by URL
  anything sent to anyone

The raw video never leaves, because YOLO and ffmpeg run locally. This is about
the derived artifacts, which is the whole of what actually gets shared.

**What this does NOT do.** It does not make it acceptable to record someone who
asked not to be recorded. If a person does not want to be in the footage, they
are not in the footage; there is no technical substitute for that, and pointing a
camera at them anyway because the faces come out blurry is not consent. This
tool is for the person who is happy to play on camera but does not want their
face shared or stored.

A person who wants no part of the identity side has a second, simpler option:
don't wear a cap. Attribution is keyed entirely on cap color, so no cap means no
identity, while shot detection carries on unaffected.

    python src/anon.py --in out/clip.mp4 --out out/clip_anon.mp4
    python src/anon.py --in out/frame.jpg --out out/frame_anon.jpg
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent

# Head keypoints in COCO pose order: nose, both eyes, both ears.
HEAD_KP = (0, 1, 2, 3, 4)
# How far past the head keypoints to blur, as a multiple of the spread between
# them. Generous on purpose: an under-blurred face is the failure that matters,
# and there is nothing on the other side of the trade to protect.
PAD = 2.6
MIN_R = 26


def blur_heads(frame, pose_model, conf=0.15):
    """Blur a disc over every head found. Returns the frame and how many."""
    import cv2
    import numpy as np

    res = pose_model.predict(frame, conf=conf, verbose=False, imgsz=1280)[0]
    if res.keypoints is None or res.keypoints.xy is None:
        return frame, 0

    boxes = res.boxes.xyxy if res.boxes is not None else None
    n = 0
    for idx, person in enumerate(res.keypoints.xy):
        pts = [(float(person[i][0]), float(person[i][1])) for i in HEAD_KP
               if i < len(person) and float(person[i][0]) > 0 and float(person[i][1]) > 0]
        if pts:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            spread = max((max(p[k] for p in pts) - min(p[k] for p in pts))
                         for k in (0, 1)) if len(pts) > 1 else 0.0
            r = int(max(MIN_R, spread * PAD))
        else:
            # No head keypoints does NOT mean no head. It usually means the head
            # is turned away, underwater, or occluded -- exactly the frames where
            # skipping would leave a face in the clear. Fall back to the top of
            # the person's box.
            #
            # This branch existed only as a comment for one commit while the code
            # did `continue`. The failure here is one-directional: blurring a bit
            # of water costs nothing, missing a face costs the thing this file is
            # for.
            if boxes is None or idx >= len(boxes):
                continue
            bx0, by0, bx1, by1 = (float(v) for v in boxes[idx])
            cx = (bx0 + bx1) / 2
            r = int(max(MIN_R, (bx1 - bx0) * 0.55))
            cy = by0 + r * 0.7

        x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
        x1, y1 = min(frame.shape[1], int(cx + r)), min(frame.shape[0], int(cy + r))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = frame[y0:y1, x0:x1]
        # Pixelate rather than gaussian-blur: a blur of a small face can be
        # partially undone, a downsample throws the information away.
        small = cv2.resize(patch, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)),
                           interpolation=cv2.INTER_LINEAR)
        patch[:] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros(patch.shape[:2], dtype=np.uint8)
        cv2.circle(mask, ((x1 - x0) // 2, (y1 - y0) // 2), r, 255, -1)
        n += 1
    return frame, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dest", default="")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    src = Path(args.src)
    if not src.exists():
        print(f"no {src}")
        return 1
    dest = Path(args.dest) if args.dest else src.with_name(src.stem + "_anon" + src.suffix)
    pose = YOLO("yolo11s-pose.pt")

    if src.suffix.lower() in (".jpg", ".jpeg", ".png"):
        fr = cv2.imread(str(src))
        fr, n = blur_heads(fr, pose)
        cv2.imwrite(str(dest), fr)
        print(f"{dest}  ({n} heads blurred)")
        return 0

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc = subprocess.Popen(
        [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
        stdin=subprocess.PIPE)
    total = frames = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr, n = blur_heads(fr, pose)
        total += n
        frames += 1
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    cap.release()
    print(f"{dest}  ({frames} frames, {total} head blurs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
