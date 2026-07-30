"""Render just the rim regions to their own videos, for detection elsewhere.

The detector never looks at a full frame. It looks at two crops around the
hoops, which together are roughly a fifth of the pixels. Uploading whole 4K
recordings to a GPU service means shipping four fifths of the data to be thrown
away on arrival, and the upload is the slowest part of that plan.

So: decode once locally (unavoidable CPU work either way), write one video per
hoop, and ship those. Encoded near-losslessly, because the entire reason the
detector works at all is native resolution -- a downscaled frame missed every
shot in the 2026-07-28 run, and that lesson cost a day. Quality here is not the
place to save bytes.

Crop coordinates come from hoops.py, so the output carries the same geometry the
full-frame pipeline uses and detections map straight back to source coordinates
by adding the crop origin.

Run: python src/croproll.py footage/x.MOV --out out/crops
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hoops import rig_for  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CRF = 16   # visually lossless; the ball is small and detail IS the signal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=Path("out/crops"))
    p.add_argument("--from", dest="t0", type=float, default=0.0)
    p.add_argument("--to", dest="t1", type=float, default=0.0, help="0 = whole video")
    return p.parse_args()


def main():
    args = parse_args()
    import cv2

    rig = rig_for(args.video)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(args.t0 * fps)
    f1 = int(args.t1 * fps) if args.t1 else total
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem

    writers = {}
    for hoop, (x1, y1, x2, y2) in rig.crops.items():
        w = (x2 - x1) // 2 * 2
        h = (y2 - y1) // 2 * 2
        path = args.out / f"{stem}_{hoop}.mp4"
        writers[hoop] = {
            "path": path, "w": w, "h": h, "box": (x1, y1, x1 + w, y1 + h),
            "proc": subprocess.Popen(
                [FFMPEG, "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}",
                 "-i", "-", "-c:v", "libx264", "-crf", str(CRF), "-preset", "veryfast",
                 "-pix_fmt", "yuv420p", str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        }

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    t0 = time.time()
    n = 0
    for f in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        for hoop, w in writers.items():
            x1, y1, x2, y2 = w["box"]
            w["proc"].stdin.write(fr[y1:y2, x1:x2].tobytes())
        n += 1
        if n % 1800 == 0:
            el = time.time() - t0
            print(f"  {n}/{f1 - f0} frames, {el:.0f}s, {n / el:.1f} fps", flush=True)
    cap.release()

    meta = {"video": args.video.name, "fps": fps, "first_frame": f0, "frames": n, "crops": {}}
    for hoop, w in writers.items():
        w["proc"].stdin.close()
        w["proc"].wait()
        size = w["path"].stat().st_size if w["path"].exists() else 0
        meta["crops"][hoop] = {"file": w["path"].name, "origin": [w["box"][0], w["box"][1]],
                               "size": [w["w"], w["h"]], "bytes": size}
        print(f"{w['path'].name}  {w['w']}x{w['h']}  {size / 1e6:.0f} MB")
    (args.out / f"{stem}_crops.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    src = args.video.stat().st_size
    out = sum(c["bytes"] for c in meta["crops"].values())
    frac = (f1 - f0) / max(1, total)
    print(f"source {src / 1e9:.2f} GB ({frac * 100:.0f}% processed) -> crops {out / 1e9:.2f} GB")
    print(f"decode+encode ran at {n / (time.time() - t0):.1f} fps")


if __name__ == "__main__":
    main()
