"""A stretch of footage with a clock on it and nothing else.

This is the ground-truth tape. It carries NO model output, so what review writes
against it is a record of what happened in the pool rather than a reaction to
what the pipeline claimed. Every rule so far has been scored against seven shots
across two windows, which is why each one looks right on one minute and falls
apart on the next.

Two properties make this the most durable artifact on the project:

  It does not need a scan. Detection takes hours per five minutes on this
  machine; this takes a couple of minutes, so marking never waits on compute.

  It cannot be invalidated. Change the detector, change the gates, retrain the
  whole thing -- a list of when shots actually happened stays true. Every
  model-calls tape becomes worthless the moment the model changes.

Deliberately different from marktape.py, which burns the model's calls in for
review to correct. That one measures a build. This one measures the game.

    python src/blindtape.py --video IMG_2528.MOV --start 900 --dur 360
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="IMG_2528.MOV")
    ap.add_argument("--start", type=float, default=900.0)
    ap.add_argument("--dur", type=float, default=360.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = ROOT / "footage" / args.video
    if not src.exists():
        print(f"no {src}")
        return 1

    # The clock reads as the position in the ORIGINAL recording, so the timestamps and the pipeline's refer to the same thing with nobody
    # converting between them.
    filters = [
        "scale=1280:-2",
        (f"drawtext=text='%{{pts\\:hms\\:{args.start:.0f}}}':"
         "x=24:y=24:fontsize=46:fontcolor=white:box=1:boxcolor=black@0.65:"
         "boxborderw=12"),
    ]

    dest = (Path(args.out) if args.out
            else ROOT / f"out/blindtape_{Path(args.video).stem}_{int(args.start)}.mp4")
    cmd = [ff, "-y", "-loglevel", "error", "-ss", f"{args.start:.3f}", "-i", str(src),
           "-t", f"{args.dur:.3f}", "-vf", ",".join(filters),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)]
    subprocess.run(cmd, check=False)
    if not dest.exists():
        print("ffmpeg produced nothing")
        return 1
    print(f"wrote {dest}  ({dest.stat().st_size / 1e6:.0f} MB, "
          f"{args.start:.0f}s to {args.start + args.dur:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
