"""Cut a short clip around one shot, with a label burned in.

Simpler than tracepic.py on purpose. A still with numbers written on it asks the
viewer to reconstruct motion from a frozen path; a clip just shows what
happened.

    python src/shotclip.py --video IMG_2528.MOV --at 1105 --label "18:25 miss"
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def esc(s):
    return str(s).replace("\\", "\\\\").replace(":", "\:").replace("'", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="IMG_2528.MOV")
    ap.add_argument("--at", type=float, required=True)
    ap.add_argument("--before", type=float, default=4.0)
    ap.add_argument("--after", type=float, default=5.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = ROOT / "footage" / args.video
    if not src.exists():
        print(f"no {src}")
        return 1

    start = max(0.0, args.at - args.before)
    filters = ["scale=1280:-2"]
    if args.label:
        filters.append(
            f"drawtext=text='{esc(args.label)}':x=24:y=24:fontsize=40:fontcolor=white:"
            "box=1:boxcolor=black@0.7:boxborderw=12")

    dest = Path(args.out) if args.out else ROOT / f"out/shot_{int(args.at)}.mp4"
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
         "-t", f"{args.before + args.after:.3f}", "-vf", ",".join(filters),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)],
        check=False)
    if not dest.exists():
        print("ffmpeg produced nothing")
        return 1
    print(f"{dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
