"""A continuous stretch of game footage with the clock and the model's shot count
burned in, so a person can mark what it got wrong.



This exists because RECALL HAS NEVER BEEN MEASURED. Every accuracy number on this
project -- make/miss, attribution, the drop-zone veto -- is computed over the
shots the detector happened to surface. If it misses a third of them, and misses
them non-randomly (the fast ones, the crowded ones), every one of those numbers
describes a biased sample and nobody knows which way.

Precision we do know: 51 of 415 judged candidates are "not a shot", about 12%.
Recall needs ground truth, and ground truth needs a person watching continuous
footage. This makes that cheap: one pass, no scrubbing, no transcribing
timestamps by hand.

    python src/marktape.py --video IMG_2529.MOV --start 140 --dur 300
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent


def shot_times(video):
    """When the pipeline says a shot happened, from whatever it has produced."""
    stem = Path(video).stem
    out = []
    idx = ROOT / f"out/clips_{stem}/index.json"
    if idx.exists():
        for r in json.loads(idx.read_text(encoding="utf-8")):
            t = r.get("t") if isinstance(r, dict) else None
            if t is not None:
                out.append((float(t), r.get("hoop", "?")))
    sh = ROOT / f"out/shooter_{stem}.json"
    if not out and sh.exists():
        for r in json.loads(sh.read_text(encoding="utf-8")):
            if r.get("t") is not None:
                out.append((float(r["t"]), r.get("hoop", "?")))
    return sorted(out)


def esc(s):
    return str(s).replace("\\", "\\\\").replace(":", "\\:").replace("'", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="IMG_2529.MOV")
    ap.add_argument("--start", type=float, default=140.0)
    ap.add_argument("--dur", type=float, default=300.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = ROOT / "footage" / args.video
    if not src.exists():
        print(f"no {src}")
        return 1

    shots = [(t, h) for t, h in shot_times(args.video)
             if args.start <= t < args.start + args.dur]
    print(f"{args.video} {args.start:.0f}s..{args.start + args.dur:.0f}s: "
          f"the model calls {len(shots)} shots")
    for t, h in shots:
        print(f"   {int(t)//60:d}:{int(t)%60:02d}  {h}")

    # The clock is the video's own timestamp, offset so it reads as the position
    # in the ORIGINAL recording. review marks against that, so his timestamps and
    # the pipeline's refer to the same thing without anyone converting.
    filters = [
        "scale=1280:-2",
        (f"drawtext=text='%{{pts\\:hms\\:{args.start:.0f}}}':"
         "x=24:y=24:fontsize=46:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=12"),
    ]
    # Running count, one label per interval, so it steps exactly on each call.
    edges = [args.start] + [t for t, _ in shots] + [args.start + args.dur + 1]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lab = esc(f"shots so far: {i}")
        filters.append(
            f"drawtext=text='{lab}':x=24:y=92:fontsize=38:fontcolor=yellow:"
            f"box=1:boxcolor=black@0.65:boxborderw=10:"
            f"enable='between(t,{lo - args.start:.3f},{hi - args.start:.3f})'")
    # A flash at the moment of each call, so a shot is unmissable while watching.
    for t, h in shots:
        r = t - args.start
        filters.append(
            f"drawtext=text='{esc('SHOT >> ' + str(h))}':x=24:y=156:fontsize=44:"
            f"fontcolor=black:box=1:boxcolor=yellow@0.95:boxborderw=12:"
            f"enable='between(t,{r:.3f},{r + 1.6:.3f})'")

    dest = Path(args.out) if args.out else ROOT / f"out/marktape_{Path(args.video).stem}_{int(args.start)}.mp4"
    cmd = [ff, "-y", "-loglevel", "error", "-ss", f"{args.start:.3f}", "-i", str(src),
           "-t", f"{args.dur:.3f}", "-vf", ",".join(filters),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)]
    subprocess.run(cmd, check=False)
    if dest.exists():
        print(f"\nwrote {dest}  ({dest.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
