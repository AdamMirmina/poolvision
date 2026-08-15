"""One clip per merged pair, to call one shot or two.

The pipeline joins consecutive descents at a hoop into a single call. That is
right for a ball bouncing off the ring and dropping in, and wrong for a rebound
someone catches and puts back -- which review says is fairly common. No threshold
on the time between them can tell the difference: across all twelve the gap runs
from 0.02s to 0.87s and the two known cases sit at 0.12s and 0.06s.

Wrists look like the real discriminator, measured at 0.75 ball-widths when white
catches a rebound against 1.18 when a ball bounces untouched. That is one example
each, and four wrong rules were built today on samples that thin, so the
threshold waits for these twelve to be labeled.

Each clip covers both descents with lead-in and follow-through, and carries the
question on screen so it can be judged without reading anything else.

    python src/mergeclips.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
LEAD = 4.0    # enough to see the first shot arrive
TAIL = 3.5    # enough to see where the ball finishes


def esc(s):
    return str(s).replace("\\", "\\\\").replace(":", "\\:").replace("'", "")


def main():
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    rows = json.loads((ROOT / "out/merges.json").read_text(encoding="utf-8"))
    out = []
    for i, r in enumerate(rows, 1):
        src = ROOT / "footage" / r["video"]
        if not src.exists():
            print(f"skip {i}: no {src.name}")
            continue
        # The clip starts before the FIRST descent, which ended just before this
        # one began -- so lead in from the second run's start, not its end.
        start = max(0.0, r["second_start"] - LEAD)
        dur = (r["second_end"] - r["second_start"]) + LEAD + TAIL
        t = r["second_start"]
        label = f"#{i}  {int(t)//60}:{int(t)%60:02d}  {r['hoop']} hoop"
        dest = ROOT / f"out/merge_{i:02d}.mp4"
        filters = [
            "scale=1180:-2",
            (f"drawtext=text='{esc(label)}':x=20:y=18:fontsize=34:fontcolor=white:"
             "box=1:boxcolor=black@0.72:boxborderw=10"),
            (f"drawtext=text='{esc('ONE shot (bounced in) or TWO (someone caught it)?')}':"
             "x=20:y=64:fontsize=27:fontcolor=yellow:box=1:boxcolor=black@0.72:boxborderw=9"),
        ]
        subprocess.run(
            [ff, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
             "-t", f"{dur:.3f}", "-vf", ",".join(filters),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)],
            check=False)
        if dest.exists():
            out.append(str(dest))
            print(f"  {dest.name}  {label}  ({dest.stat().st_size/1e6:.1f} MB)")
    print(f"\n{len(out)} clips")
    (ROOT / "out/mergeclips.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
