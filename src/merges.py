"""Every place the pipeline collapsed two descents at one hoop into a single call.

Each of these is either a rim bounce that really is one shot, or a rebound
someone caught and put back, which is two. The merge rule cannot tell them apart
and neither can any threshold on the time between them: the 5:55 and 5:58 are
0.12s apart in the data because the ball never stops being tracked.

Wrists look like the discriminator -- 0.75 ball-widths when white catches it
against 1.18 when a ball bounces untouched -- but that is one example each, and a
1.6x gap from a sample of one is how four wrong rules got built in a day.

So this lists them all, so review can say one-or-two for each and the threshold can
come from twelve labeled cases instead of a guess.

    python src/merges.py
"""

from __future__ import annotations

import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clips
import hoops
import regress

LINE = re.compile(
    r"\[bounce-merged\s*\]\s+(\w+)\s+([\d.]+)-\s*([\d.]+)\s+n=\s*(\d+).*?"
    r"gap ([\d.]+)s\s+prev ends y=(-?\d+)\s+this starts y=(-?\d+)")


def find():
    out = []
    for name, video, hits, keyfile in regress.CASES:
        if any(not (regress.ROOT / h).exists() for h in hits):
            continue
        rig = hoops.rig_for(video)
        clips.RIG_RIMS, clips.RIG_DROP = rig.rims, rig.drop or {}
        clips._ZONES.clear()
        buf = io.StringIO()
        clips.TRACE = (0, 10 ** 9, None)
        with redirect_stdout(buf):
            for hp in hits:
                raw = json.loads((regress.ROOT / hp).read_text(encoding="utf-8"))
                clips.cluster(raw["hits"], 1.0, 3, video=None)
        clips.TRACE = None
        for m in LINE.finditer(buf.getvalue()):
            hoop, t0, t1, n, gap, y_prev, y_this = m.groups()
            out.append({
                "case": name, "video": video, "hoop": hoop,
                "second_start": float(t0), "second_end": float(t1),
                "gap_s": float(gap), "n": int(n),
                # How far the ball CLIMBED between the two descents. A caught ball
                # gets lifted; a bounced one pops up on its own. Recorded because
                # it is free, not because it is known to separate them -- measured
                # at 116px on the one re-dunk there is, which is not obviously
                # different from a bounce.
                "rise_px": int(y_prev) - int(y_this),
            })
    return out


def main():
    rows = find()
    print(f"{len(rows)} merged pairs\n")
    print(f"{'case':<30} {'hoop':<6} {'at':>7} {'gap':>6} {'rise':>7}")
    for r in rows:
        t = r["second_start"]
        print(f"{r['case'][:30]:<30} {r['hoop']:<6} "
              f"{int(t)//60}:{int(t)%60:02d}   {r['gap_s']:5.2f}s {r['rise_px']:6d}px")
    (regress.ROOT / "out/merges.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote out/merges.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
