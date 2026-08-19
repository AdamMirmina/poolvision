"""Why was there no call HERE? Ask the gates directly.

review marks a shot the model missed and the useful question is which gate threw it
away, not whether the ball was seen. The detector usually saw it: the 11:00 dunk
had 256 sightings within six seconds and produced no call at all.

Guessing at that from the outside has cost several rounds. This replays the real
clustering over the real hits and prints, for every run near a time, the gate
that rejected it and by how much.

    python src/explain.py --hits out/rw_fresh.json --at 660:right --at 803:left
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clips
import hoops


def hms(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", default="out/rw_fresh.json")
    ap.add_argument("--video", default="footage/IMG_2529.MOV")
    ap.add_argument("--at", action="append", default=[],
                    help="SECONDS[:hoop], repeatable")
    ap.add_argument("--window", type=float, default=8.0)
    args = ap.parse_args()

    # cluster() reads the rims from module state, which main() normally fills in.
    # Without this every hoop raises KeyError.
    _rig = hoops.rig_for(Path(args.video).name)
    clips.RIG_RIMS = _rig.rims
    clips.RIG_DROP = _rig.drop or {}
    clips._ZONES.clear()

    hits = json.loads(Path(args.hits).read_text(encoding="utf-8"))["hits"]
    clips.EXPLAIN.clear()
    kept = clips.cluster(hits, 1.0, 3, video=args.video)

    def when(e):
        if isinstance(e, dict):
            return e.get("t")
        for part in (e if isinstance(e, (list, tuple)) else []):
            if isinstance(part, list) and part and isinstance(part[0], dict):
                return part[0].get("t")
        return None
    calls = sorted(t for t in (when(e) for e in kept) if t is not None)
    print(f"{len(kept)} calls: " + ", ".join(hms(t) for t in calls))

    targets = []
    for a in args.at:
        t, _, hoop = a.partition(":")
        targets.append((float(t), hoop or ""))
    if not targets:
        targets = [(None, "")]

    for t, hoop in targets:
        print(f"\n=== around {hms(t) if t is not None else 'everywhere'}"
              f"{' ' + hoop if hoop else ''}")
        rows = [e for e in clips.EXPLAIN
                if (t is None or abs(e[0] - t) <= args.window)
                and (not hoop or e[1] == hoop)]
        if not rows:
            print("  no run was even built here -- the gap/falling split never "
                  "made a candidate out of these sightings")
        for rt, rh, reason, detail in sorted(rows)[:14]:
            print(f"  {hms(rt)}.{int(rt % 1 * 100):02d} {rh:5s} {reason}   {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
