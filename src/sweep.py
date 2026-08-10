"""Sweep the drop-zone size against the answer key.

The gate that made the deck false positives vanish is also the single cause of
every missed shot: all seven report 'none inside the drop zone'. With precision
sitting at zero false calls there is room to loosen it, and the question is how
far before the deck comes back.

Reported as a curve rather than a chosen number, because the useful thing is
where recall stops rising and false calls start -- not whichever value happens
to score best on twenty shots.

    python src/sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clips
import hoops

ROOT = Path(__file__).resolve().parent.parent
KEY = json.loads((ROOT / "labels/answerkey_IMG_2528.json").read_text(encoding="utf-8"))


def calls_for(hits_files, video, grow):
    clips.ZONE_GROW = grow
    rig = hoops.rig_for(video)
    clips.RIG_RIMS = rig.rims
    clips.RIG_DROP = rig.drop or {}
    out = []
    for hp in hits_files:
        clips._ZONES.clear()
        raw = json.loads((ROOT / hp).read_text(encoding="utf-8"))
        for ev in clips.cluster(raw["hits"], 1.0, 3, video=None):
            t = hoop = None
            if isinstance(ev, dict):
                t, hoop = ev.get("t"), ev.get("hoop")
            else:
                for part in (ev if isinstance(ev, (list, tuple)) else []):
                    if isinstance(part, list) and part and isinstance(part[0], dict):
                        t = part[0]["t"]
                    elif isinstance(part, str):
                        hoop = part
            if t is not None:
                out.append({"t": t, "hoop": hoop})
    return sorted(out, key=lambda c: c["t"])


def score(calls, shots, tol=4.0):
    used, found = set(), 0
    for s in shots:
        for i, c in enumerate(calls):
            if i in used or abs(c["t"] - s["t"]) > tol:
                continue
            if c["hoop"] and c["hoop"] != s["hoop"]:
                continue
            used.add(i)
            found += 1
            break
    return found, len(calls) - len(used)


def main():
    shots = KEY["shots"]
    print(f"{'grow':>6} {'found':>7} {'extra':>7}   recall by outcome")
    for grow in [1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]:
        calls = calls_for(["out/key_left.json", "out/key_right.json"],
                          "IMG_2528.MOV", grow)
        found, extra = score(calls, shots)
        by = []
        for kind in ("make", "miss", "airball"):
            sub = [s for s in shots if s["outcome"] == kind]
            f, _ = score(calls, sub)
            by.append(f"{kind} {f}/{len(sub)}")
        print(f"{grow:6.1f} {found:5d}/20 {extra:7d}   " + "  ".join(by))

    # The deck false positives must stay dead at whatever value wins. Scored on
    # the window review already gave a verdict on: three calls, none of them shots.
    print("\nagainst the deck window marked (10:30-10:50, 0 shots, 3 false "
          "calls before the zone gate):")
    for grow in [1.3, 2.0, 3.0, 4.0, 5.0]:
        calls = calls_for(["out/fine_fp.json"], "IMG_2529.MOV", grow)
        print(f"{grow:6.1f} {len(calls):5d} calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
