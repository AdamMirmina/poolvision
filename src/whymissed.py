"""For every shot the pipeline misses, across every case: what does the input
look like, and which gate rejected it.

Written after four fruitless parameter sweeps. What finally cracked the IMG_2482
gap was not another knob but measuring the INPUT: those shots turned out to carry
MORE detections than the median shot the pipeline finds, which meant the loss had
to be downstream, and the cause was a per-frame distance threshold on 30fps
footage. This does that measurement for every remaining miss in one pass, so the
next cause is found by reading rather than by guessing.

Three numbers per missed shot, and they separate the possible causes cleanly:

  sightings   how often the ball was seen near it. Low means a detector problem
              and no gate change can help.
  best run    the longest run the tracker built there. Small while sightings are
              high means fragmentation, which is what 2482 turned out to be.
  reason      which gate rejected the largest nearby run, when one was built.

    python src/whymissed.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clips
import hoops
import regress
from parked import drop_parked
from tracks import build_tracks

TOL = 4.0


def runs_near(hits, hoop, t, win=3.0):
    """Every run the tracker builds near a time, longest first."""
    h = clips._one_box_per_ball(hits.get(hoop, []))
    h, _ = drop_parked(h)
    out = []
    for pts in build_tracks(h):
        runs, cur = [], [pts[0]]
        for p in pts[1:]:
            prev = cur[-1]
            if (p["t"] - prev["t"] > clips.MAX_GAP_S
                    or not (p["y"] >= prev["y"] - clips.JITTER_PX)):
                runs.append(cur)
                cur = [p]
            else:
                cur.append(p)
        runs.append(cur)
        out += [r for r in runs if abs(r[0]["t"] - t) <= win or abs(r[-1]["t"] - t) <= win]
    return sorted(out, key=len, reverse=True)


def main():
    tally = Counter()
    print(f"{'case':<16} {'shot':>6} {'hoop':<6} {'outcome':<8} "
          f"{'seen':>5} {'run':>4}  why")
    print("-" * 88)
    for name, video, hitfiles, keyfile in regress.CASES:
        if any(not (regress.ROOT / h).exists() for h in hitfiles):
            continue
        rig = hoops.rig_for(video)
        clips.RIG_RIMS, clips.RIG_DROP = rig.rims, rig.drop or {}
        clips._ZONES.clear()
        clips.EXPLAIN.clear()

        hits, calls = {}, []
        for hp in hitfiles:
            raw = json.loads((regress.ROOT / hp).read_text(encoding="utf-8"))
            for k, v in raw["hits"].items():
                hits.setdefault(k, []).extend(v)
            for h, d in clips.cluster(raw["hits"], 1.0, 3, video=None):
                calls.append({"t": d[0]["t"], "hoop": h})

        key = json.loads((regress.ROOT / "labels" / keyfile).read_text(encoding="utf-8"))
        ts = [x["t"] for v in hits.values() for x in v]
        lo, hi = (min(ts), max(ts)) if ts else (0, 0)
        for s in key.get("shots", []):
            if not (lo - 1 <= s["t"] <= hi + 1):
                continue
            if any(abs(c["t"] - s["t"]) <= TOL
                   and (not c["hoop"] or not s.get("hoop") or c["hoop"] == s["hoop"])
                   for c in calls):
                continue
            seen = len(clips._one_box_per_ball(
                [x for x in hits.get(s["hoop"], []) if abs(x["t"] - s["t"]) <= 3]))
            rs = runs_near(hits, s["hoop"], s["t"])
            best = len(rs[0]) if rs else 0
            near = [e for e in clips.EXPLAIN
                    if e[1] == s["hoop"] and abs(e[0] - s["t"]) <= TOL]
            why = near[0][2] if near else "no run built at all"
            # A shot seen fewer than a dozen times is a DETECTOR problem; no gate
            # change reaches it. Called out separately so it is never mistaken
            # for a rule that needs loosening.
            kind = ("detector: barely seen" if seen < 12 else
                    "fragmented: many sightings, tiny runs" if best <= 3 else why)
            tally[kind] += 1
            print(f"{name[:16]:<16} {s['at']:>6} {s.get('hoop',''):<6} "
                  f"{s.get('outcome','?'):<8} {seen:>5} {best:>4}  {kind}")

    print("-" * 88)
    for k, n in tally.most_common():
        print(f"  {n:2d}  {k}")
    print(f"  {sum(tally.values()):2d}  total missed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
