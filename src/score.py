"""Score a run against the answer key.

Every accuracy number this project has produced was computed over the shots the
detector happened to surface, which measures precision on a biased sample and
says nothing at all about what was missed. This scores against a list of shots
recorded by a person watching continuous footage with no model output on screen,
so recall is measurable for the first time.

    python src/score.py --hits out/key_left.json out/key_right.json

Three questions, kept separate because they fail independently and a single
combined percentage hides which one is broken:

  DETECTION   of the marked shots, how many were called, and what was called
              that nobody marked
  OUTCOME     make / miss / airball, over the shots that were detected at all
  SHOOTER     who took it, reported against the rotation baseline rather than
              against chance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clips
import hoops

ROOT = Path(__file__).resolve().parent.parent


def hms(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def _cap(s):
    """Who took it, by cap color.

    Reads `shooter_cap` and falls back to `shooter`: the answer keys used to name
    people, and were rewritten to name cap colors when the repo went public.
    """
    return s.get("shooter_cap") or s.get("shooter") or "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="labels/answerkey_IMG_2528.json")
    ap.add_argument("--hits", nargs="+", required=True)
    ap.add_argument("--video", default="IMG_2528.MOV")
    ap.add_argument("--tol", type=float, default=4.0,
                    help="seconds a call may sit from a mark and still match")
    args = ap.parse_args()

    key = json.loads((ROOT / args.key).read_text(encoding="utf-8"))
    shots = key["shots"]

    rig = hoops.rig_for(args.video)
    clips.RIG_RIMS = rig.rims
    clips.RIG_DROP = rig.drop or {}
    clips._ZONES.clear()

    # Only score inside the windows actually scanned. A shot outside them is not
    # a miss by the model, it is footage nobody looked at, and counting it as a
    # miss would understate recall for no reason.
    calls, covered = [], []
    for hp in args.hits:
        raw = json.loads(Path(hp).read_text(encoding="utf-8"))
        lo, hi = raw.get("from"), raw.get("to")
        if lo is None or hi is None or hi == 0:
            ts = [h["t"] for v in raw["hits"].values() for h in v]
            lo, hi = (min(ts), max(ts)) if ts else (0, 0)
        covered.append((lo, hi))
        clips.EXPLAIN.clear()
        for ev in clips.cluster(raw["hits"], 1.0, 3, video=str(ROOT / "footage" / args.video)):
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
                calls.append({"t": t, "hoop": hoop})
    calls.sort(key=lambda c: c["t"])

    def scanned(t):
        return any(lo - 1 <= t <= hi + 1 for lo, hi in covered)

    inplay = [s for s in shots if scanned(s["t"])]
    skipped = [s for s in shots if not scanned(s["t"])]

    print(f"scanned {', '.join(f'{hms(a)}-{hms(b)}' for a, b in covered)}")
    print(f"{len(inplay)} of {len(shots)} marked shots fall inside that")
    if skipped:
        print(f"  (outside the scan, not counted: "
              f"{', '.join(s['at'] for s in skipped)})")
    print()

    used = set()
    found, missed = [], []
    for s in inplay:
        hit = None
        for i, c in enumerate(calls):
            if i in used or abs(c["t"] - s["t"]) > args.tol:
                continue
            if c["hoop"] and c["hoop"] != s["hoop"]:
                continue
            hit = i
            break
        if hit is None:
            missed.append(s)
        else:
            used.add(hit)
            found.append((s, calls[hit]))

    extra = [c for i, c in enumerate(calls) if i not in used]

    print("=== DETECTION")
    print(f"found   {len(found)}/{len(inplay)}")
    print(f"extra   {len(extra)} calls nobody marked")
    for s, c in found:
        print(f"  HIT   {s['at']} {s['hoop']:5s} {s['outcome']:7s} "
              f"{_cap(s):5s} -> called {hms(c['t'])}")
    for s in missed:
        print(f"  MISS  {s['at']} {s['hoop']:5s} {s['outcome']:7s} {_cap(s)}")
    for c in extra:
        print(f"  EXTRA {hms(c['t'])} {c.get('hoop') or '?'}")

    # Recall split by the things most likely to break it. An airball may never
    # produce a descent into the hoop at all, and a make falls through the net
    # while a miss caroms away -- so a single recall number can hide a rule that
    # only ever sees one kind of shot.
    print()
    print("=== RECALL BY KIND")
    for field in ("outcome", "hoop"):
        vals = sorted({s[field] for s in inplay})
        for v in vals:
            tot = [s for s in inplay if s[field] == v]
            got = [s for s, _ in found if s[field] == v]
            print(f"  {field:8s} {v:8s} {len(got)}/{len(tot)}")

    print()
    print("=== OUTCOME and SHOOTER")
    print("  not scored yet: this run measures detection only. Wiring the")
    print("  existing make/miss and cap readers onto these calls is the next")
    print("  step, and doing it before detection is trustworthy would just")
    print("  produce another number computed over a biased sample.")
    if key.get("caution_on_attribution"):
        print(f"  ({key['caution_on_attribution']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
