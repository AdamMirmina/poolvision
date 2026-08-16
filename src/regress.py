"""Score EVERY piece of ground truth on the project, in one command.

This exists because the same mistake has now happened twice: a rule was measured
against the window in front of me, looked like an improvement, and quietly broke
a window review had already judged. The separation rule scored 5/5 on his marked
minute and 0 for 3 on the fresh five minutes. The drop zone killed every deck
false positive and cost three right-hoop shots.

Neither would have shipped if everything had been scored at once. So nothing is
"an improvement" until this passes.

review, asking for exactly this: "i really want to make sure we're good at shot id
before moving on to attribution. i want a bulletproof foundation."

    python src/regress.py

Add a case by adding a window to a key file and a scan for it. Do not add a case
without a scan -- a silently skipped case is worse than a missing one, because
the summary still says everything passed.
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

# Each case is a scan file plus the key that judges it. Kept explicit rather than
# discovered, so a scan going missing is a loud failure instead of a case that
# stops running.
CASES = [
    ("2528 left block, 15:00-17:30",  "IMG_2528.MOV", ["out/key_left.json"],   "answerkey_IMG_2528.json"),
    ("2528 right block, 17:55-20:20", "IMG_2528.MOV", ["out/key_right.json"],  "answerkey_IMG_2528.json"),
    ("2529 marked minute (HELD OUT)", "IMG_2529.MOV", ["out/heldout_2529.json"], "answerkey_IMG_2529.json"),
    ("2529 the deck, no shots",       "IMG_2529.MOV", ["out/fine_fp.json"],    "answerkey_IMG_2529.json"),
    ("2529 one player's dunk",               "IMG_2529.MOV", ["out/fine_dunk.json"],  "answerkey_IMG_2529.json"),
    ("2529 the backboard arc",        "IMG_2529.MOV", ["out/fine_arc.json"],   "answerkey_IMG_2529.json"),
    # The one that matters most. Different camera, five players, real game play,
    # and nothing in the pipeline has ever seen it. Everything above shares one
    # afternoon and one rig.
    ("2482 different camera (HELD OUT)", "IMG_2482.MOV", ["out/key_2482.json"], "answerkey_IMG_2482.json"),
    # The only genuinely unbiased case left. 2482 and the marked minute were both
    # consumed the moment a rule was changed BASED on their results; this footage
    # postdates every rule in the pipeline. Third camera, nine players, and dunks
    # are a third of the play -- 8 of 23, against one in everything else combined.
    ("2770 new camera + dunks (HELD OUT)", "IMG_2770.MOV", ["out/key_2770.json"], "answerkey_IMG_2770.json"),
    # THE only unfitted case. Every other key has been tuned against, including
    # the first 2770 window, which the frame-rate sweeps consumed by scoring
    # across all cases. 25 shots, 8 cap colors, 3 dunks, marked after every rule
    # currently in the pipeline was already written.
    ("2770b TRULY held out, 11:00-17:00", "IMG_2770.MOV", ["out/key_2770b.json"], "answerkey_IMG_2770b.json"),
]

TOL = 4.0


def hms(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def calls_for(video, hit_files):
    rig = hoops.rig_for(video)
    clips.RIG_RIMS = rig.rims
    clips.RIG_DROP = rig.drop or {}
    out, covered = [], []
    for hp in hit_files:
        clips._ZONES.clear()
        raw = json.loads((ROOT / hp).read_text(encoding="utf-8"))
        ts = [h["t"] for v in raw["hits"].values() for h in v]
        if ts:
            covered.append((min(ts), max(ts)))
        for hoop, dets in clips.cluster(raw["hits"], 1.0, 3, video=None):
            out.append({"t": dets[0]["t"], "hoop": hoop})
    return sorted(out, key=lambda c: c["t"]), covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    total_found = total_shots = total_named = total_extra = 0
    missing = []
    rows = []

    for name, video, hits, keyfile in CASES:
        absent = [h for h in hits if not (ROOT / h).exists()]
        if absent:
            missing.append((name, absent))
            rows.append((name, None))
            continue

        key = json.loads((ROOT / "labels" / keyfile).read_text(encoding="utf-8"))
        calls, covered = calls_for(video, hits)

        def scanned(t):
            return any(lo - 1 <= t <= hi + 1 for lo, hi in covered)

        shots = [s for s in key.get("shots", []) if scanned(s["t"])]
        nots = [s for s in key.get("not_shots", []) if scanned(s["t"])]

        # NEAREST first, not first-within-tolerance. the 5:55 and 5:58 shots
        # are three seconds apart, and the old order let the 5:55 mark swallow
        # the call at 5:57.33 -- which is 0.67s from his 5:58 and 2.33s from his
        # 5:55 -- leaving 5:58 scored as a MISS on a shot the model had found.
        # That is the scorer inventing a failure, which is worse than missing a
        # real one: it sent this session chasing a dunk problem that was partly
        # an artefact of how the two were paired.
        #
        # Pair every candidate by |dt| across the whole case, closest first.
        pairs = []
        for si, s in enumerate(shots):
            for i, c in enumerate(calls):
                if abs(c["t"] - s["t"]) > TOL:
                    continue
                if c["hoop"] and s.get("hoop") and c["hoop"] != s["hoop"]:
                    continue
                pairs.append((abs(c["t"] - s["t"]), si, i))
        pairs.sort()
        used, taken, found = set(), set(), 0
        for _, si, i in pairs:
            if si in taken or i in used:
                continue
            taken.add(si)
            used.add(i)
            found += 1
        missed = [s for si, s in enumerate(shots) if si not in taken]

        # A call landing on something review explicitly said was NOT a shot is the
        # expensive kind of wrong, and is counted separately from a call that is
        # merely unexplained.
        named_fp = []
        for nsx in nots:
            for i, c in enumerate(calls):
                if i not in used and abs(c["t"] - nsx["t"]) <= TOL:
                    used.add(i)
                    named_fp.append((nsx, c))
                    break
        extra = [c for i, c in enumerate(calls) if i not in used]

        total_found += found
        total_shots += len(shots)
        total_named += len(named_fp)
        total_extra += len(extra)
        rows.append((name, (found, len(shots), len(named_fp), len(extra), missed, named_fp)))

    print(f"{'case':<34} {'shots':>9} {'named FP':>9} {'other':>6}")
    print("-" * 62)
    for name, r in rows:
        if r is None:
            print(f"{name:<34} {'NO SCAN':>9}")
            continue
        found, nshots, nfp, nextra, missed, named_fp = r
        flag = "" if found == nshots and nfp == 0 else "   <-"
        print(f"{name:<34} {found:>4}/{nshots:<4} {nfp:>9} {nextra:>6}{flag}")
        if args.verbose or missed or named_fp:
            for s in missed:
                print(f"      MISSED {s['at']} {s.get('hoop', '')} {s.get('note', '')[:44]}")
            for s, c in named_fp:
                print(f"      CALLED {s['at']} which is: {s['what'][:48]}")
    print("-" * 62)
    print(f"{'TOTAL':<34} {total_found:>4}/{total_shots:<4} {total_named:>9} {total_extra:>6}")

    if missing:
        print("\nCASES THAT DID NOT RUN -- the total above does not include them:")
        for name, absent in missing:
            print(f"  {name}: missing {', '.join(absent)}")

    ok = total_found == total_shots and total_named == 0 and not missing
    print("\n" + ("PASS" if ok else "NOT CLEAN") +
          f": every marked shot found, no call on anything review flagged as not-a-shot"
          if ok else
          "\nNOT CLEAN: see the arrows above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
