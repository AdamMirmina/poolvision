"""Is the drop zone still earning its place, now that gravity is a gate?

The zone was added to kill the deck false positives and its size was swept
against IMG_2528 alone. On held-out footage it is the reason for 7 of 9 misses
on a different camera and 4 of 4 on the marked minute.

Gravity kills the same cases for a better reason -- a carried ball moves at
wading speed whatever the geometry says -- so the question is whether the zone is
now redundant. Scored over EVERY case, because a zone change is exactly the kind
of thing that fixes one window and breaks another.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clips, regress

for grow in [2.0, 3.0, 4.0, 6.0, 1e6]:
    clips.ZONE_GROW = grow
    tot = found = named = extra = 0
    per = []
    for name, video, hits, keyfile in regress.CASES:
        if any(not (regress.ROOT / h).exists() for h in hits):
            continue
        import json
        key = json.loads((regress.ROOT / "labels" / keyfile).read_text(encoding="utf-8"))
        clips._ZONES.clear()
        calls, covered = regress.calls_for(video, hits)
        def scanned(t): return any(lo - 1 <= t <= hi + 1 for lo, hi in covered)
        shots = [s for s in key.get("shots", []) if scanned(s["t"])]
        nots = [s for s in key.get("not_shots", []) if scanned(s["t"])]
        used, f = set(), 0
        for s in shots:
            for i, c in enumerate(calls):
                if i in used or abs(c["t"] - s["t"]) > regress.TOL: continue
                if c["hoop"] and s.get("hoop") and c["hoop"] != s["hoop"]: continue
                used.add(i); f += 1; break
        nf = 0
        for nsx in nots:
            for i, c in enumerate(calls):
                if i not in used and abs(c["t"] - nsx["t"]) <= regress.TOL:
                    used.add(i); nf += 1; break
        found += f; tot += len(shots); named += nf
        extra += len(calls) - len(used)
        per.append(f"{f}/{len(shots)}")
    label = "no zone" if grow > 1000 else f"grow {grow}"
    print(f"{label:>10}   {found:3d}/{tot}   named FP {named}   other {extra}   [{'  '.join(per)}]")
