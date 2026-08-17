"""What separates the calls that were real from the calls that were not.

Precision is the weak number on the held-out window: 23 real shots found, 9
calls nobody marked. The instinct is to hand all nine to a reviewer and ask, but the
first one turned out to rest on 4 sightings spanning 0.1 seconds, which is not a
flight, and asking a person to adjudicate that is asking them to do a
measurement by eye.

So measure it instead. For every call in a window, print the evidence it was
built from alongside whether the answer key has a shot there. If the false ones
are systematically flimsier, the fix is a gate and nobody has to watch anything.
If they look identical to the real ones, the fix is not a gate, and THAT is when
a human has to look.

    python src/callstats.py --hits out/key_2770b.json \\
        --key labels/answerkey_IMG_2770b.json --video IMG_2770.MOV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent


def hms(t):
    return f"{int(t)//60}:{int(t)%60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--tol", type=float, default=4.0)
    args = ap.parse_args()

    import clips
    import hoops

    rig = hoops.rig_for(args.video)
    clips.RIG_RIMS = rig.rims
    clips.RIG_DROP = rig.drop or {}
    clips._ZONES.clear()

    raw = json.loads(Path(args.hits).read_text(encoding="utf-8"))
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    marked = key["shots"] if isinstance(key, dict) and "shots" in key else key

    calls = clips.cluster(raw["hits"], 1.0, 3,
                          video=str(ROOT / "footage" / args.video))

    rows = []
    for hoop, dets in calls:
        t0, t1 = dets[0]["t"], dets[-1]["t"]
        ys = [d["y"] for d in dets]
        # Nearest marked shot, by the start of the run. The key records roughly
        # when the shot went up, so a real call starts near it.
        near = min((abs(s["t"] - t0), s) for s in marked) if marked else (1e9, None)
        rows.append({
            "hoop": hoop,
            "t": t0,
            "dets": len(dets),
            "dur": t1 - t0,
            "fall": max(ys) - min(ys),
            "conf": max(d["conf"] for d in dets),
            "real": near[0] <= args.tol,
            "off": near[0],
        })

    real = [r for r in rows if r["real"]]
    fake = [r for r in rows if not r["real"]]
    print(f"{len(rows)} calls: {len(real)} match a marked shot, {len(fake)} do not\n")

    def block(name, rs):
        if not rs:
            return
        print(f"=== {name} ({len(rs)})")
        print(f"  {'when':>7} {'hoop':<6} {'dets':>5} {'seconds':>8} "
              f"{'fall px':>8} {'conf':>6}")
        for r in sorted(rs, key=lambda z: z["t"]):
            print(f"  {hms(r['t']):>7} {r['hoop']:<6} {r['dets']:>5} "
                  f"{r['dur']:>8.2f} {r['fall']:>8.0f} {r['conf']:>6.2f}")
        for f in ("dets", "dur", "fall", "conf"):
            v = sorted(r[f] for r in rs)
            print(f"  {f:>7}  min {v[0]:.2f}   median {v[len(v)//2]:.2f}   max {v[-1]:.2f}")
        print()

    block("MATCHED a marked shot", real)
    block("NOBODY MARKED", fake)

    # The question a gate has to answer: is there a cut that removes false calls
    # without removing real ones? Printed as the cost in real shots for each
    # threshold, because a gate that takes a real shot with it is usually a bad
    # trade at 92% recall.
    print("=== would a gate help")
    for f, lo, hi, step in (("dets", 3, 14, 1), ("dur", 0.0, 0.9, 0.1),
                            ("fall", 0, 900, 100)):
        print(f"  keep only calls with {f} >=")
        for i in range(int((hi - lo) / step) + 1):
            thr = lo + i * step
            kr = sum(1 for r in real if r[f] >= thr)
            kf = sum(1 for r in fake if r[f] >= thr)
            if kr == len(real) and kf == len(fake):
                continue
            print(f"    {thr:>6.2f}   keeps {kr}/{len(real)} real, "
                  f"{kf}/{len(fake)} false")
            if kf == 0:
                break
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
