"""Fit a false-call gate on the in-sample videos, then test it on the held-out one.

Every threshold in this pipeline until now was chosen by sweeping it against the
same footage it was then scored on, which is why 75 of 80 was optimistic by
construction and nobody could say by how much. The held-out window answered
that for detection. This does the same for the gate itself.

The candidate signal is how far the ball actually descends across a call. Real
shots drop a long way; the false calls are balls rolling, being carried, or
sitting in the frame, and they barely descend at all. That was visible first on
the held-out window and then on all three in-sample videos, which is the right
order to find something in.

So: choose the threshold using ONLY the in-sample videos, then report what it
does to the held-out window. If the held-out result matches the in-sample
result, the gate generalizes. If it does not, the gate was fitted to noise and
this prints that rather than hiding it.

    python src/gatefit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent

# hits, key, video. The last entry is held out and is never used to choose.
FIT = [
    ("out/key_2482.json", "labels/answerkey_IMG_2482.json", "IMG_2482.MOV"),
    ("out/key_left.json", "labels/answerkey_IMG_2528.json", "IMG_2528.MOV"),
    ("out/key_2770.json", "labels/answerkey_IMG_2770.json", "IMG_2770.MOV"),
]
HELDOUT = ("out/key_2770b.json", "labels/answerkey_IMG_2770b.json", "IMG_2770.MOV")
TOL = 4.0


def rows_for(hits_p, key_p, video):
    import clips
    import hoops
    rig = hoops.rig_for(video)
    clips.RIG_RIMS = rig.rims
    clips.RIG_DROP = rig.drop or {}
    clips._ZONES.clear()
    raw = json.loads((ROOT / hits_p).read_text(encoding="utf-8"))
    key = json.loads((ROOT / key_p).read_text(encoding="utf-8"))
    marked = key["shots"] if isinstance(key, dict) and "shots" in key else key
    # Only shots the scan actually covered. Several of these scans are windows,
    # not whole videos, and counting a shot that was never looked at as a miss
    # understates recall -- it said 78% on the fit set where score.py says 94%,
    # and a threshold chosen against that number is chosen against an artefact.
    ts = [h["t"] for v in raw["hits"].values() for h in v]
    if ts:
        lo, hi = min(ts), max(ts)
        marked = [s for s in marked if lo - 1 <= s["t"] <= hi + 1]
    calls = []
    for hoop, dets in clips.cluster(raw["hits"], 1.0, 3,
                                    video=str(ROOT / "footage" / video)):
        ys = [d["y"] for d in dets]
        calls.append({"t": dets[0]["t"], "hoop": hoop,
                      "fall": max(ys) - min(ys), "real": False})

    # ONE-TO-ONE, and the hoop has to agree -- the same rule score.py uses.
    #
    # The first version of this took each call's nearest marked shot
    # independently, so two calls could both claim one shot and both count as
    # real. That reported 25 of 25 on the held-out window where score.py
    # reported 23, and the extra two were duplicates being counted as
    # successes. It also polluted the measurement this file exists for, since a
    # duplicate call sits in the "real" pile while behaving like a false one.
    pairs = sorted(
        ((abs(c["t"] - s["t"]), ci, si)
         for ci, c in enumerate(calls)
         for si, s in enumerate(marked)
         if abs(c["t"] - s["t"]) <= TOL and (not c["hoop"] or c["hoop"] == s["hoop"]))
    )
    tc, ts = set(), set()
    for _, ci, si in pairs:
        if ci in tc or si in ts:
            continue
        tc.add(ci)
        ts.add(si)
        calls[ci]["real"] = True
    out = [{"fall": c["fall"], "real": c["real"]} for c in calls]
    marked = [s for i, s in enumerate(marked) if True]
    # Marked shots the pipeline never called at all. They cannot be rescued by a
    # gate, but they belong in the recall denominator or every number here is
    # computed over the shots that happened to be surfaced -- the exact bias this
    # project has already been burned by once.
    called = sum(1 for r in out if r["real"])
    return out, len(marked) - called


def score(rows, missed, thr):
    keep = [r for r in rows if r["fall"] >= thr]
    tp = sum(1 for r in keep if r["real"])
    fp = len(keep) - tp
    total_real = sum(1 for r in rows if r["real"]) + missed
    return tp, fp, total_real


def main():
    fit_rows, fit_missed = [], 0
    for h, k, v in FIT:
        r, m = rows_for(h, k, v)
        fit_rows += r
        fit_missed += m
    ho_rows, ho_missed = rows_for(*HELDOUT)

    print(f"fit set:  {len(fit_rows)} calls, "
          f"{sum(1 for r in fit_rows if r['real'])} real, "
          f"{sum(1 for r in fit_rows if not r['real'])} false, "
          f"{fit_missed} marked shots never called")
    print(f"held out: {len(ho_rows)} calls, "
          f"{sum(1 for r in ho_rows if r['real'])} real, "
          f"{sum(1 for r in ho_rows if not r['real'])} false, "
          f"{ho_missed} marked shots never called\n")

    print("=== choosing on the FIT SET ONLY")
    print(f"  {'thr':>5} {'recall':>16} {'precision':>12}")
    best, bestf = None, -1
    for thr in range(0, 701, 25):
        tp, fp, tot = score(fit_rows, fit_missed, thr)
        rec = tp / tot
        prec = tp / max(1, tp + fp)
        f1 = 2 * rec * prec / max(1e-9, rec + prec)
        if f1 > bestf:
            best, bestf = thr, f1
        if thr % 100 == 0:
            print(f"  {thr:>5} {tp:>3}/{tot} = {rec*100:5.1f}%  "
                  f"{prec*100:9.1f}%")
    print(f"\n  best F1 on the fit set: fall >= {best}")

    print("\n=== that threshold, applied to the HELD-OUT window")
    for label, rows, missed in (("held out", ho_rows, ho_missed),):
        tp0, fp0, tot = score(rows, missed, 0)
        tp1, fp1, _ = score(rows, missed, best)
        print(f"  before:  {tp0}/{tot} recall = {tp0/tot*100:.1f}%, "
              f"precision {tp0/max(1,tp0+fp0)*100:.1f}%  ({fp0} false)")
        print(f"  after :  {tp1}/{tot} recall = {tp1/tot*100:.1f}%, "
              f"precision {tp1/max(1,tp1+fp1)*100:.1f}%  ({fp1} false)")
        print(f"  cost  :  {tp0-tp1} real shots lost, {fp0-fp1} false calls removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
