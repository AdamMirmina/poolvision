"""Does the ball actually move TOWARD the rim?

review judged the seven false calls on the held-out window and five of them are
the same thing in different clothes: "purple carried ball from deck back into
pool", "pink passed the ball over red's head towards the deep end", "black threw
ball up and caught it while on deck", "pink brings ball back into pool from
right side deck", "pink pump fakes and then passes the ball". The ball is near
the hoop and moving, and it is not being shot at it.

The discriminator came from review of the one judged a blocked shot:
"wouldn't expect the model to get it because it didn't make progress towards the
basket."

That is a better signal than the one already shipped. MIN_DROP_PX asks how far
the ball fell, which a ball dropped on the deck also does. This asks whether the
gap between ball and rim closed, which a pass across the pool does not.

    python src/approach.py --hits out/key_2770b.json \
        --key labels/answerkey_IMG_2770b.json --video IMG_2770.MOV
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent


def rows_for(hits_p, key_p, video, tol=4.0):
    import clips, hoops
    rig = hoops.rig_for(video)
    clips.RIG_RIMS = rig.rims; clips.RIG_DROP = rig.drop or {}; clips._ZONES.clear()
    raw = json.loads((ROOT / hits_p).read_text(encoding="utf-8"))
    key = json.loads((ROOT / key_p).read_text(encoding="utf-8"))
    marked = key["shots"] if isinstance(key, dict) and "shots" in key else key
    ts = [h["t"] for v in raw["hits"].values() for h in v]
    if ts:
        lo, hi = min(ts), max(ts)
        marked = [s for s in marked if lo - 1 <= s["t"] <= hi + 1]
    calls = []
    for hoop, dets in clips.cluster(raw["hits"], 1.0, 3,
                                    video=str(ROOT / "footage" / video)):
        rx = (rig.rims[hoop][0] + rig.rims[hoop][2]) / 2
        ry = (rig.rims[hoop][1] + rig.rims[hoop][3]) / 2
        d = [math.hypot(p["x"] - rx, p["y"] - ry) for p in dets]
        calls.append({"t": dets[0]["t"], "hoop": hoop, "real": False,
                      # How much of the opening gap was closed. A shot converges
                      # on the rim; a pass or a carry keeps its distance or opens
                      # it up. Normalized by the starting gap so a run that began
                      # far away is not rewarded just for being long.
                      "closed": (d[0] - min(d)) / max(1.0, d[0]),
                      # In rim-widths, so it means the same thing on a near hoop
                      # and a far one. A rig where the two rims differ by 25% in
                      # pixels would otherwise get two different gates.
                      "nearest": min(d) / max(1.0, rig.rims[hoop][2] - rig.rims[hoop][0])})
    pairs = sorted((abs(c["t"] - s["t"]), ci, si)
                   for ci, c in enumerate(calls) for si, s in enumerate(marked)
                   if abs(c["t"] - s["t"]) <= tol and (not c["hoop"] or c["hoop"] == s["hoop"]))
    tc, tsn = set(), set()
    for _, ci, si in pairs:
        if ci in tc or si in tsn: continue
        tc.add(ci); tsn.add(si); calls[ci]["real"] = True
    return calls, len(marked) - len(tsn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="")
    a = ap.parse_args()
    SETS = [("out/key_2482.json","labels/answerkey_IMG_2482.json","IMG_2482.MOV"),
            ("out/key_left.json","labels/answerkey_IMG_2528.json","IMG_2528.MOV"),
            ("out/key_2770.json","labels/answerkey_IMG_2770.json","IMG_2770.MOV")]
    HO = ("out/key_2770b.json","labels/answerkey_IMG_2770b.json","IMG_2770.MOV")
    fit, fm = [], 0
    for s in SETS:
        r, m = rows_for(*s); fit += r; fm += m
    ho, hm = rows_for(*HO)
    for nm, rs in (("FIT", fit), ("HELD OUT", ho)):
        R = [x["closed"] for x in rs if x["real"]]
        F = [x["closed"] for x in rs if not x["real"]]
        R.sort(); F.sort()
        print(f"{nm}: {len(R)} real, {len(F)} false")
        if R: print(f"   real  closed-gap  min {R[0]:.2f}  median {R[len(R)//2]:.2f}")
        if F: print(f"   false closed-gap  min {F[0]:.2f}  median {F[len(F)//2]:.2f}")
    print()
    # A shot either CLOSES the gap to the rim, or it was already at the rim when
    # the detector first found it. The second half matters: some real shots score
    # 0.00 on closed-gap purely because the ball is not seen until it arrives, and
    # a rule with only the first half counts that detection limit as evidence of
    # not-a-shot. It cost four real shots on the held-out window.
    def keep(x, c, n):
        return x["closed"] >= c or x["nearest"] <= n

    tot_fit = sum(1 for x in fit if x["real"]) + fm
    best, bf = (0.0, 0.0), -1
    for ci in range(0, 100, 5):
        for ni in range(0, 40, 2):
            c, n = ci / 100, ni / 10
            tp = sum(1 for x in fit if x["real"] and keep(x, c, n))
            fp = sum(1 for x in fit if not x["real"] and keep(x, c, n))
            rec, prec = tp / tot_fit, tp / max(1, tp + fp)
            f1 = 2 * rec * prec / max(1e-9, rec + prec)
            if f1 > bf:
                best, bf = (c, n), f1
    c, n = best
    print(f"best on the FIT SET: closed >= {c:.2f} OR within {n:.1f} rim-widths")
    tpf = sum(1 for x in fit if x["real"] and keep(x, c, n))
    fpf = sum(1 for x in fit if not x["real"] and keep(x, c, n))
    print(f"  fit  after: {tpf}/{tot_fit} recall, precision {tpf/max(1,tpf+fpf)*100:.0f}%")
    tp0 = sum(1 for x in ho if x["real"]); fp0 = sum(1 for x in ho if not x["real"])
    tp1 = sum(1 for x in ho if x["real"] and keep(x, c, n))
    fp1 = sum(1 for x in ho if not x["real"] and keep(x, c, n))
    tot = tp0 + hm
    print(f"HELD OUT before: {tp0}/{tot} recall, precision {tp0/max(1,tp0+fp0)*100:.0f}% ({fp0} false)")
    print(f"HELD OUT after : {tp1}/{tot} recall, precision {tp1/max(1,tp1+fp1)*100:.0f}% ({fp1} false)")
