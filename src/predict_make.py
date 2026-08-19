"""Score unlabelled shots for make/miss, using the model trained on labeled ones.

makemiss.py trains and evaluates in one process and keeps nothing, which is fine
for measuring the model and useless for using it. This fits on everything review
has judged and predicts on a video review has not, so the diagnostic can show what
the pipeline thinks happened alongside who it thinks shot it.

The number is a probability, and it is shown as one. At 82% accuracy and 0.875
AUC it is right far more often than not and still wrong on roughly one in five,
so rounding it to a verdict would overstate what it knows.

    python src/predict_make.py --video IMG_2482.MOV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops
import makemiss

ROOT = Path(__file__).resolve().parent.parent


def build(rows):
    """(feature dicts, the rows they came from) for anything trackable."""
    feats, kept = [], []
    for j in rows:
        try:
            tr = makemiss.anchor_descent(j["video"], j["hoop"], j["t"])
        except Exception:
            tr = None
        if not tr:
            continue
        rim = hoops.rig_for(j["video"]).rims[j["hoop"]]
        f = makemiss.features(tr, rim)
        if not f:
            continue
        feats.append(f)
        kept.append(j)
    return feats, kept


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482.MOV")
    return p.parse_args()


def main():
    args = parse_args()
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier

    judged = [j for j in json.loads((ROOT / "labels/judged.json").read_text(encoding="utf-8"))
              if j.get("label") == "make" or j.get("label") in makemiss.MISS_LABELS]
    judged = [j for j in judged if j["video"] in makemiss.RIMWATCH]
    tf, tk = build(judged)
    if len(tf) < 40:
        print(f"only {len(tf)} usable labeled shots; not enough to fit")
        return 1
    names = sorted(tf[0])
    X = np.array([[f[n] for n in names] for f in tf], dtype=float)
    y = np.array([1 if j["label"] == "make" else 0 for j in tk])
    print(f"fitting on {len(y)} judged shots ({int(y.sum())} makes)")

    model = GradientBoostingClassifier(random_state=0).fit(X, y)

    shots = [j for j in json.loads((ROOT / "labels/allshots.json").read_text(encoding="utf-8"))
             if j["video"] == args.video and j.get("label") != "notshot"]
    sf, sk = build(shots)
    if not sf:
        print("no trackable shots in that video")
        return 1
    # A feature the training set never saw cannot be used, and a feature it saw
    # that is missing here has to be filled rather than silently reordered.
    XS = np.array([[f.get(n, 0.0) for n in names] for f in sf], dtype=float)
    p = model.predict_proba(XS)[:, 1]

    # the veto, enforced: a net that does not move cannot be a make.
    #
    # One-sided on purpose. A moving net proves nothing, because a ball off the
    # iron shakes it exactly as hard and off-the-iron is the commonest miss on
    # this footage. A STILL net is the informative half, and measured on 119
    # judged shots it is: makes run at a shimmer ratio of 6.88 against 4.08 for
    # misses, and vetoing below 0.85 kills 11 misses at the cost of 2 makes.
    #
    # Applied as a ceiling on the probability rather than a hard zero, so the
    # model's own evidence still orders the shots it vetoes and a wrong veto
    # degrades a call rather than inverting it.
    import dropzone
    import hoops as _h
    import netgate
    rig = _h.rig_for(args.video)
    video = ROOT / "footage" / args.video
    rw_hits = {}
    rwp = ROOT / f"out/rimwatch_{args.video.replace('.MOV','')}.json"
    if rwp.exists():
        rw_hits[args.video] = json.loads(rwp.read_text(encoding="utf-8"))["hits"]
    out = {}
    vetoed = 0
    for j, pi in zip(sk, p):
        pi = float(pi)
        if pi >= 0.5:
            hoop = j.get("hoop", "left")
            # The stronger of the two vetoes, and the cheaper: a ball that never
            # appeared in the column under the net did not go through it. Right
            # 51 times out of 54 on the judged shots, and it fires on 45% of
            # them, against the net shimmer's 11-for-2 on a much smaller slice.
            hits = rw_hits.get(j["video"], {}).get(hoop, [])
            if hits and not dropzone.dropped(hits, rig.rims[hoop], float(j["t"]),
                                             water=(rig.water or {}).get(hoop),
                                             drop=(rig.drop or {}).get(hoop)):
                pi = min(pi, 0.25)
                vetoed += 1
            elif video.exists():
                sh = netgate.shimmer(video, rig, hoop, float(j["t"]))
                if sh and sh.get("ratio") is not None and sh["ratio"] < netgate.STILL:
                    pi = min(pi, 0.35)
                    vetoed += 1
        out[str(j["n"])] = round(pi, 3)
    if vetoed:
        print(f"  net veto pulled {vetoed} calls back from make")
    dest = ROOT / f"out/pmake_{args.video.replace('.MOV','')}.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"scored {len(out)} of {len(shots)} shots -> {dest.name}")
    print(f"  leaning make: {sum(1 for v in out.values() if v >= 0.5)}, "
          f"leaning miss: {sum(1 for v in out.values() if v < 0.5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
