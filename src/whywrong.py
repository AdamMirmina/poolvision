"""What do the shots the model gets wrong have in common?

"Make it near perfect" is only actionable if the failures share a cause. If the
wrong calls are spread evenly across every kind of shot, the answer is a better
model. If they concentrate on shots where the detector barely saw the ball, the
answer is a better detector -- a completely different piece of work, and a much
cheaper one, because it needs no new judgment from review at all.

Measured against the out-of-fold scores, so no shot is being judged by a model
that trained on it.

    python src/whywrong.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import hoops
import makemiss as M
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
judged = json.loads((ROOT / "labels/judged.json").read_text(encoding="utf-8"))
cache: dict = {}

rows, Y, extra = [], [], []
for j in judged:
    if j["video"] not in M.RIMWATCH:
        continue
    if j["label"] not in M.MISS_LABELS and j["label"] != "make":
        continue
    cache.setdefault(j["video"], M.load_hits(j["video"]))
    hits = cache[j["video"]].get(j["hoop"], [])
    tr = M.clip_track(hits, j["t"], j["tEnd"])
    if not tr:
        continue
    rim = hoops.rig_for(j["video"]).rims[j["hoop"]]
    f = M.features(tr, rim)
    if not f:
        continue
    cy = (rim[1] + rim[3]) / 2
    rw = rim[2] - rim[0]
    # How much evidence does the detector actually give at the moment that
    # decides the outcome -- the ball within one rim-width of the hoop?
    near = [p for p in tr if abs(p["y"] - cy) < rw and abs(p["x"] - (rim[0] + rim[2]) / 2) < 1.5 * rw]
    rows.append(f)
    Y.append(1 if j["label"] == "make" else 0)
    extra.append({"near": len(near), "total": len(tr), "label": j["label"], "n": j["n"],
                  "conf": max(p["conf"] for p in tr), "hoop": j["hoop"], "video": j["video"]})

names = sorted(rows[0])
X = np.array([[r[n] for n in names] for r in rows])
Y = np.array(Y)
p = cross_val_predict(GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=2,
                                                 learning_rate=0.05),
                      X, Y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                      method="predict_proba")[:, 1]
wrong = (p >= 0.5).astype(int) != Y

print(f"{len(Y)} shots, {wrong.sum()} called wrong out-of-fold ({wrong.mean():.0%})\n")

near = np.array([e["near"] for e in extra])
print("detections within one rim-width of the hoop -- the frames that decide it:")
print(f"  overall     median {np.median(near):.0f}   mean {near.mean():.1f}")
print(f"  called right median {np.median(near[~wrong]):.0f}   mean {near[~wrong].mean():.1f}")
print(f"  called wrong median {np.median(near[wrong]):.0f}   mean {near[wrong].mean():.1f}")

print("\nerror rate by how much the detector saw at the rim:")
for lo, hi, lbl in ((0, 0, "0 (never seen at the rim)"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 99, "6+")):
    m = (near >= lo) & (near <= hi)
    if m.sum() == 0:
        continue
    print(f"  {lbl:<26} {m.sum():>3} shots   {wrong[m].mean():>5.0%} wrong")

print("\nerror rate by what review said:")
for lab in sorted({e["label"] for e in extra}):
    m = np.array([e["label"] == lab for e in extra])
    if m.sum() < 3:
        continue
    print(f"  {lab:<10} {m.sum():>3} shots   {wrong[m].mean():>5.0%} wrong")

print("\nerror rate by hoop (near camera vs far):")
for h in ("left", "right"):
    m = np.array([e["hoop"] == h for e in extra])
    if m.sum():
        print(f"  {h:<6} {m.sum():>3} shots   {wrong[m].mean():>5.0%} wrong   "
              f"median {np.median(near[m]):.0f} detections at the rim")


# Hand the per-clip verdicts to a reviewer so review can look at the failures
# rather than read a table about them. Every probability here is out-of-fold, so
# no clip is being scored by a model that saw its answer.
out = []
for e, prob, truth, w in zip(extra, p, Y, wrong):
    out.append({
        "video": e["video"], "n": int(e["n"]), "label": e["label"],
        "pMake": round(float(prob), 3), "rimDets": int(e["near"]),
        "modelSaid": "make" if prob >= 0.5 else "miss",
        "wrong": bool(w),
    })
(ROOT / "out/model_calls.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"\nwrote out/model_calls.json ({sum(1 for o in out if o['wrong'])} disagreements)")
