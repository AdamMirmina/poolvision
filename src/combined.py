"""Do pixels tell us anything the trajectory doesn't?

Neither number alone settles this. Trajectory scores AUC 0.875, tight-cropped
pixels 0.799. If the pixel model is simply a worse view of the same signal,
combining them changes nothing and the answer to "should review label more" is
no. If it is a partly independent view, the combination beats both and the
answer is yes -- because a pixel model is the part that grows with more data,
while the trajectory model's own curve has already gone flat.

Aligned on (video, clip number), so every row is the same physical shot in both
feature sets. Scored out-of-fold, same folds for all three, so the comparison is
like for like rather than three separate lucky splits.

    python src/combined.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import hoops
import makemiss as M
import pixelmodel as P
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

Xp, yp, meta = P.embed_all(tight=True)
pix = {(m["video"], int(m["n"])): (x, y) for m, x, y in zip(meta, Xp, yp)}

judged = json.loads((ROOT / "labels/judged.json").read_text(encoding="utf-8"))
cache: dict = {}
rows, Xt, Y, keys = [], [], [], []
for j in judged:
    k = (j["video"], int(j["n"]))
    if k not in pix or j["video"] not in M.RIMWATCH:
        continue
    if j["label"] not in M.MISS_LABELS and j["label"] != "make":
        continue
    cache.setdefault(j["video"], M.load_hits(j["video"]))
    tr = M.clip_track(cache[j["video"]].get(j["hoop"], []), j["t"], j["tEnd"])
    if not tr:
        continue
    f = M.features(tr, hoops.rig_for(j["video"]).rims[j["hoop"]])
    if not f:
        continue
    rows.append(f)
    Y.append(1 if j["label"] == "make" else 0)
    keys.append(k)

names = sorted(rows[0])
T = np.array([[r[n] for n in names] for r in rows])
Pm = np.array([pix[k][0] for k in keys])
Y = np.array(Y)
print(f"{len(Y)} shots present in BOTH feature sets ({int(Y.sum())} makes)\n")

cv = StratifiedKFold(5, shuffle=True, random_state=0)


def score(name, X, model):
    p = cross_val_predict(model, X, Y, cv=cv, method="predict_proba")[:, 1]
    print(f"{name:<34} accuracy {((p >= .5).astype(int) == Y).mean():.1%}   AUC {roc_auc_score(Y, p):.3f}")
    return p


gb = lambda: GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=2, learning_rate=0.05)
lr = lambda: make_pipeline(StandardScaler(), PCA(n_components=32, random_state=0),
                           LogisticRegression(max_iter=4000, C=0.1))

pt = score("trajectory only", T, gb())
pp = score("pixels only (tight crop)", Pm, lr())

# Stack rather than concatenate: 1536 raw pixel dimensions bolted onto 25
# trajectory ones would let the wide half dominate the split purely by width.
# Feeding each model's out-of-fold probability into a small combiner keeps the
# two views on equal terms.
S = np.column_stack([pt, pp])
score("both, stacked", S, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))

# And the blunt version, for a sanity check that stacking isn't doing something clever.
for w in (0.3, 0.5, 0.7):
    blend = w * pt + (1 - w) * pp
    print(f"  blend {w:.0%} trajectory / {1-w:.0%} pixels      AUC {roc_auc_score(Y, blend):.3f}")

# Do they disagree? If pixels only ever repeat the trajectory's answer there is
# nothing to gain no matter how they are combined.
agree = ((pt >= .5) == (pp >= .5)).mean()
both_right = (((pt >= .5).astype(int) == Y) & ((pp >= .5).astype(int) == Y)).mean()
pix_saves = ((((pt >= .5).astype(int) != Y)) & (((pp >= .5).astype(int) == Y))).sum()
print(f"\nthe two models agree on {agree:.0%} of shots; both correct on {both_right:.0%}")
print(f"pixels are right on {pix_saves} shots the trajectory gets wrong")
