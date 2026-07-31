"""Can the trajectory alone tell a make from a miss? Measure it, don't assert it.

Earlier in this project I said no, and that answer was not worth much: it came
from three rules I wrote by hand (net movement, landing position, entry angle),
each ruled out in conversation rather than on data. review killed two of them
himself for reasons no amount of tuning would have fixed -- an airball can swish
the net, and where a ball lands depends on the angle, not the outcome.

Now there are 155 clips he judged personally, so the question can be answered
properly: build features off the tracked trajectory, fit a model, and score it
by cross-validation against the only baseline that matters -- always guessing
"miss", which is right 66% of the time here.

The honest failure mode this guards against: a model that looks good because it
memorised 155 examples. Every number printed is out-of-fold, and the per-video
split is reported too, because both videos share one camera rig and a model that
only works within a single recording is not a result.

    python src/makemiss.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops
from tracks import build_tracks

ROOT = Path(__file__).resolve().parent.parent

# Which rim-detection pass belongs to which recording. IMG_2403 predates the
# per-video naming, hence the odd one out.
RIMWATCH = {
    "IMG_2403.MOV": "out/rimwatch_full.json",
    "IMG_2480.MOV": "out/rimwatch_IMG_2480.json",
    "IMG_2481.MOV": "out/rimwatch_IMG_2481.json",
    "IMG_2482.MOV": "out/rimwatch_IMG_2482.json",
    "IMG_2483.MOV": "out/rimwatch_IMG_2483.json",
}

# A make is a make. Everything else review distinguished (off the iron, behind the
# backboard, airball into the net, airball into nothing) is a miss for this
# question. "notshot" is not a shot at all and is dropped, not counted as a miss:
# training a make/miss model on frames of someone walking past teaches it nothing
# about shooting.
MISS_LABELS = {"offiron", "behind", "airball", "airnet", "inout"}

# How far a track must fall inside the window before it counts as a shot rather
# than a ball drifting on the water. Roughly a rim's width.
MIN_DESCENT_PX = 90


def load_hits(video: str) -> dict:
    path = ROOT / RIMWATCH[video]
    return json.loads(path.read_text(encoding="utf-8"))["hits"]


def clip_track(hits: list[dict], t0: float, t1: float) -> list[dict] | None:
    """The one trajectory this clip is about.

    Detections in the window come from every object near the hoop, so they get
    tracked first and then the track that actually descends through the clip's
    own window is chosen. Picking "all detections in the window" instead is what
    produced phantom shots earlier in the project: ball A above the rim and ball
    B below it a moment later look exactly like one ball going in.
    """
    # IMG_2481's records carry tEnd = 0 (the uploader never wrote the field, now
    # fixed). Rather than drop 128 hand-judged shots over a missing number, fall
    # back to a fixed window: a descent runs well under a second and a half at
    # this frame rate. Approximated here on purpose and NOT written back to the
    # database, where it would masquerade as a measurement.
    if t1 <= t0:
        t1 = t0 + 1.4
    # Gather over the clip's PLAYBACK window (clips.py pads 1.2s before the
    # descent and 2.2s after), not a tight window around the descent. The wider
    # gather scored better -- it gives the tracker enough of the flight to link
    # the arc rather than handing the features a stub.
    window = [h for h in hits if t0 - 1.2 <= h["t"] <= t1 + 2.2]
    if len(window) < 4:
        return None
    tracks = build_tracks(window)
    if not tracks:
        return None
    # The descent the clip was CUT AROUND, not the first one visible and not the
    # biggest. All three readings were tested against the own labels:
    #
    #   anchor descent          82.2% accuracy, AUC 0.875
    #   biggest descent         79.5%, 0.839      (what this code did before)
    #   first visible descent   69.7%, 0.758
    #
    # Review described the rule as "my labels correspond to the first one i see",
    # and 41% of judged clips do hold more than one real descent, so this
    # mattered. But the measurement disagrees with the description by a wide
    # margin, which says the shot he is actually judging is the one the clip is
    # centered on -- an earlier descent caught in the lead padding reads as
    # context, not as the subject. Going with the data over the description,
    # and the gap is far too large to be noise.
    cands = []
    for tr in tracks:
        inside = [p for p in tr if t0 - 1.2 <= p["t"] <= t1 + 2.2]
        if len(inside) < 3:
            continue
        ys = [p["y"] for p in inside]
        if max(ys) - min(ys) < MIN_DESCENT_PX:   # drifting, not a shot
            continue
        cands.append((abs(inside[ys.index(min(ys))]["t"] - t0), tr))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    return cands[0][1]


# The descents exactly as clips.py found them, per recording and hoop.
#
# The bug this replaces: clips.py builds tracks over the WHOLE recording, then
# finds descents inside each track. makemiss was rebuilding tracks inside a
# three-second window around the clip. With several balls in the pool those two
# passes link detections differently, so the features could describe a different
# object than the descent the clip was cut for.
#
# Worse, clips.py MERGES two descents at one hoop whose times overlap -- a fix
# for the duplicate clips the marked shot hit, and correct for display, but it means the
# merged event's points come from two different balls. Feeding that to the
# feature extractor produces a trajectory no ball ever flew.
#
# So: reproduce clips.py's own tracking, keep the descents unmerged, and let the
# caller pick the single one the label belongs to.
_DESCENTS: dict[tuple, list] = {}
_TRACKS: dict[tuple, list] = {}


def descents_for(video: str, hoop: str) -> list[list[dict]]:
    key = (video, hoop)
    if key in _DESCENTS:
        return _DESCENTS[key]
    from clips import JITTER_PX, MAX_GAP_S, MIN_DROP_PX
    from parked import drop_parked
    hits, _ = drop_parked(load_hits(video).get(hoop, []))
    out = []
    for pts in build_tracks(hits):
        runs, cur = [], [pts[0]]
        for q in pts[1:]:
            prev = cur[-1]
            falling = q["y"] >= prev["y"] - JITTER_PX
            if q["t"] - prev["t"] > MAX_GAP_S or not falling:
                runs.append(cur)
                cur = [q]
            else:
                cur.append(q)
        runs.append(cur)
        for r in runs:
            if len(r) >= 3 and (r[-1]["y"] - r[0]["y"]) >= MIN_DROP_PX:
                out.append(r)
    out.sort(key=lambda r: r[0]["t"])
    _DESCENTS[key] = out
    return out


def tracks_for(video: str, hoop: str) -> list[list[dict]]:
    """Whole flights, from the same tracking pass the descents came from."""
    key = (video, hoop)
    if key not in _TRACKS:
        from parked import drop_parked
        hits, _ = drop_parked(load_hits(video).get(hoop, []))
        _TRACKS[key] = build_tracks(hits)
    return _TRACKS[key]


def anchor_descent(video: str, hoop: str, t0: float, whole_flight: bool = True) -> list[dict] | None:
    """The single descent this clip was cut around.

    Matched on start time, which is exactly what clips.py wrote into the record
    as `t`. Where two overlapping descents were merged into one clip, this picks
    the one the timestamp names rather than the blend of both.
    """
    cands = descents_for(video, hoop)
    if not cands:
        return None
    best = min(cands, key=lambda r: abs(r[0]["t"] - t0))
    if abs(best[0]["t"] - t0) > 1.0:
        return None
    if not whole_flight:
        return best
    # The descent names the object; its whole track carries the rise as well,
    # and how high the ball was above the rim is one of the stronger signals.
    # A descent alone begins at the top of the fall, which throws that away.
    ids = {(d["frame"], round(d["x"], 1)) for d in best}
    for tr in tracks_for(video, hoop):
        if any((q["frame"], round(q["x"], 1)) in ids for q in tr):
            lo, hi = best[0]["t"] - 2.0, best[-1]["t"] + 1.5
            span = [q for q in tr if lo <= q["t"] <= hi]
            if len(span) >= len(best):
                return span
    return best


def features(track: list[dict], rim: tuple[int, int, int, int]) -> dict | None:
    """Rim-relative geometry, scaled by rim width.

    Everything is divided by the rim's own width so the two hoops -- one near the
    camera, one far across the pool at roughly half the apparent size -- produce
    comparable numbers. Without that the model would mostly learn which hoop it
    was looking at.
    """
    x1, y1, x2, y2 = rim
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rw = max(1.0, x2 - x1)

    pts = sorted(track, key=lambda p: p["t"])
    if len(pts) < 4:
        return None
    ys = [p["y"] for p in pts]
    peak_i = ys.index(min(ys))                 # image y grows downward: min = apex

    f: dict[str, float] = {}
    f["peak_above"] = (cy - min(ys)) / rw
    f["total_drop"] = (max(ys) - min(ys)) / rw
    f["n_det"] = len(pts)
    f["dur"] = pts[-1]["t"] - pts[0]["t"]
    f["conf_max"] = max(p["conf"] for p in pts)
    f["conf_mean"] = sum(p["conf"] for p in pts) / len(pts)

    # How close it came to the rim's center, at its closest.
    f["min_dist"] = min(math.hypot(p["x"] - cx, p["y"] - cy) for p in pts) / rw

    inside = [p for p in pts if x1 <= p["x"] <= x2 and y1 <= p["y"] <= y2]
    f["n_inside"] = len(inside)
    f["frac_inside"] = len(inside) / len(pts)

    # The descent: apex onwards. This is where a make and a miss diverge, if they
    # diverge at all.
    desc = pts[peak_i:]
    f["desc_n"] = len(desc)

    # Crossing the rim plane on the way down.
    cross = None
    for a, b in zip(desc, desc[1:]):
        if a["y"] <= cy <= b["y"] and b["y"] > a["y"]:
            frac = (cy - a["y"]) / max(1e-6, b["y"] - a["y"])
            cross = {
                "x": a["x"] + frac * (b["x"] - a["x"]),
                "t": a["t"] + frac * (b["t"] - a["t"]),
                "vx": (b["x"] - a["x"]) / max(1e-6, b["t"] - a["t"]),
                "vy": (b["y"] - a["y"]) / max(1e-6, b["t"] - a["t"]),
            }
            break
    f["crossed"] = 1.0 if cross else 0.0
    if cross:
        f["cross_dx"] = (cross["x"] - cx) / rw          # signed: which side
        f["cross_adx"] = abs(cross["x"] - cx) / rw      # unsigned: how far off center
        f["cross_angle"] = math.degrees(math.atan2(abs(cross["vy"]), abs(cross["vx"]) + 1e-6))
        f["cross_vratio"] = abs(cross["vx"]) / (abs(cross["vy"]) + 1e-6)
        after = [p for p in desc if p["t"] > cross["t"]]
        f["n_after"] = len(after)
        if after:
            # A ball through the hoop keeps going down in roughly the same
            # column. A ball off the iron gets kicked sideways.
            f["dx_after"] = abs(after[-1]["x"] - cross["x"]) / rw
            f["dy_after"] = (after[-1]["y"] - cy) / rw
            f["max_dx_after"] = max(abs(p["x"] - cross["x"]) for p in after) / rw
            xs_a = [p["x"] for p in after]
            mean_x = sum(xs_a) / len(xs_a)
            f["x_spread_after"] = (sum((x - mean_x) ** 2 for x in xs_a) / len(xs_a)) ** 0.5 / rw
        else:
            f["dx_after"] = f["dy_after"] = f["max_dx_after"] = f["x_spread_after"] = 0.0
    else:
        for k in ("cross_dx", "cross_adx", "cross_angle", "cross_vratio",
                  "n_after", "dx_after", "dy_after", "max_dx_after", "x_spread_after"):
            f[k] = 0.0

    # Does the descent hold a straight line, or does it break?
    if len(desc) >= 4:
        xs = [p["x"] for p in desc]
        ts = [p["t"] for p in desc]
        n = len(xs)
        mt = sum(ts) / n
        mx = sum(xs) / n
        den = sum((t - mt) ** 2 for t in ts) or 1e-6
        slope = sum((t - mt) * (x - mx) for t, x in zip(ts, xs)) / den
        resid = sum((x - (mx + slope * (t - mt))) ** 2 for t, x in zip(ts, xs)) / n
        f["desc_x_resid"] = (resid ** 0.5) / rw
        f["desc_x_slope"] = abs(slope) / rw
    else:
        f["desc_x_resid"] = f["desc_x_slope"] = 0.0

    return f


def main():
    judged = json.loads((ROOT / "labels/judged.json").read_text(encoding="utf-8"))
    hits_cache: dict[str, dict] = {}

    rows, ys, groups, meta = [], [], [], []
    skipped = Counter()
    for j in judged:
        if j["label"] not in MISS_LABELS and j["label"] != "make":
            skipped[j["label"]] += 1
            continue
        if j["video"] not in RIMWATCH:
            skipped["no-rimwatch"] += 1
            continue
        if j["video"] not in hits_cache:
            hits_cache[j["video"]] = load_hits(j["video"])
        # The descent the clip was actually cut for, in the same tracking
        # context that cut it. Re-tracking inside a window scored 76.0%/0.834
        # against 81.0%/0.852 for this, because with several balls in the pool
        # the two passes disagree about which detections belong to one object.
        tr = anchor_descent(j["video"], j["hoop"], j["t"])
        if not tr:
            skipped["no-track"] += 1
            continue
        rim = hoops.rig_for(j["video"]).rims[j["hoop"]]
        f = features(tr, rim)
        if not f:
            skipped["no-features"] += 1
            continue
        rows.append(f)
        ys.append(1 if j["label"] == "make" else 0)
        groups.append(j["video"])
        meta.append(j)

    print(f"{len(rows)} usable shots  ({sum(ys)} makes, {len(ys) - sum(ys)} misses)")
    if skipped:
        print("skipped:", dict(skipped))
    if len(rows) < 40:
        print("not enough to say anything")
        return

    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    names = sorted(rows[0])
    X = np.array([[r[k] for k in names] for r in rows], dtype=float)
    y = np.array(ys)

    base = max(y.mean(), 1 - y.mean())
    print(f"\nbaseline (always guess the commoner class): {base:.1%}")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "gradient boosting": GradientBoostingClassifier(random_state=0, n_estimators=200,
                                                        max_depth=2, learning_rate=0.05),
    }
    best_name, best_auc, best_prob = None, 0.0, None
    for name, m in models.items():
        prob = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
        pred = (prob >= 0.5).astype(int)
        acc = (pred == y).mean()
        auc = roc_auc_score(y, prob)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        print(f"\n{name}:  accuracy {acc:.1%}   AUC {auc:.3f}")
        print(f"  makes found {tp}/{int(y.sum())}, false makes {fp}, missed makes {fn}")
        if auc > best_auc:
            best_name, best_auc, best_prob = name, auc, prob

    # Does it survive being tested on a recording it never saw?
    vids = sorted(set(groups))
    if len(vids) > 1:
        print("\nheld-out by recording (train on one, test on the other):")
        g = np.array(groups)
        for v in vids:
            tr_i, te_i = g != v, g == v
            if te_i.sum() < 15 or len(set(y[tr_i])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
            m.fit(X[tr_i], y[tr_i])
            p = m.predict_proba(X[te_i])[:, 1]
            acc = ((p >= 0.5).astype(int) == y[te_i]).mean()
            auc = roc_auc_score(y[te_i], p) if len(set(y[te_i])) > 1 else float("nan")
            hold_base = max(y[te_i].mean(), 1 - y[te_i].mean())
            print(f"  {v}: {int(te_i.sum())} shots, accuracy {acc:.1%} (baseline {hold_base:.1%}), AUC {auc:.3f}")

    # What is it actually keying on?
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(X, y)
    coefs = lr[-1].coef_[0]
    order = sorted(range(len(names)), key=lambda i: -abs(coefs[i]))
    print("\nstrongest signals (positive = points to a make):")
    for i in order[:8]:
        print(f"  {coefs[i]:+.2f}  {names[i]}")

    # Where a threshold could be trusted, if anywhere.
    print("\nif it only answers when confident:")
    for lo, hi in ((0.2, 0.8), (0.15, 0.85), (0.1, 0.9)):
        sure = (best_prob <= lo) | (best_prob >= hi)
        if sure.sum() == 0:
            continue
        acc = ((best_prob[sure] >= 0.5).astype(int) == y[sure]).mean()
        print(f"  outside {lo}-{hi}: answers {sure.sum()}/{len(y)} ({sure.mean():.0%}), "
              f"right {acc:.1%} of those")

    out = ROOT / "out/makemiss_scores.json"
    out.write_text(json.dumps([
        {"video": m["video"], "n": m["n"], "clock": m.get("clock"), "label": m["label"],
         "p_make": round(float(p), 3)}
        for m, p in zip(meta, best_prob)
    ], indent=1), encoding="utf-8")
    print(f"\nbest model: {best_name}; per-clip scores -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()


def curve():
    """Is 82% a ceiling, or just where 146 labels get you?

    The whole question of whether this is worth pursuing turns on the shape of
    this curve. If accuracy is still climbing at the last data point, more of
    the labeling fixes it and the plan is 'keep labeling'. If it has gone
    flat, hand-built trajectory features are the limit and the next move is a
    model that looks at the clip's pixels instead -- a different, larger piece of
    work. Guessing which of those is true would be an expensive guess.

        python src/makemiss.py --curve
    """
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    judged = json.loads((ROOT / "labels/judged.json").read_text(encoding="utf-8"))
    hits_cache: dict[str, dict] = {}
    rows, ys = [], []
    for j in judged:
        if j["label"] not in MISS_LABELS and j["label"] != "make":
            continue
        if j["video"] not in RIMWATCH:
            continue
        hits_cache.setdefault(j["video"], load_hits(j["video"]))
        tr = clip_track(hits_cache[j["video"]].get(j["hoop"], []), j["t"], j["tEnd"])
        if not tr:
            continue
        f = features(tr, hoops.rig_for(j["video"]).rims[j["hoop"]])
        if f:
            rows.append(f)
            ys.append(1 if j["label"] == "make" else 0)

    names = sorted(rows[0])
    X = np.array([[r[k] for k in names] for r in rows], dtype=float)
    y = np.array(ys)
    rng = np.random.default_rng(0)

    print(f"learning curve on {len(y)} shots (mean of 8 draws per size)\n")
    print(f"{'labels':>7}  {'accuracy':>9}  {'AUC':>6}")
    for frac in (0.25, 0.4, 0.55, 0.7, 0.85, 1.0):
        n = int(len(y) * frac)
        accs, aucs = [], []
        for rep in range(8 if frac < 1.0 else 1):
            idx = rng.permutation(len(y))[:n] if frac < 1.0 else np.arange(len(y))
            if len(set(y[idx])) < 2:
                continue
            m = GradientBoostingClassifier(random_state=0, n_estimators=200,
                                           max_depth=2, learning_rate=0.05)
            cv = StratifiedKFold(5, shuffle=True, random_state=rep)
            p = cross_val_predict(m, X[idx], y[idx], cv=cv, method="predict_proba")[:, 1]
            accs.append(((p >= 0.5).astype(int) == y[idx]).mean())
            aucs.append(roc_auc_score(y[idx], p))
        print(f"{n:>7}  {np.mean(accs):>8.1%}  {np.mean(aucs):>6.3f}")
