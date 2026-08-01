"""Re-read cap colors from the wide clips, using the tracked path.

attribute.py still carries the first cap reader: count bright colorless pixels
in the top of a person's box and call that white. On the game footage that
returned 44 of 89 shooters as white, which is not believable for a pool with one
white cap in it. The pool deck is bright, colorless, and in frame above the
swimmers -- the same failure the crop reader already had, fixed there and never
back-ported.

A cap is a compact, roughly round blob on top of a head. Shape decides, not
brightness.

Runs on clips already cut and a path already tracked, so it costs a minute
rather than the hour a full re-detection would.

    python src/recap2.py --video IMG_2482
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent

WATER_CV, WATER_MARGIN, MIN_SAT = 96, 16, 105
CAP_MIN_PX, CAP_MAX_PX = 60, 9000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482")
    p.add_argument("--clips", default="out/wide_clips")
    return p.parse_args()


def cap_in_box(frame, box, cv2, np):
    """The cap inside one person's box, by shape rather than by pixel count."""
    x1, y1, x2, y2 = box
    h = max(2, y2 - y1)
    # The head: the top quarter of the box, and only that. Below it is shoulders,
    # arms and water, all of which have opinions about color and none of which
    # are a cap.
    sub = frame[max(0, y1):max(2, y1 + int(h * 0.30)), max(0, x1):max(2, x2)]
    if sub.size == 0 or sub.shape[0] < 4 or sub.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    dh = np.minimum(np.abs(H - WATER_CV), 180 - np.abs(H - WATER_CV))
    deg = H * 2
    masks = [
        (((S > MIN_SAT) & (V > 70) & (dh > WATER_MARGIN) & (deg >= 32) & (deg <= 300)).astype(np.uint8), False),
        (((S < 50) & (V > 180)).astype(np.uint8), True),
    ]
    best = None
    for m, is_white in masks:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (CAP_MIN_PX <= area <= CAP_MAX_PX):
                continue
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if max(bw, bh) > 2.8 * max(1, min(bw, bh)):
                continue                       # a cap is round; the deck is a band
            if area / max(1, bw * bh) < 0.45:
                continue                       # scattered glints are not a cap
            cy = cent[i][1]
            if cy > sub.shape[0] * 0.75:
                continue                       # a cap sits high in the head box
            score = area * (1.0 if not is_white else 0.75)
            if best is None or score > best[0]:
                best = (score, -1.0 if is_white else float(np.median(deg[lab == i])))
    return None if best is None else best[1]


def main():
    args = parse_args()
    import cv2, numpy as np
    res = json.loads((ROOT / f"out/attribute_{args.video}.json").read_text(encoding="utf-8"))
    out = []
    for r in res:
        if r.get("stage") != "attributed" or not r.get("shooterPath"):
            continue
        f = ROOT / args.clips / f"{args.video}_{r['n']}.mp4"
        if not f.exists():
            continue
        path = r["shooterPath"]
        cap = cv2.VideoCapture(str(f))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        votes = []
        # Several moments across the clip, because a head turns and a cap can be
        # briefly hidden behind an arm.
        for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
            t = path[0][0] + (path[-1][0] - path[0][0]) * frac
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                continue
            p = min(path, key=lambda q: abs(q[0] - t))
            box = (int(p[1] / 100 * W), int(p[2] / 100 * H),
                   int((p[1] + p[3]) / 100 * W), int((p[2] + p[4]) / 100 * H))
            v = cap_in_box(fr, box, cv2, np)
            if v is not None:
                votes.append(v)
        cap.release()
        if not votes:
            out.append({"n": r["n"], "hue": None})
            continue
        band = {}
        for v in votes:
            band.setdefault("white" if v < 0 else int(v // 30) * 30, []).append(v)
        top = max(band.items(), key=lambda kv: len(kv[1]))
        hue = -1.0 if top[0] == "white" else float(np.median(top[1]))
        out.append({"n": r["n"], "hue": round(hue, 1), "agree": f"{len(top[1])}/{len(votes)}"})

    got = [o for o in out if o["hue"] is not None]
    bands = {}
    for o in got:
        k = "white" if o["hue"] < 0 else int(o["hue"] // 30) * 30
        bands[k] = bands.get(k, 0) + 1
    print(f"{len(got)} of {len(out)} shooters given a cap color")
    for k, v in sorted(bands.items(), key=lambda kv: -kv[1]):
        print(f"  {str(k):>6}: {v}")
    (ROOT / f"out/recap2_{args.video}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
