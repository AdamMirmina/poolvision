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

# Water sits at 192 degrees (96 on OpenCV's 0-179 scale). The margin used to be
# 16, i.e. 32 degrees, which excluded everything from 160 to 224 -- and a blue
# cap lives at 210-230. It was being rejected as pool.
#
# 10 (20 degrees) still clears the water while leaving blue readable. The shape
# test is what actually protects against the water anyway: the pool is not a
# compact round blob sitting high in a head box.
WATER_CV, WATER_MARGIN, MIN_SAT = 96, 10, 105
CAP_MIN_PX, CAP_MAX_PX = 60, 9000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482")
    p.add_argument("--clips", default="out/wide_clips")
    return p.parse_args()


# The stored path box is PADDED: attribute.py adds 0.18 x the person's height on
# every side so the tracker keeps hold of someone who moves. That padding has to
# be removed before looking for a head, and forgetting to remove it is the bug
# review found -- "the model says clip follows a different person than whatever cap
# it says". Searching the top 30% of a padded box lands above the head, on water,
# on the deck, and from this camera angle on whoever is standing BEHIND the
# shooter, since further away means higher in frame.
PAD_FRAC = 0.18


def person_core(x, y, w, h):
    """Undo the padding: the real person inside a padded box.

    Both pads are a fraction of the person's HEIGHT, including the horizontal
    one, so a tall narrow swimmer ends up with a box far wider than they are.
    Stored height is H(1 + 2p); stored width is W + 2pH.
    """
    real_h = h / (1 + 2 * PAD_FRAC)
    inset = PAD_FRAC * real_h
    return x + inset, y + inset, w - 2 * inset, real_h


def cap_in_box(frame, box, cv2, np):
    """The cap on one person, by shape rather than by pixel count."""
    x1, y1, x2, y2 = box
    cx1, cy1, cw, ch = person_core(x1, y1, x2 - x1, y2 - y1)
    x1, y1 = int(cx1), int(cy1)
    h = max(2, int(ch))
    x2 = int(cx1 + cw)
    # The head, and only the head: the top third of the REAL person, and the
    # middle 60% of their width. A cap is centered on a head; anything out at the
    # edges belongs to somebody else.
    side = cw * 0.20
    sub = frame[max(0, y1):max(2, y1 + int(h * 0.34)),
                max(0, int(x1 + side)):max(2, int(x2 - side))]
    if sub.size == 0 or sub.shape[0] < 4 or sub.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    dh = np.minimum(np.abs(H - WATER_CV), 180 - np.abs(H - WATER_CV))
    deg = H * 2
    # Three kinds of cap, three tests. A hue test alone sees only the first.
    #
    # review is planning a black and a white cap tonight, and black would have
    # come back as "no cap read" -- it is neither saturated nor bright, so
    # nothing here was looking for it. Better to find that in the code than in
    # the footage.
    #
    # The risk with black is that dark wet hair is also dark, so this leans on
    # shape harder than the others: hair is not a compact round blob sitting
    # high in the head box, a cap is. Reported separately as -2 so a black
    # reading can be audited rather than blending into the rest.
    masks = [
        # 32-345, not 32-300. The old ceiling was set to keep the red-and-white
        # ball out, and it excluded pink caps at 330 along with it. Inside a
        # head box the ball is rare and a cap is not, and the shape test already
        # rejects a ball that does wander through: it is much smaller than a cap
        # at this distance and rarely sits high in the box.
        (((S > MIN_SAT) & (V > 70) & (dh > WATER_MARGIN) & (deg >= 32) & (deg <= 345)).astype(np.uint8), "color"),
        # S<75, not S<50. A white silicone cap picks up color from the pool it
        # is sitting in, so its pixels are not as colorless as the name
        # suggests: on a real miss, only 100 pixels passed S<50 and they were
        # too scattered to form a blob. Loosening to 75 finds 222 on that same
        # cap, and the water it might be confused with sits at 139, so there is
        # plenty of room between them.
        (((S < 75) & (V > 170)).astype(np.uint8), "white"),
        (((V < 70) & (S < 120)).astype(np.uint8), "black"),
        # Blue, which hue cannot separate from the pool.
        #
        # Measured on one frame: the blue cap sits at hue 206 with saturation
        # 196 and brightness 179, the open water at hue 192 with saturation 143
        # and brightness 223. Fourteen degrees apart in hue, and a single cap
        # drifts forty-six degrees across a video, so no hue rule can do this.
        #
        # But the cap is a deep blue and the water is a pale bright one, which
        # saturation and brightness separate cleanly. Worth knowing beyond the
        # code: a PALE blue cap would be genuinely invisible in this pool.
        ((((deg >= 185) & (deg <= 265)) & (S > 170) & (V < 205)).astype(np.uint8), "color"),
    ]
    best = None
    for m, kind in masks:
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
            # A saturated color is the most trustworthy signal, white next,
            # black last -- it has the most competition from hair and shadow.
            weight = {"color": 1.0, "white": 0.75, "black": 0.55}[kind]
            score = area * weight
            if best is None or score > best[0]:
                val = {"white": -1.0, "black": -2.0}.get(kind)
                best = (score, val if val is not None else float(np.median(deg[lab == i])))
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
            key = "white" if v == -1.0 else "black" if v == -2.0 else int(v // 30) * 30
            band.setdefault(key, []).append(v)
        top = max(band.items(), key=lambda kv: len(kv[1]))
        hue = top[1][0] if top[0] in ("white", "black") else float(np.median(top[1]))
        out.append({"n": r["n"], "hue": round(hue, 1), "agree": f"{len(top[1])}/{len(votes)}"})

    got = [o for o in out if o["hue"] is not None]
    bands = {}
    for o in got:
        k = "white" if o["hue"] == -1.0 else "black" if o["hue"] == -2.0 else int(o["hue"] // 30) * 30
        bands[k] = bands.get(k, 0) + 1
    print(f"{len(got)} of {len(out)} shooters given a cap color")
    for k, v in sorted(bands.items(), key=lambda kv: -kv[1]):
        print(f"  {str(k):>6}: {v}")
    (ROOT / f"out/recap2_{args.video}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
