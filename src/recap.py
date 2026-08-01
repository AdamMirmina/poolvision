"""Re-read the cap color from the saved release clips.

The first reader counted pixels: bright and colorless meant white, saturated
and away from the water meant a color. On 57 attributed shots it called 33 of
them white, which is not believable for a pool with one white cap in it.

The cause is visible in any release clip: the pool deck is in frame above the
swimmers. Gray concrete in sun is bright and colorless, so a pixel count reads
it as a white cap every time the deck falls inside the top of a person's box.

A cap is not a pixel count. It is a compact, roughly round blob of one color
sitting on top of a head. So find connected components and test their shape and
size, which is what caps_probe did on whole frames and what this should have
done from the start.

Runs on the already-saved clips instead of reprocessing 4K footage, so it costs
seconds rather than the hour the original pass took.

    python src/recap.py --video IMG_2481
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WATER_HUE_CV = 96       # OpenCV 0-179 scale
WATER_MARGIN = 16
MIN_SAT = 105
CAP_MIN_PX = 140        # smaller than caps_probe's floor: these crops are downscaled
CAP_MAX_PX = 20000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2481")
    p.add_argument("--clips", default="out/shooter_clips")
    p.add_argument("--debug", action="store_true", help="write a contact sheet to look at")
    return p.parse_args()


def read_cap(frame, cv2, np):
    """The cap on the person nearest the middle of this crop, or nothing.

    The clip is centered on the shooter, so the cap should be near the center
    horizontally and in the upper half. Everything else -- the deck along the
    top, another player's cap at the edge, a ball -- is rejected by position,
    size and shape rather than by color alone.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    dh = np.minimum(np.abs(H - WATER_HUE_CV), 180 - np.abs(H - WATER_HUE_CV))
    deg = H * 2

    colored = ((S > MIN_SAT) & (V > 70) & (dh > WATER_MARGIN)).astype(np.uint8)
    # White is a real cap here, but so is sunlit concrete. Shape decides between
    # them, not brightness.
    white = ((S < 50) & (V > 175)).astype(np.uint8)

    best = None
    for mask, is_white in ((colored, False), (white, True)):
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (CAP_MIN_PX <= area <= CAP_MAX_PX):
                continue
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if max(bw, bh) > 2.6 * max(1, min(bw, bh)):     # a cap is round, the deck is a band
                continue
            cx, cy = cent[i]
            if abs(cx - w / 2) > w * 0.30:                  # the shooter is centered in this crop
                continue
            if cy > h * 0.62:                               # a cap is on a head, not underwater
                continue
            fill = area / max(1, bw * bh)
            if fill < 0.45:                                 # scattered glints are not a cap
                continue
            # Prefer the highest plausible blob: heads sit above shoulders.
            score = (h - cy) * (1.0 if not is_white else 0.92)
            if best is None or score > best[0]:
                sel = lab == i
                hue = -1.0 if is_white else float(np.median(deg[sel]))
                best = (score, hue, int(area), int(cx), int(cy))
    return best


def main():
    args = parse_args()
    import cv2
    import numpy as np

    res = json.loads((ROOT / f"out/attribute_{args.video}.json").read_text(encoding="utf-8"))
    clips = ROOT / args.clips
    out, tiles = [], []
    for r in res:
        n = r.get("n")
        if n is None:
            continue
        f = clips / f"{args.video}_{n}.mp4"
        if not f.exists():
            continue
        cap = cv2.VideoCapture(str(f))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        votes = []
        # Several frames, because a head turns and a cap can be briefly hidden.
        for k in (0.25, 0.45, 0.6, 0.8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * k))
            ok, fr = cap.read()
            if not ok:
                continue
            got = read_cap(fr, cv2, np)
            if got:
                votes.append(got)
                if args.debug and len(tiles) < 18 and k == 0.45:
                    t = cv2.resize(fr, (150, 150))
                    col = (240, 240, 240) if got[1] < 0 else tuple(
                        int(v) for v in cv2.cvtColor(
                            np.uint8([[[int(got[1] / 2), 220, 230]]]), cv2.COLOR_HSV2BGR)[0][0])
                    cv2.rectangle(t, (0, 0), (149, 18), col, -1)
                    cv2.putText(t, "white" if got[1] < 0 else f"{int(got[1])}", (4, 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                    tiles.append(t)
        cap.release()
        if not votes:
            out.append({"n": n, "clock": r.get("clock"), "hue": None})
            continue
        # Majority across frames, in 30-degree bands, so one bad frame cannot
        # decide a player's identity.
        band = {}
        for _, hue, area, _, _ in votes:
            k = "white" if hue < 0 else int(hue // 30) * 30
            band.setdefault(k, []).append(hue)
        top = max(band.items(), key=lambda kv: len(kv[1]))
        hue = -1.0 if top[0] == "white" else float(np.median(top[1]))
        out.append({"n": n, "clock": r.get("clock"), "hue": round(hue, 1),
                    "agree": f"{len(top[1])}/{len(votes)}"})

    got = [o for o in out if o["hue"] is not None]
    print(f"{len(got)} of {len(out)} clips gave a cap color\n")
    bands = {}
    for o in got:
        k = "white" if o["hue"] < 0 else int(o["hue"] // 30) * 30
        bands[k] = bands.get(k, 0) + 1
    for k, v in sorted(bands.items(), key=lambda kv: -kv[1]):
        print(f"  {str(k):>6}: {v}")
    (ROOT / f"out/recap_{args.video}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    if tiles:
        cv2.imwrite(str(ROOT / "out/recap_check.png"), np.hstack(tiles))
        print("\nwrote out/recap_check.png")


if __name__ == "__main__":
    main()
