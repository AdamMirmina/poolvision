"""Does the net move? the tell for a make, measured.

The rule, and the caveat that comes with it: "when it's made, the net always
moves so long as it's visible. net never moved there = not made. ofc, the net
can move if it's not made."

That asymmetry is the whole value. It is not a make detector, it is a MISS
detector with very high confidence in one direction, and the make/miss model is
at 82% precisely because it watches the ball as a moving dot and throws away the
net, the backboard and the moment of contact. This puts one of those back.

Measured on the clips already cut for review, which are rim crops, so no new
decoding of the source video is needed. The net hangs below the rim, so the
region sampled is a band under the rim box, and the measurement is how much it
changes frame to frame AFTER the ball arrives -- before that the ball itself is
moving through and would count as motion.

The ball moving through the band is the obvious confound. It is handled by
measuring change against a rolling median of the band rather than the previous
frame, so a small object passing through contributes far less than the net's
whole width swinging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops

ROOT = Path(__file__).resolve().parent.parent
OUT_W = 640          # clips are cut at this width; see clips.py


def band_in_clip(rig, hoop, clip_w, clip_h):
    """The net's band, in clip pixels.

    The clip is rig.crops[hoop] scaled to OUT_W. The rim sits inside that crop
    at a known place, so the net band below it is known too, without looking at
    a single pixel.
    """
    cx1, cy1, cx2, cy2 = rig.crops[hoop]
    rx1, ry1, rx2, ry2 = rig.rims[hoop]
    sx = clip_w / (cx2 - cx1)
    sy = clip_h / (cy2 - cy1)
    # Drawn and checked: the first version padded OUTWARD from the rim and ran
    # two rim heights down, which put most of the band on deck, pole and
    # background. The net is a fraction of that box, so its movement was diluted
    # by a majority of pixels that never move, and makes came out no different
    # from misses. Narrowed to the middle of the rim's span and shortened, so
    # what is measured is mostly net.
    cx = (rx1 + rx2) / 2
    half = (rx2 - rx1) * 0.33
    x1 = (cx - half - cx1) * sx
    x2 = (cx + half - cx1) * sx
    y1 = (ry2 - cy1) * sy                       # from the rim's bottom edge
    y2 = y1 + (ry2 - ry1) * sy * 1.4
    return (int(max(0, x1)), int(max(0, y1)),
            int(min(clip_w, x2)), int(min(clip_h, y2)))


def motion(clip_path, rig, hoop, t_from=None):
    """How much the net band moves, and when.

    Returns a dict with the peak and the total, both normalized by the band's
    area so the two hoops compare despite their very different apparent sizes.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) < 8:
        return None

    h, w = frames[0].shape[:2]
    bx1, by1, bx2, by2 = band_in_clip(rig, hoop, w, h)
    if bx2 - bx1 < 6 or by2 - by1 < 6:
        return None
    band = [cv2.cvtColor(f[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY).astype(np.float32)
            for f in frames]

    # A rolling median is the resting net. Change against THAT rather than
    # against the previous frame keeps a ball passing through from reading as
    # the net swinging: the ball is small and brief, the median barely moves.
    med = np.median(np.stack(band[:max(4, len(band) // 3)]), axis=0)
    area = band[0].size
    series = []
    for i, b in enumerate(band):
        d = np.abs(b - med)
        series.append(float((d > 18).sum()) / area)
    peak_i = int(max(range(len(series)), key=lambda i: series[i]))
    return {"peak": round(max(series), 4), "peak_frac": round(peak_i / len(series), 3),
            "mean": round(sum(series) / len(series), 4),
            "n": len(series), "band": [bx1, by1, bx2, by2]}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="IMG_2481")
    return p.parse_args()


def main():
    args = parse_args()
    judged = json.loads((ROOT / "labels/judged.json").read_text(encoding="utf-8"))
    # The vocabulary review actually used, read off the labels rather than assumed.
    MISS = {"offiron", "airball", "behind", "airnet"}
    by_key = {(j["video"], int(j["n"])): j for j in judged}

    rows = []
    for vid in args.videos.split(","):
        d = ROOT / f"out/clips_{vid}"
        if not d.exists():
            d = ROOT / f"out/clips_{vid.replace('IMG_', '')}"
        idx = d / "index.json"
        if not idx.exists():
            print(f"no clips for {vid}")
            continue
        rig = hoops.rig_for(f"{vid}.MOV")
        for c in json.loads(idx.read_text(encoding="utf-8")):
            j = by_key.get((f"{vid}.MOV", int(c["n"])))
            if not j or not j.get("label"):
                continue
            lab = j["label"]
            if lab != "make" and lab not in MISS:
                continue
            m = motion(d / c["file"], rig, c["hoop"])
            if not m:
                continue
            rows.append({"video": vid, "n": c["n"], "hoop": c["hoop"],
                         "make": 1 if lab == "make" else 0, **m})
        print(f"{vid}: {sum(1 for r in rows if r['video'] == vid)} judged clips measured")

    if not rows:
        print("nothing measured")
        return 1
    out = ROOT / "out/netmove.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    import statistics as st
    for hoop in sorted({r["hoop"] for r in rows}):
        mk = [r["peak"] for r in rows if r["make"] and r["hoop"] == hoop]
        ms = [r["peak"] for r in rows if not r["make"] and r["hoop"] == hoop]
        if not mk or not ms:
            continue
        print(f"\n{hoop} hoop: {len(mk)} makes, {len(ms)} misses")
        print(f"  peak band change, makes:  median {st.median(mk):.4f}")
        print(f"  peak band change, misses: median {st.median(ms):.4f}")
        # The claim to test is the one-sided one: a still net means not made.
        for thr in (0.02, 0.05, 0.08, 0.12):
            still_makes = sum(1 for v in mk if v < thr)
            still_misses = sum(1 for v in ms if v < thr)
            if still_makes + still_misses == 0:
                continue
            print(f"  below {thr:.2f}: {still_makes} makes, {still_misses} misses"
                  f"  -> calling those misses is right {still_misses}/{still_makes+still_misses}")
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
