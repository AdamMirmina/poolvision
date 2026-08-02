"""the make test: did the top of the ball ever sit inside the rim's ellipse?

The words: "if it's made there will always be one frame as long as no one's
blocking it where the top of the ball is inside the bounding circle as it's
going through. if we can id that frame we should always know."

It is a stronger claim than anything the current model uses, because it is about
the ball's position relative to the hoop's actual opening rather than about the
shape of its trajectory. And unlike the net, it has a clean physical basis: a
ball passing through the hoop must, at some instant, be inside the hoop.

Measured off the rimwatch detections that already exist, so no new inference is
needed. The ball's top is its center minus half its detected width, and the
ellipse is the rim's box at the tilt recorded in hoops.py.

The caveats are the own and both are real: someone blocking the rim can hide
the frame, and the detector has to have found the ball on that frame at all.
Neither is an argument against the rule, they are the reasons it will be a strong
one-sided signal rather than a perfect classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops

ROOT = Path(__file__).resolve().parent.parent


def in_ellipse(pt, rim, tilt_deg, scale=1.0):
    """Is `pt` inside the rim's ellipse, optionally grown or shrunk by `scale`."""
    x1, y1, x2, y2 = rim
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a = (x2 - x1) / 2 * scale
    b = (y2 - y1) / 2 * scale
    if a <= 0 or b <= 0:
        return False
    t = math.radians(tilt_deg)
    dx, dy = pt[0] - cx, pt[1] - cy
    # Into the ellipse's own frame.
    u = dx * math.cos(t) + dy * math.sin(t)
    v = -dx * math.sin(t) + dy * math.cos(t)
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def through(hits, rim, tilt, t0, t1, scale=1.0):
    """The best evidence of a pass-through in this window.

    Returns how deep inside the ellipse the ball's TOP got, as a fraction of the
    ellipse (0 means it never got in, 1 means dead center), and when.
    """
    x1, y1, x2, y2 = rim
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = (x2 - x1) / 2 * scale, (y2 - y1) / 2 * scale
    t = math.radians(tilt)
    best = (0.0, None)
    for h in hits:
        if not (t0 <= h["t"] <= t1):
            continue
        top = (h["x"], h["y"] - h.get("w", 30) / 2.0)
        dx, dy = top[0] - cx, top[1] - cy
        u = dx * math.cos(t) + dy * math.sin(t)
        v = -dx * math.sin(t) + dy * math.cos(t)
        r = (u / a) ** 2 + (v / b) ** 2
        if r <= 1.0:
            depth = 1.0 - r ** 0.5
            if depth > best[0]:
                best = (depth, h["t"])
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2481")
    p.add_argument("--window", type=float, default=1.2)
    return p.parse_args()


def main():
    args = parse_args()
    rw = json.loads((ROOT / f"out/rimwatch_{args.video}.json").read_text(encoding="utf-8"))
    rig = hoops.rig_for(f"{args.video}.MOV")
    judged = [j for j in json.loads((ROOT / "labels/judged.json").read_text(encoding="utf-8"))
              if j["video"] == f"{args.video}.MOV" and j.get("label")]
    MISS = {"offiron", "airball", "behind", "airnet"}
    judged = [j for j in judged if j["label"] == "make" or j["label"] in MISS]

    for scale in (1.0, 1.15, 1.3):
        mk, ms = [], []
        for j in judged:
            hits = rw["hits"].get(j["hoop"], [])
            tilt = (rig.tilt or {}).get(j["hoop"], 0.0)
            d, _ = through(hits, rig.rims[j["hoop"]], tilt,
                           float(j["t"]) - args.window, float(j["t"]) + args.window, scale)
            (mk if j["label"] == "make" else ms).append(d)
        got_mk = sum(1 for v in mk if v > 0)
        got_ms = sum(1 for v in ms if v > 0)
        print(f"\nellipse x{scale}: ball's top got inside on "
              f"{got_mk}/{len(mk)} makes and {got_ms}/{len(ms)} misses")
        if got_mk + got_ms:
            print(f"  of the shots where it did, {got_mk/(got_mk+got_ms):.0%} were makes"
                  f"  (base rate {len(mk)/(len(mk)+len(ms)):.0%})")
        # The one-sided form: never inside cannot be a make.
        never_mk = len(mk) - got_mk
        never_ms = len(ms) - got_ms
        if never_mk + never_ms:
            print(f"  never inside: {never_mk} makes, {never_ms} misses"
                  f"  -> calling those misses is right {never_ms/(never_mk+never_ms):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
