"""The net rule, measured properly and then enforced.

review, twice, and the second time after I reported a negative result: "if the net
doesn't move significantly, the ball didn't go in. that's the case i want
enforced."

The first attempt measured a band inside the review clips, which are a 640px-wide
crop of a much WIDER region -- the net there is a few dozen pixels across, and a
shimmer is a handful of pixels inside a box mostly full of deck. That measured
nothing and I wrongly read it as the rule failing.

This measures at NATIVE resolution, on a box that is only net, over the window
that starts when the ball arrives rather than the whole clip. Those are three
different things from the first attempt, and each of them matters more than the
threshold does.

The rule is enforced one-sidedly, exactly as stated: a still net vetoes a make.
A moving net proves nothing, because a ball off the iron shakes it just as hard,
and 110 of 217 judged shots on this footage are off-the-iron.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops
from frames import FFMPEG

ROOT = Path(__file__).resolve().parent.parent

# The net hangs from the rim. Its box is the rim's own width, from the rim's
# bottom edge down by about one and a half rim heights, which is where the net
# is in every frame checked by eye.
NET_DROP = 1.5
AFTER_S = 0.9        # how long after the ball arrives to watch
BEFORE_S = 0.7       # quiet window before it, for the resting baseline


def net_box(rig, hoop):
    x1, y1, x2, y2 = rig.rims[hoop]
    h = y2 - y1
    return (int(x1), int(y2), int(x2), int(y2 + h * NET_DROP))


def _grab(video, t0, dur, box, scale=2):
    """Native-resolution crop of the net, as a stack of gray frames."""
    import cv2
    import numpy as np
    x1, y1, x2, y2 = box
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "n.mp4"
        r = subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-ss", f"{max(0, t0):.3f}",
             "-i", str(video), "-t", f"{dur:.3f}",
             "-vf", f"crop={x2-x1}:{y2-y1}:{x1}:{y1},scale=iw*{scale}:ih*{scale}",
             "-c:v", "libx264", "-crf", "12", "-preset", "ultrafast", "-an", str(out)],
            capture_output=True)
        if not out.exists():
            return None
        cap = cv2.VideoCapture(str(out))
        fr = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
        cap.release()
    return fr if len(fr) >= 4 else None


def shimmer(video, rig, hoop, t_arrive):
    """How much the net moves after the ball arrives, against how much it moved before.

    Reported as a RATIO. The net is white mesh over a busy background and its
    resting noise differs completely between the two hoops and with the light,
    so an absolute pixel count is not comparable across shots. What is
    comparable is how much louder it got.
    """
    import numpy as np
    box = net_box(rig, hoop)
    before = _grab(video, t_arrive - BEFORE_S - 0.1, BEFORE_S, box)
    after = _grab(video, t_arrive - 0.15, AFTER_S, box)
    if not before or not after:
        return None

    def energy(stack):
        d = [np.abs(stack[i + 1] - stack[i]) for i in range(len(stack) - 1)]
        # The 95th percentile of per-frame change, so one bright splash does not
        # set the level and a whole-net swing still does.
        return float(np.percentile([float(np.percentile(x, 95)) for x in d], 80))

    q, l = energy(before), energy(after)
    return {"quiet": round(q, 3), "loud": round(l, 3),
            "ratio": round(l / q, 3) if q > 0.01 else None,
            "box": list(box)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2481")
    p.add_argument("--limit", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    judged = [j for j in json.loads((ROOT / "labels/judged.json").read_text(encoding="utf-8"))
              if j["video"] == f"{args.video}.MOV" and j.get("label")]
    MISS = {"offiron", "airball", "behind", "airnet"}
    judged = [j for j in judged if j["label"] == "make" or j["label"] in MISS][:args.limit]
    rig = hoops.rig_for(f"{args.video}.MOV")
    video = ROOT / "footage" / f"{args.video}.MOV"

    rows = []
    for j in judged:
        s = shimmer(video, rig, j["hoop"], float(j["t"]))
        if not s or s["ratio"] is None:
            continue
        rows.append({"n": j["n"], "hoop": j["hoop"],
                     "make": 1 if j["label"] == "make" else 0,
                     "label": j["label"], **s})
        if len(rows) % 20 == 0:
            print(f"  {len(rows)} measured", flush=True)
    (ROOT / "out/netgate.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    import statistics as st
    mk = [r["ratio"] for r in rows if r["make"]]
    ms = [r["ratio"] for r in rows if not r["make"]]
    print(f"\n{len(rows)} measured: {len(mk)} makes, {len(ms)} misses")
    if mk and ms:
        print(f"  makes  median ratio {st.median(mk):.2f}")
        print(f"  misses median ratio {st.median(ms):.2f}")
        print("\nEnforcing the one-sided rule -- a still net vetoes a make:")
        for thr in (1.0, 1.15, 1.3, 1.5, 1.8, 2.2):
            vetoed_makes = sum(1 for v in mk if v < thr)
            vetoed_misses = sum(1 for v in ms if v < thr)
            if vetoed_makes + vetoed_misses == 0:
                continue
            print(f"  below {thr:.2f}: vetoes {vetoed_makes} real makes (COST) and "
                  f"{vetoed_misses} misses (GAIN)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
