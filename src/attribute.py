"""Who took the shot? A pilot, on twenty shots, before spending hours on it.

Scoring works by watching two small crops around the rims, which is cheap. The
shooter is thirty feet away in the pool and never appears in those crops, so
attribution needs the opposite thing: the full frame.

The route is one phase 3 already measured. The ball tracks WELL in open water --
99.5% of samples correct, unbroken runs to 9.2 seconds -- and it was the rim
region that failed. So follow the ball backwards from the hoop to where it was
released, find the person nearest that point, and read their cap.

Every step of that can fail, and the point of a pilot is to find out WHICH
before committing to a full pass:

  1. is the ball detected in open water on the way to the hoop?
  2. does a release point fall out of the trajectory?
  3. is anyone detected near it?
  4. does that person carry a readable cap color?

Deliberately reports a count for each stage rather than a single success rate.
"A quarter of shots attributed" says nothing about what to fix; "the ball is
found 80% of the time but a person is only near it 30% of the time" says
exactly what to fix.

    python src/attribute.py --video IMG_2482.MOV --shots 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops

ROOT = Path(__file__).resolve().parent.parent
SPORTS_BALL, PERSON = 32, 0

# The pool, in frame coordinates. Outside it is deck, grass and foliage: all
# strongly colored, none of it a swimmer.
POOL = (1150, 250, 3500, 1750)

# Measured off this footage in caps_probe: the water sits near 192 degrees, the
# ball at 320-354, skin around 0-20. A cap has to be saturated and away from
# all of those to count.
WATER_HUE_CV = 96      # OpenCV's 0-179 scale
WATER_MARGIN = 18
MIN_SAT = 110


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482.MOV")
    p.add_argument("--shots", type=int, default=20)
    p.add_argument("--skip", type=int, default=0,
                   help="skip this many shots first. The caps go on partway through "
                        "IMG_2481, and the first run took the earliest 20 shots, which "
                        "are all bare-headed -- a clean pipeline reporting a real zero.")
    p.add_argument("--save", action="store_true",
                   help="write a crop of the identified shooter per shot, to confirm")
    p.add_argument("--lookback", type=float, default=2.0,
                   help="seconds before the descent to search for the release")
    return p.parse_args()


# Skin and the ball both live at the warm end. Measured on this footage: skin
# 0-20 degrees, the ball 320-354. The caps that showed up in the caps-only
# videos sit at 40-100. So a cap reading has to EXCLUDE the warm end, not just
# exclude the water -- the first version of this did only the latter and
# duly reported 12 of 13 shooters as hue 6-10, which is a suntan.
SKIN_BALL_LO, SKIN_BALL_HI = 300, 30      # degrees, wrapping through 0
CAP_LO, CAP_HI = 32, 300                  # everything else


def _is_cap_hue(deg):
    return CAP_LO <= deg <= CAP_HI


def cap_hue(frame, box, np, cv2):
    """The cap color on a person, or nothing.

    Samples the top FIFTH of the detection box, not the top third. A swimmer
    with their arms raised has a box whose upper third is mostly arm, and arm
    reads as a confident, saturated, completely useless hue.

    Returns None rather than a warm-end hue on purpose. "No cap seen" is a
    usable answer; "this player's cap is the color of skin" is not, and
    silently accepting it is how the phase 3 color stage came to report the
    water as everyone's identity.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    h = max(1, y2 - y1)
    crop = frame[max(0, y1):max(1, y1 + max(6, int(h * 0.20))), max(0, x1):max(1, x2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    dh = np.minimum(np.abs(H - WATER_HUE_CV), 180 - np.abs(H - WATER_HUE_CV))
    deg = H * 2
    colored = (S > MIN_SAT) & (V > 70) & (dh > WATER_MARGIN) & (deg >= CAP_LO) & (deg <= CAP_HI)
    # A white cap is in play (seen in the contact sheet) and has almost no
    # saturation, so a hue-based test throws it away. It is still perfectly
    # distinguishable: bright and colorless, which nothing else in the frame is
    # -- the water is saturated turquoise and skin is saturated warm.
    white = (S < 55) & (V > 165)
    if white.sum() > max(120, colored.sum() * 1.4):
        return -1.0, int(white.sum())          # -1 stands for white
    if colored.sum() < 40:
        return None
    return float(np.median(deg[colored])), int(colored.sum())


def main():
    args = parse_args()
    import cv2
    import numpy as np
    from ultralytics import YOLO

    import makemiss as M

    src = ROOT / "labels/allshots.json"
    if not src.exists():
        src = ROOT / "labels/judged.json"
    judged = [j for j in json.loads(src.read_text(encoding="utf-8"))
              if j["video"] == args.video and j.get("label") != "notshot"]
    judged.sort(key=lambda j: j["t"])
    shots = judged[args.skip:args.skip + args.shots]
    # The label export doesn't carry the clock string, so derive it here rather
    # than re-exporting for one cosmetic field.
    for j in shots:
        j.setdefault("clock", f"{int(j['t']) // 60}:{int(j['t']) % 60:02d}")
    print(f"{len(shots)} shots from {args.video}\n")

    rig = hoops.rig_for(args.video)
    cap = cv2.VideoCapture(str(ROOT / "footage" / args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO("yolo11s.pt")

    stage = Counter()
    rows = []
    for j in shots:
        stage["shots"] += 1
        rim = rig.rims[j["hoop"]]
        rimx, rimy = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2

        # The WHOLE clip window, not just the run-up.
        #
        # review, watching the magnifier glitch: "if the person is moving around
        # during the clip, we'd have to track them as they move and it's not
        # properly done so when they're out of the frame slice they just glitch."
        # Exactly right. A single box cannot follow someone who moves, and in a
        # pool everyone moves. So the shooter is TRACKED across the clip and the
        # output is a path, not a point.
        t1_ = float(j["tEnd"]) if float(j.get("tEnd") or 0) > float(j["t"]) else float(j["t"]) + 1.0
        f_lo = max(0, int((float(j["t"]) - 2.6) * fps))
        f_hi = int((t1_ + 1.4) * fps)
        clip_t0 = f_lo / fps
        frames = list(range(f_lo, f_hi + 1, 3))

        per_frame = []          # [(frame, [people boxes], [balls])]
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                continue
            r = model.predict(fr, conf=0.20, verbose=False, classes=[SPORTS_BALL, PERSON],
                              imgsz=1280)[0]
            ppl, balls_f = [], []
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if not (POOL[0] <= cx <= POOL[2] and POOL[1] <= cy <= POOL[3]):
                    continue
                if int(b.cls[0]) == PERSON:
                    ppl.append({"box": (x1, y1, x2, y2), "x": cx, "y": cy})
                else:
                    balls_f.append({"x": cx, "y": cy, "conf": float(b.conf[0])})
            per_frame.append((f, ppl, balls_f, fr))

        balls = [dict(b, f=f) for f, _, bs, _ in per_frame for b in bs]
        if not balls:
            rows.append({"clock": j["clock"], "n": j["n"], "video": args.video,
                         "stage": "no ball in open water"})
            continue
        stage["ball seen in flight"] += 1

        # Track people across the clip: nearest neighbour, which is enough when
        # swimmers are far apart relative to how far they move between samples.
        tracks = []
        for f, ppl, _, _ in per_frame:
            for pp in ppl:
                best, bd = None, 1e9
                for tr in tracks:
                    last = tr[-1]
                    if last["f"] >= f:
                        continue
                    gap = (f - last["f"]) / 3
                    if gap > 3:
                        continue
                    d = ((pp["x"] - last["x"]) ** 2 + (pp["y"] - last["y"]) ** 2) ** 0.5
                    if d < 220 * gap and d < bd:
                        best, bd = tr, d
                (best if best is not None else tracks.append([]) or tracks[-1]).append(dict(pp, f=f))

        # The shooter: the track holding the ball latest, before it leaves for
        # the hoop. Same rule as before -- a pass would otherwise credit whoever
        # the ball happened to be flying past.
        far = [b for b in balls if ((b["x"] - rimx) ** 2 + (b["y"] - rimy) ** 2) ** 0.5 > 400]
        if not far:
            rows.append({"clock": j["clock"], "n": j["n"], "video": args.video,
                         "stage": "ball never far from the rim"})
            continue
        # Contacts BEFORE the ball starts falling toward the hoop. Nothing that
        # happens after it can be the release.
        #
        # This rule used to be safe by accident: the search window ended at the
        # descent, so "the latest contact" could not be a later one. Widening
        # the window to track people across the whole clip removed that
        # accident, and 73 of 91 releases on IMG_2482 came back AFTER the shot
        # had begun falling -- crediting whoever caught the rebound. The tell
        # was a flight time of minus 2.27 seconds, which is not a thing.
        f_release_max = float(j["t"]) * fps
        far = [b for b in far if b["f"] <= f_release_max]
        if not far:
            rows.append({"clock": j["clock"], "n": j["n"], "video": args.video,
                         "stage": "ball never in anyone's hands before the shot"})
            continue
        best_tr, best_f, best_d = None, -1, None
        for b in far:
            for tr in tracks:
                near = [q for q in tr if abs(q["f"] - b["f"]) <= 3]
                if not near:
                    continue
                q = min(near, key=lambda z: ((z["x"] - b["x"]) ** 2 + (z["y"] - b["y"]) ** 2) ** 0.5)
                d = ((q["x"] - b["x"]) ** 2 + (q["y"] - b["y"]) ** 2) ** 0.5
                if d < 260 and b["f"] > best_f:
                    best_tr, best_f, best_d = tr, b["f"], d
        if best_tr is None:
            rows.append({"clock": j["clock"], "n": j["n"], "video": args.video,
                         "stage": "nobody was holding the ball"})
            continue
        stage["release point found"] += 1
        stage["person found near release"] += 1

        # The path, as shares of the frame, timed from the clip's own start.
        path = []
        for q in best_tr:
            x1, y1, x2, y2 = q["box"]
            pad = (y2 - y1) * 0.18
            x1, y1 = max(0.0, x1 - pad), max(0.0, y1 - pad)
            x2, y2 = min(3840.0, x2 + pad), min(2160.0, y2 + pad)
            path.append([round(q["f"] / fps - clip_t0, 2),
                         round(x1 / 3840 * 100, 2), round(y1 / 2160 * 100, 2),
                         round((x2 - x1) / 3840 * 100, 2), round((y2 - y1) / 2160 * 100, 2)])

        # Cap color, voted across the frames the shooter appears in, so one bad
        # look cannot decide who someone is.
        votes = []
        for f, _, _, fr in per_frame:
            q = next((z for z in best_tr if z["f"] == f), None)
            if not q:
                continue
            hv = cap_hue(fr, q["box"], np, cv2)
            if hv:
                votes.append(hv[0])
        hue = None
        if votes:
            band = {}
            for v in votes:
                band.setdefault("white" if v < 0 else int(v // 30) * 30, []).append(v)
            top = max(band.items(), key=lambda kv: len(kv[1]))
            hue = -1.0 if top[0] == "white" else float(np.median(top[1]))
            stage["cap color read"] += 1

        rows.append({"clock": j["clock"], "n": j["n"], "video": args.video,
                     "stage": "attributed",
                     "releaseT": round(best_f / fps, 2),
                     "dist": round(best_d),
                     "shooterPath": path,
                     "hue": (round(hue, 1) if hue is not None else None)})
    cap.release()

    # Written BEFORE anything is printed.
    #
    # An hour of compute was lost tonight because the summary crashed on a
    # `hue: None` row and the results had not been saved yet. The expensive
    # thing is the detection; a report is cheap and can be regenerated. Never
    # let a formatting bug destroy a run again.
    out_path = ROOT / f"out/attribute_{args.video.replace('.MOV','')}.json"
    out_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {out_path.name} ({len(rows)} shots)\n")

    print(f"{'stage':<34} {'shots':>6}")
    for k in ("shots", "ball seen in flight", "release point found",
              "person found near release", "cap color read"):
        print(f"{k:<34} {stage[k]:>6}")

    print("\nper shot:")
    for r in rows:
        extra = ""
        if r.get("hue") is not None:
            extra = (f"  hue {r['hue']}deg, {len(r.get('shooterPath') or [])} tracked positions, "
                     f"ball {r['dist']}px from them")
        elif "dist" in r:
            extra = f"  shooter {r['dist']}px from the ball"
        print(f"  {r['clock']:>6}  {r['stage']}{extra}")

    # `hue` is None whenever no cap could be read, and None // 30 raises. This
    # is the line that killed the IMG_2482 run.
    hues = [r["hue"] for r in rows if r.get("hue") is not None]
    if len(hues) >= 3:
        bins = Counter(int(h // 30) * 30 for h in hues)
        print(f"\ncap hues found: {sorted(bins.items())}")
        print(f"distinct 30-degree bands: {len(bins)} across {len(hues)} attributed shots")



if __name__ == "__main__":
    main()
