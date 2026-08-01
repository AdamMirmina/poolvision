"""Who took the shot, built on three observations of review's.

Every earlier attempt asked a proximity question and got 30-40%: nearest at the
descent, longest possession, the arc's origin, walk-it-back, nearest wrist. They
all converged on the same number because they were all the same question.

review changed the question three times, and each change is encoded here.

1. "The wrists are higher than the cap at the point of release."
   Hands up is a REQUIREMENT, not a bonus. The reasoning is what makes this
   safe: a shooter always has their hands up, a defender only sometimes does, so
   the test is a necessary condition. It can never exclude the true shooter, it
   only prunes people who certainly did not shoot. As a weighted bonus it lost
   to a bystander standing nearer the ball; as a filter that bystander is not
   even considered.

2. "The release should be when there gets a tiny bit of distance between wrists
   and ball."
   The release is a SHAPE, not a distance: the ball rests at the hands, then a
   gap opens and keeps opening. A bystander the ball flies past shows the
   opposite, far then near then far. Minimum distance cannot tell those apart,
   which is the failure he spotted -- the frame chosen was well after the ball
   had left the shooter's hands, with the ball merely passing someone else.

3. "That frame is significantly after the ball left white's hands."
   Sampling every third frame gave the shooter one or two sightings around the
   release, which is not a shape. Every frame is a 3x denser look: measured at
   42 of 55 frames on one clip against 14 before.

    python src/shooter.py --video IMG_2482.MOV --shots 14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops

ROOT = Path(__file__).resolve().parent.parent
POOL = (1150, 250, 3500, 1750)
BALL, PERSON = 32, 0
NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LW, RW = 0, 1, 2, 3, 4, 5, 6, 9, 10

LOOKBACK_S = 1.8      # how far before the descent to search
TOUCH = 0.9           # "at the hands", in shoulder-widths
OPENED = 0.6          # how much the gap must grow to count as released


def head_y(kp):
    pts = [kp[i] for i in (NOSE, LEYE, REYE, LEAR, REAR) if kp[i][2] > 0.25]
    return sum(p[1] for p in pts) / len(pts) if pts else None


def shoulder_w(kp):
    sh = [kp[i] for i in (LSH, RSH) if kp[i][2] > 0.25]
    if len(sh) < 2:
        return None
    return max(12.0, ((sh[0][0] - sh[1][0]) ** 2 + (sh[0][1] - sh[1][1]) ** 2) ** 0.5)


def _d(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def find_release(series):
    """The moment a gap opens between these wrists and the ball, and keeps opening.

    `series` is [(t, gap_in_shoulder_widths, lift)] in time order for ONE person.

    Returns the last such moment, because a shot is the last time that person
    let go of the ball before it went to the hoop. Requires the gap to keep
    growing rather than merely being large once, which is what separates a
    release from the ball bouncing past.
    """
    best = None
    for k in range(len(series) - 2):
        t, gap, lift = series[k]
        if gap > TOUCH:
            continue
        after = [g for _, g, _ in series[k + 1:k + 6]]
        if len(after) < 2:
            continue
        monotonic = all(after[m + 1] >= after[m] - 0.15 for m in range(len(after) - 1))
        if monotonic and after[-1] > gap + OPENED and (best is None or t > best[0]):
            best = (t, gap, lift, after[-1] - gap)
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482.MOV")
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--skip", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    src = ROOT / "labels/allshots.json"
    shots = [j for j in json.loads(src.read_text(encoding="utf-8"))
             if j["video"] == args.video and j.get("label") != "notshot"]
    shots.sort(key=lambda j: j["t"])
    shots = shots[args.skip:args.skip + args.shots]
    for j in shots:
        j.setdefault("clock", f"{int(j['t']) // 60}:{int(j['t']) % 60:02d}")

    cap = cv2.VideoCapture(str(ROOT / "footage" / args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = YOLO("yolo11s.pt")
    pos = YOLO("yolo11s-pose.pt")

    rows = []
    for j in shots:
        f_lo = max(0, int((float(j["t"]) - LOOKBACK_S) * fps))
        f_hi = int(float(j["t"]) * fps)

        # Every frame, not every third. The release is a shape and a shape needs
        # consecutive samples.
        per_person = {}
        for f in range(f_lo, f_hi + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                continue
            rb = det.predict(fr, conf=0.15, verbose=False, classes=[BALL], imgsz=1280)[0]
            ball = None
            for b in rb.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if POOL[0] <= cx <= POOL[2] and POOL[1] <= cy <= POOL[3]:
                    ball = (cx, cy)
                    break
            if ball is None:
                continue
            rp = pos.predict(fr, conf=0.25, verbose=False, imgsz=1280)[0]
            if rp.keypoints is None:
                continue
            for pi in range(len(rp.boxes)):
                bx1, by1, bx2, by2 = rp.boxes.xyxy[pi].tolist()
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                if not (POOL[0] <= cx <= POOL[2] and POOL[1] <= cy <= POOL[3]):
                    continue
                kp = rp.keypoints.data[pi].tolist()
                ws = [kp[k] for k in (LW, RW) if kp[k][2] > 0.3]
                hy, sw = head_y(kp), shoulder_w(kp)
                if not ws or hy is None or sw is None:
                    continue
                lift = max((hy - w[1]) for w in ws) / sw
                gap = min(_d((w[0], w[1]), ball) for w in ws) / sw
                # Track people by position, since pose gives no identity.
                key = None
                for k2, v in per_person.items():
                    lx, ly = v["last"]
                    if abs(cx - lx) < sw * 1.8 and abs(cy - ly) < sw * 2.2:
                        key = k2
                        break
                if key is None:
                    key = len(per_person)
                    per_person[key] = {"series": [], "last": (cx, cy), "box": None, "kp": None}
                per_person[key]["last"] = (cx, cy)
                per_person[key]["series"].append((f / fps, gap, lift))
                per_person[key]["box"] = (bx1, by1, bx2, by2)
                per_person[key]["kp"] = kp

        # Hands up is a REQUIREMENT. It can only remove people who did not shoot.
        cands = []
        for key, v in per_person.items():
            if len(v["series"]) < 4:
                continue
            v["series"].sort()
            rel = find_release(v["series"])
            if not rel:
                continue
            t, gap, lift, grew = rel
            if lift <= 0:
                continue            # their hands were not up: cannot be the shooter
            cands.append({"person": key, "t": round(t, 2), "gap": round(gap, 2),
                          "lift": round(lift, 2), "grew": round(grew, 2),
                          "box": v["box"], "kp": v["kp"], "n_seen": len(v["series"])})

        if not cands:
            rows.append({"n": j["n"], "clock": j["clock"], "video": args.video,
                         "stage": "no release with hands up"})
            continue
        # The last release before the ball fell: a shot is the last time the
        # ball left someone's hands on its way to the hoop.
        cands.sort(key=lambda c: -c["t"])
        pick = cands[0]
        rows.append({"n": j["n"], "clock": j["clock"], "video": args.video,
                     "stage": "attributed", **{k: pick[k] for k in
                     ("person", "t", "gap", "lift", "grew", "box", "n_seen")},
                     "others": len(cands) - 1})
        print(f"  #{j['n']} {j['clock']}: release t={pick['t']}, gap {pick['gap']}, "
              f"hands {pick['lift']:+.2f}, {len(cands)-1} other candidate(s)")
    cap.release()

    out = ROOT / f"out/shooter_{args.video.replace('.MOV','')}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    got = sum(1 for r in rows if r["stage"] == "attributed")
    print(f"\n{got} of {len(rows)} shots attributed -> {out.name}")


if __name__ == "__main__":
    main()
