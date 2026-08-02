"""Who took the shot, built on four observations of review's.

Every early attempt asked a proximity question and got 30-40%: nearest at the
descent, longest possession, the arc's origin, walk-it-back, nearest wrist. They
all converged on the same number because they were all the same question.

review changed the question four times, and each change is encoded here.

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

4. "Since many hands up, you would go with the person whose hands are closest to
   ball on release." And, on the same frame: "the person who shot it isn't even
   boxed as a person in this, but his hands are right next to the ball."
   Two separate fixes. The choice among hands-up candidates is now distance at
   release rather than latest release. And pose runs on a native-resolution
   window around the ball, because a shooter the detector never boxed cannot be
   chosen by any rule at all.

    python src/shooter.py --video IMG_2482.MOV --shots 14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc

ROOT = Path(__file__).resolve().parent.parent
POOL = (1150, 250, 3500, 1750)
BALL, PERSON = 32, 0
NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LW, RW = 0, 1, 2, 3, 4, 5, 6, 9, 10

LOOKBACK_S = 1.8       # how far before the descent to search
TOUCH = 0.9            # "at the hands", in shoulder-widths
OPENED = 0.6           # how much the gap must grow to count as released
PCW, PCH = 1500, 1100  # native-res pose window around the ball


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


def scan(cap, fps, t_descent, det, pos, keep=False):
    """Walk the window before a descent, measuring every person against the ball.

    Returns (ball_track, per_person). `ball_track` is [{t,x,y}] and is what the
    parabola is fitted to; `per_person` holds each person's (t, gap, lift) series
    with their box and keypoints recorded per frame.

    This is deliberately the ONE place the scan happens. The diagnostic pictures
    review reviews are drawn from this same function, so the image cannot show one
    thing while the pipeline decides on another. An earlier round of his feedback
    was wasted on exactly that mismatch.
    """
    f_lo = max(0, int((float(t_descent) - LOOKBACK_S) * fps))
    f_hi = int(float(t_descent) * fps)

    # ONE seek, then read forward. The first version seeked per frame, and on a
    # 15GB 4K60 file a single CAP_PROP_POS_FRAMES seek ran past ten minutes -- at
    # a hundred frames a shot that is not a slow step, it is one that never
    # returns. The frames wanted are consecutive anyway, so this costs nothing.
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)

    ball_track, per_person = [], {}
    for f in range(f_lo, f_hi + 1):
        ok, fr = cap.read()
        if not ok:
            break
        t = f / fps

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
        ball_track.append({"t": t, "x": ball[0], "y": ball[1]})

        # Pose on a native-resolution window around the BALL, not the whole
        # frame. review found the failure this fixes: "the person who shot it isn't
        # even boxed as a person in this, but his hands are right next to the
        # ball." No box means no candidate, so no rule can pick the one person
        # who actually shot.
        #
        # The cause is scale. A 3840-wide frame at imgsz 1280 is downscaled to
        # 0.33, and a half-submerged swimmer at the far end arrives maybe 60px
        # tall with their arms overlapping a neighbour's. A 1500x1100 window runs
        # at 0.85 instead: the same person two and a half times bigger, for the
        # same inference cost.
        #
        # Cropping around the ball cannot hide the shooter, because the ball is
        # in their hands at the moment being measured. Anyone this window leaves
        # out is by construction further from the ball than everyone it keeps,
        # and the choice below ranks by exactly that distance.
        px1 = int(max(0, min(fr.shape[1] - PCW, ball[0] - PCW / 2)))
        py1 = int(max(0, min(fr.shape[0] - PCH, ball[1] - PCH / 2)))
        rp = pos.predict(fr[py1:py1 + PCH, px1:px1 + PCW],
                         conf=0.20, verbose=False, imgsz=1280)[0]
        if rp.keypoints is None:
            continue

        for pi in range(len(rp.boxes)):
            bx1, by1, bx2, by2 = rp.boxes.xyxy[pi].tolist()
            bx1, by1, bx2, by2 = bx1 + px1, by1 + py1, bx2 + px1, by2 + py1
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if not (POOL[0] <= cx <= POOL[2] and POOL[1] <= cy <= POOL[3]):
                continue
            kp = [[k[0] + px1, k[1] + py1, k[2]] for k in rp.keypoints.data[pi].tolist()]
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
                per_person[key] = {"series": [], "last": (cx, cy), "at": {}}
            v = per_person[key]
            v["last"] = (cx, cy)
            v["series"].append((t, gap, lift))
            v["at"][round(t, 3)] = {"box": (bx1, by1, bx2, by2), "kp": kp,
                                    "gap": gap, "lift": lift, "ball": ball}
    return ball_track, per_person


def candidates(per_person):
    """Everyone who released the ball with their hands up, best first.

    Hands up is a REQUIREMENT and can only remove people who did not shoot.
    Among those who remain, the shooter is whoever's hands were NEAREST THE BALL
    at their release: the rule, replacing "the latest release", which was only
    ever a stand-in for it.

    The two rules agree whenever one person has their hands up, and diverge
    exactly where it matters. On a hands-up round several defenders release
    nothing but still show a gap opening as the ball flies past them, and the
    last of those in time is whoever the ball passed last -- systematically NOT
    the shooter, since the ball travels away from them. Distance at release
    carries no such bias: the ball starts at the shooter's hands and nowhere
    else. gap is in shoulder-widths, so a near player and a far one compare
    honestly without perspective weighing in.
    """
    out = []
    for key, v in per_person.items():
        if len(v["series"]) < 4:
            continue
        v["series"].sort()
        rel = find_release(v["series"])
        if not rel:
            continue
        t, gap, lift, grew = rel
        if lift <= 0:
            continue            # hands were not up: cannot be the shooter
        snap = v["at"].get(round(t, 3), {})
        out.append({"person": key, "t": round(t, 2), "gap": round(gap, 2),
                    "lift": round(lift, 2), "grew": round(grew, 2),
                    "box": snap.get("box"), "kp": snap.get("kp"),
                    "n_seen": len(v["series"])})
    out.sort(key=lambda c: (c["gap"], -c["t"]))
    return out


def fit_arc(ball_track):
    """The ball's parabola over this window, or None if it was carried (a dunk)."""
    if len(ball_track) < 5:
        return None
    return arc.release(ball_track)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482.MOV")
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--skip", type=int, default=0)
    return p.parse_args()


def load_shots(video, limit=200, skip=0):
    src = ROOT / "labels/allshots.json"
    shots = [j for j in json.loads(src.read_text(encoding="utf-8"))
             if j["video"] == video and j.get("label") != "notshot"]
    shots.sort(key=lambda j: j["t"])
    shots = shots[skip:skip + limit]
    for j in shots:
        j.setdefault("clock", f"{int(j['t']) // 60}:{int(j['t']) % 60:02d}")
    return shots


def main():
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    shots = load_shots(args.video, args.shots, args.skip)
    cap = cv2.VideoCapture(str(ROOT / "footage" / args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = YOLO("yolo11s.pt")
    pos = YOLO("yolo11s-pose.pt")

    rows = []
    for j in shots:
        ball_track, per_person = scan(cap, fps, float(j["t"]), det, pos)
        cands = candidates(per_person)
        flight = fit_arc(ball_track)

        base = {"n": j["n"], "clock": j["clock"], "video": args.video,
                "ball_seen": len(ball_track),
                "arc": None if not flight else {
                    "t": round(flight["t"], 2), "x": round(flight["x"], 1),
                    "y": round(flight["y"], 1), "rms": round(flight["rms"], 1),
                    "sag": flight["sag"], "n": flight["n"]}}
        if not cands:
            rows.append({**base, "stage": "no release with hands up"})
            print(f"  #{j['n']} {j['clock']}: nobody released with hands up "
                  f"({len(per_person)} people seen, ball on {len(ball_track)} frames)")
            continue
        pick = cands[0]
        rows.append({**base, "stage": "attributed",
                     **{k: pick[k] for k in ("person", "t", "gap", "lift", "grew", "box", "n_seen")},
                     "others": len(cands) - 1,
                     "runner_up_gap": cands[1]["gap"] if len(cands) > 1 else None})
        print(f"  #{j['n']} {j['clock']}: release t={pick['t']}, gap {pick['gap']}, "
              f"hands {pick['lift']:+.2f}, {len(cands)-1} other candidate(s)"
              + ("" if not flight else f", arc rms {flight['rms']:.0f}px"))
    cap.release()

    out = ROOT / f"out/shooter_{args.video.replace('.MOV','')}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    got = sum(1 for r in rows if r["stage"] == "attributed")
    print(f"\n{got} of {len(rows)} shots attributed -> {out.name}")


if __name__ == "__main__":
    main()
