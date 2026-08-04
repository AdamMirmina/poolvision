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
import capfind
import facing
import hoops

ROOT = Path(__file__).resolve().parent.parent
BALL, PERSON = 32, 0
NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LW, RW = 0, 1, 2, 3, 4, 5, 6, 9, 10

LOOKBACK_S = 1.8       # how far before the descent to search
AFTER_S = 1.2          # and how far past it, so the ball can finish landing
TOUCH = 0.9            # "at the hands", in shoulder-widths
OPENED = 0.6           # how much the gap must grow to count as released
PCW, PCH = 1500, 1100  # native-res pose window around the ball
# Five of the ten wrong attributions review commented on are the same failure, and
# it is not a ranking mistake -- the shooter was never boxed as a person, so no
# rule could pick him. The words: "shooter is not recognized as a person judging
# by the boxes", "partially submerged but cap and body and arm still visible",
# "green was behind white and not recognized as a person despite cap visible",
# "shooter was in deepend and not recognized as a person".
#
# A half-submerged swimmer at the far end is a hard detection, and 0.20 was
# discarding them. Lowered here only -- this window is already cropped tight
# around the ball, so the extra candidates are people genuinely near the shot,
# and the rules below still have to choose between them.
#
# The real fix is the cap: review twice notes it is visible when the body is not,
# which makes it a better person-detector than pose at this distance. That needs
# a cap-blob finder, which does not exist yet -- caps.py only classifies a color
# once you have one.
POSE_CONF = 0.10
# Lower again inside a cap crop: the head's location is already known, so a weak
# detection there is far more likely to be the real body than a false one.
CAP_POSE_CONF = 0.05
# How near the ball a cap must be, in cap-widths, to be worth looking at.
# Generous: a shooter's head at release is within a body-length of the ball.
CAP_NEAR_BALL = 6.0


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


FINE = ROOT / "out/balltrain/ball/weights/best.pt"
FINE_CROP = 960
# The fine-tune never exceeds 0.25 confidence -- it is undertrained and its
# scores are compressed, which is why its own precision reads badly. Measured, it
# puts a box on the ball in 60 of 60 frames the stock detector misses, so the
# useful threshold is far below any default.
FINE_CONF = 0.05
# How far from where the ball should be a box is still believable.
FINE_TOL = 55.0
_fine = [None]


def fine_model():
    """Loaded once, and only if the weights exist, so the pipeline still runs
    unchanged on a machine that has never trained one."""
    if _fine[0] is None:
        if not FINE.exists():
            _fine[0] = False
        else:
            from ultralytics import YOLO
            _fine[0] = YOLO(str(FINE))
    return _fine[0] or None


def refine_ball(fr, track):
    """Look again, in one place, at native resolution, for a ball the stock
    detector just missed.

    Every ceiling on this project traces to the same failure: the ball is lost
    exactly at the rim. Through-the-hoop is observable on 38% of makes, the
    overlap veto lands below chance, arcs stop before the ball lands. One cause.

    The fine-tune was trained on 960px native crops, so it is poor on a whole
    frame downscaled to 1280 where the ball is half the size it has ever seen.
    It is a second look, not a replacement: stock runs first, and this only fires
    where stock came up empty.

    Where to look comes from the ball's own motion. Two sightings give a velocity,
    and over one frame a thrown ball does not deviate far from it, so the crop
    lands on the ball even mid-flight. With one sighting, the last position is
    close enough at 60fps. With none, there is nothing to aim at and we pass.
    """
    m = fine_model()
    if m is None or not track:
        return None
    if len(track) >= 2:
        a, b = track[-2], track[-1]
        dt = b["t"] - a["t"]
        vx = (b["x"] - a["x"]) / dt if dt else 0.0
        vy = (b["y"] - a["y"]) / dt if dt else 0.0
        step = dt or 1 / 60
        ex, ey = b["x"] + vx * step, b["y"] + vy * step
    else:
        ex, ey = track[-1]["x"], track[-1]["y"]

    H, W = fr.shape[:2]
    cx = int(min(W - FINE_CROP, max(0, ex - FINE_CROP / 2)))
    cy = int(min(H - FINE_CROP, max(0, ey - FINE_CROP / 2)))
    crop = fr[cy:cy + FINE_CROP, cx:cx + FINE_CROP]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    r = m.predict(crop, conf=FINE_CONF, imgsz=640, verbose=False)[0]
    best, bd = None, 1e9
    for b in sorted(r.boxes, key=lambda b: -float(b.conf.item()))[:3]:
        x, y = b.xywh[0].tolist()[:2]
        gx, gy = x + cx, y + cy
        d = ((gx - ex) ** 2 + (gy - ey) ** 2) ** 0.5
        if d < bd:
            best, bd = (gx, gy), d
    # A box anywhere in a 960px crop is not evidence; one where the ball was
    # heading is. Without this the refine pass would happily label a head.
    return best if best and bd <= FINE_TOL else None


def scan(cap, fps, t_descent, det, pos, pool=(1150, 250, 3500, 1750)):
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
    # Keep watching AFTER the descent. The window used to end exactly at it,
    # which meant the flight could never contain the ball landing -- no detector
    # is able to fix that, and measuring one against it showed exactly what you
    # would expect: the arcs got denser and not one of them got longer.
    #
    # It is the cause of both of the complaints here. "Many of the parabolas
    # are unfinished, they don't go all the way until the ball lands", and the
    # make that "bounced upwards first and rolled around a bit before going in --
    # we have to keep watching after it bounces off all the way until it's no
    # longer above the rim." A rim bounce and roll takes about a second.
    f_hi = int((float(t_descent) + AFTER_S) * fps)

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
            if pool[0] <= cx <= pool[2] and pool[1] <= cy <= pool[3]:
                ball = (cx, cy)
                break
        if ball is None:
            # Second look before giving up on the frame. A dropped frame is not
            # just a missing dot on the arc: pose never runs here either, so the
            # shooter can go uncandidated in exactly the frames that matter most.
            ball = refine_ball(fr, ball_track)
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
                         conf=POSE_CONF, verbose=False, imgsz=1280)[0]
        if rp.keypoints is None:
            continue

        # A SECOND POSE PASS, on a tight crop around any cap nobody was boxed on.
        #
        # Five of ten wrong attributions are the shooter never being detected, and
        # capfind finds those players when pose cannot -- but a cap has no
        # keypoints, and both deciding rules (hands up, facing) read keypoints. So
        # instead of admitting an unrankable candidate, look again where the cap
        # says a head is. Same move that took the ball detector from useless to
        # 60-for-60: the model was never the problem, the search area was.
        extra = []
        boxes_now = [tuple(b) for b in rp.boxes.xyxy.tolist()]
        boxes_now = [(a + px1, b + py1, c + px1, d + py1) for a, b, c, d in boxes_now]
        # Named `cp`, not `cap`: `cap` is the VideoCapture this function reads
        # from, and shadowing it turned the next frame read into an AttributeError.
        for cp in capfind.find(fr, pool):
            if any(bx1 <= cp["x"] <= bx2 and by1 <= cp["y"] <= by2
                   for bx1, by1, bx2, by2 in boxes_now):
                continue
            # Only a cap NEAR THE BALL is worth a second pose pass.
            #
            # This is the own observation used as an optimisation: "if they're
            # submerged the ball would be near their head anyway." A cap across
            # the pool cannot be the shooter of this flight, so running pose on it
            # is pure cost -- and it was most of the cost. Seven caps over ~54
            # frames is why a shot went from seconds to minutes.
            if _d((cp["x"], cp["y"]), ball) > CAP_NEAR_BALL * max(28.0, cp["w"]):
                continue
            cw = max(28.0, cp["w"])
            # A body hangs BELOW its cap, so the crop starts just above the head.
            hw, hh = int(cw * 9), int(cw * 11)
            qx = int(max(0, min(fr.shape[1] - hw, cp["x"] - hw / 2)))
            qy = int(max(0, min(fr.shape[0] - hh, cp["y"] - cw * 2)))
            sub = fr[qy:qy + hh, qx:qx + hw]
            if sub.shape[0] < 40 or sub.shape[1] < 40:
                continue
            r2 = pos.predict(sub, conf=CAP_POSE_CONF, verbose=False, imgsz=960)[0]
            if r2.keypoints is None or not len(r2.boxes):
                continue
            # The person whose box actually contains the cap.
            for qi in range(len(r2.boxes)):
                ax1, ay1, ax2, ay2 = r2.boxes.xyxy[qi].tolist()
                if not (ax1 + qx <= cp["x"] <= ax2 + qx
                        and ay1 + qy <= cp["y"] <= ay2 + qy):
                    continue
                extra.append(((ax1 + qx, ay1 + qy, ax2 + qx, ay2 + qy),
                              [[k[0] + qx, k[1] + qy, k[2]]
                               for k in r2.keypoints.data[qi].tolist()]))
                break

        for pi in range(len(rp.boxes) + len(extra)):
            if pi < len(rp.boxes):
                bx1, by1, bx2, by2 = rp.boxes.xyxy[pi].tolist()
                bx1, by1, bx2, by2 = bx1 + px1, by1 + py1, bx2 + px1, by2 + py1
                kp_src = [[k[0] + px1, k[1] + py1, k[2]]
                          for k in rp.keypoints.data[pi].tolist()]
            else:
                (bx1, by1, bx2, by2), kp_src = extra[pi - len(rp.boxes)]
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if not (pool[0] <= cx <= pool[2] and pool[1] <= cy <= pool[3]):
                continue
            kp = kp_src
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


SCANS = ROOT / "out/scans"


def scan_cached(video, n, t_descent, cap, fps, det, pos, pool=(1150, 250, 3500, 1750)):
    """scan(), but written to disk the first time and read back after.

    The scan is the entire cost: a couple of hundred model calls per shot, tens
    of minutes for a handful of shots on this laptop. Every gate below it --
    what counts as hands up, what counts as a release, how much the gap must
    grow -- is arithmetic over the numbers the scan produced, and tuning those
    by re-running the scan means an hour per idea. Cached, an idea costs a
    second, so the gates can be chosen by measuring against the notes rather
    than by argument.
    """
    SCANS.mkdir(parents=True, exist_ok=True)
    f = SCANS / f"{video.replace('.MOV','')}_{n}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        per = {int(k): {"series": [tuple(s) for s in v["series"]],
                        "last": tuple(v["last"]),
                        "at": {float(kk): vv for kk, vv in v["at"].items()}}
               for k, v in d["per_person"].items()}
        return d["ball_track"], per
    ball_track, per_person = scan(cap, fps, t_descent, det, pos, pool=pool)
    f.write_text(json.dumps({"ball_track": ball_track, "per_person": per_person},
                            default=list), encoding="utf-8")
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


MIN_TRAVEL_PX = 260   # a shot crosses the frame; a ball in the water does not


def fit_arc(ball_track, t_anchor=None):
    """The ball's parabola, fitted to one continuous flight rather than to noise.

    Fitting the raw per-frame stream produced confident nonsense. On shot 8 the
    ball sat at (1350,1348) barely moving for half a second, jumped, then sat
    still again, and the fit spanned the lot: 13 points, 6.8px RMS, and a
    "flight" that traveled under 200px in 1.17 seconds. A stationary cluster
    fits a parabola beautifully, which is exactly why low residual is no evidence
    on its own.

    Three things were missing, all of which already existed elsewhere in this
    repo and none of which had reached here:

      a ball at rest in the water or on the deck is furniture, and parked.py
      knows how to drop it. It has now bitten in three separate places.

      more than one ball is in the pool, so the per-frame stream hops between
      objects. tracks.py builds continuous tracks; a fit across a hop describes
      a path no ball took.

      curvature alone does not separate a flight from a hover. sag grows with
      the square of the span, so a nearly flat fit over a long window clears any
      fixed pixel threshold. A shot also has to GO somewhere.
    """
    if len(ball_track) < 5:
        return None
    from parked import drop_parked
    from tracks import build_tracks

    # tracks.py counts gaps in FRAMES, so the frame index has to be a real frame
    # index. The sampling rate is recoverable from the track itself: consecutive
    # sightings are one frame apart, so the smallest positive gap is 1/fps.
    dts = sorted(b["t"] - a["t"] for a, b in zip(ball_track, ball_track[1:])
                 if b["t"] - a["t"] > 1e-6)
    fps = 1.0 / dts[0] if dts else 30.0
    hits = [{"conf": b.get("conf", 0.5), **b, "frame": int(round(b["t"] * fps))}
            for b in ball_track]
    hits, _ = drop_parked(hits)
    # THE track that ends at the hoop, not the best-looking track anywhere in the
    # window. Choosing by point count picked whichever ball had the longest clean
    # flight, and with more than one ball in the pool that is regularly a
    # different shot entirely: review saw a parabola drawn across open water far
    # from the ball in frame, and a release frame from before the shooter had
    # even collected the rebound. The window ends at the descent this clip was
    # cut for, so the right track is the one still in the air at the end of it.
    best = None
    # Anchor on the DESCENT, not on the end of the window. They used to be the
    # same instant, so max(t) was a fine stand-in. Now that the scan keeps
    # watching past the descent, they are not: the shot's own ball reaches the
    # rim and stops moving, while some other ball in the pool carries on and
    # ends later. Ranking by "closest to the end of the window" would start
    # handing the arc to whichever ball happened to still be traveling, which
    # is the same wrong-ball failure review already caught once.
    t_end = float(t_anchor) if t_anchor is not None else max(b["t"] for b in ball_track)
    for pts in build_tracks(hits):
        got = arc.release(pts)
        if not got:
            continue
        travel = ((pts[-1]["x"] - pts[0]["x"]) ** 2 + (pts[-1]["y"] - pts[0]["y"]) ** 2) ** 0.5
        if travel < MIN_TRAVEL_PX:
            continue
        lateness = t_end - pts[-1]["t"]
        if best is None or lateness < best[1] - 0.15 or (
                abs(lateness - best[1]) <= 0.15 and got["n"] > best[0]["n"]):
            best = (got, lateness, travel)
    if best:
        got, _lateness, travel = best
        got["travel"] = round(travel, 1)
        return got
    return None


REACH_PX = 110      # how close the arc's origin must come to count as "their hands"
NEVER_HELD = 4.0    # if the ball never came this close, in shoulder-widths, they did not shoot
ADMIT_PX = 260      # wider admission, so facing has something to arbitrate
FACE_TOWARD = 0.20  # decisively facing the hoop
FACE_AWAY = -0.20   # decisively turned away from it
NEAR_S = 0.25       # how stale a person's last sighting may be when matched


def _people_at(per_person, t, hands_up_only=False):
    """[(id, box)] for everyone seen within NEAR_S of t.

    `hands_up_only` keeps the first rule in force. It was silently dropped
    when attribution moved to the ball's flight: the origin was matched against
    whoever was nearest it, hands or no hands, and he caught the result straight
    away -- "in LOTS of the images you're showing to me, it's not iding the right
    person even when they're the only one with hands up."

    The reasoning is what makes the filter safe, and it did not stop being true
    when the evidence changed. A shooter always has their hands up; a bystander
    the flight happens to start near often does not. The test can never exclude
    the real shooter, only people who certainly did not shoot.
    """
    out = []
    for key, v in per_person.items():
        snaps = [(abs(float(tt) - t), v["at"][tt]) for tt in v["at"]]
        if not snaps:
            continue
        dt, s = min(snaps, key=lambda z: z[0])
        if dt > NEAR_S or not s.get("box"):
            continue
        if hands_up_only and not (s.get("lift") or 0) > 0:
            continue
        # Square on AND wrists down is an elimination, not a low score.
        #
        # review on two separate wrong picks: "the person selected is square and
        # wrists down", and "should have gotten yellow because of wrists being
        # above, and facing hooop". Both times the right shooter WAS in frame and
        # was out-ranked by someone who cannot have taken the shot. The original
        # framing of hands-up applies to facing too -- a shooter always turns
        # toward the hoop and raises the ball, so someone doing neither is not a
        # weaker candidate, they are not a candidate.
        kp = s.get("kp")
        if kp:
            side, strength = facing.side(kp)
            if side == 0 and not (s.get("lift") or 0) > 0:
                continue
        # The ball has to have come near them at some point in the window.
        #
        #
        # The person chosen there had the ball come no closer than 5 shoulder-
        # widths all window, while three others got inside one. They were picked
        # only because they happened to stand nearest where the arc began, and
        # standing near where a flight started is not the same as throwing it.
        #
        # Filtered here rather than vetoed after the choice, so the next-best
        # candidate still gets considered instead of the shot going unanswered.
        near = min((g for _t2, g, _l in v["series"]), default=99)
        if near > NEVER_HELD:
            continue
        out.append((key, s["box"]))
    return out


def attribute(ball_track, per_person, rig=None, hoop=None, trace=None, t_descent=None):
    """Who shot it, by the arc first and the release shape second.

    The release rule answers 5 of 17 shots on IMG_2482 and the reason is not a
    threshold. On half the shots the closest anyone's wrists ever come to the
    ball is over two shoulder-widths, because THE DETECTOR NEVER SEES THE BALL
    WHILE SOMEONE IS HOLDING IT -- two hands around it, against skin, at 4K
    downscaled to 1280. There is no gap-opening shape to find when the ball only
    exists once it is already in the air.

    Which leaves the flight as the only evidence, and review said so first: "follow
    the ball backwards through the air until it's clearly in someone's
    possession." The parabola is fitted to the samples that DO exist and run back
    to where it began. Measured on the same 17 shots that finds a person on 10,
    and on four of them the origin lands inside someone's box to the pixel.

    Falls back to the release shape when no arc fits, which is not a lesser
    answer but a different case: a ball carried to the rim has no parabola, and
    that absence is how a dunk is recognized.

    Returns (pick, how, extras) where `how` names the evidence used, so the
    diagnostic can say which one answered and review can judge them separately.
    """
    cands = candidates(per_person)
    flight = fit_arc(ball_track, t_anchor=t_descent)
    # The reasoning, recorded rather than discarded. The
    # second half is the point -- a wrong answer tells you a rule misfired, but
    # only the trail tells you a rule never ran.
    T = trace if trace is not None else []
    T.append(("ball", f"seen on {len(ball_track)} frames in the window"))
    if flight:
        T.append(("flight", f"parabola fits {flight['n']} sightings, travels "
                            f"{flight['travel']:.0f}px, residual {flight['rms']:.0f}px"))
    else:
        T.append(("flight", "no parabola fits: the ball was carried, so a dunk"))
    # Three different facts, reported separately, because collapsing them into
    # one number produced a line that was simply false. It counted `cands`, which
    # requires hands up AND a detected release shape AND enough sightings, then
    # described that count as "released with hands above their head" -- so a shot
    # where the release shape was never found read as "0 of 2 people released with
    # hands above their head" over a frame plainly showing the shooter with both
    # hands overhead.
    #
    # Hands up is the necessary condition and the first rule in the cascade.
    # Whether it held has to be readable on its own, separately from whether the
    # release was located, or a failure in one is read as a finding about the
    # other.
    n_up = sum(1 for v in per_person.values()
               if any(l is not None and l > 0 for _, _, l in v["series"]))
    n_rel = sum(1 for v in per_person.values()
                if len(v["series"]) >= 4 and find_release(v["series"]))
    T.append(("hands up", f"{n_up} of {len(per_person)} people had both hands "
                          f"above their head at some point"))
    T.append(("release", f"{n_rel} of {len(per_person)} showed a release: the ball "
                         f"at the hands, then a gap that keeps opening"))
    T.append(("candidates raw", f"{len(cands)} had hands up AND a release, "
                                f"which is what a candidate needs"))

    if flight:
        # Hands up first, everyone only if nobody had them up. Two passes rather
        # than one, because the filter is a necessary condition and not a
        # preference: if anyone in reach of the throw had their hands up, the
        # shooter is among them and a bystander closer to the origin is not a
        # candidate at all.
        hit = None
        for hands_up_only in (True, False):
            people = _people_at(per_person, flight["t"], hands_up_only)
            if hands_up_only:
                T.append(("candidates", f"{len(people)} people with hands up, near the ball, "
                                        f"and in frame at the release"))
            elif not people:
                T.append(("candidates", "nobody with hands up qualified; falling back to everyone"))

            # Facing decides WHO IS A CANDIDATE, not just how they rank.
            #
            # As a tiebreak it was inert: of 21 arc-attributed shots only one had
            # more than one candidate within reach of the origin, so there was
            # almost never a tie to break.
            #
            # So admission widens, and facing prunes the clearly-turned-away.
            # Deliberately soft, on his caveat that a "shooter could theoretically
            # shoot backwards": it only prunes when someone else is clearly facing
            # the hoop, it needs a decisive reading on both sides rather than a
            # marginal one, and it never empties the candidate set.
            near = []
            for pid, box in people:
                x1, y1, x2, y2 = box
                dx = max(x1 - flight["x"], 0, flight["x"] - x2)
                dy = max(y1 - flight["y"], 0, flight["y"] - y2)
                d = (dx * dx + dy * dy) ** 0.5
                if d <= ADMIT_PX:
                    near.append((pid, box, d))
                    T.append(("in reach", f"person {pid} is {d:.0f}px from where the throw began"))
            if len(near) > 1 and rig is not None and hoop:
                r = rig.rims[hoop]
                tgt = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
                scored = []
                for pid, box, d in near:
                    snaps = sorted(per_person[pid]["at"], key=lambda tt: abs(float(tt) - flight["t"]))
                    kp = per_person[pid]["at"][snaps[0]].get("kp") if snaps else None
                    # the left/right reading, which measures better than the
                    # front/back one it replaces (89% against 81%) and does not
                    # need a face the camera cannot see.
                    bc = facing.body_center(kp, box) if kp else None
                    fv = facing.side_toward(kp, tgt[0], bc[0] if bc else None) if kp else None
                    scored.append((pid, box, d, fv))
                toward = [z for z in scored if z[3] is not None and z[3] > FACE_TOWARD]
                # Turned away, OR square on. the rule has three states, not
                # two: "if they're opposite sides, they're facing forward or back
                # and probably aren't the shooter." Square was being treated as
                # neutral, which meant a straddler could still be chosen over
                # someone squarely pointed at the hoop -- he caught exactly that,
                # a green box reading "square" while a neighbour read 100%.
                out_ = {z[0] for z in scored
                        if z[3] is not None and (z[3] < FACE_AWAY or z[3] == 0.0)}
                for pid, _b, _d, fv in scored:
                    if fv is None:
                        T.append(("facing", f"person {pid}: not readable"))
                    elif fv == 0.0:
                        T.append(("facing", f"person {pid}: square on, so probably not shooting"))
                    else:
                        T.append(("facing", f"person {pid}: turned "
                                            f"{'toward' if fv > 0 else 'away from'} the hoop "
                                            f"({abs(fv)*100:.0f}%)"))
                if toward and out_:
                    kept = [z[:3] for z in scored if z[0] not in out_]
                    if kept:
                        T.append(("facing", f"dropped {len(out_)} turned away or square on"))
                        near = kept
            for pid, box, d in near:
                if d > REACH_PX and len(near) > 1:
                    continue          # the wider radius is only for the pruning above
                if d > ADMIT_PX:
                    continue
                # Facing the hoop breaks ties, where it can be trusted. the heuristic: two people with a hand up, both beside the ball, and
                # the one turned toward the basket the shot went to is the
                # shooter. It is independent of everything else here, which are
                # all about WHERE someone is rather than which way they face.
                face = None
                if rig is not None and hoop and facing.usable_for(rig, hoop):
                    snaps = sorted(per_person[pid]["at"], key=lambda tt: abs(float(tt) - flight["t"]))
                    snap = per_person[pid]["at"][snaps[0]] if snaps else {}
                    kp = snap.get("kp")
                    if kp:
                        r = rig.rims[hoop]
                        face = facing.toward(kp, ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2),
                                             facing.body_center(kp, snap.get("box")))
                        if face is not None:
                            face *= facing.orientation(rig, hoop)
                score = d - (0 if face is None else face * 60.0)
                if hit is None or score < hit[5]:
                    hit = (pid, d, flight["t"], flight["x"], flight["y"], score)
            if hit is None:
                walked = arc.origin_at_person(
                    flight["fit"], flight["t"],
                    lambda t, hu=hands_up_only: _people_at(per_person, t, hu),
                    back_s=1.2, reach_px=REACH_PX)
                if walked:
                    hit = (walked["person"], walked["dist"], walked["t"],
                           walked["x"], walked["y"], walked["dist"])
            if hit:
                break
        if hit:
            pid, d, t, x, y, _score = hit
            # Walk the arc BACK to the earliest moment it is still at this
            # person, and call THAT the release.
            #
            # flight["t"] is where the TRACK starts, not where the throw does. If
            # the detector only picks the ball up once it is clear of the hands,
            # the frame drawn is well after the ball has gone -- review, on one:
            # "this frame is long after the shooter released it and the shooter's
            # hands have gone down." The parabola describes the part before the
            # first sighting too, so it can be run back until it leaves them.
            box = None
            for pid2, b2 in _people_at(per_person, t):
                if pid2 == pid:
                    box = b2
                    break
            if box:
                # Stop at the HANDS, not at the edge of the box.
                #
                # Walking back until the curve leaves the box overshoots: the box
                # is a person-sized target and the curve keeps going for another
                # hundred pixels past the hands before it exits. review saw the
                # result -- "should that star be on the ball at the point of
                # release?" -- with the marker sitting beside the shooter rather
                # than on the ball he was holding. The release is where the ball
                # was, so the walk-back ends at the closest approach to the
                # wrists, and to the box only when no wrist was read.
                snaps = sorted(per_person[pid]["at"], key=lambda tt: abs(float(tt) - t))
                snap = per_person[pid]["at"][snaps[0]] if snaps else {}
                kp = snap.get("kp")
                hands = [(kp[k][0], kp[k][1]) for k in (LW, RW)
                         if kp and kp[k][2] > 0.3] if kp else []
                bx1, by1, bx2, by2 = box
                best_t, best_d = t, None
                tt = t
                for _ in range(int(0.9 * 60)):
                    tt -= 1 / 60.0
                    cx_, cy_ = arc.at(flight["fit"], tt)
                    if hands:
                        d2 = min(((cx_ - hx) ** 2 + (cy_ - hy) ** 2) ** 0.5 for hx, hy in hands)
                    else:
                        ddx = max(bx1 - cx_, 0, cx_ - bx2)
                        ddy = max(by1 - cy_, 0, cy_ - by2)
                        d2 = (ddx * ddx + ddy * ddy) ** 0.5
                    if best_d is None or d2 < best_d:
                        best_d, best_t = d2, tt
                    elif d2 > best_d + REACH_PX:
                        break          # past them and moving away; stop
                if best_t < t and best_d is not None and best_d <= REACH_PX * 1.6:
                    t, (x, y) = best_t, arc.at(flight["fit"], best_t)
            snaps = sorted(per_person[pid]["at"], key=lambda tt: abs(float(tt) - t))
            snap = per_person[pid]["at"][snaps[0]] if snaps else {}
            T.append(("chosen", f"person {pid}, nearest the start of the throw at {d:.0f}px"))
            return ({"person": pid, "t": round(t, 2), "dist": round(d, 1),
                     "gap": snap.get("gap"), "lift": snap.get("lift"),
                     "box": snap.get("box"), "kp": snap.get("kp")},
                    "arc", {"flight": flight, "cands": cands, "trace": T})

    if cands:
        T.append(("chosen", f"person {cands[0]['person']}, whose hands the ball left last "
                            f"({cands[0]['gap']:.2f} shoulder-widths away)"))
        return cands[0], "release", {"flight": flight, "cands": cands, "trace": T}
    T.append(("chosen", "nobody: no candidate survived"))
    return None, "none", {"flight": flight, "cands": cands, "trace": T}


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

    import numpy as np
    from frames import frame_at
    from recap2 import cap_in_box

    video_path = ROOT / "footage" / args.video
    # The scene box comes from the rig, never from a constant. It is what filters
    # every ball and person detection, and the one that used to be hardcoded here
    # excluded this session's entire left hoop.
    pool = hoops.rig_for(args.video).pool
    rows = []
    for j in shots:
        ball_track, per_person = scan_cached(args.video, j["n"], float(j["t"]),
                                             cap, fps, det, pos, pool=pool)
        pick, how, extra = attribute(ball_track, per_person,
                                     rig=hoops.rig_for(args.video), hoop=j.get("hoop"),
                                     t_descent=float(j["t"]))
        flight, cands = extra["flight"], extra["cands"]

        base = {"n": j["n"], "clock": j["clock"], "video": args.video,
                "ball_seen": len(ball_track), "people_seen": len(per_person),
                "hands_up": len(cands),
                "arc": None if not flight else {
                    "t": round(flight["t"], 2), "x": round(flight["x"], 1),
                    "y": round(flight["y"], 1), "rms": round(flight["rms"], 1),
                    "travel": flight.get("travel"), "n": flight["n"]}}
        if not pick:
            rows.append({**base, "stage": "no answer", "how": how})
            print(f"  #{j['n']} {j['clock']}: no answer "
                  f"({'no flight fitted' if not flight else 'the flight began with nobody near it'})")
            continue

        # The cap color, read off the chosen person. Kept in a SEPARATE field
        # from the attribution and never fused with it: a wrong color on a
        # correct person and a right color on a wrong person look identical
        # once combined, which is what made a previous round of ground truth
        # unusable. score_rotation.py scores the two apart for the same reason.
        hue = None
        if pick.get("box"):
            fr = frame_at(video_path, pick["t"])
            if fr is not None:
                try:
                    hue = cap_in_box(fr, pick["box"], cv2, np, padded=False)
                except Exception:
                    hue = None

        rows.append({**base, "stage": "attributed", "how": how, "hue": hue,
                     **{k: pick.get(k) for k in ("person", "t", "gap", "lift", "box")},
                     "dist": pick.get("dist"), "others": max(0, len(cands) - 1)})
        print(f"  #{j['n']} {j['clock']}: {how}"
              + (f", flight starts {pick['dist']:.0f}px away, {flight['travel']:.0f}px traveled"
                 if how == "arc" else f", release gap {pick['gap']:.2f}")
              + f", cap {hue if hue is None else round(hue)}")
    cap.release()

    out = ROOT / f"out/shooter_{args.video.replace('.MOV','')}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    got = sum(1 for r in rows if r["stage"] == "attributed")
    print(f"\n{got} of {len(rows)} shots attributed -> {out.name}")


if __name__ == "__main__":
    main()
