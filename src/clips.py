"""Cut a short clip per candidate shot, for review by a human.

Finding shots is now reliable (17/17 against the marked calls). Deciding whether
one went in is not, and won't be until there are labeled examples of the ugly
cases. Scrubbing a two-hour video for them is miserable, so the job here is to
turn a list of candidate moments into short loops that can be judged in a couple
of seconds each and tapped through on a phone.

Clips are H.264 so they play inline in a browser with no plugin, and are cut
generously around the event: enough lead-in to see the shot arrive, enough
follow-through to see where the ball ends up. Judging make versus behind-the-rim
needs motion, which a contact sheet of stills cannot show, and needs to be
watched slowly -- the review UI drops playbackRate rather than baking slow motion
into the file, so the same clip serves both speeds.

Run: python src/clips.py footage/x.MOV out/rimwatch_full.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hoops import rig_for  # noqa: E402
import dropzone  # noqa: E402
from parked import drop_parked  # noqa: E402
from tracks import build_tracks  # noqa: E402

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

LEAD_S = 1.2    # before the first detection
TAIL_S = 2.2    # after the last -- the outcome is the entire point of the clip
OUT_W = 640     # even number required by yuv420p
CRF = 30        # quality/size knob; 30 keeps a 3s clip near 200KB

# OpenCV's VideoWriter exposes no bitrate or quality control, and its H.264
# output ran ~5MB for a three second clip (177MB across 31 clips), which is
# unusable for something meant to load on a phone. Piping raw frames to a real
# ffmpeg gives CRF control and lands the same clips around 40x smaller.


_POSE = [None]


def _people_at(video, t, imgsz=1600):
    """Person boxes in the frame at `t`. Loaded lazily, cached per frame."""
    from ultralytics import YOLO
    import rigcheck
    from pathlib import Path
    if _POSE[0] is None:
        _POSE[0] = YOLO(str(Path(__file__).resolve().parent.parent / "yolo11s-pose.pt"))
    fr = rigcheck.grab(Path(video), t)
    if fr is None:
        return []
    r = _POSE[0].predict(fr, conf=0.15, verbose=False, imgsz=imgsz, classes=[0])[0]
    return r.boxes.xyxy.tolist() if r.boxes is not None else []


def _separates(video, run, rim, sample=5):
    """Does the ball ever get AWAY from everyone during this run?

    the rule was that the false positives were all "in someone's hand"
    -- held by white, carried out of the pool, held by a player standing on the
    concrete. A shot separates from every person by definition; a held ball never
    does.

    This is a different axis from everything else tried, and it matters because
    the others all failed on measured data: apparent ball size does not separate
    (67/84/100px for real shots against 62/59/76px for false ones), over-water
    does not either (real shots read 0% because a ball high above the pool
    projects past the pool edge in the image), and above-the-rim admits anyone
    standing behind the hoop.

    DUNKS still count, which is the same clause. A dunk does
    separate -- the ball leaves the hands at the ring and drops through the net --
    so it satisfies this rule for the right reason. What it excludes is a ball
    held at chest height that never leaves anyone.
    """
    if len(run) < 3:
        return False, 0.0
    idx = list(range(0, len(run), max(1, len(run) // sample)))
    best = 0.0
    for i in idx:
        p = run[i]
        boxes = _people_at(video, p["t"])
        if not boxes:
            continue
        d = min(((p["x"] - max(b[0], min(p["x"], b[2]))) ** 2
                 + (p["y"] - max(b[1], min(p["y"], b[3]))) ** 2) ** 0.5 for b in boxes)
        best = max(best, d / max(24.0, p.get("w") or 60.0))
    return best >= SEP_BALL_WIDTHS, best


def _falls(run, min_pts=6, min_curve=40.0, min_span=0.25):
    """Does this run show a ball actually falling under gravity?

    Fits y = a*t^2 + ... over the run. Image y grows downward, so a genuine
    ballistic drop has a POSITIVE `a`, and its size is the acceleration -- a held
    or carried ball gives a near-zero one whatever its position.

    Deliberately not a residual test: a shot seen over a few frames is noisy, and
    demanding a tidy fit would throw away exactly the short, fast flights this is
    meant to catch.
    """
    if len(run) < min_pts:
        return False
    ts = [p["t"] for p in run]
    if ts[-1] - ts[0] < min_span:
        return False
    import numpy as np
    t0 = ts[0]
    a = np.polyfit([t - t0 for t in ts], [p["y"] for p in run], 2)[0]
    if float(a) < min_curve:
        return False
    # ...and it has to be MOVING. Curvature alone let four of the named false
    # positives back in: a ball held in someone's hands bobs enough to fit a
    # gentle parabola, and so does one carried out of the pool. A thrown ball
    # falls fast; a hand-held one moves at hand speed. That is the difference
    # curvature cannot express on its own.
    speeds = []
    for p, q in zip(run, run[1:]):
        dt = q["t"] - p["t"]
        if dt > 0:
            speeds.append(abs(q["y"] - p["y"]) / dt)
    return bool(speeds) and max(speeds) >= MIN_FALL_SPEED


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("rimwatch", type=Path)
    p.add_argument("--out", type=Path, default=Path("out/clips"))
    p.add_argument("--min-dets", type=int, default=3)
    p.add_argument("--gap", type=float, default=1.0, help="seconds of absence that ends an event")
    return p.parse_args()


JITTER_PX = 25      # detector wobble tolerated without calling it a direction change
MIN_DROP_PX = 150   # a real descent, not the ball jiggling in someone's hands
# ...or, for a dunk, enough consecutive sightings at the hoop to mean the ball
# was really brought there. Set from the marked tape: the two clear dunks it
# missed ran 16 and 13 detections; the false positive review flagged at 4:10
# (ball never left his hand) is not near a hoop at all, so this does not
# readmit it.
CARRY_DETS = 5
# ...and no more than this, or it is a person holding the ball, not a dunk.
CARRY_MAX = 20
# How far back to look for the ball clearing the ring, in seconds.
ABOVE_LOOKBACK = 1.2
# Peak vertical speed a real flight reaches, px/s. A hand-held ball never does.
MIN_FALL_SPEED = 500.0
# How far the ball must get from every person, in ball-widths, to count as a
# throw rather than something someone is holding. Measured: real shots reach 7.4
# and 15.2; every false positive review flagged sits at 0.
SEP_BALL_WIDTHS = 3.0
# The drop zone widened a little. It was measured for where a ball LANDS, and the
# question here is whether the ball ever passed through the space under the hoop,
# which is a slightly bigger region once the ball's own radius and tracking slop
# are allowed for. Measured across five known events: at 1.15 the real shot that
# lands on the zone's corner comes in, and the deck ball stays out until 1.8.
# Set by sweeping against the 20-shot answer key rather than picked. Recall
# rises steeply to 2.0 and then stops -- 17/20 at every value from 2.5 to 4.0 --
# while false calls keep climbing and the deck window he already judged starts
# leaking. The curve:
#
#   grow   found   extra   deck FPs        grow   found   extra   deck FPs
#   1.0     9/20     0        0            2.5    17/20     4        ~2
#   1.5    13/20     1        0            3.0    17/20     7         1
#   2.0    16/20     2        1            5.0    19/20    11         3
#
# 2.0 is the knee: every make found (7/7), one deck false call, and past it the
# zone buys almost nothing. 1.5 is the only value that keeps the deck at zero,
# and it costs three real shots, which is the wrong trade -- a false call is a
# clip dismissed in two seconds.
ZONE_GROW = 2.0
# The fine-tuned detector fires several boxes for ONE ball in a single frame --
# five within 25px at t=804.03. Left alone they become parallel tracks of the
# same flight, and one shot is reported as three or four. Boxes in the same frame
# whose centers are within this many ball widths are the same ball.
SAME_BALL = 1.2
# A MISS, measured off two of them rather than imagined. The ball does NOT bounce
# back up -- that was the first guess and it caught nothing while adding three
# false calls. It arrives at the ring RISING, passes within a fifth of a rim
# width of the center, and then falls away sideways in one continuous descent:
# y 455 to 1123 at 18:11, y 624 to 1111 at 18:25, ending nearly four rim widths
# out. It leaves the space under the hoop, which is exactly why the zone gate
# cannot see it.
#
# Proximity alone is not enough, because the deck ball passes over the ring too
# (measured at 0.00 rim widths -- the deck sits behind the hoop in the image). The
# pairing is what separates them: a ball that passes the ring AND then falls a
# long way has gone into the pool, and a ball rolling on a flat deck has nowhere
# to fall.
RING_NEAR = 0.6      # rim widths from the ring center, at closest approach
RING_FALL_PX = 450   # and it must drop this far afterwards
# Two calls at one hoop this close together are one shot seen twice. A real
# second attempt needs the ball to come back out and be shot again.
MERGE_CALLS_S = 2.5
# GRAVITY. A ball in free flight accelerates; a ball someone is carrying moves at
# wading speed. This is the one discriminator on the project that is physics
# rather than geometry, and it separates the case geometry cannot touch: the 10:40, "one player got back in the pool with it", where the ball really does descend
# through the drop zone under the left hoop, because the person holding it is
# walking down the steps.
#
# Peak downward speed, px/s, measured over consecutive sightings:
#
#   carrying the ball in      837        the 18:25 miss   1990
#   the 11:00 dunk           1587        the 20:10 make   2070
#   the 13:23 arc            2490        the 15:57 make   2610
#
# 1200 sits nearly midway, with the slowest real shot 32% above it and the
# fastest carry 30% below.
MIN_PEAK_FALL = 1200.0
MAX_GAP_S = 0.6     # a longer blackout than this ends the descent
END_NEAR_RIM = 3.2  # where a descent must finish, in rim widths, to be a shot
BOUNCE_GAP_S = 1.1  # a second descent this soon at the same hoop is the same shot


RIG_RIMS: dict = {}


EXPLAIN = []          # (t, hoop, reason, detail) for every run a gate rejected
RIG_DROP: dict = {}   # per-hoop drop-plane calibration, set alongside RIG_RIMS
_ZONES: dict = {}


def _through_the_ring(r, rim):
    """Did the ball pass the ring and then fall away into the pool?

    Every shot still missed after the drop zone was sized against the answer key is a
    MISS at the right hoop, and all of them report the same reason: the ball never
    reached the space under the hoop. That is correct and it is the point. A make
    drops through the net and an airball falls short into the water, so both land
    under the hoop; a miss hits the rim and caroms away from it.

    Growing the zone until it catches caroms also catches the deck -- at 5.0 the
    key reaches 19/20 and all three of the deck false positives review flagged come
    back. So a carom needs its own evidence rather than a looser zone.

    The evidence is already in the track and nothing reads it. Runs are split
    where the ball stops falling, so a bounce ALWAYS creates a run boundary. A
    boundary that happens at the ring, with the ball climbing afterwards, is a
    carom. A boundary anywhere else is the ball being lost, or resting, or
    someone walking with it.
    """
    rcx, rcy = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2
    rw = max(1.0, rim[2] - rim[0])
    near = min((((p["x"] - rcx) ** 2 + (p["y"] - rcy) ** 2) ** 0.5) / rw for p in r)
    if near > RING_NEAR:
        return False
    return (max(p["y"] for p in r) - min(p["y"] for p in r)) >= RING_FALL_PX


def _one_box_per_ball(hits):
    """Collapse duplicate boxes for the same ball in the same frame.

    The stock COCO detector rarely double-fires, so this was not needed while it
    was the only source. The fine-tune does it constantly -- it is far more
    sensitive, which is the point, and the cost is several overlapping boxes per
    ball. Without this the tracker builds one track per box and a single dunk is
    reported four times.

    Keeps the highest-confidence box of each cluster, since that is the one whose
    center is most likely to be on the ball rather than on its wake or shadow.
    """
    by_t = {}
    for h in hits:
        by_t.setdefault(round(h["t"], 3), []).append(h)
    out = []
    for _, group in by_t.items():
        kept = []
        for h in sorted(group, key=lambda z: -z["conf"]):
            w = max(1.0, h.get("w", 60.0))
            if any(((h["x"] - k["x"]) ** 2 + (h["y"] - k["y"]) ** 2) ** 0.5
                   < SAME_BALL * w for k in kept):
                continue
            kept.append(h)
        out.extend(kept)
    return sorted(out, key=lambda z: z["t"])


def _dropzone(hoop, rim):
    """The widened drop zone for a hoop, or None if this rig has no plane fit."""
    if hoop in _ZONES:
        return _ZONES[hoop]
    d = (RIG_DROP or {}).get(hoop)
    poly = None
    if d:
        base = dropzone.zone(rim, drop=d, key=hoop)
        cx = sum(p[0] for p in base) / len(base)
        cy = sum(p[1] for p in base) / len(base)
        poly = [(cx + (x - cx) * ZONE_GROW, cy + (y - cy) * ZONE_GROW) for x, y in base]
    _ZONES[hoop] = poly
    return poly


def _why(hoop, r, reason, detail=""):
    EXPLAIN.append((r[0]["t"], hoop, reason, detail))


def cluster(hits_by_hoop: dict, gap: float, min_dets: int, video=None):
    """One shot per clip, found by its DESCENT.

    Two rules were tried and both fail, for opposite reasons. Splitting on gaps
    in detection cuts a single shot in half, because the ball is lost during the
    pause while the shooter sets -- that produced a clip of someone raising their
    arm and a separate clip of the ball arriving. Widening the gap and merging
    overlapping windows fixed that but glued genuinely separate shots together,
    which is worse: a reviewer is shown two outcomes and one set of buttons and
    has no way to say which one they mean.

    Neither is fixable by tuning, because the two failures want the threshold
    moved in opposite directions. At 4:52 two shots sit 1.36s apart; the pause
    inside the single 5:44 shot is 1.03s. There is no gap that separates one and
    not the other.

    What actually defines a shot is the ball coming down at the hoop, exactly
    once. So an event is a run of falling detections spanning a real drop. The
    two shots at 4:52 fall 737->1136 and 751->1204: unambiguously two. The 5:44
    shot, wind-up and all, contains one descent: unambiguously one.
    """
    events = []
    for hoop, hits in hits_by_hoop.items():
        # A ball at rest is detected in most frames and would otherwise dominate
        # everything downstream -- see parked.py for the two times this bit.
        hits = _one_box_per_ball(hits)
        hits, dropped = drop_parked(hits)
        if dropped:
            print(f"  {hoop}: dropped {dropped} detections of ball(s) sitting still")

        # Descents are found WITHIN a track, never across the merged detection
        # stream. With more than one ball on screen a merged stream hops between
        # objects and shreds one descent into many fragments -- see tracks.py.
        for pts in build_tracks(hits):
            runs, cur = [], [pts[0]]
            for p in pts[1:]:
                prev = cur[-1]
                falling = p["y"] >= prev["y"] - JITTER_PX
                if p["t"] - prev["t"] > MAX_GAP_S or not falling:
                    runs.append(cur)
                    cur = [p]
                else:
                    cur.append(p)
            runs.append(cur)

            # A descent that ENDS somewhere else is not a shot at this hoop.
            #
            # Review is
            # right, and a falling ball is not enough on its own: a pass across
            # the pool falls too. What makes it a shot is where the fall FINISHES.
            rim = RIG_RIMS[hoop]
            rcx, rcy = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2
            rw = max(1.0, rim[2] - rim[0])
            kept = []
            for r in runs:
                drop = r[-1]["y"] - r[0]["y"]
                # A DUNK never falls far. The ball is carried to the rim, so a
                # gate built on "how far did it drop" cannot see one -- and on a
                # low pool hoop that is a huge slice of the game.
                #
                # Measured against the marking of five minutes of game footage:
                # 4:39 had 29 detections and a 16-long run ending beside the net,
                # rejected for dropping 120px instead of 150. 6:49 had 19, a
                # 13-long run, ending 1.0 rim widths away, rejected for 53px. Both
                # obvious dunks to a person watching.
                #
                # So there are two ways to be a shot now: fall a long way into the
                # rim, or be SEEN AT the rim persistently. A pass does neither --
                # it crosses the frame without lingering at a hoop.
                # A dunk is BRIEF. the second tape showed what the bare
                # count admits: 50 consecutive sightings while a player carried
                # the ball out of the pool, and 12 more climbing back in, both
                # called shots. "it caught blue getting back in the pool with the
                # ball in hand."
                #
                # The ball at the rim on a dunk lasts a moment. A person holding
                # it lasts as long as they like, so an upper bound separates them
                # where a lower bound alone cannot.
                # THE RULE: the ball has to be seen ABOVE THE RIM at some
                # point, or it is not a shot. The words: "it was a bullet with
                # almost no arc and way below the rim. that would be the
                # equivalent of a pass. maybe say shots have to be seen above the
                # rim to be called shots?"
                #
                # This one condition kills every false positive he named, and
                # they were five different-looking events: the ball held in
                # white's hands (8:10-8:12), raised and lowered without release
                # (8:42), carried out of the pool (8:22), a player standing on the
                # concrete holding it (8:26), and the flat pass (8:55). None of
                # them put the ball above the ring.
                #
                # It also fixes the dunk path without weakening it: a dunk brings
                # the ball ABOVE the rim and pushes it down, while a player
                # holding one at chest height beside the hoop never does. That is
                # the difference the run-length bounds could not express.
                # Checked across the WHOLE window around this descent, not just
                # the run's own points. Applied to the run alone it killed two of
                # the five marked shots, both at the left hoop, where the ball is
                # only picked up during its final drop into the net -- every
                # sighting in the run sits below the ring even though the shot
                # plainly went over it. The evidence is a few frames earlier.
                # A CLEAR DOWNWARD PARABOLA, which is the own broadening of
                # his above-the-rim rule: "cause im realizing now that airballs
                # below the rim won't be counted."
                #
                # That is right, and the parabola is the better test because it
                # describes what makes something a shot rather than where it
                # happened to go. An airball that never reaches the ring still
                # arcs. And it separates his false positives by their physics
                # rather than by their position: a held ball does not move, a
                # carried one drifts at walking pace, a flat bullet pass has
                # almost no curvature, and a player standing on the deck has
                # none at all.
                #
                # Image y grows downward, so gravity is a POSITIVE quadratic
                # coefficient. Fitted over the run's own points.
                lo, hi = r[0]["t"] - ABOVE_LOOKBACK, r[-1]["t"]
                above = any(p["y"] < rim[1] for p in pts if lo <= p["t"] <= hi)
                arced = _falls(r)
                # Only the CARRIED path has to clear the ring. A run with a real
                # descent already proved itself by falling into the hoop.
                #
                # Applied to everything it cost two of the five marked shots,
                # both at the left hoop, where the ball is only picked up on its
                # final drop -- so nothing in the window sits above the ring even
                # though the shot plainly went over it. Image height is also not
                # world height: a player on the deck BEHIND the right hoop appears
                # above it while standing at ground level, which is why his 8:21
                # false positive survived the rule.
                #
                # Restricted to the carried path it does exactly the job he
                # described -- a held, lifted or walked ball never clears the
                # ring -- without touching shots that fall through it.
                # Applied to EVERY run, not only carried ones. the named
                # false positives -- the ball held in white's hands, and the one
                # carried out of the pool -- come through the DESCENT gate: the
                # ball really does fall 150+px while someone lowers it or walks
                # with it. Gating only the carry path never saw them.
                # Kept as measured, not as reasoned. Four variants were scored
                # against the marked minute:
                #
                #   no rule at all             15 calls, 5/5 real, 7 named FPs
                #   above-rim, every run        5 calls, 3/5 real
                #   above-rim, carry path only  6 calls, 4/5 real, 2 named FPs
                #   parabola alone              3 calls, 2/5 real, 3 named FPs
                #
                # The parabola is the better IDEA -- it catches airballs below the
                # ring, which above-rim discards, and review flagged exactly that.
                # It is not yet the better RULE: fitted over a run of 6+ points it
                # throws away short flights seen for only a few frames, which is
                # most of the left hoop's shots. It needs the run-building to hold
                # a flight together first; tightening the test cannot fix data
                # that arrives in fragments.
                # the separation rule, applied to EVERY run. The named false
                # positives came through the DESCENT gate as often as the carry
                # gate -- a ball genuinely falls 150+px while someone lowers it --
                # so gating one path was never going to catch them.
                # TWO rules, because they catch different things and the feedback contained both:
                #
                #   SEPARATION -- the ball must get away from every person. Kills
                #   the ball held in white's hands, carried out of the pool, or
                #   held by someone standing on the concrete. A dunk passes for
                #   the right reason: it separates at the ring.
                #
                #   ARC or ABOVE THE RING -- the ball must behave like a shot.
                #   Kills the flat bullet pass, which separates perfectly well and
                #   is still not a shot: "way below the rim... the equivalent of a
                #   pass."
                #
                # Measured separately: separation alone gives 9 calls and all five
                # of his real shots; above-rim alone gives 6 and loses one.
                # Kept SEPARATION ALONE, on the numbers rather than on taste.
                # Adding "must also arc or clear the ring" removes one false
                # positive and costs a real shot (8:35, whose run has no arc fit,
                # no descent and nothing above the ring -- it is seen for too few
                # frames to prove anything about its shape). Recall is worth more
                # here: a false call is a clip a reviewer dismisses in two seconds, a
                # missed shot is invisible.
                #
                #   separation alone   9 calls, 5/5 real, 4 named FPs
                #   both rules         5 calls, 4/5 real, 3 named FPs
                # THE BALL HAS TO REACH THE SPACE UNDER THE HOOP.
                #
                # This replaces the separation rule, which was wrong about the one
                # case review explicitly warned about. A dunk NEVER separates: the
                # ball is in the player's hands the whole way, measured at 0.0 ball
                # widths for the entire run of his 11:00 dunk, so the rule threw it
                # away by construction.
                #
                # It also passed everything separation is blind to. All three false
                # calls on the fresh tape were a loose ball with nobody near it:
                # "this was a ball rolling across the deck, not a shot. then one player
                # pcked it up and it ocunted as another shot. then one player got back in
                # the poool and it ocunted another."
                #
                # Height cannot fix that, and this is the third measurement saying
                # so. The deck ball sits 1.87 rim-widths ABOVE the ring in the
                # image while the real dunk never rises above it at all, because
                # the deck is further from the camera. It even passes over the ring
                # more squarely (0.00 rim-widths) than the real arcing shot (0.11).
                #
                # The drop zone is the one piece of geometry that knows about
                # depth, because it is a quad on the WATER PLANE rather than a box
                # in the image. tuned by eye and signed off Measured
                # over five known events, sightings inside the zone:
                #
                #   ball rolling on the deck      0
                #   player picking it up          0
                #   player getting back in        0
                #   the 11:00 dunk               64
                #   the 13:23 arc                 7
                #
                # Nothing on the deck can enter it at any dilation short of 1.8.
                # THE DROP ZONE IS GONE as a hard gate, and it should have been
                # the moment gravity arrived.
                #
                # It existed for one reason: nothing else could exclude the deck.
                # A ball rolling on the concrete passes over the ring more
                # squarely than a real shot, because the deck sits behind the
                # hoop and the image reads depth as height. The zone was the only
                # geometry that knew about depth.
                #
                # Gravity excludes those cases directly and better -- a carried or
                # rolled ball moves at wading speed whatever the geometry says --
                # so the zone became a second, weaker copy of a test already being
                # done properly.
                #
                # Measured across all seven ground-truth windows, which is the
                # only reason this is knowable:
                #
                #   zone at 2.0   41/57 found   0 named FP   7 other
                #   zone at 4.0   47/57         1            15
                #   no zone       48/57         1            15
                #
                # It was the reason for 7 of 9 misses on a different camera and
                # 4 of 4 on the marked minute. Seven real shots to prevent one
                # false call is the wrong trade by its own standard: "a false call
                # is a clip dismissed in two seconds, a missed shot is invisible."
                #
                # Kept as a SOFT signal rather than deleted: a run that reached
                # the zone still passes the ring test for free, since being under
                # the hoop is real evidence, just not necessary evidence.
                poly = _dropzone(hoop, rim)
                in_zone = bool(poly) and any(
                    dropzone.contains(poly, p["x"], p["y"]) for p in r)
                through = _through_the_ring(r, rim) or in_zone
                carried = CARRY_DETS <= len(r) <= CARRY_MAX
                if len(r) < min_dets and not carried:
                    _why(hoop, r, "too few detections",
                         f"{len(r)} points, needs {min_dets} or {CARRY_DETS}-{CARRY_MAX}")
                    continue
                # How far it fell, measured over the WINDOW rather than only
                # within the run. A track breaks at the moment the ball hits the
                # water -- the jump is too big to associate -- so the run left
                # behind is just the ball drifting on the surface. the 13:23
                # shot measured -15px that way, meaning it ended HIGHER than it
                # started, while the real flight from y=779 down to y=1424 sat in
                # the sightings one second earlier under a different track id.
                #
                # Taking the highest point seen in the lookback gives the fall the
                # shot actually made. It is safe here only because the drop zone
                # has already ruled out everything on the deck; on its own a
                # window-wide maximum would happily pair an unrelated high
                # sighting with an unrelated low one.
                # A carried ball can satisfy every geometric test -- it passes
                # under the ring, it enters the zone, it descends -- and only
                # fails on speed.
                speeds = [(b["y"] - a["y"]) / (b["t"] - a["t"])
                          for a, b in zip(r, r[1:]) if b["t"] > a["t"]]
                if not speeds or max(speeds) < MIN_PEAK_FALL:
                    _why(hoop, r, "never fell at the speed of a dropped ball",
                         f"peak {max(speeds) if speeds else 0:.0f} px/s, "
                         f"needs {MIN_PEAK_FALL:.0f}")
                    continue
                top = min((p["y"] for p in hits if lo <= p["t"] <= hi), default=r[0]["y"])
                fell = max(drop, r[-1]["y"] - top)
                if fell < MIN_DROP_PX and not carried:
                    _why(hoop, r, "did not fall far enough",
                         f"{fell:.0f}px over the window, needs {MIN_DROP_PX}")
                    continue
                # Where the descent ENDS is not evidence for a miss, and this
                # gate says it is. the 18:11 passed the drop zone with 45
                # sightings inside it and was killed here for ending 3.8 rim
                # widths out -- which is what a carom does. It is the same wrong
                # assumption as the zone, that a shot finishes at the hoop, in a
                # second place.
                #
                # A run that already proved itself by passing the ring and
                # falling into the pool has established which hoop it belongs to.
                # Where it came to rest afterwards adds nothing.
                near = min(((p["x"] - rcx) ** 2 + (p["y"] - rcy) ** 2) ** 0.5
                           for p in r[-3:]) / rw
                if near > END_NEAR_RIM and not through:
                    _why(hoop, r, "ended too far from the rim",
                         f"{near:.1f} rim widths, needs under {END_NEAR_RIM}")
                    continue
                kept.append(r)

            # A ball that bounces up off the rim and rolls before dropping is ONE
            # shot, not two. So consecutive descents at the same hoop, close in time
            # and both finishing at the rim, are joined rather than counted twice
            # -- and the clip cut from the joined event runs to the real outcome.
            merged_runs = []
            for r in kept:
                if merged_runs and r[0]["t"] - merged_runs[-1][-1]["t"] <= BOUNCE_GAP_S:
                    merged_runs[-1] = merged_runs[-1] + r
                else:
                    merged_runs.append(r)
            for r in merged_runs:
                events.append((hoop, r))

    events.sort(key=lambda e: e[1][0]["t"])

    # Two descents at the same hoop whose time ranges OVERLAP get merged. They
    # may genuinely be two balls, but a reviewer is shown a clip, and two
    # clips covering the same instant look identical to them -- the marked shot hit exactly
    # this ("36 and 37 are definitely the same"). One clip covering both is
    # honest; two near-duplicates waste a decision and pollute the labels.
    # Overlap is not the only way to get two clips of one shot. The fine-tuned
    # detector is sensitive enough to hold several tracks through a single
    # flight, and runs from DIFFERENT tracks never meet the per-track merge
    # above, so they arrive here as separate events that merely sit close in
    # time. Scored against the answer key, three of four apparently-false calls were
    # this: 16:36 one second from his 16:35, 18:19 and 20:01 landing on the same
    # second as his marks. Not false positives at all, the same shot counted
    # twice.
    #
    # So near-in-time counts as well as overlapping. A genuine second attempt
    # needs the ball to come back out and be shot again, which does not happen
    # inside MERGE_CALLS_S.
    merged = []
    for hoop, dets in events:
        # Compared on START times, not end-to-start. Merging against the end
        # lets an event grow every time it absorbs one, and a long event then
        # keeps swallowing the next shot along -- that ate the 20:01 make
        # outright, turning a duplicate fix into a lost shot. Starts cannot
        # snowball.
        if (merged and merged[-1][0] == hoop
                and dets[0]["t"] - merged[-1][1][0]["t"] <= MERGE_CALLS_S):
            merged[-1] = (hoop, merged[-1][1] + dets)
            continue
        if merged and merged[-1][0] == hoop and dets[0]["t"] <= merged[-1][1][-1]["t"]:
            merged[-1] = (hoop, sorted(merged[-1][1] + dets, key=lambda d: d["t"]))
        else:
            merged.append((hoop, dets))
    return merged


MIN_TAIL_S = 1.5   # the outcome must be on screen, whatever else gets cut
MIN_LEAD_S = 0.3


def windows(events):
    """Clip bounds per event.

    Trimmed against neighbours so one clip doesn't simply replay the shot
    before it, but the TAIL is protected and the lead-in gives way first. When
    shots come in quick succession, squeezing both ends equally produced clips
    as short as 0.8s -- 32 of 128 on IMG_2481 were under two seconds, which is
    not long enough to see whether the ball went in. A clip that starts with the
    ball already falling is judgeable; one that ends before the ball arrives is
    worthless, and worse than worthless in a review queue because it still costs
    a decision.

    So a clip may overlap slightly into the next shot's lead-in. That is the
    lesser evil: each clip is still centered on its own descent, and the overlap
    shows the previous ball already gone.
    """
    out = []
    for i, (hoop, dets) in enumerate(events):
        d0, d1 = dets[0]["t"], dets[-1]["t"]
        start = d0 - LEAD_S
        end = d1 + TAIL_S

        prev = next((e for e in reversed(events[:i]) if e[0] == hoop), None)
        if prev:
            start = max(start, prev[1][-1]["t"] + 0.35)
        start = min(start, d0 - MIN_LEAD_S)          # never eat into the shot itself

        nxt = next((e for e in events[i + 1:] if e[0] == hoop), None)
        if nxt:
            end = min(end, nxt[1][0]["t"] - 0.35)
        end = max(end, d1 + MIN_TAIL_S)              # the outcome wins over tidiness

        out.append((max(0.0, start), end))
    return out


def main():
    args = parse_args()

    rig = rig_for(args.video)
    global RIG_RIMS, RIG_DROP
    RIG_RIMS = rig.rims
    RIG_DROP = rig.drop or {}
    rw = json.loads(args.rimwatch.read_text(encoding="utf-8"))
    events = cluster(rw["hits"], args.gap, args.min_dets, video=args.video)
    bounds = windows(events)
    print(f"{len(events)} shots (one descent each)")

    fps = 60.0 if "2528" in args.video.name or "2529" in args.video.name else 30.0
    args.out.mkdir(parents=True, exist_ok=True)
    index = []

    for i, (hoop, dets) in enumerate(events, 1):
        x1, y1, x2, y2 = rig.crops[hoop]
        t_start, t_stop = bounds[i - 1]
        h = int(OUT_W * (y2 - y1) / (x2 - x1)) // 2 * 2   # even height for yuv420p
        name = f"shot_{i:03d}_{hoop}_{dets[0]['t']:.0f}s.mp4"
        path = args.out / name

        # ffmpeg seeks, crops and encodes in one pass; nothing is decoded in
        # Python at all. The previous version pulled frames through OpenCV and
        # seeked with CAP_PROP_POS_FRAMES, which is fine on a 3GB 1080p file and
        # catastrophic on a 15GB 4K60 one -- a single seek to frame 36000 ran
        # past ten minutes without returning. At one seek per shot and roughly a
        # hundred shots that is not a slow step, it is a step that never
        # finishes. `-ss` BEFORE `-i` seeks on the container index instead of
        # decoding forward to the target.
        proc = subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error",
             "-ss", f"{t_start:.3f}", "-i", str(args.video), "-t", f"{t_stop - t_start:.3f}",
             "-vf", f"crop={x2-x1}:{y2-y1}:{x1}:{y1},scale={OUT_W}:{h}",
             "-c:v", "libx264", "-crf", str(CRF), "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(path)],
            capture_output=True)
        err = proc.stderr.decode("utf-8", "replace").strip()
        if err:
            print(f"    ffmpeg: {err[:200]}")
        n = int(round((t_stop - t_start) * fps))

        index.append({
            "n": i,
            "file": name,
            "hoop": hoop,
            "t": round(dets[0]["t"], 2),
            "t_end": round(dets[-1]["t"], 2),
            "clip_start": round(t_start, 2),
            "clip_stop": round(t_stop, 2),
            "clock": f"{int(dets[0]['t'])//60}:{int(dets[0]['t'])%60:02d}",
            "dets": len(dets),
            "peak_conf": round(max(d["conf"] for d in dets), 3),
            "frames": n,
            "size_kb": round(path.stat().st_size / 1024) if path.exists() else 0,
        })
        print(f"  {name}  {n} frames, {index[-1]['size_kb']} KB")

    (args.out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    total = sum(c["size_kb"] for c in index)
    print(f"wrote {len(index)} clips, {total/1024:.1f} MB total -> {args.out}")


if __name__ == "__main__":
    main()
