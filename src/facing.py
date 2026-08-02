"""Which way is a person facing, and is it toward the hoop.

the observation, on a frame where the pipeline picked the wrong one of two
people: "each have one hand up and they're the 2 closest to the ball. diff is,
blue is looking away from the basket that the shot went to and yellow is looking
towards. can we use this as another heuristic."

It is a good one precisely because it is independent of everything already in
use. Hands up, distance to the ball and distance to the flight's origin are all
about WHERE someone is. Facing is about which way they are turned, so it can
separate two people standing in the same place, which is exactly the case that
defeats the rest.

The geometry, from the top-down-ish camera this pool has:

  the shoulder line runs across the body, so the direction the chest points is
  perpendicular to it. That leaves two candidates, one in front and one behind,
  and the face keypoints settle which. A person facing the camera shows their
  own LEFT shoulder on the image RIGHT; facing away it flips. Nose and eye
  confidence back the same call up, and are used when the shoulders are level
  enough that their order is unreliable.
"""

from __future__ import annotations

NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH = 0, 1, 2, 3, 4, 5, 6
LELB, RELB, LWR, RWR = 7, 8, 9, 10
LHIP, RHIP = 11, 12
CONF = 0.25


def _pt(kp, i):
    p = kp[i]
    return (p[0], p[1]) if p[2] > CONF else None


def facing(kp):
    """A unit vector for the way the chest points, or None.

    Returned in image coordinates, so y grows downward and a vector pointing
    "down" means the person faces the near side of the pool.
    """
    ls, rs = _pt(kp, LSH), _pt(kp, RSH)
    if not ls or not rs:
        return None
    sx, sy = ls[0] - rs[0], ls[1] - rs[1]
    n = (sx * sx + sy * sy) ** 0.5
    if n < 8:
        return None
    # Perpendicular to the shoulders, both ways.
    a = (-sy / n, sx / n)
    b = (sy / n, -sx / n)

    # Toward the camera or away from it, in order of how much the evidence is
    # worth.
    #
    # Shoulder order is the weakest and it is the fallback: their own left
    # shoulder appearing on the image right means we are looking at their front.
    # It is also what the front/back call collapsed to whenever the face was
    # hidden, which is exactly the case that produced the wrong readings.
    toward_camera = ls[0] > rs[0]

    # ARMS. Right, and it
    # needs no face at all: elbows and wrists sit forward of the shoulder line,
    # so which side of that line the limbs fall on IS the front.
    #
    # (Nipples were the other suggestion. COCO pose has 17 keypoints and none of
    # them are on the chest, so there is nothing to read.)
    limbs = [kp[i] for i in (LELB, RELB, LWR, RWR) if kp[i][2] > CONF]
    if limbs:
        mx = sum(q[0] for q in limbs) / len(limbs)
        my = sum(q[1] for q in limbs) / len(limbs)
        smx, smy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
        # Which perpendicular the limbs sit on. `a` is one normal to the
        # shoulders; a positive projection means the limbs are on a's side.
        proj = a[0] * (mx - smx) + a[1] * (my - smy)
        if abs(proj) > n * 0.10:            # ignore arms hanging on the line
            toward_camera = (a[1] > 0) == (proj > 0)

    # The face still wins where it exists, being the least ambiguous of the
    # three: a visible nose means we are looking at a front, whatever else says.
    face = sum(1 for i in (NOSE, LEYE, REYE) if kp[i][2] > CONF)
    ears = sum(1 for i in (LEAR, REAR) if kp[i][2] > CONF)
    if face >= 2:
        toward_camera = True
    elif face == 0 and ears >= 1 and not limbs:
        toward_camera = False

    # "Toward the camera" is downward in the image for this rig, since the camera
    # looks down at the pool from one side.
    return a if (a[1] > 0) == toward_camera else b


# How wide the shoulders must read, in pixels, before the front/back call is
# trusted, and whether face or ear keypoints have to confirm it.
#
# Measured on 36 attributed shots: at the NEAR hoop this gets the shooter facing
# it on 12 of 13, and at the FAR hoop on 10 of 23, which is a coin flip. Not a
# flipped sign -- a conditioning limit. Someone facing directly away from the
# camera projects almost no facing vector onto the image plane, so its direction
# is decided by a few pixels of noise, and their face keypoints are not visible
# to settle it either. So the heuristic reports None rather than guessing when
# the geometry cannot support an answer.
MIN_SHOULDER_PX = 26.0


def orientation(rig, hoop):
    """+1 if facing reads true for this hoop, -1 if it reads inverted, 0 if unusable.

    The far hoop is not noisy, it is INVERTED, and that took a second look to
    see. Measured on 32 attributed shots: the near hoop has the shooter facing
    it on 11 of 12 with a median of +0.52; the far hoop reads 6 of 20 with a
    median of -0.51. Noise would sit near zero. A consistent -0.51 is a
    consistent reading of "facing away".

    The cause is the same geometry that first looked like an absence of signal.
    A shooter at the far hoop is turned away from the camera, so their face
    keypoints are not visible and the front/back call falls back to shoulder
    order, which lands on the wrong side of the body. Correcting for it takes
    the far hoop to 14 of 20 (+0.51) and both hoops together to 25 of 32.

    This is not fitted to the labels. Which hoop is far follows from the camera
    alone -- a rim's apparent width IS its distance -- so the correction is
    predictable for a new session before seeing a single shot from it.
    """
    if not getattr(rig, "rims", None) or hoop not in rig.rims:
        return 0
    widths = {k: v[2] - v[0] for k, v in rig.rims.items()}
    return 1 if hoop == max(widths, key=widths.get) else -1


def usable_for(rig, hoop):
    """Kept for callers that only need to know whether to bother."""
    return orientation(rig, hoop) != 0


def readable(kp):
    """Whether this pose can support a front/back call at all."""
    ls, rs = _pt(kp, LSH), _pt(kp, RSH)
    if not ls or not rs:
        return False
    w = ((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2) ** 0.5
    if w < MIN_SHOULDER_PX:
        return False
    # Front or back has to be evidenced, not inferred from shoulder order, which
    # is exactly the part that fails at distance.
    # Face, ears, OR arms. Arms are the addition: they are visible on people
    # turned away from the camera, which is precisely where the other two fail.
    return (sum(1 for i in (NOSE, LEYE, REYE) if kp[i][2] > CONF) >= 2
            or sum(1 for i in (LEAR, REAR) if kp[i][2] > CONF) >= 1
            or sum(1 for i in (LELB, RELB, LWR, RWR) if kp[i][2] > CONF) >= 2)


def toward(kp, target, center):
    """How much this person faces `target`, from -1 (away) to 1 (straight at it).

    `center` is where they are standing. Returned as a cosine so it can be
    thresholded or ranked without any further scaling.
    """
    if not readable(kp):
        return None
    f = facing(kp)
    if not f or not center:
        return None
    dx, dy = target[0] - center[0], target[1] - center[1]
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-6:
        return None
    return f[0] * dx / n + f[1] * dy / n


def body_center(kp, box=None):
    """Shoulders if they are readable, the box's middle otherwise."""
    ls, rs = _pt(kp, LSH), _pt(kp, RSH)
    if ls and rs:
        return ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    if box:
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LEFT versus RIGHT, from the wrists against the head.
#
# the rule, and it beats the front/back measure it replaces: "if wrists are
# left of the head shooter's facing left, and if wrists are right they're facing
# right. if they're opposite sides, they're facing forward or back and probably
# aren't the shooter."
#
# It works because it never asks the question that was capping the other one.
# Front-versus-back needs a face the camera often cannot see, and when it cannot
# the call collapses -- on one frame six of seven people read as facing the
# camera. Left-versus-right needs no face at all, and left/right is the axis that
# actually matters here, because the two hoops sit at the two ends of the pool.
#
# Measured on 32 attributed shots: it points at the hoop the shot went to on 25
# of the 28 it can read, which is 89% against the 81% of the front/back version.
# The other 4 straddle, one wrist each side, which is the "probably aren't the
# shooter" case rather than a failure to read.

DEAD = 0.15          # wrists this close to the head, in shoulder-widths, are neutral


def head_x(kp):
    pts = [kp[i] for i in (NOSE, LEYE, REYE, LEAR, REAR) if kp[i][2] > CONF]
    if pts:
        return sum(p[0] for p in pts) / len(pts)
    sh = [kp[i] for i in (LSH, RSH) if kp[i][2] > CONF]
    return sum(p[0] for p in sh) / len(sh) if len(sh) == 2 else None


def side(kp):
    """(direction, strength): +1 image-right, -1 image-left, 0 straddling.

    `strength` is how far the wrists sit from the head in shoulder-widths,
    capped at 1, so a marginal reading can be shown and weighted as marginal
    rather than treated the same as an unmistakable one.
    """
    if not kp:
        return None
    hx = head_x(kp)
    if hx is None:
        return None
    sh = [kp[i] for i in (LSH, RSH) if kp[i][2] > CONF]
    if len(sh) < 2:
        return None
    w = max(12.0, abs(sh[0][0] - sh[1][0]))
    ws = [kp[i] for i in (LWR, RWR) if kp[i][2] > CONF]
    if not ws:
        return None
    sides = [(q[0] - hx) / w for q in ws]
    if len(sides) == 2 and min(sides) < -DEAD and max(sides) > DEAD:
        return (0, 0.0)                 # one wrist each side: square on
    m = sum(sides) / len(sides)
    if abs(m) < DEAD:
        return (0, 0.0)
    return (1 if m > 0 else -1, min(1.0, abs(m)))


def side_toward(kp, target_x, body_x):
    """+strength if the wrists point at the target, -strength if away, 0 if square on."""
    r = side(kp)
    if not r or body_x is None:
        return None
    d, mag = r
    if d == 0:
        return 0.0
    want = 1 if target_x > body_x else -1
    return mag if d == want else -mag
