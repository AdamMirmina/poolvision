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

    # Toward the camera or away from it. Their own left shoulder appearing on
    # the image right means we are looking at their front.
    toward_camera = ls[0] > rs[0]

    # The face keypoints are the more reliable signal when they exist at all: a
    # visible nose means we are looking at a front, whatever the shoulders say.
    face = sum(1 for i in (NOSE, LEYE, REYE) if kp[i][2] > CONF)
    ears = sum(1 for i in (LEAR, REAR) if kp[i][2] > CONF)
    if face >= 2:
        toward_camera = True
    elif face == 0 and ears >= 1:
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
    return sum(1 for i in (NOSE, LEYE, REYE) if kp[i][2] > CONF) >= 2 or         sum(1 for i in (LEAR, REAR) if kp[i][2] > CONF) >= 1


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
