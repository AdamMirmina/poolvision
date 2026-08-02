"""Was it a three, from the deck markers and the shooter's hips.

the rule was that the two posts "denote where the 3pt line is on each
side. if someone's shooting on right hoop and they're behind the left post it's
a 3."

So each hoop has a boundary, the boundary is anchored on a marker, and a shot
counts as a three when the shooter's body is on the far side of it from that
hoop. Body, not the ball and not their hands: hips and shoulders, because a
shooter leaning or reaching forward is still standing where they are standing.
Pose gives both, and the hip midpoint is the steadier of the two when someone is
half submerged with their arms up.

WHERE THE MARKERS ARE IS NOT GUESSED. Like the rims, they are measured per
camera setup and stored in hoops.py, and a session without them raises rather
than inventing a line. A fabricated boundary would produce threes that look
authoritative and mean nothing, which is worse than no threes at all.

Checked 2026-08-02: the markers are visible in the 2026-07-29 footage but
sitting stowed together on the deck at the bottom-right, not deployed, and they
do not appear in the 2026-08-01 shootout at all. So no session has usable
positions yet.
"""

from __future__ import annotations

LHIP, RHIP, LSH, RSH = 11, 12, 5, 6


def body_point(kp, conf=0.25):
    """Where the shooter is standing: hip midpoint, shoulders as the fallback.

    Wrists and the ball are deliberately not used. A shot released from an
    outstretched arm leaves the ball a foot or more past the body, and on a
    boundary call that foot is exactly the difference being argued about.
    """
    for a, b in ((LHIP, RHIP), (LSH, RSH)):
        pts = [kp[i] for i in (a, b) if kp[i][2] > conf]
        if len(pts) == 2:
            return ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
        if pts:
            return (pts[0][0], pts[0][1])
    return None


def side_of(line, pt):
    """Signed side of an infinite line through two points. Sign is the answer."""
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)


def is_three(line, hoop_center, body):
    """True when `body` is on the opposite side of `line` from the hoop.

    Expressed relative to the hoop rather than as "left of the post", because
    which side counts flips between the two hoops and hardcoding it once would
    silently be wrong for one of them.
    """
    if body is None:
        return None
    a, b = side_of(line, hoop_center), side_of(line, body)
    if a == 0 or b == 0:
        return None
    return (a > 0) != (b > 0)


def margin(line, body, unit):
    """How far past the line they were, in rim widths, signed toward the hoop.

    A call this close to a boundary needs its own confidence. A body a tenth of a
    rim width from the line is a coin flip that should be shown as one rather
    than reported as a three.
    """
    if body is None:
        return None
    (x1, y1), (x2, y2) = line
    dx, dy = x2 - x1, y2 - y1
    den = (dx * dx + dy * dy) ** 0.5 or 1e-6
    return abs(side_of(line, body)) / den / max(1.0, unit)


def call(rig, hoop, kp):
    """(is_three, margin_in_rim_widths) for one shooter at one hoop.

    Returns (None, None) when the session has no measured markers, which is the
    honest answer and not a two-point call.
    """
    lines = getattr(rig, "three_lines", None)
    if not lines or hoop not in lines:
        return None, None
    x1, y1, x2, y2 = rig.rims[hoop]
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    body = body_point(kp) if kp else None
    return is_three(lines[hoop], center, body), margin(lines[hoop], body, x2 - x1)
