"""Did the ball drop into the water directly under the net?

 And, on why: "if it went thru the net it
would likely drop there."

The recall is what makes it work. A ball through the hoop falls in the column
below it, and 34 of 37 judged makes do. A ball off the iron is kicked sideways
and lands somewhere else. So the informative event is a shot that NEVER appears
in that column, and on 119 judged shots that veto is right 51 times out of 54.

Measured on the rimwatch detections that already exist, so it costs nothing to
apply. It is a different question from anything else in the model, which all
concern the descent's shape near the rim rather than where the ball ended up.
"""

from __future__ import annotations

# The patch of water under the net, in rim widths. 1.10 wide by 1.30 deep, so it
# reads as a small square under the net rather than a wide band.
#
# Re-swept 2026-08-03 (src/dropsweep.py) rather than reusing the old number, for
# two reasons that both invalidated it:
#
#   The old "34 of 37 makes" was measured against box() -- the axis-aligned
#   rectangle -- because that is what dropped() actually tested, while the quad
#   was only ever what got DRAWN. The picture showed one region and the rule
#   tested another.
#
#   The shape is now correct (a patch of the water plane running out from the
#   wall), so the old sweep does not describe this zone at all.
#
# Measured on 66 judged makes and 125 judged misses, with a real point-in-polygon
# test. Recall is far lower than the old figure implied and does not depend on
# width nearly as much as feared: 1.80x1.60 catches 73% of makes, 1.10x1.30
# catches 64%. What matters is the veto's precision -- "never appeared here, so
# it missed" -- and that barely moves: 83.5% at the widest against 81.8% here.
#
# So a visibly square zone costs about 1.7 points of veto precision. Worth it,
# and stated rather than hidden.
HALF_W = 0.55
# Half-depth, because the square is CENTERED on the point under the rim.
# 0.65 keeps the same 1.30 total depth the sweep was run at.
HALF_D = 0.65
# Kept for the legacy quad() path used by sessions with no measured axes.
DEPTH_LO = 0.3
DEPTH_HI = 1.3
WINDOW_S = 1.6

# Compass, review's: NORTH is the diving board end, away from the camera and
# toward the top of the frame. SOUTH is toward the camera. Used in review
# notes so a direction can be said out loud without ambiguity -- "too far
# north" is unmistakable in a way that "too far up" is not.


def box(rim):
    """The drop zone as a rectangle: (x1, y1, x2, y2). Used for the test itself."""
    x1, y1, x2, y2 = rim
    cx, w = (x1 + x2) / 2, x2 - x1
    return (int(cx - HALF_W * w), int(y2 + DEPTH_LO * w),
            int(cx + HALF_W * w), int(y2 + DEPTH_HI * w))


def quad(rim, water):
    """The drop zone as a PARALLELOGRAM that sits in the water and follows the wall.



    Both halves matter. Starting at the rim's bottom edge put the top of the zone
    on deck, above the waterline, where a ball can never be. And the pool wall
    runs at about 35 degrees in this view, so an axis-aligned box either clips
    into the deck at one end or misses water at the other.

    `water` is {"water_y_at_center": ..., "slope": ...} for this hoop. Falls back
    to the plain rectangle when a session has not measured it.
    """
    x1, y1, x2, y2 = rim
    cx, w = (x1 + x2) / 2, x2 - x1
    if not water:
        b = box(rim)
        return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
    m = water["slope"]
    top = water["water_y_at_center"] + DEPTH_LO * w * 0.5
    bot = top + (DEPTH_HI - DEPTH_LO) * w
    lx, rx = cx - HALF_W * w, cx + HALF_W * w
    # Shear both edges along the wall, so the zone lies flat on the water.
    return [(int(lx), int(top + m * (lx - cx))), (int(rx), int(top + m * (rx - cx))),
            (int(rx), int(bot + m * (rx - cx))), (int(lx), int(bot + m * (lx - cx)))]


def quad_on_plane(rim, drop):
    """The drop zone from measured water-plane axes: the correct construction.

    `drop` is {"p", "along", "into"} from src/dropfit.py. The zone is a patch of
    the water surface under the net, so one side runs ALONG the wall and the
    other runs AWAY from it, into the pool.

    The version above shears to the wall and then extends straight down the
    image, which is the same thing only if the camera looks straight down. It
    does not, so that zone ran along the pool's length rather than out from the
    wall: on the left hoop it walked up onto the concrete, on the right it leaned
    down the wall instead of reaching into the water.

    Sizes are unchanged. HALF_W and the depths were measured against real makes
    (34 of 37 land inside; tightening to 0.5 catches 16) and this is a fix to the
    zone's SHAPE, not an invitation to re-tune numbers that were earned.
    """
    x1, y1, x2, y2 = rim
    w = x2 - x1
    px, py = drop["p"]
    ax, ay = drop["along"]
    ix, iy = drop["into"]
    # CENTERED on the point below the rim, not started there and pushed outward.
    #
    # The offset version put the near edge DEPTH_LO out and the far edge DEPTH_HI
    # out, so the whole square sat to one side of the net and any error in the
    # "into" direction slid it further. On the shootout's left hoop that left it
    # visibly off: review, "left zone is not aligned under the backboard... it's too
    # far away from the camera."
    #
    # Centering is also the more faithful model. A ball through the net falls
    # essentially straight down, so the water it enters is the patch AROUND the
    # point under the rim, not a patch beginning some distance out from it. As a
    # bonus it makes the sign of "into" nearly irrelevant, which removes the one
    # step in this measurement that can silently come out backwards.
    # Centered where balls from MADE shots actually land, measured (src/
    # dropcentre.py) rather than derived. Both derivations were wrong, in
    # different directions:
    #
    #   Centered on the point under the rim -- the "ball falls straight down"
    #   model -- recall collapsed from 64% to 36%. The ball does not land under
    #   the net; it carries through and out into the pool by about 1.2 rim
    #   widths, consistently, at both hoops across two sessions.
    #
    #   Offset outward but centered at along=0, it still sat wrong, because the
    #   landing is ALSO offset along the wall: +0.88 rim widths at the left hoop
    #   and -1.13 at the right, opposite signs. Nothing about the geometry
    #   predicts that; only the judged makes show it.
    ca, ci = drop.get("center", (0.0, (DEPTH_LO + DEPTH_HI) / 2))
    corners = []
    for s in (ca - HALF_W, ca + HALF_W):
        for d in (ci - HALF_D, ci + HALF_D):
            corners.append((px + ax * s * w + ix * d * w,
                            py + ay * s * w + iy * d * w))
    # Order them around the patch rather than by the loop, so it draws as a quad
    # and not as a bow tie.
    a, b, c, e = corners[0], corners[1], corners[3], corners[2]
    return [(int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
            (int(c[0]), int(c[1])), (int(e[0]), int(e[1]))]


def zone(rim, water=None, drop=None):
    """The drop zone, measured axes when the session has them."""
    return quad_on_plane(rim, drop) if drop else quad(rim, water)


def contains(poly, x, y):
    """Point in polygon, so the rule tests the shape that gets drawn."""
    n = len(poly)
    hit = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xx = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xx:
                hit = not hit
    return hit


def dropped(hits, rim, t_descent, window=WINDOW_S, water=None, drop=None):
    """Did any sighting land in the zone under the net, around the descent?

    Tests the ZONE, not a rectangle. It used to test box() -- an axis-aligned
    column -- while quad()/zone() was only what got drawn, so the review frames
    showed one region and the veto fired on another. That is exactly the drift
    posediag exists to prevent one file over, and it survived because a rectangle
    under a rim looks plausible enough in a picture.

    Falls back to the rectangle only when a session has no measured axes, which
    is the honest behavior for footage nobody has set up yet.
    """
    poly = zone(rim, water, drop) if (water or drop) else None
    bx1, by1, bx2, by2 = box(rim)
    for h in hits:
        if not (t_descent - 0.3 <= h["t"] <= t_descent + window):
            continue
        if poly:
            if contains(poly, h["x"], h["y"]):
                return True
        elif bx1 <= h["x"] <= bx2 and by1 <= h["y"] <= by2:
            return True
    return False


def clip_to_water(quad_pts, water_mask, center):
    """Pull any corner that is not on water back toward `center` until it is.

 The parallelogram follows the waterline correctly along
    its top edge, but the pool turns a corner under one hoop, so an end of the
    zone can still cross onto the deck.

    Trimming per corner rather than shrinking the whole box keeps as much real
    water as possible, which matters: this rule's value is its recall, and every
    square foot of water given up is a make it can no longer catch.
    """
    h, w = water_mask.shape[:2]

    def wet(pt):
        x, y = int(pt[0]), int(pt[1])
        return 0 <= x < w and 0 <= y < h and water_mask[y, x] > 0

    out = []
    for pt in quad_pts:
        if wet(pt):
            out.append((int(pt[0]), int(pt[1])))
            continue
        # Walk toward the center until the water starts, then stop.
        px, py = float(pt[0]), float(pt[1])
        cx, cy = float(center[0]), float(center[1])
        found = (int(cx), int(cy))
        for i in range(1, 41):
            f = i / 40.0
            q = (px + (cx - px) * f, py + (cy - py) * f)
            if wet(q):
                found = (int(q[0]), int(q[1]))
                break
        out.append(found)
    return out
