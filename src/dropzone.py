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

# The column under the rim, in rim widths. Swept: a half-width of 0.9 catches 34
# of 37 makes, and tightening it to 0.5 drops that to 16 while barely improving
# precision, which is the wrong trade for a rule whose whole value is its recall.
HALF_W = 0.9
DEPTH_LO = 0.3
DEPTH_HI = 1.6
WINDOW_S = 1.6


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


def dropped(hits, rim, t_descent, window=WINDOW_S):
    """Did any sighting land in the column below the rim, around the descent?"""
    bx1, by1, bx2, by2 = box(rim)
    for h in hits:
        if not (t_descent - 0.3 <= h["t"] <= t_descent + window):
            continue
        if bx1 <= h["x"] <= bx2 and by1 <= h["y"] <= by2:
            return True
    return False
