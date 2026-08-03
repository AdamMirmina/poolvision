"""Find the three-point posts in the frame being judged, not in a stored guess.



Right, and it is not a tuning problem: the posts stand in floats that drift on
the water all afternoon. A single stored anchor is correct at one moment of one
recording and slowly wrong everywhere else, so the boundary has to be measured
per shot.

Found by what makes them distinctive rather than by a template: a white vertical
pole standing in a blue float. The float gives the search a place to look, and
the pole's lowest white pixel inside it is the base -- the same point the stored
anchors were read off a 5.5x zoom by hand.

Falls back to the rig's stored anchor when they cannot be found, which is the
honest answer for a frame where they are occluded or out of shot. Returning a
guess would be worse than declining, because the boundary decides whether a shot
is worth 2 or 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent

# The float: a strong blue, more saturated than the pool water behind it.
FLOAT_H = (95, 125)
FLOAT_S = 110
FLOAT_V = 60
# The pole: bright and unsaturated.
POLE_S = 70
POLE_V = 185
MIN_FLOAT_AREA = 2500
SEARCH_PAD = 90


def find(frame, near=None, radius=520):
    """Post bases in this frame, as [(x, y)], nearest-first from `near`.

    `near` is where they were last seen, which keeps the search off the many
    other blue things in a pool scene. Without it the whole frame is searched.
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    floats = ((h >= FLOAT_H[0]) & (h <= FLOAT_H[1]) & (s >= FLOAT_S)
              & (v >= FLOAT_V)).astype(np.uint8)
    floats = cv2.morphologyEx(floats, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    if near is not None:
        m = np.zeros_like(floats)
        cv2.circle(m, (int(near[0]), int(near[1])), radius, 1, -1)
        floats = floats * m

    cnts, _ = cv2.findContours(floats, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pole = ((s <= POLE_S) & (v >= POLE_V)).astype(np.uint8)

    found = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_FLOAT_AREA:
            continue
        x, y, w, hh = cv2.boundingRect(c)
        # The pole rises out of the float, so look in a band above and across it.
        x0 = max(0, x - SEARCH_PAD)
        x1 = min(frame.shape[1], x + w + SEARCH_PAD)
        y0 = max(0, y - int(2.6 * hh))
        y1 = min(frame.shape[0], y + hh)
        sub = pole[y0:y1, x0:x1]
        if sub.size == 0:
            continue
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        pc, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for p in pc:
            px, py, pw, ph = cv2.boundingRect(p)
            # Tall and narrow, and tall enough to be a pole rather than a splash.
            if ph < 60 or pw > ph * 0.8:
                continue
            if best is None or ph > best[3]:
                best = (px, py, pw, ph)
        if best is None:
            continue
        px, py, pw, ph = best
        # The BASE: where the pole enters the float, not the box center and not
        # the blue base beside it. That distinction has been got wrong before.
        found.append((float(x0 + px + pw / 2), float(y0 + py + ph)))

    if near is not None:
        found.sort(key=lambda p: (p[0] - near[0]) ** 2 + (p[1] - near[1]) ** 2)
    return found


def anchors_for(frame, rig):
    """{side: (anchor, direction)} for this frame, stored values as the fallback.

    The direction is kept from the rig: the boundary runs the length of the pool
    and that does not change when a float drifts a foot. Only where it passes
    through does.
    """
    sides = list((rig.three_lines or {}).items())
    out = {side: tl for side, tl in sides}
    if not sides:
        return out

    # ONE search, then assign distinct posts. Searching per side found the same
    # float twice: the two anchors sit about 160px apart, so each side's nearest
    # blue thing is the same one, and both boundaries collapsed onto one post.
    mid = (sum(tl[0][0] for _, tl in sides) / len(sides),
           sum(tl[0][1] for _, tl in sides) / len(sides))
    cands = find(frame, near=mid, radius=700)
    if not cands:
        return out

    # Nearest first, without replacement, so one post cannot serve both sides.
    pairs = sorted(
        ((( tl[0][0] - c[0]) ** 2 + (tl[0][1] - c[1]) ** 2, side, i)
         for side, tl in sides for i, c in enumerate(cands)),
        key=lambda z: z[0])
    used_side, used_c = set(), set()
    for d2, side, i in pairs:
        if side in used_side or i in used_c:
            continue
        # A post found far from where one was last seen is a different post.
        # Accepting it would move the boundary by metres and look plausible.
        if d2 > 420 ** 2:
            continue
        used_side.add(side)
        used_c.add(i)
        out[side] = ((cands[i][0], cands[i][1]), dict(sides)[side][1])
    return out
