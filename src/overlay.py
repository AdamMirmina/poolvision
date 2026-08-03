"""The rig's geometry, burned into a clip for its whole duration.



That is a different job from the review still. The still marks one instant and
answers "who shot it"; this answers "where did the ball actually go", which needs
the regions present in every frame so any pause is readable. Without them a
paused frame shows a ball somewhere near a hoop and nothing to judge it against,
which is most of what a rim bounce looks like.

The camera does not move within a recording, so the geometry is identical on
every frame. It is drawn once at output scale and composited, rather than
redrawn per frame: one blend per frame instead of a dozen draw calls.

Thin lines and no text. This is a backdrop to watch the ball against, not a
diagram -- the review still is the diagram.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dropzone

RIM = (60, 235, 245)     # yellow, as on the stills
NET = (200, 200, 200)
DROP = (90, 90, 90)
ALPHA = 0.85             # solid enough to read over bright water


def layer(rig, out_w, out_h, src_w=3840, src_h=2160):
    """One BGR image of the geometry plus its mask, at output scale.

    Drawn at OUTPUT resolution rather than drawn big and shrunk, so a one-pixel
    line stays a one-pixel line instead of dissolving into gray.
    """
    import cv2
    import numpy as np

    img = np.zeros((out_h, out_w, 3), np.uint8)
    sx, sy = out_w / src_w, out_h / src_h

    def P(p):
        return (int(round(p[0] * sx)), int(round(p[1] * sy)))

    for hoop, rim in rig.rims.items():
        x1, y1, x2, y2 = rim
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rw, rh = (x2 - x1) / 2, max(6, (y2 - y1) / 2)
        tilt = (rig.tilt or {}).get(hoop, 0.0)
        cv2.ellipse(img, P((cx, cy)),
                    (max(2, int(rw * sx)), max(2, int(rh * sy))),
                    tilt, 0, 360, RIM, 2, cv2.LINE_AA)

        b = dropzone.box(rim)
        cv2.rectangle(img, P((b[0], b[1])), P((b[2], b[3])), NET, 1, cv2.LINE_AA)

        q = dropzone.zone(rim, (rig.water or {}).get(hoop), (rig.drop or {}).get(hoop))
        pts = np.array([P(p) for p in q], np.int32)
        cv2.polylines(img, [pts], True, DROP, 2, cv2.LINE_AA)

    mask = (img.sum(axis=2) > 0)
    return img, mask


def burn(frame, lay):
    """Composite the layer onto a frame, in place."""
    img, mask = lay
    if frame.shape[:2] != img.shape[:2]:
        return frame
    frame[mask] = (ALPHA * img[mask] + (1 - ALPHA) * frame[mask]).astype(frame.dtype)
    return frame
