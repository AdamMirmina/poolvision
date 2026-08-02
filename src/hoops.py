"""Where the rims are, per recording.

These were hardcoded as one pair of boxes until 2026-07-30, on the assumption
that a camera at a fixed window stays put. It doesn't: the 2026-07-29 session
was framed noticeably differently from 2026-07-27, and the old coordinates
pointed at open water. Running the pipeline on that would have burned hours and
returned zero shots, and "zero shots detected" looks exactly like a detector
problem rather than a coordinates problem -- which is the expensive kind of
wrong.

So: measured per video, and a recording with no entry here fails loudly instead
of silently using someone else's numbers.

To add a session: grab a mid-video frame, find the rims, and check the boxes
land on them by drawing and LOOKING. Two minutes, and it protects an overnight
run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rig:
    """One camera setup: the rim boxes, and the two crops taken around them.

    `rims` are tight on the hoop, for geometry.

    `crops` are wide, and they are what a person watches. A clip cropped tight on
    the rim shows a ball appearing and disappearing with no shooter and no
    approach, which is useless for judging who shot or whether it went in.

    `dets` are tight, and they are what the detector sees. These were the same
    box until 2026-08-01, and sharing it was costing accuracy and time at once: a
    1063x888 crop fed to a 640 model is downscaled to 0.60, so a 35px ball
    arrives as 21px, right at the edge of what the detector holds on to. Tight
    crops arrive near native size AND stack into a single inference per frame
    instead of one per hoop.

    `pool` is the part of the frame anything real can happen in. It filters both
    ball and person detections, and it lived as a hardcoded constant in
    shooter.py until 2026-08-01, when measuring it against this session's frame
    showed it excluded the ENTIRE left hoop and the whole bottom-left of the
    water, where several people stand. That does not fail loudly; it silently
    drops those detections and reads in the morning as "attribution found
    nothing." Exactly the failure this module exists to prevent, one file over.
    """
    rims: dict[str, tuple[int, int, int, int]]
    crops: dict[str, tuple[int, int, int, int]]
    dets: dict[str, tuple[int, int, int, int]] | None = None
    pool: tuple[int, int, int, int] = (1150, 250, 3500, 1750)

    def det_boxes(self) -> dict[str, tuple[int, int, int, int]]:
        """The detector's crops, falling back to the review crops for old rigs."""
        return self.dets if self.dets else self.crops


def _crop_around(rim: tuple[int, int, int, int], pad_x: int, pad_y: int,
                 w: int = 3840, h: int = 2160) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rim
    return (max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y))


# 2026-07-27 session (IMG_2403). First footage; the camera sat lower/wider.
_S0727_RIMS = {
    "left": (615, 838, 820, 928),
    "right": (3535, 562, 3717, 655),
}

# 2026-07-29 session (IMG_2480-2483). Camera moved: the whole scene shifted and
# both hoops sit higher in frame than they did two days earlier.
_S0729_RIMS = {
    "left": (897, 992, 1065, 1082),
    "right": (3405, 640, 3577, 707),
}

# 2026-08-01 session (IMG_2528 shootout, IMG_2529 game). Camera moved again: the
# whole scene is framed from further round the corner.
#
# Measured off an averaged frame -- several frames blended so swimmers and
# splashes average away and only the fixed furniture remains. Three hand-drawn
# passes got boxes that contained both rims but sat loose around them, and loose
# is not good enough: the box center feeds the strongest make/miss feature and
# the box width is the unit everything else is normalized by, so slack in the box
# is slack in every number downstream. The hand-drawn right box was 204x108 for a
# rim that is actually 164x60.
#
# So the ring is found by its own color instead. The right rim fits an ellipse
# cleanly. The LEFT does not, and the reason is worth keeping: its red mask
# catches only part of the ring (the far arc washes out against the deck), and
# cv2.fitEllipse on a partial arc extrapolates a much bigger ellipse than the one
# you can see -- 265px wide for a contour spanning 146px. Its box is measured off
# the visible ring in a 4x zoom rather than trusting that fit.
_S0801_RIMS = {
    "left": (837, 1026, 1040, 1114),
    "right": (3378, 676, 3542, 736),
}

RIGS: dict[str, Rig] = {
    "IMG_2403.MOV": Rig(
        rims=_S0727_RIMS,
        crops={"left": (188, 453, 1248, 1293), "right": (2860, 178, 3840, 1038)},
    ),
}
for _name in ("IMG_2528.MOV", "IMG_2529.MOV"):
    RIGS[_name] = Rig(
        rims=_S0801_RIMS,
        crops={k: _crop_around(v, 430, 400) for k, v in _S0801_RIMS.items()},
        # 250x180 of padding is about five ball diameters clear of the rim on
        # every side, which is all the detector needs to see a descent through.
        dets={k: _crop_around(v, 250, 180) for k, v in _S0801_RIMS.items()},
        # Measured off the averaged frame by segmenting the water, then padded
        # out to take in the left hoop and the heads and arms that rise above
        # the waterline. Everything left of this is grass and deck.
        pool=(832, 0, 3840, 2160),
    )
for _name in ("IMG_2480.MOV", "IMG_2481.MOV", "IMG_2482.MOV", "IMG_2483.MOV"):
    RIGS[_name] = Rig(
        rims=_S0729_RIMS,
        crops={k: _crop_around(v, 430, 400) for k, v in _S0729_RIMS.items()},
    )


def rig_for(video: str) -> Rig:
    """Look up a recording's camera setup by filename.

    Raises rather than guessing. A wrong rim box produces a confident-looking
    run that finds nothing, and that failure is far more expensive to diagnose
    than a missing-key error at startup.
    """
    from pathlib import Path

    key = Path(video).name
    if key not in RIGS:
        raise KeyError(
            f"No hoop coordinates for {key}. Measure them off a mid-video frame "
            f"and add a RIGS entry -- do NOT fall back to another session's "
            f"numbers. Known: {', '.join(sorted(RIGS))}"
        )
    return RIGS[key]
