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
    """One camera setup: the rim boxes, and the generous crop fed to the detector.

    `rims` are tight on the hoop, for geometry. `crops` are much wider, because
    the detector needs the ball's approach and its exit in frame, and a tight
    crop on the rim hides exactly the evidence a human reviewer needs too.
    """
    rims: dict[str, tuple[int, int, int, int]]
    crops: dict[str, tuple[int, int, int, int]]


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

RIGS: dict[str, Rig] = {
    "IMG_2403.MOV": Rig(
        rims=_S0727_RIMS,
        crops={"left": (188, 453, 1248, 1293), "right": (2860, 178, 3840, 1038)},
    ),
}
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
