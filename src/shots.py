"""Made-basket detection from a ball track and a fixed hoop region.

Pure geometry, no learning. The camera doesn't move, so the hoop is a rectangle
defined by hand once (see hoop.py) rather than something to detect per frame.

A make is: the ball is above the hoop, passes through it, and comes out below,
traveling downward throughout. That last part is what separates a make from the
ball being thrown across the frame at hoop height, or bouncing up off the rim.

Gaps in the track are expected and fine. The ball gets lost behind splash and
bodies constantly, so this works on whatever samples exist rather than requiring
a continuous trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hoop:
    """Hoop bounding box in pixels, defined once on a reference frame."""
    x1: float
    y1: float
    x2: float
    y2: float

    def contains_x(self, x: float) -> bool:
        return self.x1 <= x <= self.x2

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class BallPoint:
    frame: int
    x: float
    y: float


@dataclass(frozen=True)
class Make:
    """A detected made basket. `frame` is when the ball passed through."""
    frame: int
    entry_frame: int
    exit_frame: int


def detect_makes(
    track: list[BallPoint],
    hoop: Hoop,
    *,
    max_pass_frames: int = 45,
    min_gap_frames: int = 60,
    above_margin: float = 0.0,
    below_margin: float = 0.0,
) -> list[Make]:
    """Find made baskets in a ball track.

    track:            ball positions, any gaps allowed, need not be sorted
    max_pass_frames:  how long the ball may take to go from above to below the
                      hoop and still count as one pass. Too generous and a ball
                      that drifts down outside the net registers; too tight and
                      a slow swish is missed.
    min_gap_frames:   makes closer together than this are treated as one. Guards
                      against a single basket being counted twice when the ball
                      re-enters the region on the way out of the net.
    above/below_margin: extra pixels the ball must clear above/below the hoop
                      before a pass counts. Raise these if the rim area is noisy.
    """
    pts = sorted(track, key=lambda p: p.frame)
    if not pts:
        return []

    above_y = hoop.y1 - above_margin
    below_y = hoop.y2 + below_margin

    makes: list[Make] = []
    for i, p in enumerate(pts):
        # Anchor on the ball being clearly BELOW the hoop and horizontally
        # aligned with it -- that point IS the exit. Requiring a separate later
        # sample to prove the exit breaks on gappy tracks, where the first frame
        # after the splash is the only evidence the ball came out at all.
        if not (hoop.contains_x(p.x) and p.y > below_y):
            continue

        # Look back for the ball clearly ABOVE the hoop, horizontally aligned.
        entry = None
        for q in reversed(pts[:i]):
            if p.frame - q.frame > max_pass_frames:
                break
            if q.y < above_y and hoop.contains_x(q.x):
                entry = q
                break
        if entry is None:
            continue

        # The whole pass must be downward. Without this, a ball bouncing up off
        # the rim and back down, or thrown horizontally past the hoop, registers.
        span = [q for q in pts if entry.frame <= q.frame <= p.frame]
        if not _monotonic_down(span):
            continue

        makes.append(Make(frame=p.frame, entry_frame=entry.frame, exit_frame=p.frame))

    return _dedupe(makes, min_gap_frames)


def _monotonic_down(span: list[BallPoint], tolerance: float = 8.0) -> bool:
    """True if the ball is broadly descending. `tolerance` absorbs jitter in the
    detector without allowing a real upward bounce."""
    for a, b in zip(span, span[1:]):
        if b.y < a.y - tolerance:
            return False
    return True


def _dedupe(makes: list[Make], min_gap: int) -> list[Make]:
    out: list[Make] = []
    for m in makes:
        if out and m.frame - out[-1].frame < min_gap:
            continue
        out.append(m)
    return out
