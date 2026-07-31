"""Drop detections of balls that never move.

A ball lying on the deck is sharp, unobstructed and stationary, so the detector
finds it in almost every frame. The ball in play is wet, fast and blurred, so it
is found in far fewer. Any count that treats them equally is dominated by the
one nobody is using.

This has now bitten twice, in two different places:

  2026-07-27 footage, tracking stage -- a parked ball won all 1,950 frames of a
  window and the pipeline reported "100% ball detection" for something that moved
  four pixels in 65 seconds.

  2026-07-29 footage, clustering stage -- a ball on the ground by the right hoop
  produced 3,202 of 7,013 detections (present in 54% of frames) and helped turn
  9.4 minutes of shooting into "397 shots".

The first was fixed inside analyze.py and the fix did not reach rimwatch/clips,
which were written afterwards and independently. Hence this module: one
implementation, imported by everything, so the next new code path gets it for
free instead of rediscovering it.

The test is occupancy, not motion: bin detections into coarse cells and treat a
cell that is occupied across a large share of frames as furniture. A ball in play
passes through a cell for a handful of frames; a ball at rest sits in one for
the entire recording.
"""

from __future__ import annotations

CELL_PX = 40
PARKED_FRAC = 0.25


def parked_cells(hits: list[dict], cell: int = CELL_PX, frac: float = PARKED_FRAC) -> set[tuple[int, int]]:
    """Grid cells that hold a detection in more than `frac` of observed frames."""
    frames = {h["frame"] for h in hits}
    if not frames:
        return set()
    seen: dict[tuple[int, int], set[int]] = {}
    for h in hits:
        key = (int(h["x"] // cell), int(h["y"] // cell))
        seen.setdefault(key, set()).add(h["frame"])
    limit = frac * len(frames)
    return {k for k, fs in seen.items() if len(fs) > limit}


def drop_parked(hits: list[dict], cell: int = CELL_PX, frac: float = PARKED_FRAC) -> tuple[list[dict], int]:
    """Return (hits worth keeping, how many were dropped)."""
    dead = parked_cells(hits, cell, frac)
    if not dead:
        return hits, 0
    kept = [h for h in hits if (int(h["x"] // cell), int(h["y"] // cell)) not in dead]
    return kept, len(hits) - len(kept)
