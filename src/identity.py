"""Assign tracklets to players.

The key move: identity is decided per TRACKLET, not per frame. Any single frame
of someone half-submerged is often ambiguous, but a four-second track usually
contains at least one clear look at the cap, and that one look labels the whole
sequence. Published jersey-number accuracy is quoted on tracklets for exactly
this reason -- frame-level numbers are much worse.

Two constraints make this better than matching each tracklet independently:

1. **Closed set.** We know exactly who's in the pool, because the app knows who
   checked in and which cap color each was given.
2. **One person can't be in two places.** Tracklets that overlap in time must be
   different people, so assigning them together (rather than greedily one at a
   time) lets a confident assignment resolve an ambiguous one by elimination.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Tracklet:
    """One continuous run of a tracked person, with whatever cap-color
    observations were gathered along it."""
    id: int
    start_frame: int
    end_frame: int
    # color name -> summed confidence across the tracklet
    votes: dict[str, float] = field(default_factory=dict)

    def overlaps(self, other: "Tracklet") -> bool:
        return self.start_frame <= other.end_frame and other.start_frame <= self.end_frame


@dataclass(frozen=True)
class Assignment:
    tracklet_id: int
    player: str | None  # None = genuinely couldn't tell; ask the human
    confidence: float


def assign(
    tracklets: list[Tracklet],
    color_to_player: dict[str, str],
    *,
    min_confidence: float = 0.15,
) -> list[Assignment]:
    """Assign each tracklet to a player.

    color_to_player: the cap colors handed out at check-in, e.g.
                     {"red": "hand", "cyan": "ben"}.
    min_confidence:  below this, return None rather than guess. An unassigned
                     tracklet is a question for the review screen; a wrongly
                     assigned one is a silently wrong stat, which is worse.
    """
    out: list[Assignment] = []
    for group in _overlapping_groups(tracklets):
        out.extend(_assign_group(group, color_to_player, min_confidence))
    return sorted(out, key=lambda a: a.tracklet_id)


def _overlapping_groups(tracklets: list[Tracklet]) -> list[list[Tracklet]]:
    """Cluster tracklets into sets that overlap in time, transitively. The
    one-person-one-place constraint only applies within a group, so solving each
    separately is both correct and much cheaper than one global solve."""
    ordered = sorted(tracklets, key=lambda t: t.start_frame)
    groups: list[list[Tracklet]] = []
    for t in ordered:
        for g in groups:
            if any(t.overlaps(x) for x in g):
                g.append(t)
                break
        else:
            groups.append([t])
    return groups


def _assign_group(
    group: list[Tracklet],
    color_to_player: dict[str, str],
    min_confidence: float,
) -> list[Assignment]:
    """Within one time-overlapping group, each player may be used at most once."""
    # Score every (tracklet, player) pair from the tracklet's color votes.
    scores: dict[int, dict[str, float]] = {}
    for t in group:
        total = sum(t.votes.values())
        per_player: dict[str, float] = defaultdict(float)
        if total > 0:
            for color, weight in t.votes.items():
                player = color_to_player.get(color)
                if player:
                    per_player[player] += weight / total
        scores[t.id] = dict(per_player)

    # Greedy over globally-best pairs. With at most ~8 players and a handful of
    # concurrent tracklets, this matches what Hungarian would give while keeping
    # the dependency list to nothing.
    taken_players: set[str] = set()
    assigned: dict[int, Assignment] = {}
    pairs = sorted(
        ((score, tid, player) for tid, ps in scores.items() for player, score in ps.items()),
        key=lambda p: -p[0],
    )
    for score, tid, player in pairs:
        if tid in assigned or player in taken_players or score < min_confidence:
            continue
        assigned[tid] = Assignment(tid, player, round(score, 3))
        taken_players.add(player)

    # Anything left is either unseen or too ambiguous. If exactly one tracklet
    # and one player remain, elimination settles it -- that's the whole point of
    # solving the group together.
    leftover = [t for t in group if t.id not in assigned]
    remaining = [p for p in set(color_to_player.values()) if p not in taken_players]
    if len(leftover) == 1 and len(remaining) == 1:
        assigned[leftover[0].id] = Assignment(leftover[0].id, remaining[0], 0.5)
        leftover = []

    for t in leftover:
        assigned[t.id] = Assignment(t.id, None, 0.0)

    return list(assigned.values())
