"""Tests for the footage-independent core.

Shot geometry, cap color matching and tracklet assignment are pure logic, so
they can be verified now against synthetic cases. Real footage will tune the
thresholds; it shouldn't change any of these behaviors.

Run: python tests/test_core.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caps import CapColor, calibrate, classify  # noqa: E402
from identity import Tracklet, assign  # noqa: E402
from shots import BallPoint, Hoop, detect_makes  # noqa: E402

HOOP = Hoop(x1=100, y1=100, x2=160, y2=140)
failures: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  pass  {name}")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        failures.append(name)


def drop(start_frame: int, x: float = 130) -> list[BallPoint]:
    """A clean swish: above the hoop, through it, out the bottom."""
    return [BallPoint(start_frame + i, x, 40 + i * 25) for i in range(7)]


print("shots")
check("clean make", len(detect_makes(drop(0), HOOP)), 1)

# Thrown across the frame at hoop height. Never above, never below: not a make.
across = [BallPoint(i, 40 + i * 20, 120) for i in range(10)]
check("horizontal pass is not a make", len(detect_makes(across, HOOP)), 0)

# Comes down, hits the rim, bounces back up. The upward leg must disqualify it.
brick = ([BallPoint(i, 130, 40 + i * 25) for i in range(4)]
         + [BallPoint(4 + i, 130, 130 - i * 25) for i in range(4)])
check("rim bounce-out is not a make", len(detect_makes(brick, HOOP)), 0)

# Ball drops well to the side of the hoop.
wide = [BallPoint(i, 300, 40 + i * 25) for i in range(7)]
check("miss wide is not a make", len(detect_makes(wide, HOOP)), 0)

# Gappy track: the ball is lost behind splash mid-flight. Still a make.
gappy = [BallPoint(0, 130, 40), BallPoint(1, 130, 65), BallPoint(5, 130, 190)]
check("make survives a tracking gap", len(detect_makes(gappy, HOOP)), 1)

# Two genuine makes far apart stay two.
check("two separate makes", len(detect_makes(drop(0) + drop(200), HOOP)), 2)
# Two passes 20 frames apart is physically one basket seen twice (the ball
# re-entering the region below the net), so it must collapse.
check("near-duplicates collapse to one", len(detect_makes(drop(0) + drop(20), HOOP)), 1)
check("empty track", detect_makes([], HOOP), [])

print("caps")
REFS = calibrate({
    "red": [(2, 220, 200), (178, 210, 190)],      # straddles the 0/180 wrap
    "cyan": [(90, 200, 210)],
    "lime": [(40, 200, 200)],
    "white": [(0, 10, 245)],
    "black": [(0, 10, 20)],
})
by_name = {r.name: r for r in REFS}

# Red must average across the hue wrap to ~0, not to the middle of the circle.
check("red hue wraps correctly", by_name["red"].h < 5 or by_name["red"].h > 175, True)
check("white is achromatic", by_name["white"].is_achromatic, True)
check("cyan is chromatic", by_name["cyan"].is_achromatic, False)

check("matches red", classify((1, 215, 195), REFS).name, "red")
check("matches cyan", classify((92, 190, 200), REFS).name, "cyan")
check("shaded cyan still cyan", classify((92, 170, 120), REFS).name, "cyan")
check("white not confused with a color", classify((0, 8, 240), REFS).name, "white")
check("black separated from white", classify((0, 8, 25), REFS).name, "black")

# A blown-out highlight on a colored cap reads near-white; it must not be
# matched to a chromatic reference on hue noise alone.
check("glare highlight goes achromatic", classify((150, 5, 250), REFS).name, "white")

# Two references close in hue should report low confidence rather than a
# confident coin-flip.
close = calibrate({"a": [(60, 200, 200)], "b": [(64, 200, 200)]})
check("ambiguous colors report low confidence", classify((62, 200, 200), close).confidence < 0.3, True)
check("no references", classify((60, 200, 200), []), None)

print("identity")
C2P = {"red": "hand", "cyan": "ben", "lime": "another player"}

solo = [Tracklet(1, 0, 100, {"red": 5.0})]
check("clear tracklet assigns", assign(solo, C2P)[0].player, "hand")

# Overlapping tracklets can't be the same person even if color votes disagree.
conflict = [
    Tracklet(1, 0, 100, {"red": 9.0, "cyan": 1.0}),
    Tracklet(2, 50, 150, {"red": 4.0, "cyan": 3.0}),
]
res = {a.tracklet_id: a.player for a in assign(conflict, C2P)}
check("strongest claim wins the color", res[1], "hand")
check("overlapping tracklet gets someone else", res[2] != "hand", True)

# Elimination: two overlapping tracklets, one confident, one with no color seen
# at all. The second is resolved by who's left.
elim = [
    Tracklet(1, 0, 100, {"red": 9.0}),
    Tracklet(2, 10, 90, {}),
]
res2 = {a.tracklet_id: a.player for a in assign(elim, {"red": "hand", "cyan": "ben"})}
check("elimination resolves the unseen tracklet", res2[2], "ben")

# Same color twice but NOT overlapping is fine: one person, two tracks.
seq = [Tracklet(1, 0, 50, {"red": 5.0}), Tracklet(2, 60, 100, {"red": 5.0})]
res3 = {a.tracklet_id: a.player for a in assign(seq, C2P)}
check("non-overlapping tracks share a player", (res3[1], res3[2]), ("hand", "hand"))

# Ambiguous with nothing to eliminate against -> None, not a guess.
amb = [
    Tracklet(1, 0, 100, {}),
    Tracklet(2, 10, 90, {}),
    Tracklet(3, 20, 80, {}),
]
check("hopeless cases return None", all(a.player is None for a in assign(amb, C2P)), True)

# ---- picking the ball that's actually in play -------------------------------
# Regression for the real failure in the 2026-07-27 run: a ball resting on the
# deck was detected at 0.88 in all 1,950 frames while the ball being played with
# was blurred and weaker, so "keep the most confident ball" tracked furniture for
# 65 seconds and reported a 100% detection rate for it.
from analyze import pick_moving_ball  # noqa: E402

parked_plus_play = {}
for f in range(200):
    balls = [(424.0, 2084.0, 0.88)]           # never moves, always confident
    if f % 3 == 0:
        balls.append((500.0 + f * 12, 900.0 - f * 2, 0.31))   # in play, weaker
    parked_plus_play[f] = balls

track, dropped = pick_moving_ball(parked_plus_play)
check("the parked ball is dropped", dropped, 200)
check("the ball in play is kept", len(track), 67)
check("no sample sits on the parked ball", all(abs(p.x - 424) > 50 for p in track), True)

# A window with no stationary ball must be left alone.
only_play = {f: [(500.0 + f * 12, 900.0 - f * 2, 0.31)] for f in range(60)}
t2, d2 = pick_moving_ball(only_play)
check("a clean window is untouched", (len(t2), d2), (60, 0))

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all passed")
