"""Player identity from colored swim caps.

Color classification, not learned re-identification. Shirtless, wet,
half-submerged people at distance don't carry the appearance cues re-ID models
depend on, so the caps put that signal back deliberately.

Two things make this work rather than be brittle:

1. **Per-party calibration.** Colors are matched against a reference clip shot
   in the same light as the game, never a fixed color table. Outdoor light
   shifts hue badly between sun and cloud, and the literature is blunt that
   color thresholding is very sensitive to illumination.

2. **Achromatic colors are handled separately.** White, gray and black have no
   meaningful hue -- their hue channel is basically noise. Comparing them by hue
   like everything else is the classic way this breaks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CapColor:
    """A calibrated reference color, in OpenCV HSV ranges (H 0-179, S/V 0-255)."""
    name: str
    h: float
    s: float
    v: float

    @property
    def is_achromatic(self) -> bool:
        """White/gray/black: saturation too low for hue to mean anything."""
        return self.s < 60


@dataclass(frozen=True)
class Match:
    name: str
    confidence: float  # 0-1; low means the observation is ambiguous, not wrong


def calibrate(samples: dict[str, list[tuple[float, float, float]]]) -> list[CapColor]:
    """Build reference colors from the calibration clip.

    samples: {color name: [(h, s, v), ...]} sampled from cap pixels in the
    close-up reference footage. Uses the median, which shrugs off the specular
    highlights that glossy silicone throws in direct sun.
    """
    refs: list[CapColor] = []
    for name, pts in samples.items():
        if not pts:
            continue
        refs.append(CapColor(
            name=name,
            h=_median_hue([p[0] for p in pts]),
            s=_median([p[1] for p in pts]),
            v=_median([p[2] for p in pts]),
        ))
    return refs


def classify(hsv: tuple[float, float, float], refs: list[CapColor]) -> Match | None:
    """Match one observed cap patch against the calibrated colors.

    Confidence is the margin between best and runner-up, so two colors that are
    genuinely hard to tell apart report low confidence rather than a confident
    guess. Downstream, low-confidence observations should be outvoted by the rest
    of the tracklet rather than trusted on their own.
    """
    if not refs:
        return None
    scored = sorted(((_distance(hsv, r), r) for r in refs), key=lambda t: t[0])
    best_d, best = scored[0]
    if len(scored) == 1:
        return Match(best.name, 1.0)
    runner_d = scored[1][0]
    # Identical distances -> no information. A big gap -> confident.
    margin = (runner_d - best_d) / max(runner_d, 1e-6)
    return Match(best.name, max(0.0, min(1.0, margin)))


def _distance(hsv: tuple[float, float, float], ref: CapColor) -> float:
    h, s, v = hsv
    observed_achromatic = s < 60

    # Never match a colored cap to white/black or vice versa. Mixing these is
    # the main way naive color matching goes wrong under glare, since a blown-out
    # highlight on a red cap reads as near-white.
    if observed_achromatic != ref.is_achromatic:
        return 1e6

    if ref.is_achromatic:
        # Only brightness separates white / gray / black.
        return abs(v - ref.v)

    # Hue dominates, with saturation and value as weak tiebreakers so that a
    # shaded version of a color still matches the sunlit reference.
    return _hue_gap(h, ref.h) * 4.0 + abs(s - ref.s) * 0.35 + abs(v - ref.v) * 0.2


def _hue_gap(a: float, b: float) -> float:
    """Circular hue distance on OpenCV's 0-179 scale."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _median_hue(hues: list[float]) -> float:
    """Median on a circle. A plain median is wrong for red, which straddles the
    0/180 wrap and would average to cyan."""
    if not hues:
        return 0.0
    xs = sum(math.cos(h * math.pi / 90) for h in hues)
    ys = sum(math.sin(h * math.pi / 90) for h in hues)
    ang = math.atan2(ys, xs) * 90 / math.pi
    return ang % 180
