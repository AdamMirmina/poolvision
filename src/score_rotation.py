"""Score attribution against the rotation, with no labeling by anyone.

This is the first footage in the project where the correct answer exists without
someone judging clips. review, one player and another player shot in strict rotation -- blue, pink,
white, repeating -- so shot k was taken by (k mod 3). Every previous number came
from ten notes review said while doing something else, which was never enough to
tell a real improvement from luck. It is how five different rules were tried and
all scored the same.

Two things are scored separately, because fusing them is what made every earlier
score uninterpretable. A hit needed the right person AND the right color, and
those fail independently, so two 30% steps read as 0 of 10 no matter which half
improved:

  attribution -- does the chosen person follow the rotation
  color      -- does the cap read as that person's actual cap

The rotation is anchored rather than assumed: the offset that best fits the
read colors is chosen, since the first detected shot is not necessarily the
first shot taken.

    python src/score_rotation.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The rotation review found, in order.
ROTATION = [("hand", "blue"), ("one player", "pink"), ("another player", "white")]


def color_name(h):
    if h is None:
        return None
    if h == -1:
        return "white"
    if h == -2:
        return "black"
    if 30 <= h < 75:
        return "yellow"
    if 75 <= h < 165:
        return "green"
    if 165 <= h < 265:
        return "blue"
    return "pink"


def main():
    src = ROOT / "out/shooter_IMG_2528.json"
    if not src.exists():
        print("no attribution output yet")
        return 1
    rows = [r for r in json.loads(src.read_text(encoding="utf-8")) if r.get("stage") == "attributed"]
    rows.sort(key=lambda r: r["t"])
    print(f"{len(rows)} shots attributed\n")
    if not rows:
        return 0

    caps = [color_name(r.get("hue")) for r in rows]
    read = sum(1 for c in caps if c)
    print(f"cap color read on {read} of {len(rows)}")
    print("colors seen:", dict(Counter(c for c in caps if c)))

    # Anchor the rotation: the first DETECTED shot need not be the first shot
    # taken, so try each offset and keep the one that fits best. Reported with
    # the runner-up, because an offset that wins by one is not an anchor.
    scores = []
    for off in range(3):
        hit = tot = 0
        for i, c in enumerate(caps):
            if not c:
                continue
            tot += 1
            if c == ROTATION[(i + off) % 3][1]:
                hit += 1
        scores.append((hit / tot if tot else 0.0, off, hit, tot))
    scores.sort(reverse=True)
    best, off, hit, tot = scores[0]
    print(f"\nbest rotation offset {off}: {hit} of {tot} colors match the expected shooter ({best:.0%})")
    print(f"runner-up offset {scores[1][1]}: {scores[1][2]} of {scores[1][3]} ({scores[1][0]:.0%})")
    if best - scores[1][0] < 0.15:
        print("  the offsets are too close to call -- treat the anchor as unproven")

    print("\nby expected shooter:")
    per = {}
    for i, c in enumerate(caps):
        who, want = ROTATION[(i + off) % 3]
        d = per.setdefault(who, {"want": want, "hit": 0, "tot": 0, "got": Counter()})
        if c:
            d["tot"] += 1
            d["got"][c] += 1
            if c == want:
                d["hit"] += 1
    for who, d in per.items():
        pct = 100 * d["hit"] / d["tot"] if d["tot"] else 0
        print(f"  {who:>5} (should read {d['want']:>5}): {d['hit']}/{d['tot']} = {pct:.0f}%   saw {dict(d['got'])}")

    # A run of consecutive shots landing on the right person is far stronger
    # evidence than the same total scattered, because the rotation is a sequence.
    runs, cur = [], 0
    for i, c in enumerate(caps):
        if c and c == ROTATION[(i + off) % 3][1]:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    if runs:
        print(f"\nlongest correct streak: {max(runs)} consecutive shots")
        print(f"streaks: {sorted(runs, reverse=True)[:8]}")
    print("\nchance for three shooters is 33%. Anything near that is the pipeline guessing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
