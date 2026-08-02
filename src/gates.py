"""Which gate is throwing shots away, measured rather than guessed.

Attribution answers a shot only when someone clears three gates in a row: seen
often enough, a release-shaped gap, and hands above their own head. Five of nine
shots came back "nobody released with hands up", and that sentence does not say
WHICH of the three did the rejecting. Loosening the wrong one buys nothing and
loosening all of them turns the pick into a coin flip.

Runs entirely off cached scans, so trying an idea costs a second instead of the
forty minutes a fresh pass of the models takes.

    python src/gates.py --video IMG_2482.MOV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shooter

ROOT = Path(__file__).resolve().parent.parent


def load(video):
    out = {}
    for f in sorted(shooter.SCANS.glob(f"{video.replace('.MOV','')}_*.json")):
        n = int(f.stem.rsplit("_", 1)[1])
        d = json.loads(f.read_text(encoding="utf-8"))
        per = {int(k): {"series": [tuple(s) for s in v["series"]]}
               for k, v in d["per_person"].items()}
        out[n] = (d["ball_track"], per)
    return out


def release_with(series, touch, opened, window, slack):
    """find_release with the thresholds exposed, so they can be swept."""
    best = None
    for k in range(len(series) - 2):
        t, gap, lift = series[k]
        if gap > touch:
            continue
        after = [g for _, g, _ in series[k + 1:k + 1 + window]]
        if len(after) < 2:
            continue
        mono = all(after[m + 1] >= after[m] - slack for m in range(len(after) - 1))
        if mono and after[-1] > gap + opened and (best is None or t > best[0]):
            best = (t, gap, lift, after[-1] - gap)
    return best


def report(scans):
    print(f"{'shot':>5} {'ball':>5} {'ppl':>4}   per person: frames / closest gap / highest hands / release?")
    for n, (track, per) in sorted(scans.items()):
        bits = []
        for key, v in sorted(per.items()):
            s = sorted(v["series"])
            if len(s) < 2:
                continue
            mingap = min(g for _, g, _ in s)
            maxlift = max(l for _, _, l in s)
            rel = release_with(s, shooter.TOUCH, shooter.OPENED, 5, 0.15)
            mark = "-" if not rel else ("OK" if rel[2] > 0 else f"handsdown({rel[2]:+.2f})")
            bits.append(f"[{len(s):>2}f {mingap:>5.2f} {maxlift:>+5.2f} {mark}]")
        print(f"{n:>5} {len(track):>5} {len(per):>4}   " + " ".join(bits))


def sweep(scans, notes):
    """How many shots get an answer under each setting, and how contested it is.

    Coverage alone is not the goal. A gate loose enough to answer everything
    answers with whoever happened to be nearest, so the number of people who
    also clear it is reported beside it: one candidate is a decision, four is a
    guess wearing a decision's clothes.
    """
    print(f"\n{'touch':>6} {'opened':>7} {'lift':>6} {'win':>4} | {'answered':>9} {'avg cands':>10}")
    best = []
    for touch in (0.9, 1.2, 1.6):
        for opened in (0.6, 0.4, 0.25):
            for liftmin in (0.0, -0.25, -0.6):
                for window in (5, 8):
                    got, cands_tot = 0, 0
                    for n, (track, per) in scans.items():
                        cs = []
                        for key, v in per.items():
                            s = sorted(v["series"])
                            if len(s) < 4:
                                continue
                            rel = release_with(s, touch, opened, window, 0.15)
                            if rel and rel[2] > liftmin:
                                cs.append((rel[1], key))
                        if cs:
                            got += 1
                            cands_tot += len(cs)
                    best.append((got, -cands_tot / max(1, got), touch, opened, liftmin, window))
                    print(f"{touch:>6} {opened:>7} {liftmin:>6} {window:>4} | "
                          f"{got:>4}/{len(scans):<4} {cands_tot/max(1,got):>10.1f}")
    best.sort(reverse=True)
    g, negc, t, o, l, w = best[0]
    print(f"\nmost answered: touch {t}, opened {o}, lift > {l}, window {w} "
          f"-> {g} of {len(scans)} shots, {-negc:.1f} candidates each on average")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482.MOV")
    p.add_argument("--sweep", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    scans = load(args.video)
    if not scans:
        print("no cached scans yet; run posediag.py first")
        return 1
    notes = {int(x["n"]): x for x in
             json.loads((ROOT / "labels/shooter_notes.json").read_text(encoding="utf-8"))}
    print(f"{len(scans)} cached scans for {args.video}\n")
    report(scans)
    sweep(scans, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
