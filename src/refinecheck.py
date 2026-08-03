"""What the refine pass actually changes on real shots, before re-running everything.

The hole test said the fine-tune recovers 60 of 60 frames the stock detector
misses. That was measured on frames chosen because they were holes. This asks the
question the pipeline cares about instead: run the real scan, twice, on the same
shots, and see whether the arcs come out longer and better conditioned.

Same video, same shots, same code path. The only difference is whether the second
look is allowed to happen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoops
import shooter

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482.MOV")
    p.add_argument("--shots", type=int, default=8)
    args = p.parse_args()

    import cv2
    from ultralytics import YOLO

    # Descent times come from the shot records, which are the same times the
    # reviewer shows -- so this measures the exact shots review is judging, not a
    # re-derived set that could drift from them.
    tf = ROOT / f"out/t{Path(args.video).stem.replace('IMG_', '')}.txt"
    if not tf.exists():
        print(f"no descent times at {tf}")
        return 1

    video = ROOT / "footage" / args.video
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    rig = hoops.rig_for(args.video)
    det = YOLO(str(ROOT / "yolo11s.pt"))
    pos = YOLO(str(ROOT / "yolo11s-pose.pt"))

    times = [float(s) for s in tf.read_text(encoding="utf-8").split() if s.strip()]
    times = times[:args.shots]
    if not times:
        print("no descent times found")
        return 1

    print(f"{len(times)} shots from {args.video}\n")
    print(f"{'shot':>5} | {'stock pts':>10} | {'refined pts':>12} | {'gained':>7} | {'span s':>14}")
    print("-" * 62)

    tot_a = tot_b = 0
    for i, t in enumerate(times, 1):
        # The old pipeline: window stops dead at the descent, no second look.
        shooter._fine[0] = False
        shooter.AFTER_S = 0.0
        a, _ = shooter.scan(cap, fps, t, det, pos, pool=rig.pool)
        # The new one: keep watching past the descent, and look again where the
        # stock detector came up empty.
        shooter._fine[0] = None
        shooter.AFTER_S = 1.2
        b, _ = shooter.scan(cap, fps, t, det, pos, pool=rig.pool)
        fa = shooter.fit_arc(a, t_anchor=t)
        fb = shooter.fit_arc(b, t_anchor=t)
        sa = f"{fa['span']:.2f}" if fa else "-"
        sb = f"{fb['span']:.2f}" if fb else "-"
        tot_a += len(a)
        tot_b += len(b)
        print(f"{i:>5} | {len(a):>10} | {len(b):>12} | {len(b)-len(a):>+7} | {sa:>6} -> {sb:<6}")

    print(f"\ntotal sightings {tot_a} -> {tot_b}  ({100*(tot_b-tot_a)/max(tot_a,1):+.0f}%)")
    print("Longer arcs are the point: the through-the-hoop rule and the overlap")
    print("veto were both being judged on flights that stopped early.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
