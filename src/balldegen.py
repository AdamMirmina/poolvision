"""Is the fine-tune finding the ball, or has it just learned "draw a box in the
middle"?

Every training crop was built centered on the predicted ball position, so the ball
sits dead center in all of them. A model that ignores the image entirely and
always emits a center box would score near-perfect recall on the validation split
without having learned anything at all. The 99% number is untrustworthy until
that is ruled out.

The test: roll each validation frame so the ball moves to a known off-center
position, and ask whether the prediction follows it. A real detector follows. A
degenerate one stays in the middle, and the distance between its box and the
ball will equal the distance the ball was moved.

Rolling rather than cropping keeps the image the same size and keeps every pixel,
so nothing changes except where the ball is.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "out/balldata/images/val"
LAB = ROOT / "out/balldata/labels/val"
TUNED = ROOT / "out/balltrain/ball/weights/best.pt"
CONF = 0.05
SHIFTS = ((0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (-0.25, 0.25))


def main():
    import cv2
    import numpy as np
    from ultralytics import YOLO

    imgs = sorted(VAL.glob("*.jpg"))
    if not imgs or not TUNED.exists():
        print("nothing to test")
        return 1
    m = YOLO(str(TUNED))

    print("does the prediction follow the ball when the ball moves?\n")
    print(f"{'ball moved by':>16} | {'follows ball':>13} | {'stuck at center':>16}")
    print("-" * 52)

    for dx, dy in SHIFTS:
        follow = stuck = seen = 0
        for im in imgs:
            lf = LAB / (im.stem + ".txt")
            if not lf.exists():
                continue
            p = lf.read_text(encoding="utf-8").split()
            if len(p) < 5:
                continue
            lx, ly = float(p[1]), float(p[2])
            a = cv2.imread(str(im))
            if a is None:
                continue
            h, w = a.shape[:2]
            sx, sy = int(dx * w), int(dy * h)
            a = np.roll(np.roll(a, sx, axis=1), sy, axis=0)
            tx, ty = (lx + dx) % 1.0, (ly + dy) % 1.0

            r = m.predict(a, imgsz=640, conf=CONF, device="cpu", verbose=False)[0]
            if not len(r.boxes):
                continue
            b = max(r.boxes, key=lambda b: float(b.conf.item()))
            px, py = b.xywhn[0].tolist()[:2]
            seen += 1
            d_ball = ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
            d_ctr = ((px - 0.5) ** 2 + (py - 0.5) ** 2) ** 0.5
            if d_ball < d_ctr:
                follow += 1
            else:
                stuck += 1
        if not seen:
            continue
        lab = "not moved" if (dx or dy) == 0 else f"{dx:+.2f},{dy:+.2f}"
        print(f"{lab:>16} | {follow:>4}/{seen:<3} {100*follow/seen:>5.0f}% | "
              f"{stuck:>4}/{seen:<3} {100*stuck/seen:>6.0f}%")

    print("\nIf the shifted rows stay high in the first column, the detector is")
    print("reading the image. If they collapse, the 99% was an artifact of every")
    print("training crop being centered on the ball, and the number means nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
