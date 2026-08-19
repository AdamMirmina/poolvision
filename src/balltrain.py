"""Fine-tune the ball detector on this pool's own footage.

The single highest-leverage thing left. Every ceiling reached so far is one
failure wearing three hats: the detector loses the ball where it matters. the through-the-hoop rule is observable on 38% of makes, the overlap veto lands below
chance, arcs stop before the ball lands.

COCO's "sports ball" is a generic class trained on clean photographs of balls.
This pool's ball is wet, motion-blurred, half-inside a net, against skin or
turquoise water, at 4K downscaled for inference. A model that has seen a few
hundred of those is a different proposition.

Trained from the COCO weights rather than from scratch, because 323 images is
nowhere near enough to learn what a ball is -- only enough to learn what THIS
ball looks like here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "out/balldata/data.yaml"


def main():
    if not DATA.exists():
        print("no dataset; run src/balldata.py first")
        return 1
    from ultralytics import YOLO

    model = YOLO("yolo11s.pt")
    # CPU-only on a 15W laptop chip, so the settings are chosen for a night
    # rather than for a GPU hour: modest resolution, small batch, patience so it
    # stops when it stops improving instead of grinding to a fixed epoch count.
    model.train(
        data=str(DATA),
        epochs=80,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=4,
        patience=15,
        project=str(ROOT / "out/balltrain"),
        name="ball",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=False,
    )
    best = ROOT / "out/balltrain/ball/weights/best.pt"
    print(f"\nbest weights: {best}  exists={best.exists()}")
    if best.exists():
        m = YOLO(str(best))
        r = m.val(data=str(DATA), device="cpu", imgsz=640, verbose=False)
        try:
            print(f"validation mAP50 {r.box.map50:.3f}  mAP50-95 {r.box.map:.3f}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
