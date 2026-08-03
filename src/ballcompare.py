"""Stock YOLO against the fine-tune, on the same frames, scored the same way.

mAP is the wrong question for this project. The pipeline never needs a clean
detection set -- it needs the ball found in the frames where it currently is
not, because that single failure is what caps the through-the-hoop rule at 38%
recall, drags the overlap veto below chance, and leaves arcs ending in mid-air.

So the measure here is the one the pipeline actually cares about: given a frame
and a rough idea of where the ball should be, does the model put a box there?

Both models are scored against the same labels and neither has ever trained on
the validation split, so the comparison is fair even though the labels come from
parabola fits rather than a person drawing boxes. A label that is slightly off
is slightly off for both.

Stock reports the ball as COCO class 32 (sports ball); the fine-tune has one
class. Confidence is swept because the useful operating point for a detector
feeding a tracker is much lower than the 0.25 default -- a weak detection in the
right place is worth more than a miss.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "out/balldata/images/val"
LAB = ROOT / "out/balldata/labels/val"
TUNED = ROOT / "out/balltrain/ball/weights/best.pt"
STOCK = ROOT / "yolo11s.pt"

# A hit is a box centered near the label. Generous on purpose: the tracker only
# needs the ball localised well enough to fit an arc through, not a tight box.
TOL = 1.5  # in label-box widths
CONFS = (0.03, 0.05, 0.10, 0.25)


def labels_for(img: Path):
    f = LAB / (img.stem + ".txt")
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append(tuple(float(x) for x in p[1:5]))
    return out


def run(model_path: Path, imgs, cls_filter=None):
    from ultralytics import YOLO
    m = YOLO(str(model_path))
    per = {}
    for im in imgs:
        r = m.predict(str(im), imgsz=640, conf=min(CONFS), device="cpu", verbose=False)[0]
        boxes = []
        for b in r.boxes:
            c = int(b.cls.item())
            if cls_filter is not None and c != cls_filter:
                continue
            x, y, w, h = b.xywhn[0].tolist()
            boxes.append((x, y, w, h, float(b.conf.item())))
        per[im.name] = boxes
    return per


def score(per, imgs, conf):
    hits = 0
    total = 0
    errs = []
    fp = 0
    for im in imgs:
        labs = labels_for(im)
        if not labs:
            continue
        total += len(labs)
        boxes = [b for b in per[im.name] if b[4] >= conf]
        used = set()
        for (lx, ly, lw, lh) in labs:
            best, bd = None, 1e9
            for i, (x, y, w, h, c) in enumerate(boxes):
                if i in used:
                    continue
                d = ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
                if d < bd:
                    best, bd = i, d
            if best is not None and bd <= TOL * max(lw, 0.01):
                hits += 1
                used.add(best)
                errs.append(bd / max(lw, 0.01))
        fp += len(boxes) - len(used)
    med = sorted(errs)[len(errs) // 2] if errs else float("nan")
    return hits, total, fp, med


def main():
    imgs = sorted(VAL.glob("*.jpg"))
    if not imgs:
        print("no validation images; run src/balldata.py first")
        return 1
    if not TUNED.exists():
        print(f"no fine-tuned weights at {TUNED}")
        return 1

    print(f"{len(imgs)} validation frames, neither model trained on them\n")
    print("finding the ball, by confidence threshold")
    print(f"{'conf':>6} | {'stock recall':>22} | {'fine-tuned recall':>22} | {'stock FP':>9} | {'tuned FP':>9}")
    print("-" * 82)

    stock = run(STOCK, imgs, cls_filter=32)
    tuned = run(TUNED, imgs, cls_filter=None)

    rows = []
    for c in CONFS:
        sh, st, sfp, sme = score(stock, imgs, c)
        th, tt, tfp, tme = score(tuned, imgs, c)
        rows.append((c, sh, st, sfp, sme, th, tt, tfp, tme))
        print(f"{c:>6.2f} | {sh:>4}/{st:<4} {100*sh/max(st,1):>13.0f}% | "
              f"{th:>4}/{tt:<4} {100*th/max(tt,1):>13.0f}% | {sfp:>9} | {tfp:>9}")

    c, sh, st, sfp, sme, th, tt, tfp, tme = rows[1]
    print(f"\nat conf {c:.2f}, the operating point a tracker would use:")
    print(f"  stock      {100*sh/max(st,1):.0f}% of balls found, "
          f"median center error {sme:.2f} ball-widths")
    print(f"  fine-tuned {100*th/max(tt,1):.0f}% of balls found, "
          f"median center error {tme:.2f} ball-widths")
    d = th - sh
    if d > 0:
        print(f"\n  the fine-tune finds {d} more of {st}. That is the number that "
              f"matters:\n  every one is a frame the arc currently has a hole in.")
    elif d == 0:
        print("\n  no difference. The fine-tune is not worth adopting.")
    else:
        print(f"\n  the fine-tune finds {-d} FEWER. Keep the stock detector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
