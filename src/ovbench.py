"""Is the Intel iGPU faster than the CPU here, and does it see the same thing?

Every scan so far has run on the CPU, because torch is the CPU-only build and
this laptop has no CUDA card. It has an Iris Xe iGPU that has never been used.
A held-out window takes about twenty hours at CPU speed, and the two long videos
waiting are longer still, so the runtime is what bounds how much footage can ever
be evaluated.

The speed answer alone is not enough. Swapping runtimes under a held-out test is
only legitimate if the detector still sees the same things: same weights, same
input size, same boxes. A faster scan that finds a different set of balls is a
different experiment, and its score would not be comparable to the 75 of 80 the
CPU pipeline produced. So this measures agreement as well as time, and prints
both.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def canvases(video, rig, n, stride):
    """Rebuild the stacked rim canvas exactly as rimwatch does."""
    import json
    r = json.loads(Path(rig).read_text(encoding="utf-8"))
    det = {k: v for k, v in r["det_boxes"].items()} if "det_boxes" in r else r["crops"]
    order = sorted(det)
    hs = {k: det[k][3] - det[k][1] for k in order}
    ws = {k: det[k][2] - det[k][0] for k in order}
    offs, y = {}, 0
    for k in order:
        offs[k] = y
        y += hs[k]
    canvas = np.zeros((y, max(ws.values()), 3), np.uint8)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(700 * fps))
    out, f = [], 0
    while len(out) < n:
        ok, fr = cap.read()
        if not ok:
            break
        if f % stride == 0:
            canvas[:] = 0
            for k in order:
                x1, y1, x2, y2 = det[k]
                canvas[offs[k]:offs[k] + (y2 - y1), 0:x2 - x1] = fr[y1:y2, x1:x2]
            out.append(canvas.copy())
        f += 1
    cap.release()
    return out


def boxes_of(res):
    if res.boxes is None or not len(res.boxes):
        return np.zeros((0, 4))
    return res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") \
        else np.asarray(res.boxes.xyxy)


def agree(a, b, tol=6.0):
    """Fraction of CPU boxes with a GPU box within `tol` px on every corner.

    Reported both ways, because a runtime that finds the same boxes plus extra
    ones is a different failure from one that misses them, and a single
    percentage hides which happened.
    """
    if not len(a) and not len(b):
        return 1.0, 1.0
    if not len(a) or not len(b):
        return 0.0, 0.0
    d = np.abs(a[:, None, :] - b[None, :, :]).max(axis=2)
    return float((d.min(axis=1) <= tol).mean()), float((d.min(axis=0) <= tol).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="footage/IMG_2770.MOV")
    p.add_argument("--rig", default="out/key_2770.json")
    p.add_argument("--weights", default="out/balltrain/ball/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=1600)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--stride", type=int, default=17)
    args = p.parse_args()

    from ultralytics import YOLO

    print("building canvases...", flush=True)
    cv = canvases(ROOT / args.video, ROOT / args.rig, args.n, args.stride)
    print(f"{len(cv)} canvases of {cv[0].shape[1]}x{cv[0].shape[0]}", flush=True)

    w = ROOT / args.weights
    ov_dir = w.parent / (w.stem + "_openvino_model")
    if not ov_dir.exists():
        print("exporting to openvino (one time, slow)...", flush=True)
        YOLO(str(w)).export(format="openvino", imgsz=args.imgsz, half=False)
    print(f"openvino model: {ov_dir.name}", flush=True)

    torch_m = YOLO(str(w))
    ov_m = YOLO(str(ov_dir), task="detect")

    runs = {}
    for label, model, dev in (("cpu  (torch)", torch_m, None),
                              ("igpu (openvino)", ov_m, "intel:gpu"),
                              ("cpu  (openvino)", ov_m, "intel:cpu")):
        kw = dict(conf=args.conf, verbose=False, imgsz=args.imgsz)
        if dev:
            kw["device"] = dev
        try:
            model.predict(cv[0], **kw)          # warm up, excluded from timing
            t0 = time.time()
            res = [model.predict(c, **kw)[0] for c in cv]
            dt = time.time() - t0
            runs[label] = (dt / len(cv), [boxes_of(r) for r in res])
            print(f"{label:16s} {dt/len(cv)*1000:7.0f} ms/frame", flush=True)
        except Exception as e:
            print(f"{label:16s} FAILED: {type(e).__name__}: {e}", flush=True)

    base = runs.get("cpu  (torch)")
    if not base:
        return
    print()
    for label, (per, bl) in runs.items():
        if label == "cpu  (torch)":
            continue
        print(f"{label} vs cpu(torch):  {base[0]/per:.1f}x faster")
        fwd = [agree(a, b)[0] for a, b in zip(base[1], bl)]
        rev = [agree(a, b)[1] for a, b in zip(base[1], bl)]
        nb, nn = sum(len(a) for a in base[1]), sum(len(a) for a in bl)
        print(f"   boxes: cpu {nb}, this {nn}")
        print(f"   of the cpu boxes, {np.mean(fwd)*100:.0f}% matched within 6px")
        print(f"   of its own boxes, {np.mean(rev)*100:.0f}% matched a cpu box")


if __name__ == "__main__":
    main()
