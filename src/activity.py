"""Where in the recording is anything actually happening?

Cheap motion timeline: sample a frame every N frames, downscale hard, and
measure how much the pool area changed since the last sample. No model, no GPU,
so it can run alongside a YOLO pass. The point is to derive the windows of real
play from the footage itself instead of from a description of it -- getting
those windows wrong is what produced both a "no basketball was played" verdict
and a 100%-ball-detection number that was really a ball lying on the deck.
"""
import sys
import cv2
import numpy as np

STEP = 60  # sample every 2s at 30fps


def main(path: str) -> None:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev = None
    rows = []
    for f in range(0, total, STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            rows.append((f / fps, float(np.mean(cv2.absdiff(small, prev)))))
        prev = small
    cap.release()

    if not rows:
        print("no frames read")
        return
    vals = np.array([r[1] for r in rows])
    thresh = float(np.percentile(vals, 60))
    print(f"samples {len(rows)} | motion median {np.median(vals):.2f} p60 {thresh:.2f} max {vals.max():.2f}\n")
    print("active stretches (motion above p60 for 3+ consecutive samples):")
    run = []
    for t, v in rows:
        if v > thresh:
            run.append((t, v))
        else:
            if len(run) >= 3:
                print(f"  {fmt(run[0][0])} - {fmt(run[-1][0])}  ({run[-1][0]-run[0][0]:5.0f}s, peak {max(x[1] for x in run):.1f})")
            run = []
    if len(run) >= 3:
        print(f"  {fmt(run[0][0])} - {fmt(run[-1][0])}  ({run[-1][0]-run[0][0]:5.0f}s, peak {max(x[1] for x in run):.1f})")


def fmt(s: float) -> str:
    return f"{int(s)//60}:{int(s)%60:02d}"


if __name__ == "__main__":
    main(sys.argv[1])
