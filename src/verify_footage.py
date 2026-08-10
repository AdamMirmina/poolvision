"""Does this video actually open, or does it just have the right file size?

A stalled MTP copy leaves a file at its full pre-allocated size, unchanging, and
unreadable. It passes every size-based check. An iPhone MOV keeps its index at
the END, so a truncated copy has bytes but no index, and the only honest test is
opening it and reading real frames.

    python src/verify_footage.py IMG_2528.MOV IMG_2529.MOV
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def check(name):
    import cv2
    p = ROOT / "footage" / name
    if not p.exists():
        return False, f"{name}: missing"
    gb = p.stat().st_size / 1e9
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n <= 0 or fps <= 0:
        cap.release()
        return False, f"{name}: {gb:.1f} GB but NO INDEX -- the copy did not finish"
    # read a frame from the far end: the index can exist while the tail is short
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 30))
    ok, _ = cap.read()
    cap.release()
    if not ok:
        return False, f"{name}: index present but the end is unreadable -- truncated"
    return True, f"{name}: OK  {w}x{h} @ {fps:.0f}fps  {n} frames  {n/fps/60:.1f} min  {gb:.1f} GB"

ok_all = True
# Default to EVERY video present, not two names from one session. The old
# default silently checked the wrong files after a new transfer: two freshly
# copied 16GB videos were verified by a run that reported OK for a different
# pair entirely, which is the stale-artifact failure this project keeps hitting.
_all = sorted(p.name for p in (ROOT / "footage").glob("*.MOV")) if (ROOT / "footage").exists() else []
for name in (sys.argv[1:] or _all):
    good, msg = check(name)
    print(msg)
    ok_all = ok_all and good
raise SystemExit(0 if ok_all else 1)
