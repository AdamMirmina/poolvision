"""Full analysis pass over a recording. Designed to run unattended.

Answers the question the project actually turns on: can shots be detected, and
can they be attributed to the person who took them?

Four stages, each writing its results to disk as it finishes so a crash late on
doesn't lose the expensive earlier work:

  A  shooting   -- per-frame ball + person tracking over the shooting windows,
                   then made-shot geometry, then attribution to a shooter
  B  color     -- sample the colors held up during the color tests, at two
                   target sizes, and check they separate
  C  swap       -- follow track ids through the deliberate underwater swap, the
                   headline identity failure mode
  D  report     -- roll it all up, and cut annotated clips around every detected
                   shot so the result can be checked by eye rather than trusted

Nothing here is believed without a picture to go with it: three separate results
on this project looked right and were wrong until the pixels were inspected.

Usage:
    python src/analyze.py footage/IMG_2403.MOV
    python src/analyze.py footage/x.mov --stages A,D
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shots import BallPoint, Hoop, detect_makes  # noqa: E402

PERSON, SPORTS_BALL = 0, 32

# Hoop rims, located by hand off a clean frame at native 3840x2160. The camera
# is fixed, so these hold for the whole recording. Re-measure if it ever moves.
HOOPS = {
    "left": Hoop(x1=615, y1=838, x2=820, y2=928),
    "right": Hoop(x1=3535, y1=562, x2=3717, y2=655),
}

# Windows of interest, in seconds. From the own description of the session.
SHOOTING = [(280, 400, "shoot_a"), (555, 620, "shoot_b")]
COLOR = [(435, 465, "noodles"), (490, 510, "small_balls")]
SWAP = (515, 555, "swap")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=Path("out/analysis"))
    p.add_argument("--stages", default="A,B,C,D")
    p.add_argument("--imgsz", type=int, default=1920)
    p.add_argument("--model", default="yolo11s.pt")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────────────────────── stage A: shooting ─────────────────────────

def stage_a(args, cv2, YOLO):
    """Per-frame ball + person tracking, then shots, then attribution."""
    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results = {}

    for a, b, name in SHOOTING:
        log(f"A/{name}: frames {int(a*fps)}-{int(b*fps)} ({b-a}s)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(a * fps))
        ball_track: list[BallPoint] = []
        # frame -> [(track_id, cx, cy)] so a shot can be matched to whoever was
        # nearest when it went up
        people: dict[int, list] = {}
        f = int(a * fps)
        end = int(b * fps)
        t0 = time.time()
        while f < end:
            ok, fr = cap.read()
            if not ok:
                break
            r = model.track(fr, persist=True, conf=0.15, verbose=False,
                            classes=[PERSON, SPORTS_BALL], imgsz=args.imgsz)[0]
            best_ball = None
            here = []
            if r.boxes is not None and len(r.boxes):
                ids = r.boxes.id
                for i, bx in enumerate(r.boxes):
                    cls = int(bx.cls.item())
                    cf = float(bx.conf.item())
                    x1, y1, x2, y2 = bx.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if cls == SPORTS_BALL:
                        # keep only the most confident ball in the frame
                        if best_ball is None or cf > best_ball[2]:
                            best_ball = (cx, cy, cf)
                    else:
                        tid = int(ids[i].item()) if ids is not None else -1
                        here.append((tid, cx, cy))
            if best_ball:
                ball_track.append(BallPoint(f, best_ball[0], best_ball[1]))
            people[f] = here
            f += 1
            if (f - int(a * fps)) % 300 == 0:
                done = f - int(a * fps)
                tot = end - int(a * fps)
                log(f"  {done}/{tot} frames, {time.time()-t0:.0f}s, ball seen {len(ball_track)}x")
        # Shots, against each hoop. Frame indices are absolute, and at 30fps a
        # pass takes well under a second, so the defaults hold.
        makes = []
        for hname, hoop in HOOPS.items():
            for m in detect_makes(ball_track, hoop, max_pass_frames=30, min_gap_frames=45):
                shooter = nearest_person(people, m.entry_frame, ball_track)
                makes.append({"hoop": hname, "frame": m.frame, "entry": m.entry_frame,
                              "t": round(m.frame / fps, 1), "shooter_track": shooter})
        makes.sort(key=lambda m: m["frame"])
        results[name] = {
            "window_s": [a, b],
            "frames": end - int(a * fps),
            "ball_detected_frames": len(ball_track),
            "ball_detect_rate_pct": round(100 * len(ball_track) / max(1, end - int(a * fps)), 1),
            "makes": makes,
            "ball_track": [[p.frame, round(p.x), round(p.y)] for p in ball_track],
        }
        log(f"A/{name}: ball in {len(ball_track)} frames, {len(makes)} make(s) detected")
    cap.release()
    return results


def nearest_person(people, frame, ball_track):
    """Whoever was closest to the ball just before it went up. Crude on purpose:
    a proper release-point estimate needs a working shot detector first, and
    this is enough to test whether attribution is even plausible."""
    ball = next((p for p in ball_track if p.frame == frame), None)
    if ball is None:
        return None
    best, bestd = None, 1e12
    for off in range(0, 20):
        row = people.get(frame - off)
        if not row:
            continue
        for tid, cx, cy in row:
            d = (cx - ball.x) ** 2 + (cy - ball.y) ** 2
            if d < bestd:
                bestd, best = d, tid
        if best is not None:
            break
    return best


# ───────────────────────── stage B: color ─────────────────────────

def stage_b(args, cv2, YOLO, np):
    """Sample colors held up during the color tests.

    Looks at the region just ABOVE each detected person -- that's where a held-up
    object (and, later, a cap) sits. Reports how far apart the sampled hues are,
    which is what decides whether color identity can work at all.
    """
    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = {}
    for a, b, name in COLOR:
        log(f"B/{name}: {a}-{b}s")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(a * fps))
        samples = []
        f, end = int(a * fps), int(b * fps)
        step = 5
        while f < end:
            ok, fr = cap.read()
            if not ok:
                break
            if (f - int(a * fps)) % step == 0:
                r = model.predict(fr, conf=0.3, verbose=False, classes=[PERSON], imgsz=args.imgsz)[0]
                if r.boxes is not None:
                    for bx in r.boxes:
                        x1, y1, x2, y2 = [int(v) for v in bx.xyxy[0].tolist()]
                        # band above the head, where a held-up object sits
                        bh = max(12, (y2 - y1) // 3)
                        yy1, yy2 = max(0, y1 - bh), max(1, y1)
                        patch = fr[yy1:yy2, x1:x2]
                        if patch.size == 0:
                            continue
                        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                        # the most saturated pixels are the colored object,
                        # not the sky/deck behind it
                        s = hsv[:, :, 1]
                        thr = np.percentile(s, 92)
                        m = s >= max(thr, 90)
                        if m.sum() < 20:
                            continue
                        samples.append([float(np.median(hsv[:, :, 0][m])),
                                        float(np.median(hsv[:, :, 1][m])),
                                        float(np.median(hsv[:, :, 2][m]))])
            f += 1
        out[name] = {"window_s": [a, b], "n_samples": len(samples), "hsv": samples[:400]}
        hues = sorted(round(x[0]) for x in samples)
        log(f"B/{name}: {len(samples)} color samples, hue range {hues[:3]}..{hues[-3:] if hues else []}")
    cap.release()
    return out


# ───────────────────────── stage C: the swap ─────────────────────────

def stage_c(args, cv2, YOLO):
    """Follow track ids through the deliberate underwater swap."""
    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    a, b, name = SWAP
    log(f"C/{name}: {a}-{b}s")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a * fps))
    seen = defaultdict(list)  # track id -> [(frame, cx)]
    counts = []
    f, end = int(a * fps), int(b * fps)
    while f < end:
        ok, fr = cap.read()
        if not ok:
            break
        r = model.track(fr, persist=True, conf=0.3, verbose=False, classes=[PERSON], imgsz=args.imgsz)[0]
        n = 0
        if r.boxes is not None and len(r.boxes):
            ids = r.boxes.id
            n = len(r.boxes)
            for i, bx in enumerate(r.boxes):
                if ids is None:
                    continue
                x1, _, x2, _ = bx.xyxy[0].tolist()
                seen[int(ids[i].item())].append((f, round((x1 + x2) / 2)))
        counts.append(n)
        f += 1
    cap.release()
    tracks = {str(k): {"frames": len(v), "first": v[0][0], "last": v[-1][0],
                       "x_start": v[0][1], "x_end": v[-1][1]} for k, v in seen.items()}
    log(f"C/{name}: {len(tracks)} track ids across the swap")
    return {"window_s": [a, b], "mean_people": round(sum(counts) / max(1, len(counts)), 2),
            "tracks": tracks}


# ───────────────────────── stage D: evidence clips ─────────────────────────

def stage_d(args, cv2, res):
    """Cut annotated stills around every detected make, so each can be checked."""
    shots = []
    for name, r in (res.get("A") or {}).items():
        for m in r["makes"]:
            shots.append((name, m))
    if not shots:
        log("D: no makes detected, nothing to illustrate")
        return {"clips": 0}
    cap = cv2.VideoCapture(str(args.video))
    d = args.out / "shots"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, m in shots[:40]:
        for k, off in enumerate((-12, -6, 0, 6, 12)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, m["frame"] + off))
            ok, fr = cap.read()
            if not ok:
                continue
            h = HOOPS[m["hoop"]]
            cv2.rectangle(fr, (int(h.x1), int(h.y1)), (int(h.x2), int(h.y2)), (0, 255, 255), 4)
            fr = cv2.resize(fr, (1280, 720))
            cv2.imwrite(str(d / f"{name}_t{m['t']}_{k}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            n += 1
    cap.release()
    log(f"D: wrote {n} evidence frames to {d}")
    return {"clips": n}


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np
    from ultralytics import YOLO

    stages = [s.strip().upper() for s in args.stages.split(",")]
    res = {}
    p = args.out / "analysis.json"
    if p.exists():
        try:
            res = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            res = {}

    def save():
        p.write_text(json.dumps(res, indent=1), encoding="utf-8")

    t0 = time.time()
    if "A" in stages:
        res["A"] = stage_a(args, cv2, YOLO); save()
    if "B" in stages:
        res["B"] = stage_b(args, cv2, YOLO, np); save()
    if "C" in stages:
        res["C"] = stage_c(args, cv2, YOLO); save()
    if "D" in stages:
        res["D"] = stage_d(args, cv2, res); save()
    log(f"done in {(time.time()-t0)/60:.1f} min -> {p}")


if __name__ == "__main__":
    main()
