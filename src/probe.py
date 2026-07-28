"""Step 1: does this footage support the project at all?

Run this on ONE recorded party before any real work happens. It answers the four
questions that decide whether poolvision is a few weeks or a few months, and
what to change about the camera before the next party:

  1. Are people reliably DETECTED in the water?      -> person detection rate
  2. Is the BALL reliably detected?                  -> ball detection rate
  3. Do TRACKS survive, or fragment constantly?      -> tracks per person, track length
  4. Do the CAP COLORS separate cleanly?             -> sampled cap-region colors

Everything downstream (training, identity, shot detection) is guesswork until
these four numbers exist, which is why this is the first thing that runs.

Deliberately uses an OFF-THE-SHELF COCO model. The point is to measure the
starting position, not to be good yet. If off-the-shelf already detects people
well, that's a whole training pass we don't have to do.

Usage:
    python src/probe.py footage/party-2026-07-26.mov
    python src/probe.py footage/party.mov --frames 400 --model yolo11m.pt

Writes out/report.md, out/report.json, and out/frames/*.jpg (annotated samples
to actually LOOK at -- the numbers alone won't tell you the camera was too low).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# COCO class ids we care about. The hoop is NOT in COCO, which is fine: the
# camera is fixed, so the hoop gets defined by hand once rather than detected.
PERSON, SPORTS_BALL = 0, 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure whether pool footage is workable.")
    p.add_argument("video", type=Path, help="Path to a recorded party (.mov/.mp4)")
    p.add_argument("--frames", type=int, default=300, help="How many frames to sample across the video")
    p.add_argument("--model", default="yolo11m.pt", help="Ultralytics model (n/s/m/l/x)")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence floor")
    p.add_argument("--out", type=Path, default=Path("out"), help="Output directory")
    p.add_argument("--save-every", type=int, default=20, help="Save an annotated frame every N sampled frames")
    # Real recordings have long dead stretches (nobody in the pool). Sampling
    # those wastes the budget and drags every average toward zero, so the window
    # of actual play can be selected explicitly.
    p.add_argument("--start", type=float, default=0.0, help="Start at this minute")
    p.add_argument("--end", type=float, default=None, help="Stop at this minute")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise SystemExit(f"No such video: {args.video}")

    # Imported here so --help works without the heavy deps installed.
    import cv2
    from ultralytics import YOLO

    out_frames = args.out / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    first = int(args.start * 60 * fps)
    last = int(args.end * 60 * fps) if args.end is not None else total
    first = max(0, min(first, total - 1))
    last = max(first + 1, min(last, total))
    span = last - first
    step = max(1, span // max(1, args.frames))
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    print(f"{args.video.name}: {w}x{h} @ {fps:.1f}fps, {total} frames "
          f"({total / fps / 60:.1f} min). Window {args.start:.1f}-"
          f"{(last / fps / 60):.1f} min, every {step}th frame.")

    model = YOLO(args.model)

    person_counts: list[int] = []
    person_confs: list[float] = []
    ball_frames = 0
    ball_confs: list[float] = []
    person_areas: list[float] = []
    # track id -> how many sampled frames it appeared in
    track_seen: dict[int, int] = defaultdict(int)
    sampled = 0

    idx = first
    while idx < last:
        ok = cap.grab()
        if not ok:
            break
        if (idx - first) % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            # persist=True keeps tracker state across our sampled frames. Note
            # we're tracking on SAMPLED frames, so fragmentation here is a
            # pessimistic proxy -- real per-frame tracking will do better. It
            # still surfaces whether people vanish constantly.
            res = model.track(frame, persist=True, conf=args.conf, verbose=False,
                              classes=[PERSON, SPORTS_BALL])[0]

            n_person, found_ball = 0, False
            if res.boxes is not None and len(res.boxes):
                ids = res.boxes.id
                for i, box in enumerate(res.boxes):
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    if cls == PERSON:
                        n_person += 1
                        person_confs.append(conf)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        person_areas.append(((x2 - x1) * (y2 - y1)) / (w * h))
                        if ids is not None:
                            track_seen[int(ids[i].item())] += 1
                    elif cls == SPORTS_BALL:
                        found_ball = True
                        ball_confs.append(conf)

            person_counts.append(n_person)
            ball_frames += int(found_ball)

            if sampled % args.save_every == 0:
                cv2.imwrite(str(out_frames / f"f{idx:07d}.jpg"), res.plot())
            sampled += 1
        idx += 1
    cap.release()

    lengths = sorted(track_seen.values(), reverse=True)
    report = {
        "video": str(args.video),
        "resolution": f"{w}x{h}",
        "fps": round(fps, 1),
        "duration_min": round(total / fps / 60, 1) if fps else None,
        "window_min": [round(first / fps / 60, 1), round(last / fps / 60, 1)],
        "frames_sampled": sampled,
        "model": args.model,
        "people": {
            "mean_per_frame": round(statistics.fmean(person_counts), 2) if person_counts else 0,
            "max_per_frame": max(person_counts) if person_counts else 0,
            "frames_with_none_pct": round(100 * sum(1 for c in person_counts if c == 0) / max(1, sampled), 1),
            "mean_confidence": round(statistics.fmean(person_confs), 3) if person_confs else 0,
            # A person occupying a tiny share of frame means the camera is too
            # far or too wide, which caps everything downstream.
            "median_box_pct_of_frame": round(100 * statistics.median(person_areas), 3) if person_areas else 0,
        },
        "ball": {
            "detected_in_pct_of_frames": round(100 * ball_frames / max(1, sampled), 1),
            "mean_confidence": round(statistics.fmean(ball_confs), 3) if ball_confs else 0,
        },
        "tracking": {
            "unique_track_ids": len(track_seen),
            "longest_tracks": lengths[:12],
            "median_track_len": statistics.median(lengths) if lengths else 0,
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(render(report), encoding="utf-8")
    print(render(report))
    print(f"\nAnnotated frames: {out_frames}  <- LOOK at these, don't just read the numbers.")


def render(r: dict) -> str:
    p, b, t = r["people"], r["ball"], r["tracking"]
    # Rough reads, not verdicts. The annotated frames decide.
    #
    # Judge person detection on CONFIDENCE only. An earlier version also required
    # frames_with_none_pct < 10 and reported "needs fine-tuning" on footage where
    # detection was in fact working well -- because a real recording has stretches
    # with nobody in the pool, and "the detector found no one" is indistinguishable
    # here from "there was no one to find". That metric can't separate the two, so
    # it's reported but not used to judge.
    person_ok = p["mean_confidence"] >= 0.6
    # Likewise, a ball that's merely PRESENT in frame (sitting on the deck) is not
    # the same as tracking a thrown one. Low mean confidence usually means it's
    # locking onto a static object rather than the game ball, so require both.
    ball_ok = b["detected_in_pct_of_frames"] >= 40 and b["mean_confidence"] >= 0.5
    return f"""# poolvision probe

`{r['video']}`
{r['resolution']} @ {r['fps']}fps, {r['duration_min']} min, {r['frames_sampled']} frames sampled, model `{r['model']}`

## People
- {p['mean_per_frame']} detected per frame on average (peak {p['max_per_frame']})
- {p['frames_with_none_pct']}% of frames found nobody at all
- mean confidence {p['mean_confidence']}
- median person fills {p['median_box_pct_of_frame']}% of the frame
- **{'looks usable off-the-shelf' if person_ok else 'needs fine-tuning, or the camera is too far/low'}**

## Ball
- found in {b['detected_in_pct_of_frames']}% of sampled frames (confidence {b['mean_confidence']})
- **{'usable off-the-shelf' if ball_ok else 'needs its own training pass -- expect to label a few hundred frames'}**

## Tracking
- {t['unique_track_ids']} unique track ids; median track length
  {t['median_track_len']} sampled frames; longest {t['longest_tracks']}
- **Only meaningful if sampled frames are CONSECUTIVE.** This probe samples
  sparsely across a long window to measure detection, which puts seconds between
  samples -- no tracker can associate identities across that, so these numbers
  say nothing about real fragmentation. Measure tracking separately on a short
  run of consecutive frames.

## Next
Open `out/frames/` and look. Numbers won't tell you the camera was mounted too
low, that glare washes out one end of the pool, or that the hoop is cut off.
"""


if __name__ == "__main__":
    main()
