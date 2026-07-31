"""Cut a short clip per candidate shot, for review by a human.

Finding shots is now reliable (17/17 against the marked calls). Deciding whether
one went in is not, and won't be until there are labeled examples of the ugly
cases. Scrubbing a two-hour video for them is miserable, so the job here is to
turn a list of candidate moments into short loops that can be judged in a couple
of seconds each and tapped through on a phone.

Clips are H.264 so they play inline in a browser with no plugin, and are cut
generously around the event: enough lead-in to see the shot arrive, enough
follow-through to see where the ball ends up. Judging make versus behind-the-rim
needs motion, which a contact sheet of stills cannot show, and needs to be
watched slowly -- the review UI drops playbackRate rather than baking slow motion
into the file, so the same clip serves both speeds.

Run: python src/clips.py footage/x.MOV out/rimwatch_full.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hoops import rig_for  # noqa: E402
from parked import drop_parked  # noqa: E402
from tracks import build_tracks  # noqa: E402

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

LEAD_S = 1.2    # before the first detection
TAIL_S = 2.2    # after the last -- the outcome is the entire point of the clip
OUT_W = 640     # even number required by yuv420p
CRF = 30        # quality/size knob; 30 keeps a 3s clip near 200KB

# OpenCV's VideoWriter exposes no bitrate or quality control, and its H.264
# output ran ~5MB for a three second clip (177MB across 31 clips), which is
# unusable for something meant to load on a phone. Piping raw frames to a real
# ffmpeg gives CRF control and lands the same clips around 40x smaller.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("rimwatch", type=Path)
    p.add_argument("--out", type=Path, default=Path("out/clips"))
    p.add_argument("--min-dets", type=int, default=3)
    p.add_argument("--gap", type=float, default=1.0, help="seconds of absence that ends an event")
    return p.parse_args()


JITTER_PX = 25      # detector wobble tolerated without calling it a direction change
MIN_DROP_PX = 150   # a real descent, not the ball jiggling in someone's hands
MAX_GAP_S = 0.6     # a longer blackout than this ends the descent


def cluster(hits_by_hoop: dict, gap: float, min_dets: int):
    """One shot per clip, found by its DESCENT.

    Two rules were tried and both fail, for opposite reasons. Splitting on gaps
    in detection cuts a single shot in half, because the ball is lost during the
    pause while the shooter sets -- that produced a clip of someone raising their
    arm and a separate clip of the ball arriving. Widening the gap and merging
    overlapping windows fixed that but glued genuinely separate shots together,
    which is worse: a reviewer is shown two outcomes and one set of buttons and
    has no way to say which one they mean.

    Neither is fixable by tuning, because the two failures want the threshold
    moved in opposite directions. At 4:52 two shots sit 1.36s apart; the pause
    inside the single 5:44 shot is 1.03s. There is no gap that separates one and
    not the other.

    What actually defines a shot is the ball coming down at the hoop, exactly
    once. So an event is a run of falling detections spanning a real drop. The
    two shots at 4:52 fall 737->1136 and 751->1204: unambiguously two. The 5:44
    shot, wind-up and all, contains one descent: unambiguously one.
    """
    events = []
    for hoop, hits in hits_by_hoop.items():
        # A ball at rest is detected in most frames and would otherwise dominate
        # everything downstream -- see parked.py for the two times this bit.
        hits, dropped = drop_parked(hits)
        if dropped:
            print(f"  {hoop}: dropped {dropped} detections of ball(s) sitting still")

        # Descents are found WITHIN a track, never across the merged detection
        # stream. With more than one ball on screen a merged stream hops between
        # objects and shreds one descent into many fragments -- see tracks.py.
        for pts in build_tracks(hits):
            runs, cur = [], [pts[0]]
            for p in pts[1:]:
                prev = cur[-1]
                falling = p["y"] >= prev["y"] - JITTER_PX
                if p["t"] - prev["t"] > MAX_GAP_S or not falling:
                    runs.append(cur)
                    cur = [p]
                else:
                    cur.append(p)
            runs.append(cur)
            for r in runs:
                if len(r) >= min_dets and (r[-1]["y"] - r[0]["y"]) >= MIN_DROP_PX:
                    events.append((hoop, r))

    events.sort(key=lambda e: e[1][0]["t"])
    return events


MIN_TAIL_S = 1.5   # the outcome must be on screen, whatever else gets cut
MIN_LEAD_S = 0.3


def windows(events):
    """Clip bounds per event.

    Trimmed against neighbours so one clip doesn't simply replay the shot
    before it, but the TAIL is protected and the lead-in gives way first. When
    shots come in quick succession, squeezing both ends equally produced clips
    as short as 0.8s -- 32 of 128 on IMG_2481 were under two seconds, which is
    not long enough to see whether the ball went in. A clip that starts with the
    ball already falling is judgeable; one that ends before the ball arrives is
    worthless, and worse than worthless in a review queue because it still costs
    a decision.

    So a clip may overlap slightly into the next shot's lead-in. That is the
    lesser evil: each clip is still centered on its own descent, and the overlap
    shows the previous ball already gone.
    """
    out = []
    for i, (hoop, dets) in enumerate(events):
        d0, d1 = dets[0]["t"], dets[-1]["t"]
        start = d0 - LEAD_S
        end = d1 + TAIL_S

        prev = next((e for e in reversed(events[:i]) if e[0] == hoop), None)
        if prev:
            start = max(start, prev[1][-1]["t"] + 0.35)
        start = min(start, d0 - MIN_LEAD_S)          # never eat into the shot itself

        nxt = next((e for e in events[i + 1:] if e[0] == hoop), None)
        if nxt:
            end = min(end, nxt[1][0]["t"] - 0.35)
        end = max(end, d1 + MIN_TAIL_S)              # the outcome wins over tidiness

        out.append((max(0.0, start), end))
    return out


def main():
    args = parse_args()
    import cv2

    rig = rig_for(args.video)
    rw = json.loads(args.rimwatch.read_text(encoding="utf-8"))
    events = cluster(rw["hits"], args.gap, args.min_dets)
    bounds = windows(events)
    print(f"{len(events)} shots (one descent each)")

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    args.out.mkdir(parents=True, exist_ok=True)
    index = []

    for i, (hoop, dets) in enumerate(events, 1):
        x1, y1, x2, y2 = rig.crops[hoop]
        t_start, t_stop = bounds[i - 1]
        f0, f1 = int(t_start * fps), int(t_stop * fps)
        h = int(OUT_W * (y2 - y1) / (x2 - x1)) // 2 * 2   # even height for yuv420p
        name = f"shot_{i:03d}_{hoop}_{dets[0]['t']:.0f}s.mp4"
        path = args.out / name

        proc = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{h}", "-r", f"{fps:.3f}",
             "-i", "-",
             "-c:v", "libx264", "-crf", str(CRF), "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        n = 0
        for f in range(f0, f1):
            ok, fr = cap.read()
            if not ok:
                break
            proc.stdin.write(cv2.resize(fr[y1:y2, x1:x2], (OUT_W, h)).tobytes())
            n += 1
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace").strip()
        proc.wait()
        if err:
            print(f"    ffmpeg: {err[:200]}")

        index.append({
            "n": i,
            "file": name,
            "hoop": hoop,
            "t": round(dets[0]["t"], 2),
            "t_end": round(dets[-1]["t"], 2),
            "clip_start": round(t_start, 2),
            "clip_stop": round(t_stop, 2),
            "clock": f"{int(dets[0]['t'])//60}:{int(dets[0]['t'])%60:02d}",
            "dets": len(dets),
            "peak_conf": round(max(d["conf"] for d in dets), 3),
            "frames": n,
            "size_kb": round(path.stat().st_size / 1024) if path.exists() else 0,
        })
        print(f"  {name}  {n} frames, {index[-1]['size_kb']} KB")

    cap.release()
    (args.out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    total = sum(c["size_kb"] for c in index)
    print(f"wrote {len(index)} clips, {total/1024:.1f} MB total -> {args.out}")


if __name__ == "__main__":
    main()
