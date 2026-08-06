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
# ...or, for a dunk, enough consecutive sightings at the hoop to mean the ball
# was really brought there. Set from the marked tape: the two clear dunks it
# missed ran 16 and 13 detections; the false positive review flagged at 4:10
# (ball never left his hand) is not near a hoop at all, so this does not
# readmit it.
CARRY_DETS = 5
# ...and no more than this, or it is a person holding the ball, not a dunk.
CARRY_MAX = 20
MAX_GAP_S = 0.6     # a longer blackout than this ends the descent
END_NEAR_RIM = 3.2  # where a descent must finish, in rim widths, to be a shot
BOUNCE_GAP_S = 1.1  # a second descent this soon at the same hoop is the same shot


RIG_RIMS: dict = {}


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

            # A descent that ENDS somewhere else is not a shot at this hoop.
            #
            # Review is
            # right, and a falling ball is not enough on its own: a pass across
            # the pool falls too. What makes it a shot is where the fall FINISHES.
            rim = RIG_RIMS[hoop]
            rcx, rcy = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2
            rw = max(1.0, rim[2] - rim[0])
            kept = []
            for r in runs:
                drop = r[-1]["y"] - r[0]["y"]
                # A DUNK never falls far. The ball is carried to the rim, so a
                # gate built on "how far did it drop" cannot see one -- and on a
                # low pool hoop that is a huge slice of the game.
                #
                # Measured against the marking of five minutes of game footage:
                # 4:39 had 29 detections and a 16-long run ending beside the net,
                # rejected for dropping 120px instead of 150. 6:49 had 19, a
                # 13-long run, ending 1.0 rim widths away, rejected for 53px. Both
                # obvious dunks to a person watching.
                #
                # So there are two ways to be a shot now: fall a long way into the
                # rim, or be SEEN AT the rim persistently. A pass does neither --
                # it crosses the frame without lingering at a hoop.
                # A dunk is BRIEF. the second tape showed what the bare
                # count admits: 50 consecutive sightings while a player carried
                # the ball out of the pool, and 12 more climbing back in, both
                # called shots. "it caught blue getting back in the pool with the
                # ball in hand."
                #
                # The ball at the rim on a dunk lasts a moment. A person holding
                # it lasts as long as they like, so an upper bound separates them
                # where a lower bound alone cannot.
                carried = CARRY_DETS <= len(r) <= CARRY_MAX
                if len(r) < min_dets and not carried:
                    continue
                if drop < MIN_DROP_PX and not carried:
                    continue
                near = min(((p["x"] - rcx) ** 2 + (p["y"] - rcy) ** 2) ** 0.5
                           for p in r[-3:]) / rw
                if near > END_NEAR_RIM:
                    continue
                kept.append(r)

            # A ball that bounces up off the rim and rolls before dropping is ONE
            # shot, not two. So consecutive descents at the same hoop, close in time
            # and both finishing at the rim, are joined rather than counted twice
            # -- and the clip cut from the joined event runs to the real outcome.
            merged_runs = []
            for r in kept:
                if merged_runs and r[0]["t"] - merged_runs[-1][-1]["t"] <= BOUNCE_GAP_S:
                    merged_runs[-1] = merged_runs[-1] + r
                else:
                    merged_runs.append(r)
            for r in merged_runs:
                events.append((hoop, r))

    events.sort(key=lambda e: e[1][0]["t"])

    # Two descents at the same hoop whose time ranges OVERLAP get merged. They
    # may genuinely be two balls, but a reviewer is shown a clip, and two
    # clips covering the same instant look identical to them -- the marked shot hit exactly
    # this ("36 and 37 are definitely the same"). One clip covering both is
    # honest; two near-duplicates waste a decision and pollute the labels.
    merged = []
    for hoop, dets in events:
        if merged and merged[-1][0] == hoop and dets[0]["t"] <= merged[-1][1][-1]["t"]:
            merged[-1] = (hoop, sorted(merged[-1][1] + dets, key=lambda d: d["t"]))
        else:
            merged.append((hoop, dets))
    return merged


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

    rig = rig_for(args.video)
    global RIG_RIMS
    RIG_RIMS = rig.rims
    rw = json.loads(args.rimwatch.read_text(encoding="utf-8"))
    events = cluster(rw["hits"], args.gap, args.min_dets)
    bounds = windows(events)
    print(f"{len(events)} shots (one descent each)")

    fps = 60.0 if "2528" in args.video.name or "2529" in args.video.name else 30.0
    args.out.mkdir(parents=True, exist_ok=True)
    index = []

    for i, (hoop, dets) in enumerate(events, 1):
        x1, y1, x2, y2 = rig.crops[hoop]
        t_start, t_stop = bounds[i - 1]
        h = int(OUT_W * (y2 - y1) / (x2 - x1)) // 2 * 2   # even height for yuv420p
        name = f"shot_{i:03d}_{hoop}_{dets[0]['t']:.0f}s.mp4"
        path = args.out / name

        # ffmpeg seeks, crops and encodes in one pass; nothing is decoded in
        # Python at all. The previous version pulled frames through OpenCV and
        # seeked with CAP_PROP_POS_FRAMES, which is fine on a 3GB 1080p file and
        # catastrophic on a 15GB 4K60 one -- a single seek to frame 36000 ran
        # past ten minutes without returning. At one seek per shot and roughly a
        # hundred shots that is not a slow step, it is a step that never
        # finishes. `-ss` BEFORE `-i` seeks on the container index instead of
        # decoding forward to the target.
        proc = subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error",
             "-ss", f"{t_start:.3f}", "-i", str(args.video), "-t", f"{t_stop - t_start:.3f}",
             "-vf", f"crop={x2-x1}:{y2-y1}:{x1}:{y1},scale={OUT_W}:{h}",
             "-c:v", "libx264", "-crf", str(CRF), "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(path)],
            capture_output=True)
        err = proc.stderr.decode("utf-8", "replace").strip()
        if err:
            print(f"    ffmpeg: {err[:200]}")
        n = int(round((t_stop - t_start) * fps))

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

    (args.out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    total = sum(c["size_kb"] for c in index)
    print(f"wrote {len(index)} clips, {total/1024:.1f} MB total -> {args.out}")


if __name__ == "__main__":
    main()
