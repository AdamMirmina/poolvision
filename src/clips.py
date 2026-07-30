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
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Crop boxes, matching rimwatch's. Wider than the rim so the ball's approach and
# its exit are both in frame; a tight crop on the rim hides exactly the evidence
# a human needs.
RIM_BOXES = {
    "left": (188, 453, 1248, 1293),
    "right": (2860, 178, 3840, 1038),
}
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


def cluster(hits_by_hoop: dict, gap: float, min_dets: int):
    """Group detections into candidate events, then merge any whose CLIP WINDOWS
    would overlap.

    Clustering on detection gaps alone splits a single shot in two. A real shot
    goes: ball held low, a pause while the shooter sets, then the throw, the
    arc, and the fall. The detector loses the ball during the pause, so a
    gap rule cuts the shot in half and produces one clip of someone raising
    their arm and a separate clip of the ball arriving -- which is exactly what
    the marked shot hit on the very first clip he reviewed.

    Padding each event out to its clip window and merging overlaps fixes it,
    because the two halves of one shot are always closer together than the
    padding. The cost is that two genuinely separate shots close in time end up
    in one clip; those are flagged rather than silently merged, and review
    can say so in the notes.
    """
    events = []
    for hoop, hits in hits_by_hoop.items():
        hits = sorted(hits, key=lambda h: h["t"])
        cur: list = []
        for h in hits:
            if cur and h["t"] - cur[-1]["t"] > gap:
                events.append((hoop, cur))
                cur = []
            cur.append(h)
        if cur:
            events.append((hoop, cur))
    events = [(h, e) for h, e in events if len(e) >= min_dets]

    merged = []
    for hoop, dets in sorted(events, key=lambda e: (e[0], e[1][0]["t"])):
        start, end = dets[0]["t"] - LEAD_S, dets[-1]["t"] + TAIL_S
        if merged and merged[-1][0] == hoop and start <= merged[-1][2]:
            merged[-1][1].extend(dets)
            merged[-1][2] = max(merged[-1][2], end)
        else:
            merged.append([hoop, list(dets), end])
    out = [(h, sorted(d, key=lambda x: x["t"])) for h, d, _ in merged]
    out.sort(key=lambda e: e[1][0]["t"])
    return out


def main():
    args = parse_args()
    import cv2

    rw = json.loads(args.rimwatch.read_text(encoding="utf-8"))
    events = cluster(rw["hits"], args.gap, args.min_dets)
    print(f"{len(events)} candidate events")

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    args.out.mkdir(parents=True, exist_ok=True)
    index = []

    for i, (hoop, dets) in enumerate(events, 1):
        x1, y1, x2, y2 = RIM_BOXES[hoop]
        t_start = max(0.0, dets[0]["t"] - LEAD_S)
        t_end = dets[-1]["t"] + TAIL_S
        f0, f1 = int(t_start * fps), int(t_end * fps)
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
