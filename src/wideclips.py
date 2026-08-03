"""The whole play, release to outcome, for confirming attribution.

review, looking at the shooter crops: "some of these are not shots at all. can we
show the full clip at these times so i can see if it was really a shot, not just
zoomed on the person?"

That is right and the crop was the wrong evidence. A tight box around a person
shows who was holding the ball; it cannot show whether the ball then went at a
hoop, so it cannot distinguish a shot from a pass, a throw-back, or someone
messing about. Confirming attribution means seeing the whole play.

Full frame, downscaled. Deliberately not a clever crop spanning shooter and
hoop: the shooter can be anywhere and a crop that guesses wrong hides exactly
the thing being checked. The whole frame can never be wrong about what is in it.

Runs from before the release through the outcome at the rim, so one clip answers
both "was this a shot" and "who took it".

    python src/wideclips.py --video IMG_2481
"""

from __future__ import annotations

import argparse
import json

import hoops
import overlay
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_W = 960          # a quarter of 4K: the pool reads clearly, caps stay visible
LEAD_S = 2.6         # far enough back to catch the release
TAIL_S = 1.4         # far enough forward to see the ball reach the rim
CRF = 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2481")
    p.add_argument("--out", default="out/wide_clips")
    return p.parse_args()


def main():
    args = parse_args()
    import cv2
    import imageio_ffmpeg

    attributed = json.loads((ROOT / f"out/attribute_{args.video}.json").read_text(encoding="utf-8"))
    wanted = {int(r["n"]) for r in attributed if r.get("stage") == "attributed" and r.get("n") is not None}
    judged = {int(j["n"]): j for j in json.loads(((ROOT / "labels/allshots.json" if (ROOT / "labels/allshots.json").exists() else ROOT / "labels/judged.json")).read_text(encoding="utf-8"))
              if j["video"] == args.video + ".MOV"}

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / "footage" / f"{args.video}.MOV"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h = int(OUT_W * H / W) // 2 * 2
    ff = imageio_ffmpeg.get_ffmpeg_exe()

    # Drawn once: the camera does not move within a recording.
    lay = overlay.layer(hoops.rig_for(args.video + '.MOV'), OUT_W, h)
    made = 0
    for n in sorted(wanted):
        j = judged.get(n)
        if not j:
            continue
        t0 = max(0.0, float(j["t"]) - LEAD_S)
        t1 = float(j["tEnd"] if j["tEnd"] > j["t"] else j["t"] + 1.0) + TAIL_S
        f0, f1 = int(t0 * fps), int(t1 * fps)
        dest = out / f"{args.video}_{n}.mp4"
        proc = subprocess.Popen(
            [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{OUT_W}x{h}", "-r", f"{fps:.3f}", "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
            stdin=subprocess.PIPE)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        wrote = 0
        for _ in range(f1 - f0):
            ok, fr = cap.read()
            if not ok:
                break
            small = cv2.resize(fr, (OUT_W, h))
            # The rig's geometry on every frame, so any pause is readable.
            #
            overlay.burn(small, lay)
            proc.stdin.write(small.tobytes())
            wrote += 1
        proc.stdin.close()
        proc.wait()
        if wrote:
            made += 1
        if made % 10 == 0 and wrote:
            print(f"  {made} clips...")
    cap.release()
    total_mb = sum(f.stat().st_size for f in out.glob("*.mp4")) / 1e6
    print(f"{made} wide clips, {total_mb:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
