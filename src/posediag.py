"""Draw every assumption the attribution makes, on one frame, for one shot.



So one picture per shot carries the lot: who was seen, whose hands were up, how
far each pair of hands was from the ball, which release was chosen, and the
parabola the ball actually flew. Anything the pipeline believes and this image
does not show is a thing he cannot catch, and he has caught several.

Drawn from shooter.scan/candidates directly rather than from a saved summary.
The picture and the decision come out of the same call, so they cannot drift
apart -- which they did once before, and it cost a whole round of his feedback:
he judged a label that disagreed with a box, and every "no" became unusable
because there was no telling which half he was rejecting.

    python src/posediag.py --video IMG_2482.MOV --shots 1,4,7,8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc
import shooter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path("C:/dev/poolean/web/public/assets")

# BGR. Named, because a previous version labeled a color key in the legend and
# got the channel order backwards -- "cyan = above" was drawn in yellow, and review
# spent a round of review on a picture whose own key was wrong. Every marker here
# is labeled with WORDS on the image, not by color alone.
GREEN = (80, 220, 80)
RED = (60, 60, 235)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)
AMBER = (40, 190, 250)


def draw(fr, cands, pick, flight, ball_track, t_rel, meta):
    import cv2

    h, w = fr.shape[:2]

    # The parabola, when there is one. A dunk has no arc to fit, and that
    # absence is itself the dunk detector, so it gets said in words rather than
    # left blank.
    if flight:
        pts = arc.polyline(flight["fit"], ball_track[0]["t"], ball_track[-1]["t"], 40)
        prev = None
        for _t, x, y in pts:
            p = (int(x), int(y))
            if prev:
                cv2.line(fr, prev, p, AMBER, 3, cv2.LINE_AA)
            prev = p
        ox, oy = int(flight["x"]), int(flight["y"])
        cv2.circle(fr, (ox, oy), 13, AMBER, 3, cv2.LINE_AA)
        cv2.putText(fr, "arc starts here", (ox + 18, oy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMBER, 2, cv2.LINE_AA)

    # The ball where the release was called.
    at = next((b for b in ball_track if abs(b["t"] - t_rel) < 1e-6), None)
    if at:
        cv2.circle(fr, (int(at["x"]), int(at["y"])), 16, WHITE, 3, cv2.LINE_AA)
        cv2.putText(fr, "ball", (int(at["x"]) + 20, int(at["y"]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

    for c in cands:
        if not c.get("box"):
            continue
        chosen = pick is not None and c["person"] == pick["person"]
        col = GREEN if chosen else (RED if c["lift"] > 0 else GRAY)
        x1, y1, x2, y2 = (int(v) for v in c["box"])
        cv2.rectangle(fr, (x1, y1), (x2, y2), col, 3 if chosen else 2)
        for k in (shooter.LW, shooter.RW):
            wpt = c["kp"][k] if c.get("kp") else None
            if wpt and wpt[2] > 0.3:
                cv2.circle(fr, (int(wpt[0]), int(wpt[1])), 7, col, -1, cv2.LINE_AA)
        tag = ("CHOSEN  " if chosen else "") + f"hands {c['lift']:+.2f} up, {c['gap']:.2f} from ball"
        cv2.putText(fr, tag, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)

    # Caption band. Words, not a color key.
    band = 118
    fr[0:band] = (fr[0:band] * 0.25).astype(fr.dtype)
    lines = [meta["title"], meta["sub"], meta["rule"]]
    for i, s in enumerate(lines):
        cv2.putText(fr, s, (18, 34 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78 if i == 0 else 0.62, WHITE, 2, cv2.LINE_AA)
    return fr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="IMG_2482.MOV")
    p.add_argument("--shots", default="", help="comma-separated shot numbers; blank = all")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--out", type=Path, default=ASSETS)
    return p.parse_args()


def main():
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    want = {int(s) for s in args.shots.split(",") if s.strip()}
    shots = [j for j in shooter.load_shots(args.video, args.limit)
             if not want or int(j["n"]) in want]
    notes = {int(x["n"]): x for x in
             json.loads((ROOT / "labels/shooter_notes.json").read_text(encoding="utf-8"))}

    cap = cv2.VideoCapture(str(ROOT / "footage" / args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = YOLO("yolo11s.pt")
    pos = YOLO("yolo11s-pose.pt")

    args.out.mkdir(parents=True, exist_ok=True)
    made = []
    for j in shots:
        n = int(j["n"])
        ball_track, per_person = shooter.scan(cap, fps, float(j["t"]), det, pos)
        cands = shooter.candidates(per_person)
        flight = shooter.fit_arc(ball_track)
        pick = cands[0] if cands else None
        if not pick:
            print(f"  #{n}: nobody released with hands up, no frame drawn")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(pick["t"] * fps)))
        ok, fr = cap.read()
        if not ok:
            print(f"  #{n}: could not read the release frame")
            continue

        note = (notes.get(n, {}).get("note") or "").strip()
        meta = {
            "title": f"#{n}  release at {pick['t']:.2f}s"
                     + ("  ·  no arc fitted (carried, i.e. a dunk)" if not flight
                        else f"  ·  arc fits {flight['n']} points to {flight['rms']:.0f}px"),
            "sub": (f"you said: {note}" if note else "no note from you on this one")
                   + f"   ·   {len(cands)} people released with hands up"
                   + f"   ·   ball seen on {len(ball_track)} frames",
            "rule": "green = chosen (hands up, nearest the ball).  "
                    "red = hands up but further away.  gray = hands down, ruled out.",
        }
        out = args.out / f"pose-diag-{n}.jpg"
        cv2.imwrite(str(out), draw(fr, cands, pick, flight, ball_track, pick["t"], meta),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        made.append({"n": n, "chosen_gap": pick["gap"], "others": len(cands) - 1,
                     "arc": bool(flight), "note": note})
        print(f"  #{n}: chose gap {pick['gap']:.2f}, {len(cands)-1} others, "
              f"arc {'yes' if flight else 'no'} -> {out.name}")
    cap.release()
    print(f"\n{len(made)} diagnostic frames written to {args.out}")
    (ROOT / "out/posediag.json").write_text(json.dumps(made, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
