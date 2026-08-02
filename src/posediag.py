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
from frames import frame_at

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
    # Everything scales off the frame width. The first version used fixed point
    # sizes tuned on a preview and they came out microscopic on a 3840-wide
    # frame, which is the whole failure this file exists to prevent: an image
    # whose own labels cannot be read is not evidence.
    S = w / 1700.0
    th = max(2, int(3 * S))

    if flight:
        # Over the arc's OWN span, not the whole window. Extrapolating a
        # parabola across the full 1.8s throws it hundreds of pixels off frame
        # in a few steps, so almost all of the drawn curve is clipped away and
        # what survives is a stub that looks like a bug. The honest curve is the
        # stretch the fit was actually made over.
        t_from = flight["t"]
        t_to = flight["t"] + flight["span"]
        prev = None
        for _t, x, y in arc.polyline(flight["fit"], t_from, t_to, 48):
            p = (int(x), int(y))
            if prev:
                cv2.line(fr, prev, p, AMBER, th, cv2.LINE_AA)
            prev = p
        # The samples the parabola was fitted to, so a fit resting on four
        # points cannot pass for one resting on twenty.
        for b in ball_track:
            if t_from - 0.02 <= b["t"] <= t_to + 0.02:
                cv2.circle(fr, (int(b["x"]), int(b["y"])), max(4, int(5 * S)), AMBER, -1, cv2.LINE_AA)
        ox, oy = int(flight["x"]), int(flight["y"])
        cv2.circle(fr, (ox, oy), int(20 * S), AMBER, th, cv2.LINE_AA)
        cv2.putText(fr, "the throw starts here", (ox + int(28 * S), oy + int(8 * S)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95 * S, AMBER, th, cv2.LINE_AA)

    at = next((b for b in ball_track if abs(b["t"] - t_rel) < 1e-6), None)
    if at:
        cv2.circle(fr, (int(at["x"]), int(at["y"])), int(24 * S), WHITE, th, cv2.LINE_AA)
        cv2.putText(fr, "ball", (int(at["x"]) + int(30 * S), int(at["y"]) - int(18 * S)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95 * S, WHITE, th, cv2.LINE_AA)

    placed = []
    for c in cands:
        if not c.get("box"):
            continue
        chosen = pick is not None and c["person"] == pick["person"]
        lift, gap = c.get("lift"), c.get("gap")
        col = GREEN if chosen else (RED if (lift or 0) > 0 else GRAY)
        x1, y1, x2, y2 = (int(v) for v in c["box"])
        cv2.rectangle(fr, (x1, y1), (x2, y2), col, th + (2 if chosen else 0))
        for k in (shooter.LW, shooter.RW):
            wpt = c["kp"][k] if c.get("kp") else None
            if wpt and wpt[2] > 0.3:
                cv2.circle(fr, (int(wpt[0]), int(wpt[1])), int(11 * S), col, -1, cv2.LINE_AA)
        tag = ("CHOSEN  " if chosen else "")
        tag += "hands not read" if lift is None else (
            f"hands {lift:+.2f} above head" + ("" if gap is None else f", {gap:.2f} from ball"))
        # Kept inside the frame, and nudged off any label already placed. Boxes
        # cluster in this footage, so labels drawn at each box's own corner land
        # on top of each other and clip off the right edge.
        (tw, tht), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.85 * S, th)
        tx = min(max(int(8 * S), x1), w - tw - int(8 * S))
        ty = max(int(26 * S), y1 - int(16 * S))
        while any(abs(ty - py) < tht * 1.4 and abs(tx - px) < max(tw, pw) * 0.7
                  for px, py, pw in placed):
            ty += int(tht * 1.5)
        placed.append((tx, ty, tw))
        cv2.putText(fr, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * S, col, th, cv2.LINE_AA)

    # Caption band. Words, not a color key -- an earlier version keyed markers
    # by color, got BGR backwards, and review reviewed a picture whose own legend
    # was wrong.
    lines = [meta["title"], meta["sub"], meta["rule"]]
    band = int(56 * S * len(lines) + 30 * S)
    fr[0:band] = (fr[0:band] * 0.22).astype(fr.dtype)
    for i, s in enumerate(lines):
        cv2.putText(fr, s, (int(22 * S), int(52 * S + i * 54 * S)), cv2.FONT_HERSHEY_SIMPLEX,
                    (1.25 if i == 0 else 1.0) * S, WHITE, th, cv2.LINE_AA)
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
        ball_track, per_person = shooter.scan_cached(args.video, n, float(j["t"]), cap, fps, det, pos)
        pick, how, extra = shooter.attribute(ball_track, per_person)
        flight, cands = extra["flight"], extra["cands"]
        if not pick:
            print(f"  #{n}: no answer ({'no arc fitted' if not flight else 'arc began with nobody near it'}"
                  f", and nobody released with hands up)")
            continue

        fr = frame_at(ROOT / "footage" / args.video, pick["t"])
        if fr is None:
            print(f"  #{n}: could not read the frame")
            continue

        note = (notes.get(n, {}).get("note") or "").strip()
        if how == "arc":
            title = (f"#{n}  the throw starts {pick['dist']:.0f}px from this person, "
                     f"at {pick['t']:.2f}s")
            rule = (f"amber = the ball's flight: {flight['n']} sightings, {flight['travel']:.0f}px "
                    f"traveled, fitted to {flight['rms']:.0f}px. green = who it came from.")
        else:
            title = (f"#{n}  no flight fitted, so the ball was carried: a dunk. "
                     f"release at {pick['t']:.2f}s")
            rule = ("green = chosen: hands up, and nearest the ball as the gap opened.  "
                    "red = hands up but further away.")
        meta = {
            "title": title,
            "sub": (f"you said: {note}" if note else "no note from you on this one")
                   + f"   ·   ball seen {len(ball_track)}x   ·   {len(per_person)} people seen",
            "rule": rule,
        }
        # Everyone seen gets a box, not only those who cleared the release test,
        # so a shooter the rule missed is still visible as a box review can point
        # at. Showing only the survivors hides exactly the failure worth finding.
        shown = list(cands)
        picked_ids = {c["person"] for c in cands}
        if pick["person"] not in picked_ids and pick.get("box"):
            shown.append({**pick, "lift": pick.get("lift") or 0.0, "gap": pick.get("gap") or 0.0})
        out = args.out / f"pose-diag-{n}.jpg"
        cv2.imwrite(str(out), draw(fr, shown, pick, flight, ball_track, pick["t"], meta),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        made.append({"n": n, "how": how, "others": len(cands) - 1,
                     "arc": bool(flight), "note": note})
        print(f"  #{n}: {how}, {len(cands)} hands-up candidates, "
              f"arc {'yes' if flight else 'no'} -> {out.name}")
    cap.release()
    print(f"\n{len(made)} diagnostic frames written to {args.out}")
    (ROOT / "out/posediag.json").write_text(json.dumps(made, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
