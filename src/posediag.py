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

import numpy as np

import arc
import hoops
import shooter
import threept
from frames import frame_at

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path("C:/dev/poolean/web/public/assets")

# BGR. Named, because a previous version labeled a color key in the legend and
# got the channel order backwards -- "cyan = above" was drawn in yellow, and review
# spent a round of review on a picture whose own key was wrong. Every marker here
# is labeled with WORDS on the image, not by color alone.
GREEN = (80, 220, 80)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
AMBER = (40, 190, 250)
# Wrists are colored by what they SAY, not by whose box they sit in. Right: a marker
# that repeats the box's color carries no information the box did not already
# give, and hands up or down is the single reading the whole rule turns on.
WRIST_UP = (60, 235, 245)     # yellow
WRIST_DOWN = (235, 150, 60)   # blue
LINE3 = (200, 120, 255)


def everyone_at(per_person, t):
    """Every person seen, with their reading at the moment nearest `t`.

    candidates() returns only those who cleared the release test, and drawing
    only those hides the failure worth finding: the whole point was that the
    real shooter was sometimes not boxed at all. A person the rule discarded
    still has to appear, so he can point at them.
    """
    out = []
    for key, v in per_person.items():
        snaps = [(abs(float(tt) - t), float(tt), v["at"][tt]) for tt in v["at"]]
        if not snaps:
            continue
        dt, tt, s = min(snaps, key=lambda z: z[0])
        if dt > 0.30 or not s.get("box"):
            continue
        out.append({"person": key, "box": s["box"], "kp": s.get("kp"),
                    "gap": s.get("gap"), "lift": s.get("lift")})
    return out


def others_in_frame(fr, known, pos, pool, conf=0.25):
    """People the decision never saw, for the picture only.

    Matched against the boxes already found by overlap, so a person seen by both
    passes is not drawn twice with two slightly different boxes.
    """
    r = pos.predict(fr, conf=conf, verbose=False, imgsz=1280)[0]
    if r.keypoints is None:
        return []
    out = []
    for i in range(len(r.boxes)):
        bx1, by1, bx2, by2 = r.boxes.xyxy[i].tolist()
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if not (pool[0] <= cx <= pool[2] and pool[1] <= cy <= pool[3]):
            continue
        dup = False
        for k in known:
            kx1, ky1, kx2, ky2 = k["box"]
            ox = max(0, min(bx2, kx2) - max(bx1, kx1))
            oy = max(0, min(by2, ky2) - max(by1, ky1))
            # Overlap alone was not enough. The decision pass boxes a shooter
            # from raised hand to hip while the drawing pass boxes the torso, so
            # the same person came back as two boxes whose overlap fell under any
            # sane area threshold -- review saw one person wearing a black box and
            # a green one. Center containment catches it: two boxes on one person
            # always contain each other's middle, however differently they are
            # cropped.
            if ox * oy > 0.25 * min((bx2 - bx1) * (by2 - by1), (kx2 - kx1) * (ky2 - ky1)):
                dup = True
                break
            mx, my = (bx1 + bx2) / 2, (by1 + by2) / 2
            kmx, kmy = (kx1 + kx2) / 2, (ky1 + ky2) / 2
            if (kx1 <= mx <= kx2 and ky1 <= my <= ky2) or (bx1 <= kmx <= bx2 and by1 <= kmy <= by2):
                dup = True
                break
        if dup:
            continue
        out.append({"person": f"x{i}", "box": (bx1, by1, bx2, by2),
                    "kp": r.keypoints.data[i].tolist(), "gap": None, "lift": None})
    return out


def crop_to_action(fr, people, flight, ball_track, t_rel, pad=180, pool=None):
    """The whole pool, plus anything happening outside it.

    An earlier version cropped tight to the people and the ball, which made the
    marks bigger but hid the thing being judged: review wants to see everyone in
    the water, because a shooter the pipeline never boxed is exactly the failure
    worth catching and a tight crop can leave them out of frame entirely. "on
    these images i want whole pool visible, at least, showing all people in the
    pool."

    So the crop starts from the pool and only ever grows.

    Returns the cropped frame and the (dx, dy) to subtract from every drawing
    coordinate.
    """
    xs, ys = [], []
    if pool:
        xs += [pool[0], pool[2]]
        ys += [pool[1], pool[3]]
    for p in people:
        x1, y1, x2, y2 = p["box"]
        xs += [x1, x2]
        ys += [y1, y2]
    for b in ball_track:
        if flight and not (flight["t"] - 0.05 <= b["t"] <= flight["t"] + flight["span"] + 0.05):
            continue
        xs.append(b["x"])
        ys.append(b["y"])
    if flight:
        xs.append(flight["x"])
        ys.append(flight["y"])
    if not xs:
        return fr, (0, 0)

    h, w = fr.shape[:2]
    x1, x2 = max(0, int(min(xs)) - pad), min(w, int(max(xs)) + pad)
    y1, y2 = max(0, int(min(ys)) - pad), min(h, int(max(ys)) + pad)

    # Hold 16:9 so the still and the clip below it read as the same scene rather
    # than two differently-shaped pictures of it.
    cw, ch = x2 - x1, y2 - y1
    if cw / max(1, ch) < 16 / 9:
        want = int(ch * 16 / 9)
        grow = (want - cw) // 2
        x1, x2 = max(0, x1 - grow), min(w, x2 + grow)
    else:
        want = int(cw * 9 / 16)
        grow = (want - ch) // 2
        y1, y2 = max(0, y1 - grow), min(h, y2 + grow)
    return fr[y1:y2, x1:x2].copy(), (x1, y1)


def draw(fr, people, pick, flight, ball_track, t_rel, off=(0, 0), three=None,
         hoop_center=None):
    import cv2

    dx, dy = off
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
            p = (int(x) - dx, int(y) - dy)
            if prev:
                cv2.line(fr, prev, p, AMBER, th, cv2.LINE_AA)
            prev = p
        # The samples the parabola was fitted to, so a fit resting on four
        # points cannot pass for one resting on twenty.
        for b in ball_track:
            if t_from - 0.02 <= b["t"] <= t_to + 0.02:
                cv2.circle(fr, (int(b["x"]) - dx, int(b["y"]) - dy), max(4, int(5 * S)),
                           AMBER, -1, cv2.LINE_AA)
        ox, oy = int(flight["x"]) - dx, int(flight["y"]) - dy
        r0 = int(13 * S)
        cv2.line(fr, (ox - r0, oy), (ox + r0, oy), AMBER, th, cv2.LINE_AA)
        cv2.line(fr, (ox, oy - r0), (ox, oy + r0), AMBER, th, cv2.LINE_AA)

    # ONE boundary, for the hoop this shot went to, shaded on the side away from
    # that hoop. The side test runs per pixel rather than by building a polygon:
    # the line crosses the frame at an arbitrary angle and clipping a half-plane
    # to a rectangle by hand is where the off-by-one bugs live.
    if three and hoop_center is not None:
        (lx1, ly1), (lx2, ly2) = three
        ax, ay = lx1 - dx, ly1 - dy
        bx_, by_ = lx2 - dx, ly2 - dy
        yy, xx = np.mgrid[0:fr.shape[0], 0:fr.shape[1]]
        side = (bx_ - ax) * (yy - ay) - (by_ - ay) * (xx - ax)
        hx, hy_ = hoop_center[0] - dx, hoop_center[1] - dy
        hside = (bx_ - ax) * (hy_ - ay) - (by_ - ay) * (hx - ax)
        far = (side > 0) if hside < 0 else (side < 0)
        shade = fr.copy()
        shade[far] = LINE3
        cv2.addWeighted(shade, 0.20, fr, 0.80, 0, fr)
        cv2.line(fr, (int(ax), int(ay)), (int(bx_), int(by_)), LINE3, th + 1, cv2.LINE_AA)
        cv2.circle(fr, (int(bx_), int(by_)), int(14 * S), LINE3, -1, cv2.LINE_AA)
        cv2.putText(fr, "3 from this side", (int(ax) - int(300 * S), int(ay) - int(20 * S)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * S, LINE3, th, cv2.LINE_AA)

    at = next((b for b in ball_track if abs(b["t"] - t_rel) < 1e-6), None)
    if at:
        bx, by = int(at["x"]) - dx, int(at["y"]) - dy
        cv2.circle(fr, (bx, by), int(18 * S), WHITE, th, cv2.LINE_AA)

    # EVERY person, not only the ones that cleared the release test.
    #
    # The prose caption that used to sit in a black band across the top is gone.
    # It said in three long lines what the marks already say, it covered the top
    # of the frame, and he called it wordy and messy. Those facts now render as
    # HTML under the picture, where they wrap and can be read without competing
    # with the image.
    placed = []
    for c in sorted(people, key=lambda z: -(z.get("lift") or -9)):
        chosen = pick is not None and c["person"] == pick["person"]
        lift, gap, kp = c.get("lift"), c.get("gap"), c.get("kp")
        up = (lift or 0) > 0
        col = GREEN if chosen else BLACK
        x1, y1, x2, y2 = (int(v) for v in c["box"])
        x1, x2, y1, y2 = x1 - dx, x2 - dx, y1 - dy, y2 - dy
        cv2.rectangle(fr, (x1, y1), (x2, y2), col, th + (2 if chosen else 0))

        # Head level, which is the line "hands up" is measured against. Without
        # it the lift number is a claim rather than something checkable by eye.
        hy = shooter.head_y(kp) if kp else None
        if hy is not None:
            hy -= dy
        if hy is not None:
            cv2.line(fr, (x1, int(hy)), (x2, int(hy)), col, max(1, th - 1), cv2.LINE_AA)

        # Wrists, marked up or down against that line rather than by color
        # alone. A previous version keyed this by color, got BGR backwards, and
        # review reviewed a picture whose own legend was wrong.
        for k in (shooter.LW, shooter.RW):
            wpt = kp[k] if kp else None
            if not wpt or wpt[2] <= 0.3:
                continue
            wx, wy = int(wpt[0]) - dx, int(wpt[1]) - dy
            above = hy is not None and wy < hy
            wcol = WRIST_UP if above else WRIST_DOWN
            cv2.circle(fr, (wx, wy), int(10 * S), wcol, -1, cv2.LINE_AA)
            cv2.circle(fr, (wx, wy), int(10 * S), (20, 20, 20), max(1, th - 2), cv2.LINE_AA)

        # Only the shooter is labeled. Everyone else is a box, a head line and
        # two wrist markers, which already say everything the numbers did.
        tag = "SHOOTER" if chosen else ""
        if not tag:
            continue
        (tw, tht), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.62 * S, max(1, th - 1))
        tx = min(max(int(8 * S), x1), w - tw - int(8 * S))
        ty = max(int(22 * S), y1 - int(12 * S))
        while any(abs(ty - py) < tht * 1.5 and abs(tx - px) < max(tw, pw) * 0.75
                  for px, py, pw in placed):
            ty += int(tht * 1.6)
        placed.append((tx, ty, tw))
        cv2.putText(fr, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62 * S, col,
                    max(1, th - 1), cv2.LINE_AA)
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

    rig = hoops.rig_for(args.video)
    pool = rig.pool
    # The make/miss model's own call on each shot, so the page can show what the
    # pipeline thinks happened next to who it thinks shot it.
    pm = ROOT / f"out/pmake_{args.video.replace('.MOV','')}.json"
    pmake = json.loads(pm.read_text(encoding="utf-8")) if pm.exists() else {}
    args.out.mkdir(parents=True, exist_ok=True)
    made = []
    for j in shots:
        n = int(j["n"])
        ball_track, per_person = shooter.scan_cached(args.video, n, float(j["t"]),
                                                     cap, fps, det, pos, pool=pool)
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
        people = everyone_at(per_person, pick["t"])
        # Everyone ELSE in the pool, from a full-frame pass at this one frame.
        #
        # Because the
        # decision runs pose on a 1500x1100 window around the ball, at native
        # resolution, and anyone outside that window is never seen. That window
        # is right for DECIDING -- it makes distant swimmers two and a half times
        # bigger and the shooter is by definition next to the ball -- but it is
        # wrong for a picture meant to show the whole pool.
        #
        # So the picture gets its own pass. These extra people are drawn and
        # never voted on, which keeps the two jobs apart: the decision still
        # comes from the high-resolution window, and the image still shows
        # everybody. One extra inference per shot.
        people += others_in_frame(fr, people, pos, pool)
        # Two or three, from the shooter's hips against this hoop's boundary.
        three, marg = threept.call(rig, j.get("hoop", "left"), pick.get("kp"))
        points = None if three is None else (3 if three else 2)
        if pick.get("box") and not any(p["person"] == pick["person"] for p in people):
            people.append({k: pick.get(k) for k in ("person", "box", "kp", "gap", "lift")})

        sub, off = crop_to_action(fr, people, flight, ball_track, pick["t"], pool=rig.pool)
        hoop = j.get("hoop", "left")
        out = args.out / f"pose-diag-{n}.jpg"
        cv2.imwrite(str(out), draw(sub.copy(), people, pick, flight, ball_track, pick["t"], off),
                    [cv2.IMWRITE_JPEG_QUALITY, 86])
        line = (rig.three_lines or {}).get(hoop)
        rr = rig.rims[hoop]
        rim_center = ((rr[0] + rr[2]) / 2, (rr[1] + rr[3]) / 2)
        if line:
            cv2.imwrite(str(args.out / f"pose-diag-{n}-3pt.jpg"),
                        draw(sub.copy(), people, pick, flight, ball_track, pick["t"], off,
                             three=line, hoop_center=rim_center),
                        [cv2.IMWRITE_JPEG_QUALITY, 86])
        made.append({
            "n": n, "how": how, "note": note, "t": round(pick["t"], 2),
            "people": len(people), "handsUp": sum(1 for p in people if (p.get("lift") or 0) > 0),
            "ballSeen": len(ball_track),
            "dist": None if pick.get("dist") is None else round(pick["dist"]),
            "arcN": flight["n"] if flight else None,
            "arcTravel": round(flight["travel"]) if flight else None,
            "arcRms": round(flight["rms"], 1) if flight else None,
            "pMake": pmake.get(str(n)),
            "points": points, "threeMargin": None if marg is None else round(marg, 2),
            "hoop": j.get("hoop", "left"),
            "has3pt": bool((rig.three_lines or {}).get(j.get("hoop", "left")) and rig.quad),
        })
        print(f"  #{n}: {how}, {len(people)} people boxed, "
              f"{'arc' if flight else 'no arc'}, pMake {pmake.get(str(n))} -> {out.name}")
    cap.release()
    print(f"\n{len(made)} diagnostic frames written to {args.out}")
    (ROOT / "out/posediag.json").write_text(json.dumps(made, indent=1), encoding="utf-8")
    # The facts the picture no longer carries. They render as HTML under the
    # image, where they wrap and can be read, instead of as a black band across
    # the top of the frame competing with the thing being judged.
    (args.out / "pose-diag.json").write_text(
        json.dumps({str(m["n"]): m for m in made}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
