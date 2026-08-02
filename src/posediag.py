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
GREEN = (40, 150, 30)   # darker, to hold against the water
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
AMBER = (40, 190, 250)
# Wrists are colored by what they SAY, not by whose box they sit in. Right: a marker
# that repeats the box's color carries no information the box did not already
# give, and hands up or down is the single reading the whole rule turns on.
WRIST_UP = (60, 235, 245)     # yellow
WRIST_DOWN = (235, 150, 60)   # blue
LINE3 = (200, 120, 255)
RIMC = (60, 235, 245)      # yellow, on the ask


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
    # Compare against what this pass has ALREADY accepted, not only against the
    # caller's list. Since the drawing moved to a single pose pass, `known`
    # arrives empty, so nothing was checking a detection against its own
    # neighbours and one person could come back as two overlapping boxes with
    # two pairs of wrists --
    for i in range(len(r.boxes)):
        bx1, by1, bx2, by2 = r.boxes.xyxy[i].tolist()
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if not (pool[0] <= cx <= pool[2] and pool[1] <= cy <= pool[3]):
            continue
        dup = False
        for k in list(known) + out:
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
            # Still not enough. The two passes crop a shooter very differently --
            # one to the raised hand, one to the torso -- so neither center need
            # fall inside the other box while both plainly describe one person.
            # Horizontal agreement is the reliable part: two people standing
            # apart do not share a column.
            xo = max(0, min(bx2, kx2) - max(bx1, kx1))
            if xo > 0.55 * min(bx2 - bx1, kx2 - kx1) and oy > 0:
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
         hoop_center=None, rim=None, rim_tilt=0.0, ball_now=None):
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
        # Start the curve at the RELEASE, not at the first sighting.
        #
        # It should. flight["t"] is where the detector
        # first caught the ball, which is above and after the hands; the release
        # is where the parabola is now walked back to, and the frame being drawn
        # is that moment. Extending the curve back to it puts its start on the
        # ball in the shooter's hands, which is the thing the picture is claiming.
        t_from = min(flight["t"], t_rel)
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
        ox_, oy_ = arc.at(flight["fit"], t_from)
        # Snap to the real ball when it is visible in this frame.
        #
        # The extrapolated point is where the FITTED curve says the ball was, and
        # a parabola fitted to the flight is not a perfect description of the
        # moment it was still being held -- When the detector can see the ball in the frame
        # being drawn, that is better evidence than the fit, so the marker moves
        # onto it.
        if ball_now is not None:
            bnx, bny = ball_now
            if ((bnx - ox_) ** 2 + (bny - oy_) ** 2) ** 0.5 < 220:
                ox_, oy_ = bnx, bny
        ox, oy = int(ox_) - dx, int(oy_) - dy
        r0 = int(13 * S)
        cv2.line(fr, (ox - r0, oy), (ox + r0, oy), AMBER, th, cv2.LINE_AA)
        cv2.line(fr, (ox, oy - r0), (ox, oy + r0), AMBER, th, cv2.LINE_AA)

    # The boundary as a WALL standing on the pool, not a wash of color over
    # half the picture. with "walls and
    # ceiling to that vertical wall."
    #
    # A real vertical wall photographed from above projects as a band above its
    # base line, taller at the near end than the far one, because the near end is
    # closer to the camera. The base runs from the deck post (near) out past the
    # diving board (far), so the height tapers along it. That taper is what makes
    # it read as standing up out of the water rather than lying flat on it.
    if three:
        (lx1, ly1), (lx2, ly2) = three
        ax, ay = int(lx1) - dx, int(ly1) - dy
        bx_, by_ = int(lx2) - dx, int(ly2) - dy
        # The far end of the line is an extrapolation well past the diving board
        # and lands off the top of the frame, so the wall built on it ran off
        # screen and read as a slanted band rather than a wall. Clipped to the
        # picture, leaving headroom for the wall's own height.
        ok_, pa, pb = cv2.clipLine((0, int(160 * S), fr.shape[1], fr.shape[0]),
                                   (ax, ay), (bx_, by_))
        if ok_:
            (ax, ay), (bx_, by_) = pa, pb
        h_near, h_far = int(300 * S), int(96 * S)
        top_a = (ax, ay - h_near)
        top_b = (bx_, by_ - h_far)
        poly = np.array([[ax, ay], [bx_, by_], top_b, top_a], np.int32)
        face = fr.copy()
        cv2.fillPoly(face, [poly], LINE3)
        cv2.addWeighted(face, 0.26, fr, 0.74, 0, fr)
        # The edges are what sell it: base on the water, two uprights, and the
        # top edge that is the "ceiling" of the wall.
        cv2.line(fr, (ax, ay), (bx_, by_), LINE3, th + 1, cv2.LINE_AA)          # base
        cv2.line(fr, top_a, top_b, LINE3, th + 1, cv2.LINE_AA)                  # top
        cv2.line(fr, (ax, ay), top_a, LINE3, th + 1, cv2.LINE_AA)               # near upright
        cv2.line(fr, (bx_, by_), top_b, LINE3, th, cv2.LINE_AA)                 # far upright
        # A few verticals across it, so the surface reads as a plane rather than
        # an outlined shape.
        for k in range(1, 6):
            f_ = k / 6.0
            bxk = int(ax + (bx_ - ax) * f_), int(ay + (by_ - ay) * f_)
            txk = bxk[0], int(bxk[1] - (h_near + (h_far - h_near) * f_))
            cv2.line(fr, bxk, txk, LINE3, max(1, th - 2), cv2.LINE_AA)

    # The rim as a tilted ellipse, and the net box under it. Both are the shapes
    # the make rules are measured against, so both get drawn on every frame: a
    # rule is only as trustworthy as the box it reads, and drawing them is how
    # that gets checked by eye instead of assumed.
    if rim is not None:
        rx1, ry1, rx2, ry2 = rim
        cv2.ellipse(fr, (int((rx1 + rx2) / 2) - dx, int((ry1 + ry2) / 2) - dy),
                    (int((rx2 - rx1) / 2), int((ry2 - ry1) / 2)), rim_tilt, 0, 360,
                    RIMC, max(2, th - 1), cv2.LINE_AA)
        cv2.rectangle(fr, (int(rx1) - dx, int(ry2) - dy),
                      (int(rx2) - dx, int(ry2 + (ry2 - ry1) * 1.5) - dy),
                      RIMC, max(2, th - 2))

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
        # Pose read from THIS frame, not from a cached snapshot at another time.
        #
        # Because the release is
        # now walked back to the shooter's hands, the frame drawn can be up to
        # 0.3s earlier than the nearest pose sample, and in 0.3s an arm coming
        # down travels most of its own length. The box and the keypoints were
        # honest readings of a different instant than the pixels underneath them.
        #
        # The scan's samples still decide WHO shot -- they are the high-resolution
        # look around the ball. They just stop being what gets drawn.
        # Where the ball actually is in the frame being drawn, for the marker.
        ball_now = None
        rb = det.predict(fr, conf=0.10, verbose=False, classes=[shooter.BALL], imgsz=1280)[0]
        bestd = None
        for b in rb.boxes:
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            cxb, cyb = (bx1 + bx2) / 2, (by1 + by2) / 2
            if not (pool[0] <= cxb <= pool[2] and pool[1] <= cyb <= pool[3]):
                continue
            if pick.get("box"):
                px1, py1, px2, py2 = pick["box"]
                d0 = ((cxb - (px1 + px2) / 2) ** 2 + (cyb - (py1 + py2) / 2) ** 2) ** 0.5
            else:
                d0 = 0
            if bestd is None or d0 < bestd:
                bestd, ball_now = d0, (cxb, cyb)

        people = others_in_frame(fr, [], pos, pool)
        if pick.get("box"):
            px1, py1, px2, py2 = pick["box"]
            best, bi = 0.0, None
            for i, c in enumerate(people):
                bx1, by1, bx2, by2 = c["box"]
                ox = max(0, min(bx2, px2) - max(bx1, px1))
                oy = max(0, min(by2, py2) - max(by1, py1))
                small = min((bx2 - bx1) * (by2 - by1), (px2 - px1) * (py2 - py1)) or 1
                if ox * oy / small > best:
                    best, bi = ox * oy / small, i
            if bi is not None and best > 0.2:
                # The chosen person, now described by this frame's own pose.
                people[bi] = {**people[bi], "person": pick["person"],
                              "gap": pick.get("gap"), "lift": pick.get("lift")}
            elif pick.get("box"):
                people.append({k: pick.get(k) for k in ("person", "box", "kp", "gap", "lift")})
        # One more pass, over the FINAL list. The earlier dedupe only guarded the
        # drawing pass against the decision pass, and review kept seeing a shooter
        # wearing both a green box and a black one -- because the decision pass
        # itself sometimes tracks one person as two, so both boxes come from the
        # same place and neither is "extra". Anything that heavily overlaps the
        # chosen person is the chosen person.
        if pick.get("box"):
            px1, py1, px2, py2 = pick["box"]
            keep = []
            for c in people:
                if c["person"] == pick["person"]:
                    keep.append(c)
                    continue
                bx1, by1, bx2, by2 = c["box"]
                ox = max(0, min(bx2, px2) - max(bx1, px1))
                oy = max(0, min(by2, py2) - max(by1, py1))
                inter = ox * oy
                small = min((bx2 - bx1) * (by2 - by1), (px2 - px1) * (py2 - py1))
                if small > 0 and inter / small > 0.35:
                    continue
                if ox > 0.5 * min(bx2 - bx1, px2 - px1) and oy > 0.25 * min(by2 - by1, py2 - py1):
                    continue
                keep.append(c)
            people = keep
        # Two or three, from the shooter's hips against this hoop's boundary.
        three, marg = threept.call(rig, j.get("hoop", "left"), pick.get("kp"))
        points = None if three is None else (3 if three else 2)
        if pick.get("box") and not any(p["person"] == pick["person"] for p in people):
            people.append({k: pick.get(k) for k in ("person", "box", "kp", "gap", "lift")})

        sub, off = crop_to_action(fr, people, flight, ball_track, pick["t"], pool=rig.pool)
        hoop = j.get("hoop", "left")
        rr = rig.rims[hoop]
        rim_center = ((rr[0] + rr[2]) / 2, (rr[1] + rr[3]) / 2)
        tilt = (rig.tilt or {}).get(hoop, 0.0)
        out = args.out / f"pose-diag-{n}.jpg"
        cv2.imwrite(str(out), draw(sub.copy(), people, pick, flight, ball_track, pick["t"], off, rim=rr,
                             rim_tilt=tilt, ball_now=ball_now),
                    [cv2.IMWRITE_JPEG_QUALITY, 86])
        line = (rig.three_lines or {}).get(hoop)
        if line:
            cv2.imwrite(str(args.out / f"pose-diag-{n}-3pt.jpg"),
                        draw(sub.copy(), people, pick, flight, ball_track, pick["t"], off,
                             three=line, hoop_center=rim_center, rim=rr, rim_tilt=tilt,
                             ball_now=ball_now),
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
            "rim": [round(v) for v in rr], "tilt": tilt,
            "net": [round(rr[0]), round(rr[3]), round(rr[2]),
                    round(rr[3] + (rr[3] - rr[1]) * 1.5)],
            "wide": list(rig.pool),
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
