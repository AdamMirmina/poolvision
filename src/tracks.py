"""Link ball detections into per-object tracks.

The descent finder used to sort every detection by time and walk the list as
though it described one object. That holds while there is one ball on screen and
falls apart the moment there are two: on IMG_2481, 14% of frames carry more than
one detection, so the walk hops between different balls, reverses direction
constantly, and shreds a single descent into a dozen three-sample fragments.
That is how 9.4 minutes of shooting became "397 shots".

Tracking first, descents second. A descent belongs to ONE object or it isn't a
shot, and confirmed two balls in play is normal for a real game, so this
isn't an edge case to tolerate -- it's the common case.

It also removes a way to invent a basket outright: with a single merged
trajectory, ball A above the rim and ball B below it a few frames later look
exactly like one ball passing through.

Deliberately simple: nearest-neighbour association with constant-velocity
prediction. No Kalman filter, no appearance model. The objects are far apart
relative to how far they move between frames, which is the regime where simple
association is enough.
"""

from __future__ import annotations

from collections import defaultdict

MATCH_PX = 150      # a ball moves ~15-60 px/frame here; 150 tolerates a fast throw
MAX_MISS = 8        # frames a track survives unmatched (~0.27s of occlusion)
MIN_TRACK = 3       # shorter than this is noise, not an object
PREDICT_CAP = 5     # frames of velocity to extrapolate across a gap, at most
GAP_SLACK_PX = 30   # extra match radius per frame of gap
MAX_RADIUS_PX = 260 # ceiling on that growth -- see below


def build_tracks(hits: list[dict], match_px: float | None = None,
                 max_miss: int | None = None, min_track: int | None = None) -> list[list[dict]]:
    """Group detections into tracks, one per physical object.

    Returns tracks as lists of detection dicts, sorted by time.

    Defaults resolve from the module globals at CALL time, not definition time.
    Python binds default arguments once when the function is defined, so writing
    `max_miss: int = MAX_MISS` makes the value un-tunable from outside -- a
    parameter sweep that reassigns tracks.MAX_MISS then silently measures the
    same setting over and over and reports it as insensitive, which is exactly
    what happened the first time this was tuned.
    """
    # MATCH_PX is a distance PER FRAME, and the comment on it -- "a ball moves
    # ~15-60 px/frame here" -- was measured on 60fps footage. IMG_2482 is 30fps,
    # so the ball travels twice as far between consecutive sightings and a fast
    # throw exceeds the radius, breaking the track every frame. Six of its shots
    # were missed with runs of 1-2 points despite having MORE detections around
    # them than the median shot the pipeline finds (71-138 against a median 99).
    #
    # Raising the constant globally recovers two of them and costs three false
    # calls on the 60fps footage, which is the wrong trade. Scaling it by the
    # actual frame interval is the same fix without the cost: 30fps footage gets
    # a proportionally larger radius and 60fps footage is unchanged.
    #
    # Capped at 1.5x, swept: 1.0 and 1.3 recover nothing, 1.5 recovers both shots
    # with the named-false-call count UNCHANGED at 2, and 2.0 and 3.0 recover no
    # more while 3.0 adds a named false call. The cost of 1.5 is ten more
    # anonymous extra calls, which is the own trade -- a false call is a clip
    # dismissed in two seconds, a missed shot is invisible.
    #
    # The interval is inferred from the detections themselves rather than plumbed
    # through from the video, so nothing upstream has to remember to pass it.
    if match_px is None:
        ts = sorted({h["t"] for h in hits})
        dts = [b - a for a, b in zip(ts, ts[1:]) if 0 < b - a < 0.2]
        dt = min(dts) if dts else 1 / 60.0
        match_px = MATCH_PX * max(1.0, min(1.5, dt * 60.0))
    match_px = MATCH_PX if match_px is None else match_px
    max_miss = MAX_MISS if max_miss is None else max_miss
    min_track = MIN_TRACK if min_track is None else min_track
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for h in hits:
        by_frame[h["frame"]].append(h)

    open_tracks: list[dict] = []   # {"pts": [...], "vx": float, "vy": float, "miss": int}
    done: list[list[dict]] = []

    for frame in sorted(by_frame):
        dets = sorted(by_frame[frame], key=lambda d: -d["conf"])

        # Where each open track expects to find its object this frame.
        # Extrapolation is CAPPED. A ball rising at 27 px/frame, extrapolated
        # honestly across a 14-frame dropout, is predicted ~500 px from where it
        # actually is -- so the track fails to match, splits, and each half is
        # too short a drop to count as a shot. That is how the 6:28 make went
        # missing. Past a few frames, velocity says nothing useful: trust the
        # last known position and widen the search instead.
        preds = []
        for t in open_tracks:
            last = t["pts"][-1]
            gap = frame - last["frame"]
            step = min(gap, PREDICT_CAP)
            preds.append((last["x"] + t["vx"] * step, last["y"] + t["vy"] * step, gap))

        # Greedy nearest-neighbour, best (track, detection) pair first. Greedy is
        # fine here: two balls are rarely within a match radius of each other.
        pairs = []
        for ti, (px, py, gap) in enumerate(preds):
            # Ceiling matters. Widening the radius without one fixed a missed
            # make (a track that split across a dropout) and immediately caused
            # the opposite failure: after a 10-frame gap the radius reached
            # 420 px, wide enough to grab the OTHER ball, and one track then
            # stitched 80 detections across 7.4 seconds of separate shots.
            radius = min(match_px + GAP_SLACK_PX * max(0, gap - 1), MAX_RADIUS_PX)
            for di, d in enumerate(dets):
                dist = ((d["x"] - px) ** 2 + (d["y"] - py) ** 2) ** 0.5
                if dist <= radius:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t: set[int] = set()
        used_d: set[int] = set()
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = open_tracks[ti]
            prev = t["pts"][-1]
            gap = max(1, frame - prev["frame"])
            # Smooth the velocity so one noisy detection doesn't throw the next
            # prediction far enough to lose the object.
            t["vx"] = 0.5 * t["vx"] + 0.5 * (dets[di]["x"] - prev["x"]) / gap
            t["vy"] = 0.5 * t["vy"] + 0.5 * (dets[di]["y"] - prev["y"]) / gap
            t["pts"].append(dets[di])
            t["miss"] = 0

        for di, d in enumerate(dets):
            if di not in used_d:
                open_tracks.append({"pts": [d], "vx": 0.0, "vy": 0.0, "miss": 0})

        still_open = []
        for ti, t in enumerate(open_tracks):
            if ti < len(preds) and ti not in used_t:
                t["miss"] += 1
            if t["miss"] > max_miss:
                done.append(t["pts"])
            else:
                still_open.append(t)
        open_tracks = still_open

    done.extend(t["pts"] for t in open_tracks)
    return [sorted(p, key=lambda d: d["frame"]) for p in done if len(p) >= min_track]
