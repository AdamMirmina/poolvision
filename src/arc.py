"""Where did the ball come from? Fit its flight and run it backwards.

Every previous attempt asked "who was near the ball at some chosen instant".
That question has no good answer at a contested shot: the defender's hands are
beside the ball at exactly the moment it leaves, often nearer than the shooter
who just released it. the notes said so in plain language -- "blue shot it,
green defending", and the pipeline picked green. One in twelve correct.

A thrown ball follows a parabola, and a parabola has an origin. Fitting the
whole flight and extrapolating back to where it began uses every sample instead
of one frame, and a defender standing beside the ball's path does not sit at the
start of it. That is a different question with a well-defined answer.

In image coordinates the horizontal is linear in time and the vertical is
quadratic. Perspective bends a real parabola into a conic, but over the second
or so a pool shot lasts, and across a pool this shallow relative to the camera
distance, the quadratic form holds well enough to locate an origin.

Dunks have no parabola, which is a feature rather than a problem: a ball carried
to the rim will not fit an arc, so failing to fit IS the dunk detector.
"""

from __future__ import annotations


def fit_xy(pts: list[dict]) -> tuple[dict, float] | None:
    """Least squares: x linear in t, y quadratic in t.

    Returns the coefficients and the RMS residual in pixels. The residual is the
    part that matters -- it says whether this was a flight at all, and it is what
    separates a shot from a ball being carried, bounced or dribbled.
    """
    n = len(pts)
    if n < 4:
        return None
    t0 = pts[0]["t"]
    ts = [p["t"] - t0 for p in pts]
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]

    # x = d*t + e
    st = sum(ts); stt = sum(t * t for t in ts)
    sx = sum(xs); stx = sum(t * x for t, x in zip(ts, xs))
    den = n * stt - st * st
    if abs(den) < 1e-9:
        return None
    d = (n * stx - st * sx) / den
    e = (sx - d * st) / n

    # y = a*t^2 + b*t + c, by normal equations on [t^2, t, 1]
    s1, s2, s3, s4 = st, stt, sum(t ** 3 for t in ts), sum(t ** 4 for t in ts)
    sy = sum(ys); sty = sum(t * y for t, y in zip(ts, ys)); stty = sum(t * t * y for t, y in zip(ts, ys))
    A = [[s4, s3, s2], [s3, s2, s1], [s2, s1, float(n)]]
    B = [stty, sty, sy]
    # 3x3 Gaussian elimination with partial pivoting
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(A[r][i]))
        if abs(A[piv][i]) < 1e-12:
            return None
        A[i], A[piv] = A[piv], A[i]
        B[i], B[piv] = B[piv], B[i]
        for r in range(i + 1, 3):
            f = A[r][i] / A[i][i]
            for c in range(i, 3):
                A[r][c] -= f * A[i][c]
            B[r] -= f * B[i]
    sol = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        acc = B[i] - sum(A[i][c] * sol[c] for c in range(i + 1, 3))
        sol[i] = acc / A[i][i]
    a, b, c = sol

    resid = 0.0
    for t, x, y in zip(ts, xs, ys):
        dx = (d * t + e) - x
        dy = (a * t * t + b * t + c) - y
        resid += dx * dx + dy * dy
    rms = (resid / n) ** 0.5
    return {"t0": t0, "a": a, "b": b, "c": c, "d": d, "e": e}, rms


def at(fit: dict, t: float) -> tuple[float, float]:
    tt = t - fit["t0"]
    return fit["d"] * tt + fit["e"], fit["a"] * tt * tt + fit["b"] * tt + fit["c"]


# How far the arc must bend away from a straight line, in pixels, before it
# counts as a flight. A straight path fits a parabola perfectly well with a
# curvature of nearly zero, which is how a ball carried to the rim came back as
# a thrown one. Gravity is not optional in a real shot: at these distances a
# 0.7s flight sags tens of pixels below the chord, and a carried ball sags none.
MIN_SAG_PX = 14.0


def sag(fit: dict, t_from: float, t_to: float) -> float:
    """Greatest distance between the arc and the straight line across it."""
    span = t_to - t_from
    return abs(fit["a"]) * span * span / 8.0


def best_arc(track: list[dict], tol: float = 34.0, min_pts: int = 5):
    """The longest stretch of this track that behaves like a thrown ball.

    Grown by testing each candidate point against the fit so far, rather than
    against a residual averaged over the whole segment. That distinction
    matters: a synthetic ball held still for half a second and then thrown was
    picked up 0.17s too early by an average-residual rule, because five
    stationary samples hidden among sixteen flying ones barely move the mean.
    Judging each new point on its own distance from the arc stops exactly where
    the ball stopped flying, which is the moment it was in someone's hands.

    Every end point is considered, not only the last sample. A ball that gets
    blocked, or that bounces off the rim and carries on, has a real flight
    followed by something else, and insisting the arc reach the final sample
    finds nothing at all in those cases.

    A rising ball must also actually rise: `a` is vertical acceleration in image
    coordinates where y grows downward, so gravity is POSITIVE and a fit with
    a <= 0 curves the wrong way.
    """
    pts = sorted(track, key=lambda p: p["t"])
    n = len(pts)
    if n < min_pts:
        return None

    best = None
    for end in range(n, min_pts - 1, -1):
        start = end - min_pts
        got = fit_xy(pts[start:end])
        if not got:
            continue
        fit, rms = got
        if fit["a"] <= 0 or rms > tol:
            continue
        # Extend backwards one sample at a time, keeping a point only if it lies
        # on the arc the later samples already describe.
        while start > 0:
            cand = pts[start - 1]
            px, py = at(fit, cand["t"])
            if ((px - cand["x"]) ** 2 + (py - cand["y"]) ** 2) ** 0.5 > tol:
                break
            trial = fit_xy(pts[start - 1:end])
            if not trial or trial[0]["a"] <= 0 or trial[1] > tol:
                break
            start -= 1
            fit, rms = trial
        seg = pts[start:end]
        if sag(fit, seg[0]["t"], seg[-1]["t"]) < MIN_SAG_PX:
            continue                      # straight enough to be carried, not thrown
        if best is None or len(seg) > len(best["pts"]) or (
                len(seg) == len(best["pts"]) and rms < best["rms"]):
            best = {"start": start, "end": end, "fit": fit, "rms": rms, "pts": seg}
    return best


def release(track: list[dict], **kw):
    """Where and when the ball was let go, from the arc's own beginning.

    Returns the release point, the fit, and how confident the fit is. `None`
    means no ballistic segment was found at all, which for a shot that reached
    the rim means it was carried there: a dunk.
    """
    arc = best_arc(track, **kw)
    if not arc:
        return None
    first = arc["pts"][0]
    x, y = at(arc["fit"], first["t"])
    return {
        "t": first["t"], "x": x, "y": y,
        # The samples the fit was actually made from. Drawing every detection in
        # the time window instead put dots on the picture that the curve does not
        # explain, because the window can hold a second ball.
        "pts": [{"t": p["t"], "x": p["x"], "y": p["y"]} for p in arc["pts"]],
        "fit": arc["fit"], "rms": arc["rms"],
        "sag": round(sag(arc["fit"], first["t"], arc["pts"][-1]["t"]), 1),
        "span": arc["pts"][-1]["t"] - first["t"],
        "n": len(arc["pts"]),
    }


def polyline(fit: dict, t_from: float, t_to: float, steps: int = 24):
    """The fitted arc as points, for drawing over the clip."""
    out = []
    for i in range(steps + 1):
        t = t_from + (t_to - t_from) * i / steps
        x, y = at(fit, t)
        out.append((round(t, 3), round(x, 1), round(y, 1)))
    return out


def origin_at_person(fit, t_first, people_at, back_s: float = 1.2, step: float = 1 / 60,
                     reach_px: float = 90.0):
    """Run the parabola backwards until it meets someone's hands.

    The missing step, and the reason a correct arc still produced a wrong
    answer. `t_first` is where the TRACK begins, not where the THROW begins: if
    the detector does not pick the ball up until it is already in the air, the
    arc's first point floats in open water with nobody near it. That is exactly
    what one verified case looked like.

    The parabola describes the whole flight, including the part before the ball
    was first seen, so it can be run backwards. The throw began at the last
    moment before the ball was in anyone's hands, so walk back until the curve
    comes within arm's reach of a person and stop there.

    `people_at(t)` returns [(id, box)] at that moment. Boxes rather than centers:
    a shooter's arms are up and the ball leaves above their head, far from their
    middle and no distance at all from their outline.
    """
    best = None
    t = t_first
    n = int(back_s / step)
    for _ in range(n):
        t -= step
        x, y = at(fit, t)
        for pid, box in people_at(t):
            x1, y1, x2, y2 = box
            dx = max(x1 - x, 0, x - x2)
            dy = max(y1 - y, 0, y - y2)
            d = (dx * dx + dy * dy) ** 0.5
            if d <= reach_px:
                # STOP at the first person met walking backwards.
                #
                # Taking the earliest contact instead was wrong and verifiably
                # so: on a blocked shot it walked the full window past the
                # blocker and landed on an unrelated swimmer. The ball came from
                # whoever it touched most recently before the flight, and if that
                # was a blocker rather than the shooter, that is the honest
                # answer for the flight actually observed -- extrapolating a
                # parabola back THROUGH a deflection describes a trajectory the
                # ball never took.
                return {"t": t, "x": x, "y": y, "person": pid, "dist": d,
                        "walked": round(t_first - t, 2)}
    return None
