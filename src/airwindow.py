"""When is the ball in the air, as offsets into each kind of clip.

First version marked the DESCENT, which is only the falling half of a shot, so
the gold arrived after the ball had already peaked. Correct, and by construction rather than by accident.

Now it marks the whole flight -- the anchored object's own track, rise included
-- which is what "in the air" means to someone watching.

Two clip types need two answers. The rim clip is cut with a 1.2s lead before the
descent; the wide clip (wideclips.py) uses 2.6s. Storing one offset and letting
the player guess which it belongs to is how the band ends up in the wrong place
on one of them, so both are computed here and stored separately.

    python src/airwindow.py
"""
import json, sys, os
sys.path.insert(0, "src")
import makemiss as M

# The release time, per shot, from the attribution pass.
#
# This is the piece that was missing. The rim detector only ever sees the ball
# inside the two crops around the hoops, so a track built from it starts when
# the ball ARRIVES near the rim -- which is why the band could not start early
# enough no matter how it was tuned. The release only exists in the full-frame attribution pass, and
# the first version of that script did not save it.
RELEASE = {}
for _f in os.listdir("out"):
    if _f.startswith("attribute_") and _f.endswith(".json"):
        for _r in json.load(open("out/" + _f)):
            if _r.get("releaseT") is not None and _r.get("n") is not None:
                RELEASE[(_r.get("video"), int(_r["n"]))] = float(_r["releaseT"])
print(f"{len(RELEASE)} shots have a known release time")

RIM_LEAD, MIN_LEAD = 1.2, 0.3      # clips.py
WIDE_LEAD = 2.6                    # wideclips.py
MAX_FLIGHT = 3.0                   # longer than this is the tracker, not a ball

judged = json.load(open("labels/allshots.json"))
out, missing = [], 0
for j in judged:
    if j["video"] not in M.RIMWATCH:
        continue
    flight = M.anchor_descent(j["video"], j["hoop"], j["t"], whole_flight=True)
    desc = M.anchor_descent(j["video"], j["hoop"], j["t"], whole_flight=False)
    if not flight or not desc:
        missing += 1
        continue
    # The flight lookup widens around the descent; keep it honest by starting no
    # earlier than the track really goes and ending at the descent's end, which
    # is the outcome.
    # Prefer the real release. Fall back to the track's first sighting near the
    # hoop, and say so in the output rather than pretending they are the same.
    known = RELEASE.get((j["video"], int(j["n"])))
    # A release after the descent has begun is not a release; refuse it rather
    # than drawing a band that runs backwards.
    if known is not None and known >= float(j["t"]):
        known = None
    t0 = known if known is not None else min(p["t"] for p in flight)
    src = "release" if known is not None else "first sighting near the hoop"
    t1 = max(desc[-1]["t"], max(p["t"] for p in flight if p["t"] <= desc[-1]["t"] + 0.5))
    if t1 - t0 > MAX_FLIGHT:
        t0 = t1 - MAX_FLIGHT
        src += " (capped)"

    lead = max(MIN_LEAD, min(RIM_LEAD, j["t"]))
    if j.get("clipStart"):
        lead = j["t"] - float(j["clipStart"])
    rim_start = j["t"] - lead
    wide_start = j["t"] - WIDE_LEAD
    out.append({
        "video": j["video"], "n": j["n"],
        "airStart": round(max(0.0, t0 - rim_start), 2),
        "airEnd": round(max(0.15, t1 - rim_start), 2),
        "wideAirStart": round(max(0.0, t0 - wide_start), 2),
        "wideAirEnd": round(max(0.15, t1 - wide_start), 2),
        "flight": round(t1 - t0, 2), "from": src,
    })

f = sorted(o["flight"] for o in out)
print(f"{len(out)} clips, {missing} with no matching track")
print(f"whole flight: min {f[0]:.2f}s  median {f[len(f)//2]:.2f}s  max {f[-1]:.2f}s")
import collections
print("start of the band came from:", dict(collections.Counter(o["from"] for o in out)))
json.dump(out, open("out/airwindows.json", "w"), indent=1)
print("wrote out/airwindows.json")
