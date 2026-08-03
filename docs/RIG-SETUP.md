# Setting up a new camera angle

Every session the camera moves, and every geometric thing the pipeline believes
has to be re-measured against the new view. This is the order to do it in, what
"measured" means for each, and the mistakes that have actually been made on each
one so they are not repeated.

review, after a long round of me getting these wrong one at a time: "you should be
documenting how im asking you to set up these boxes and walls so on a new angle
you'll be able to do it first try."

Everything lives in `src/hoops.py` as one `Rig` per recording. A recording with
no entry raises rather than borrowing another session's numbers, because a wrong
box produces a confident-looking run that finds nothing, and "zero shots
detected" reads as a detector problem rather than a coordinates problem.

## The rule that governs all of it

**Draw it on a real frame and look at it before trusting a single number that
comes out of it.** Every item below was got wrong at least once, and every one
was caught by drawing rather than by reasoning. An automatic fit that returns a
plausible number is not evidence; the picture is.

Use an averaged frame for anything fixed (`src/newrig.py` blends several so
swimmers and splashes wash out and only furniture survives). Use a real frame for
anything involving people or water surface.

---

## 1. Rim boxes (`rims`)

Tight on the hoop. The box center feeds the strongest make/miss feature and its
width is the unit everything else is normalized by, so slack here is slack in
every number downstream.

**How:** mask the orange-red steel (`(H<=25 | H>=335) & S>70 & V>60`) on an
averaged frame, take the largest contour near the expected position, use its
bounding box.

**What went wrong:** three hand-drawn passes produced boxes that contained both
rims but sat loose around them: 204x108 for a rim that is actually 164x60.
Hand-drawing is not good enough.

**The trap:** `cv2.fitEllipse` on a partial arc extrapolates wildly. The left rim
gave a 265px-wide ellipse from a contour spanning 146px, because its far side
washes out against the deck. If the mask only catches part of the ring, measure
the visible ring off a 4x zoom instead of trusting the fit.

## 2. Rim tilt (`tilt`)

A rim seen from above and to one side is a *tilted* ellipse. Drawing it
axis-aligned is visibly wrong along its edges, and it matters because the
ball-through-the-hoop test is a containment check against exactly this shape.

**How:** do NOT fit it. Fitting produced 168x180 at 56 degrees for a rim that is
plainly wide and flat, the same partial-arc failure as above. Take the correction
by eye: on this camera the near rim wants its right side lifted and the far rim
its left, about 9 degrees each, opposite signs. Image y grows downward, so
lifting the right side is a NEGATIVE angle.

## 3. Detector crops (`dets`) and review crops (`crops`)

Two different jobs, and they were one box until they were measured apart.

- `dets` are tight (rim + ~250x180). They stack into ONE inference per frame and
  arrive near native scale. Measured: 168s against 308s for the same 1800 frames,
  and 117 ball detections against 15.
- `crops` are wide (rim + ~430x400) and are what a person watches. A clip cropped
  tight on the rim shows a ball appearing and disappearing with no shooter.

**Settled question, do not re-litigate:** a taller detector crop does NOT help.
Same video, same minute, tight against one twice as tall: 233 detections against
229, and the far hoop collapsed from 21 to 3 because a taller canvas is
downscaled harder for the same imgsz.

## 4. Scene box (`pool`)

Filters every ball and person detection. Measure it by segmenting the water
(`H 165-215, S>60, V>90`), take the largest component, then pad out generously:
heads and arms rise above the waterline and a hoop can sit outside the water
entirely.

**What went wrong:** this was a hardcoded constant from a previous camera
position. Drawn on the new session's frame it excluded the ENTIRE left hoop and
the whole bottom-left of the water. It fails silently: detections outside it are
simply dropped.

## 5. Waterline and wall slope (`water`)

Per hoop: where the water begins below it, and which way the pool wall runs
there.

**How:** walk down several columns under the rim until the water mask starts, fit
a line through those entry points. Its intercept is the waterline, its slope is
the wall. On this camera the wall runs about 35 degrees, opposite signs at the
two ends.

**Why it matters:** the drop zone must sit fully IN the water and follow the
wall. Starting it at the rim's bottom edge put its top on the deck, where a ball
can never be, and an axis-aligned box either clips into the deck at one end or
misses water at the other.

## 6. Three-point wall (`three_lines`, `quad`)

Two measured points define each boundary: the deck post, and where the line ends
by the diving board.

- **Anchor at the BASE of the white pole**, where it enters the float. Not the
  detection box's center, which sits on the blue base well left of the pole. Not
  the box's top edge either. If an automatic search keeps latching onto the cord
  at the float's edge, read it off a 3x zoom.
- **Which post serves which hoop is crossed over**, and this is the rule
  rather than an inference: "if someone is shooting on right hoop and they're
  behind the left post it's a 3."
- **The line runs the LENGTH of the pool**, from the post out past the diving
  board, separating the deep end from the shallow end. The first version ran it
  across the pool's width, which is orthogonal to correct.
- **Render it as a WALL, not a shaded half-plane**: base line, top edge, two
  uprights and cross members, height tapering from the near end to the far one
  because that is what a vertical plane does under this camera. Clip the base to
  the frame or the far end lands off screen and it reads as a slanted band.
- **Only the relevant hoop's wall is drawn**, behind a toggle.

## Saying where something is

the compass, so a correction can be given out loud without ambiguity:

- **north** = the diving board end, away from the camera, toward the top of frame
- **south** = toward the camera, bottom of frame
- east/west across the frame as usual

"The left zone is too far north" is unmistakable. "Too far up" is not, and cost a
round of guessing.

## 7. Drop zone: the full procedure, and every way it has been got wrong

**Run these four in order and paste the results in. Do not derive any of it by
reasoning about the geometry -- that has now failed four separate times, each
time producing something that looked plausible in a picture.**

```
py src/waterfit.py  --video IMG_XXXX.MOV     # waterline + wall, per hoop
py src/poolplane.py --video IMG_XXXX.MOV     # the water plane's vanishing points
py src/dropfit.py   --video IMG_XXXX.MOV     # along/into axes, per hoop
py src/dropcentre.py                         # where made balls actually land
py src/dropsweep.py                          # size, against judged shots
py src/rigcheck.py  --video IMG_XXXX.MOV     # LOOK at it
```

### The four wrong turns, so they are not repeated

**1. Sheared box extended straight down the image.** The zone is a patch of the
water SURFACE, so its sides run along the wall and away from it. "Away from the
wall" equals "down the image" only if the camera looks straight down. It doesn't,
so the zone ran along the pool's length: at the left hoop it walked onto the
concrete, at the right it slid down the wall.

**2. Picking "into" as whichever direction stays over water longest.** Walking
ALONG a wall also stays over water the whole way, so it can win. It did, and the
shootout's left zone drifted north. `dropfit.py` now decides "along" by matching
the water boundary's local tangent at that hoop and takes "into" as the other
family; only its SIGN comes from the water.

**3. Assuming the two hoops are on parallel walls.** They are not. The right
stands on a long wall, the left on the step-notch edge, so which vanishing point
means "along" swaps between them. This is also why the left hoop's waterline
slope never fitted stably (-0.43 one frame, -0.73 another, residuals past 500px):
its wall is near-vertical in the image, so columns walked downward cross it at a
glancing angle and the slope is meaningless. Measure per hoop, never per session.

**4. Centering the zone under the net.** The "ball falls straight down" model is
wrong and measurably so: centering recall from 64% to 36%. A made ball carries
through the net and out into the pool by about 1.2 rim widths, consistently, both
hoops, two sessions (`dropcentre.py`). The zone spans 0.30 to 1.60 rim widths out
from the waterline point for that reason.

**A fifth thing was tested and rejected**, which is worth recording so nobody
re-discovers it: made balls also land offset ALONG the wall (+0.88 rim widths at
the left hoop, -1.13 at the right). Shifting the zone by those medians makes
recall much WORSE, 67% to 29%, because the spread straddles zero and the median
does not represent it. Along stays centered at 0.

### The decision must test the shape that gets drawn

`dropped()` tested `box()` -- an axis-aligned rectangle -- while the quad was
only ever what got DRAWN. Review frames showed one region while the veto fired on
another, for weeks, because a rectangle under a rim looks plausible enough in a
picture. It is a real point-in-polygon test now. **If you change the zone's
shape, check that the test changed with it.**

### Size

1.10 x 1.30 rim widths, a small square, per review. Measured on 66 judged makes and
125 judged misses: the widest setting tried (1.80 x 1.60) catches 73% of makes
against this one's 67%, but what the rule trades on is the veto's precision
("never appeared here, so it missed"), and that only moves 83.3% to 82.0%. The
square costs about 1.3 points. Stated rather than buried.

Note the old "34 of 37 makes" quoted for years described the RECTANGLE, over a
smaller sample, before the shape was corrected. Recall was never that high.

## 7a. Drop zone axes (`drop`, from `src/dropfit.py`)

**Run `src/dropfit.py` and paste `drop=` into the rig. Do not derive the zone
from the waterline slope alone.**

The zone is a patch of the water SURFACE under the net, so its sides run ALONG
the wall and AWAY from the wall. The first version sheared to the wall and then
extended straight down the image, which is the same thing only if the camera
looks straight down. It doesn't, so the zone ran along the pool's length instead
of out from the wall: at the left hoop it walked onto the concrete, at the right
it leaned down the wall instead of reaching into the water.

Under perspective neither direction is constant across the image, because
parallel world lines converge. So `src/poolplane.py` measures the two vanishing
points of the water plane from the pool's own boundary, and at any point "along
the wall" is toward one and "away from the wall" is toward the other. That is
exact, not an approximation.

**The trap that makes this per-hoop rather than per-session: THE TWO HOOPS ARE
NOT ON PARALLEL WALLS.** The right hoop stands on a long wall; the left stands on
the step-notch edge, which belongs to the perpendicular family. So which
vanishing point means "along" swaps between them. `dropfit.py` does not decide
that by hand: of the four candidate directions it picks "into" as the one that
stays over water longest, which is self-verifying and cannot come out backwards.

This is also the explanation for something that looked like noise for days: the
LEFT hoop's waterline slope never fitted stably (-0.43 one frame, -0.73 another,
residuals over 500px). Its wall is nearly vertical in the image, so walking
columns downward to find the water crosses the boundary at a glancing angle and
the fitted slope is meaningless. The plane-based axes give it -11.5, i.e. very
nearly vertical, which is correct and which no slope fit was ever going to say.

Sizes are separate and were earned: half-width 0.9 rim widths, depth 0.3 to 1.6.
34 of 37 makes land inside; at half-width 0.5 only 16 do. Fixing the SHAPE is not
an invitation to re-tune those.

## 7b. Drop zone sizes (`dropzone.py`)

The column of water under the net. This is the strongest make/miss signal on the
project: 34 of 37 makes put the ball in it, so a shot that never appears there is
a miss, right 51 times out of 54.

Half-width 0.9 rim widths, from the waterline down about 1.3 rim widths, sheared
to the wall slope. Do not tighten it: at half-width 0.5 it catches 16 of 37 makes
instead of 34 and barely improves precision, which is the wrong trade for a rule
whose whole value is recall.

---

## What to draw on every review frame

All of it, always, in these colors, because a rule is only as trustworthy as the
region it reads and drawing it is how that gets checked:

| Mark | Color | Notes |
|---|---|---|
| Rim ellipse | yellow | tilted |
| Net box | yellow | under the rim |
| Drop zone | dark gray | parallelogram, in the water, following the wall |
| Ball flight | amber | only the samples the fit was made from |
| Release marker | amber cross | snapped to the real ball when visible |
| Shooter box | dark green | |
| Everyone else | black | never red; red reads as a verdict |
| Wrists | yellow up / blue down | colored by what they SAY, not by whose box |
| Head line | box color | the line "hands up" is measured against |
| Facing arrow | black | left/right, with strength as a percentage |
| Three-point wall | pink | only the relevant hoop, toggled |

Label markers with WORDS, never by color alone. A previous version keyed them by
color, got BGR backwards, and a whole round of review was spent on a picture
whose own legend was wrong.

## Facing, specifically

Use LEFT/RIGHT from the wrists against the head, not front/back.

Front-versus-back needs a face the camera often cannot see, and when it cannot,
the call collapses: printing the vectors for one frame showed six of seven people
reading as facing the camera. Left-versus-right needs no face and is the axis
that matters, because the hoops sit at the two ends of the pool. Measured 89%
against 81%.

Three states, not two, and the third is load-bearing: wrists both left, both
right, or one each side. One each side means square on, and per review that is
"probably aren't the shooter" — it must count AGAINST a candidate, not be treated
as neutral.
