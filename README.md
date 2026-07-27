# poolvision

Automatic scoring and per-player shot stats for pool basketball, from one fixed
camera. Results feed into [poolean](https://github.com/AdamMirmina/poolean)'s
existing PocketBase, so the app stays the place everyone actually looks.

Separate from the poolean repo on purpose: poolean's repo is uploaded to EAS on
every iOS build, and a Python project with torch, model weights and thousands of
training frames has no business in there.

## Where this actually is

**Nothing is proven yet.** No footage exists. Every estimate below is provisional
until `probe.py` has run on a real party. See `docs/STATE.md`.

## The plan

1. **Record one party.** One iPhone, mounted high, locked exposure and white
   balance. Setup details in `docs/STATE.md`.
2. **Run the probe.** `python src/probe.py footage/party.mov`. It measures
   whether people are detected, whether the ball is detected, whether tracks
   survive, and how the cap colors separate. Those four numbers decide the
   shape of everything after this.
3. Everything else depends on step 2, so it isn't written yet.

## What it's meant to do, eventually

- Detect made baskets from ball trajectory through a hand-defined hoop region
- Identify who shot, using colored swim caps assigned at check-in
- Process after the party (more accurate than live, since identity can
  propagate backward and forward through a track)
- Push results to poolean, with a review screen to fix what it gets wrong

## What it is not meant to do

Assists, rebounds, and defensive stats. The published research doesn't support
doing those reliably in these conditions, so they're out of scope rather than
quietly attempted and wrong.

## Running the probe

```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/probe.py footage/your-party.mov
```

Then read `out/report.md` **and look at `out/frames/`**. The annotated frames
catch things the numbers can't, like the camera being mounted too low.
