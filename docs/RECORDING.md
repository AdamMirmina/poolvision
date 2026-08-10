# Recording checklist — the last sessions of the season

Code can be rewritten in January. Footage cannot. Anything not recorded in the
remaining sessions is unavailable until next summer, and that asymmetry is the
whole reason this file exists.

Storage is not a constraint: 567 GB free against 12-30 GB per session (checked
2026-08-10). **Record everything, the whole time, and never stop recording to
save space.**

## The one rule that matters most

**Do not move the camera during a session.** Every geometric constant in this
project — both rim boxes, both drop zones, the detector windows, the water plane
— is measured per camera position. Move it and all of them are wrong, silently,
in a way that reads as the model getting worse. See `RIG-SETUP.md`.

If the camera has to move, that is fine, but treat it as a **new rig**: it needs
its own measurement pass before any of that footage can be scored.

## What we already have too much of

Three players, bright overcast daylight, one of two camera positions, orderly
play. Twenty-seven marked shots come from a single afternoon. More of that adds
volume, not evidence — review spotted this himself: *"that whole video is the
structured rotating shootout so it would be more of a similar type of data."*

## What we have NONE of, in priority order

1. **Evening, dusk, or under lights.** This is the biggest hole by far. Every
   frame on disk is daytime. Low light changes contrast, color and motion blur
   all at once, which hits the ball detector and the cap reader together. If
   there is any chance of an evening session, that footage is worth more than
   everything else on this list.
2. **Different weather.** Overcast versus direct sun changes water glare
   completely, and glare is what the ball has to be picked out against.
3. **More people in the water.** Five players already look different from three.
   Six or seven, with bystanders, is the crowding case.
4. **Different cap colors**, especially two that are close to each other, and
   anyone with no cap at all. The identity work assumes caps are distinguishable
   and has never been tested when they are not.
5. **A wet camera lens, or a splash on it.** Currently unhandled and certain to
   happen eventually.

## Two things to do every session, both about 30 seconds

**A calibration line-up, at the start and end.** Everyone stands in a row facing
the camera, in the water, for ten seconds. This gives a per-party color
reference under that day's actual light, which is the thing a fixed color table
cannot replace. Two of these turned up by accident in IMG_2528 and are the best
identity data on the project. Do it on purpose.

**Say the date and who is playing, out loud, at the start.** The audio is
discarded during processing but the recording keeps it, and it removes any doubt
later about which session a file is.

## Practical

- **Restart the recording promptly if it stops.** The longest file on disk is 35
  minutes, which is probably a thermal or storage cutoff on the phone. A gap
  costs whatever happened during it.
- **Same resolution and frame rate as before** (4K, 60fps on the 2528/2529
  sessions). Changing it is not fatal but every pixel threshold in the pipeline
  is in pixels, so it is one more thing to re-verify.
- **Copy the files off the phone the same day.** A full phone at the next session
  is a session lost.
- **Consent, every time.** This is a camera on a party full of real people
  running body detection. Everyone in the water agrees first, the footage stays
  local, and it is deleted after processing. Non-negotiable, and it is in
  `AGENTS.md` for the same reason.

## What NOT to worry about

Do not try to make the play tidy for the camera, and do not stage shots. The
messy footage is the valuable footage: the model already handles a drill and
falls over on a ball rolling across the deck. Rebounds, dunks, people climbing
in and out, someone throwing the ball back in from the concrete — those are the
cases that break it, and they only exist in real play.
