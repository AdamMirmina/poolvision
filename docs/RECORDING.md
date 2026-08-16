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
- **A phone that fills up mid-recording writes a broken file, and it is
  recoverable.** IMG_2770 arrived with its index unreadable at the end and looked
  like 25 minutes of lost footage. The video data was entirely intact; only the
  index was bad. `ffmpeg -err_detect ignore_err -i broken.MOV -c copy -movflags
  +faststart fixed.MOV` rebuilt it and recovered all 92,213 frames. Do that
  BEFORE concluding anything is lost, and check the size on the phone against the
  size on disk first -- if they match, the transfer was fine and the file itself
  is the problem, so re-copying will never help.
- **Consent, every time.** This is a camera on a party full of real people
  running body detection. Everyone in the water agrees first, the footage stays
  local, and it is deleted after processing. Non-negotiable, and it is in
  `AGENTS.md` for the same reason.

## If someone does not want their face used

Three different asks, three different answers. Do not blur one and call the
others handled.

**"I don't want to be recorded."** Then they are not recorded. There is no
technical substitute for that. Pointing the camera at someone because the faces
come out blurry is not consent.

**"I'll play, but I don't want my face stored or shared."** `src/anon.py` blurs
every face before a frame leaves the laptop. This costs NOTHING: the pipeline has
never used a face. Shot detection uses the ball and the hoop geometry; attribution
uses cap color and wrist position from pose keypoints. Verified on a five-player
frame -- all five faces pixelated, all five caps still perfectly readable.

**Do NOT run it on everything by default.** It was written for a hypothetical
nobody had raised, then applied to every artifact, which taxed each one for
nothing -- blurring twelve short review clips cost over half an hour of CPU and
delayed handing them to a reviewer. Review asked the right question: "why are we blurring
faces."

Run it when someone has actually asked, and on stills opened in a CHAT, which go
to Anthropic as images and are the one genuinely third-party surface here. Clips
uploaded to poolean land on the own VPS at an unguessable URL and are watched
by review; that is not a reason to blur. The raw video never leaves the laptop
either way, because YOLO and ffmpeg run locally.

**"I don't want to be identified by the model."** Simplest of the three: don't
wear a cap. Attribution is keyed entirely on cap color, so no cap means no
identity, and shot detection carries on unaffected.

## What NOT to worry about

Do not try to make the play tidy for the camera, and do not stage shots. The
messy footage is the valuable footage: the model already handles a drill and
falls over on a ball rolling across the deck. Rebounds, dunks, people climbing
in and out, someone throwing the ball back in from the concrete — those are the
cases that break it, and they only exist in real play.
