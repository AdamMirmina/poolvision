#!/usr/bin/env bash
# The 2026-08-01 session, processed end to end.
#
# IMG_2528 is the shootout and it is the first footage this project has ever had
# where the correct answer is known without anyone labeling anything: review,
# one player and another player shot in strict rotation (blue, pink, white) throughout, lining up
# facing the camera before each position change. Every position was two rounds
# with defenders' hands down and two with hands up, and the last two rounds were
# dunks, left hoop then right.
#
# IMG_2529 is a game. No rotation, so no free ground truth -- it is the realism
# check rather than the measurement.
#
# WHERE THE FRAMES GO. 4K60 doubles the work and a uniform pass over both videos
# would run past morning. Shot detection ran at 30fps before and found 27 of 27,
# so it gets every SECOND frame here and loses nothing. The full 60 is spent
# only in the ~2s before each detected shot, which is where the release lives
# and the entire reason for shooting at 60 in the first place.
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*"; }

log "STEP 1 rim positions"
# The camera moves between sessions. Pointing the previous session's boxes at
# open water burns the whole night and looks exactly like a detector failure,
# which is why hoops.py raises on an unknown recording rather than guessing.
py src/newrig.py --video IMG_2528.MOV || { log "ABORT: no rim boxes"; exit 1; }

for V in IMG_2528 IMG_2529; do
  log "STEP 2 $V: shot detection (every 2nd frame)"
  py src/rimwatch.py "footage/$V.MOV" --stride 2 --out "out/rimwatch_$V.json" 2>&1 | tail -3

  log "STEP 3 $V: cutting clips"
  py src/clips.py "footage/$V.MOV" "out/rimwatch_$V.json" --out "out/clips_$V" 2>&1 | grep -vE "^\[|^  shot_" | tail -2
done

log "STEP 4 shooter attribution on the shootout, at full 60fps near each release"
py src/shooter.py --video IMG_2528.MOV --shots 500 2>&1 | tail -4

log "STEP 5 scoring against the known rotation"
py src/score_rotation.py 2>&1 | tail -30

log "DONE"
