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

# Rim boxes for this session are already measured, drawn and checked by eye --
# see hoops.py _S0801_RIMS. Verified here rather than re-derived, because a
# wrong box costs the night and reads as a detector failure in the morning.
py -c "import sys; sys.path.insert(0,'src'); import hoops; hoops.rig_for('IMG_2528.MOV')"   || { log "ABORT: no rim boxes for this session"; exit 1; }

log "STEP 1 shot detection on the shootout, every 2nd frame"
# Every 2nd frame of 60fps is the 30fps that found 27 of 27 previously, so this
# loses nothing. The full 60 is spent in the release windows instead, which is
# the only place the extra frames actually buy anything.
py src/rimwatch.py footage/IMG_2528.MOV --from 0 --to 99999 --step 2   --out out/rimwatch_IMG_2528.json 2>&1 | tail -3

log "STEP 2 cutting clips for the shootout"
py src/clips.py footage/IMG_2528.MOV out/rimwatch_IMG_2528.json   --out out/clips_IMG_2528 2>&1 | grep -vE "^\[|^  shot_" | tail -2

log "STEP 3 shooter attribution, full 60fps around each release"
py src/export_shots.py --video IMG_2528 2>&1 | tail -2
py src/shooter.py --video IMG_2528.MOV --shots 500 2>&1 | tail -6

log "STEP 4 scoring against the rotation review found"
py src/score_rotation.py 2>&1 | tail -32

# The game only if the shootout finished with time to spare. It carries no
# rotation, so it is the realism check rather than the measurement, and the
# measurement is what the night is for.
if py src/verify_footage.py IMG_2529.MOV >/dev/null 2>&1; then
  log "STEP 5 the game: shot detection"
  py src/rimwatch.py footage/IMG_2529.MOV --from 0 --to 99999 --step 2     --out out/rimwatch_IMG_2529.json 2>&1 | tail -3
  py src/clips.py footage/IMG_2529.MOV out/rimwatch_IMG_2529.json     --out out/clips_IMG_2529 2>&1 | grep -vE "^\[|^  shot_" | tail -2
else
  log "the game did not verify; skipping it"
fi

log "DONE"
