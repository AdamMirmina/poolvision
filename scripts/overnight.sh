#!/usr/bin/env bash
# The 2026-08-01 session, processed end to end.
#
# IMG_2528 is the shootout and it is the first footage this project has ever had
# where the correct answer is known without anyone labeling anything: review, one player
# and another player shot in strict rotation (blue, pink, white) throughout, lining up
# facing the camera before each position change. Every position was two rounds
# with defenders' hands down and two with hands up, and the last two rounds were
# dunks, left hoop then right.
#
# IMG_2529 is a game. No rotation, so no free ground truth -- it is the realism
# check rather than the measurement.
#
# BOTH VIDEOS ARE DETECTED FIRST, then everything else. Detection is the long
# pole and it is the step whose absence blocks every other step, so if the night
# runs short the morning still has shots for both recordings rather than one
# finished video and one untouched.
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*"; }

# Rim boxes for this session are measured, fitted and checked by eye -- see
# hoops.py _S0801_RIMS. Verified here rather than re-derived, because a wrong box
# costs the night and reads as a detector failure in the morning.
py -c "import sys; sys.path.insert(0,'src'); import hoops; hoops.rig_for('IMG_2528.MOV')" || { log "ABORT: no rim boxes for this session"; exit 1; }

for V in IMG_2528 IMG_2529; do
  if [ -s "out/rimwatch_$V.json" ]; then log "$V already detected, skipping"; continue; fi
  log "DETECT $V (every 2nd frame, both hoops in one stacked inference)"
  py src/rimwatch.py "footage/$V.MOV" --from 0 --to 99999 --step 2 \
     --out "out/rimwatch_$V.json" 2>&1 | tail -4
done

for V in IMG_2528 IMG_2529; do
  [ -s "out/rimwatch_$V.json" ] || { log "no detections for $V, skipping the rest of it"; continue; }
  log "CLIPS $V"
  py src/clips.py "footage/$V.MOV" "out/rimwatch_$V.json" --out "out/clips_$V" 2>&1 | tail -3
done

log "ATTRIBUTION on the shootout (full 60fps around each release)"
py src/export_shots.py --video IMG_2528 2>&1 | tail -2
py src/shooter.py --video IMG_2528.MOV --shots 500 2>&1 | tail -4

log "SCORING against the rotation review found"
py src/score_rotation.py 2>&1 | tail -32

log "DONE"
