#!/usr/bin/env bash
# Everything that can run without review, in the order that gets the most in front
# of him by morning.
#
# Quick work first, training last. The ball fine-tune will take the CPU for
# hours, so anything that produces something reviewable runs before it rather
# than queuing behind it.
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*"; }

log "1/7 make/miss scores for the shootout, with both vetoes"
py -u src/predict_make.py --video IMG_2528.MOV 2>&1 | tail -3

log "2/7 attribution on the shootout, all of today's rules"
py -u src/shooter.py --video IMG_2528.MOV --shots 500 2>&1 | tail -3

log "3/7 the rotation score, which is the measurement the shootout exists for"
py -u src/score_rotation.py 2>&1 | tail -20

log "4/7 review frames for the shootout, so there is more than one video to judge"
py -u src/posediag.py --video IMG_2528.MOV --limit 60 2>&1 | tail -3

log "5/7 the game: clips, with the pass filter and bounce merge"
py -u src/clips.py footage/IMG_2529.MOV out/rimwatch_IMG_2529.json \
   --out out/clips_IMG_2529 2>&1 | tail -3

log "6/7 figures for the writeup"
py -u src/figures.py 2>&1 | tail -4

log "7/7 fine-tuning the ball detector -- this one takes the night"
py -u src/balltrain.py 2>&1 | tail -12

log "DONE"
