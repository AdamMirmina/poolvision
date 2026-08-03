#!/usr/bin/env bash
# The 32 shootout shots reached review with a frame but no footage, so most
# of the queue was a picture with nothing to check it against.
#
# The scans have to be rebuilt anyway: the scan window now runs past the descent,
# so every cached scan predates the fix and would rebuild the same truncated
# flight. Clearing them first is the point, not a side effect.
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*"; }

log "1/4 clearing stale scans for IMG_2528 (cached before the window fix)"
rm -f out/scans/IMG_2528_*.json
log "    $(ls out/scans/IMG_2528_*.json 2>/dev/null | wc -l) left"

log "2/4 attribution, which rebuilds the scans on the corrected window"
py -u src/shooter.py --video IMG_2528.MOV --shots 500 2>&1 | tail -4

log "3/4 wide clips from those shots"
py -u src/wideclips.py --video IMG_2528 2>&1 | tail -4

log "4/4 attaching them to the records"
node scripts/upload-wide.mjs --video IMG_2528 2>&1 | tail -4

log "DONE"
