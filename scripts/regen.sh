#!/usr/bin/env bash
# Everything poolean shows is stale, and none of today's work reaches review until
# this runs.
#
# The frames on the site were rendered before the drop zones were corrected, the
# traces were written before hands-up was split from release, and every cached
# scan predates the window fix, so re-rendering without re-scanning would just
# redraw the same truncated flights. It all has to come from the bottom up.
#
#   scans -> attribution -> review frames -> wide clips -> upload
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*"; }

VIDEOS="${1:-IMG_2482 IMG_2528}"

for V in $VIDEOS; do
  log "=== $V ==="

  log "  1/5 clearing scans cached before the window fix"
  rm -f "out/scans/${V}_"*.json

  log "  2/5 attribution (rebuilds scans on the corrected window)"
  py -u src/shooter.py --video "${V}.MOV" --shots 500 2>&1 | tail -3

  log "  3/5 review frames: measured drop zones, hands/release split"
  py -u src/posediag.py --video "${V}.MOV" --limit 60 2>&1 | tail -3

  log "  4/5 wide clips, with rim/net/zone burned in for the whole duration"
  py -u src/wideclips.py --video "$V" 2>&1 | tail -3
done

log "5/5 uploading frames, manifest and clips"
node scripts/upload-diag.mjs 2>&1 | tail -3
for V in $VIDEOS; do
  node scripts/upload-wide.mjs --video "$V" 2>&1 | tail -2
done

log "DONE -- redeploy poolean/web afterwards so the manifest is served"
