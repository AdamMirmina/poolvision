#!/usr/bin/env bash
# The full attribution pipeline for the real game footage.
#
#
#
# That is right that IMG_2481 was the wrong target. It is a shooting drill with
# several balls in the water, which is the hardest case and not the one the
# product is for. IMG_2482 is a real game: caps on, one live ball, the rest
# sitting at the side.
#
# Order matters -- wide clips are only cut for shots that attribution actually
# resolved, so attribution runs first.
set -u
cd "$(dirname "$0")/.."
V="${1:-IMG_2482}"

echo "[$(date +%H:%M)] attribution + shooter tracking for $V"
py src/attribute.py --video "$V.MOV" --shots 200 --save 2>&1 | grep -E "^shots|ball seen|release point|person found|cap color|^[0-9]+ shots" | tail -6

echo "[$(date +%H:%M)] wide clips"
py src/wideclips.py --video "$V" 2>&1 | tail -2

echo "[$(date +%H:%M)] in-air windows"
py src/airwindow.py 2>&1 | tail -3

echo "[$(date +%H:%M)] hoop boxes"
py src/rimoverlay.py 2>&1 | tail -2

echo "[$(date +%H:%M)] uploading"
node scripts/upload-wide.mjs --video "$V" 2>&1 | tail -1
node scripts/push-shooters.mjs --video "$V" 2>&1 | tail -1
node scripts/push-air.mjs 2>&1 | tail -1
node scripts/push-rims.mjs 2>&1 | tail -1
echo "[$(date +%H:%M)] DONE $V"
