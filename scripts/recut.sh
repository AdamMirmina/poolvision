#!/usr/bin/env bash
# Re-cut the un-labeled videos with the capped match radius, and replace what's
# in the review queue. 2481 is deliberately NOT here: review labeled 127 of its
# 128 clips, and regenerating would strand ~39 of those answers on boundaries
# that no longer exist. Duplicates review has already judged are cheaper than work
# review has to redo.
set -u
cd "$(dirname "$0")/.."
for v in IMG_2482 IMG_2483 IMG_2480; do
  rim="out/rimwatch_${v}.json"; dir="out/recut_${v}"
  [ -f "$rim" ] || { echo "[$(date +%H:%M)] SKIP $v (no detections)"; continue; }
  echo "[$(date +%H:%M)] CUT $v"
  rm -rf "$dir"
  py src/clips.py "footage/$v.MOV" "$rim" --out "$dir" 2>&1 | grep -vE "^\[|^  shot_" | tail -2
  echo "[$(date +%H:%M)] REPLACE $v"
  node scripts/migrate-clips.mjs --video "$v.MOV" --clips "$dir" --rim "$rim" --apply 2>&1 | tail -3
done
echo "[$(date +%H:%M)] RECUT DONE"
