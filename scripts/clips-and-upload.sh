#!/usr/bin/env bash
# Cut clips for the remaining videos and push them into the review queue.
#
# Each video gets its own output directory, otherwise the second run overwrites
# the first's clips while the index.json still names them -- the upload would
# then attach the wrong video to every record, silently.
#
# Ordered by review value: the game with caps first, since that's the closest
# thing to the real target and the case where two balls in play actually bites.
#
# Needs PB_EM / PB_PW in the environment.
set -u
cd "$(dirname "$0")/.."

for v in IMG_2482 IMG_2480 IMG_2483; do
  rim="out/rimwatch_${v}.json"
  dir="out/clips_${v}"
  if [ ! -f "$rim" ]; then
    echo "[$(date +%H:%M)] SKIP $v -- no detection pass at $rim"
    continue
  fi
  if [ -f "$dir/index.json" ]; then
    echo "[$(date +%H:%M)] clips already cut for $v, reusing"
  else
    echo "[$(date +%H:%M)] CUT $v"
    py src/clips.py "footage/$v.MOV" "$rim" --out "$dir" 2>&1 | grep -vE "^\[|^  shot_" | tail -3
  fi

  echo "[$(date +%H:%M)] UPLOAD $v"
  node scripts/upload-clips.mjs --video "$v.MOV" --clips "$dir" --labels /dev/null 2>&1 | tail -2
done

echo "[$(date +%H:%M)] ALL DONE"
