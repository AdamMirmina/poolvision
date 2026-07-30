#!/usr/bin/env bash
# Detection pass over the 2026-07-29 session, sequenced by value.
#
# Sequential on purpose: two YOLO passes on one CPU take longer together than
# either does alone, and finishing the most useful video first beats finishing
# all four at once. If the night gets cut short, the order decides what's ready.
#
#   IMG_2481  shooting WITH caps   -- dense shots, first real cap footage
#   IMG_2482  game WITH caps       -- the actual target, and the two-ball case
#   IMG_2480  shooting, no caps    -- shot detection without identity noise
#   IMG_2483  game, no caps        -- realistic play, no cap signal
#
# Skips anything already done, so it can be re-run after an interruption.
set -u
cd "$(dirname "$0")/.."

for v in IMG_2481 IMG_2482 IMG_2480 IMG_2483; do
  out="out/rimwatch_${v}.json"
  if [ -f "$out" ]; then
    echo "[$(date +%H:%M)] SKIP $v (already done)"
    continue
  fi
  # NOTE: no pgrep guard here. Git Bash's pgrep does not see native Windows
  # processes, so a check like `pgrep -f rimwatch.py` silently returns nothing
  # and happily starts a second pass on a video already being processed -- which
  # is exactly what happened the first time this ran. The output-file check above
  # is the real guard; don't add a process check that can't see the processes.
  echo "[$(date +%H:%M)] START $v"
  py src/rimwatch.py "footage/$v.MOV" --from 0 --to 99999 --out "$out" 2>&1 \
    | grep -vE "^\[[0-9]" | tail -4
  echo "[$(date +%H:%M)] DONE $v"
done

echo "[$(date +%H:%M)] ALL DONE"
for f in out/rimwatch_IMG_24*.json; do
  [ -f "$f" ] && py -c "
import json,sys
d=json.load(open(sys.argv[1]))
n={k:len(v) for k,v in d['hits'].items()}
print(f\"{sys.argv[1].split('/')[-1]:28s} left {n.get('left',0):5d}  right {n.get('right',0):5d}\")
" "$f"
done
