#!/usr/bin/env bash
# Wait for the attribution re-run, then push everything that depends on it.
#
# The re-run exists to fix three things review reported at once: the person crop
# now spans the same window as the wide clip so the two can be played in
# lockstep, and the release time is finally saved so the in-air band can start
# when the ball leaves someone's hands rather than when it arrives near the rim.
set -u
cd "$(dirname "$0")/.."
while [ "$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*attribute*' } | Measure-Object).Count" | tr -d '\r\n ')" != "0" ]; do
  sleep 30
done
echo "[$(date +%H:%M)] attribution done"

echo "[$(date +%H:%M)] recomputing the in-air band from the real release"
py src/airwindow.py 2>&1 | tail -4

echo "[$(date +%H:%M)] uploading"
node scripts/push-shooters.mjs --video IMG_2481 2>&1 | tail -2
node scripts/push-air.mjs 2>&1 | tail -2
echo "[$(date +%H:%M)] DONE"
