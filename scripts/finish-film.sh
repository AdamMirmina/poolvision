#!/usr/bin/env bash
# Wait for the upload to finish, then register what landed.
#
# Chained as a separate step rather than folded into wait-then-upload.sh,
# because that script is already running and editing a file bash is part way
# through reading is its own kind of bug.
#
# Waits on the upload's own output rather than on a timer: the marker is the
# line upload-film.sh prints after its last put.
set -u
cd "$(dirname "$0")/.."
LOG="${1:-}"
[ -z "$LOG" ] && { echo "usage: finish-film.sh <path-to-upload-job-output>"; exit 1; }

for _ in $(seq 1 720); do
  if grep -q "Total in bucket" "$LOG" 2>/dev/null; then
    break
  fi
  sleep 15
done

if ! grep -q "Total in bucket" "$LOG" 2>/dev/null; then
  echo "upload has not finished after two hours; not registering anything"
  exit 1
fi

# A failed put prints FAILED rather than a URL. Registering a recording whose
# object is not actually in the bucket would give a reviewer a player that 404s,
# which looks like the film route being broken rather than one upload missing.
if grep -q "FAILED" "$LOG"; then
  echo "at least one upload failed, so nothing is being registered:"
  grep "FAILED" "$LOG"
  exit 1
fi

echo "uploads finished. Registering:"
node scripts/register-film.mjs --write
