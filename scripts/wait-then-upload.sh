#!/usr/bin/env bash
# Wait for make-review-copies.sh to finish, then upload.
#
# Waits on two conditions, not one. A file appearing in out/film only means
# ffmpeg has opened it, so uploading on file count alone would push a partial
# copy of whichever recording is still encoding, and a truncated mp4 plays right
# up to the point it stops.
set -u
cd "$(dirname "$0")/.."

want=$(ls footage/*.MOV 2>/dev/null | wc -l)
[ "$want" -eq 0 ] && { echo "no source footage"; exit 1; }

while :; do
  have=$(ls out/film/*.mp4 2>/dev/null | wc -l)
  running=$(ps -W 2>/dev/null | grep -ci ffmpeg)
  if [ "$have" -ge "$want" ] && [ "$running" -eq 0 ]; then
    break
  fi
  sleep 30
done

echo "transcode finished: $have file(s), $(du -sm out/film | cut -f1) MB"
exec bash scripts/upload-film.sh
