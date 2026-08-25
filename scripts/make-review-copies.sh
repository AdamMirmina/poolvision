#!/usr/bin/env bash
# Review-quality copies of the game footage.
#
# The source is 4K60 and ninety gigabytes across ten games. Nobody logging a box
# score needs that: 720p30 is comfortably enough to see a ball go through a hoop,
# and it takes the whole season to well under a gigabyte, which is inside every
# free storage tier and cheap to move.
#
# Measured, not assumed: 3.7 MB per minute at crf 26.
set -u
FF=$(py -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p out/film
for src in footage/*.MOV; do
  base=$(basename "$src" .MOV)
  out="out/film/${base}.mp4"
  [ -s "$out" ] && { echo "skip $base (already done)"; continue; }
  echo "encoding $base ..."
  # -movflags +faststart so it can be streamed and scrubbed without downloading
  # the whole file first, which is the entire point of hosting it.
  "$FF" -y -loglevel error -i "$src" \
    -vf "scale=1280:-2,fps=30" -c:v libx264 -crf 26 -preset veryfast \
    -pix_fmt yuv420p -movflags +faststart -an "$out" \
    && echo "  $base -> $(du -m "$out" | cut -f1) MB" \
    || echo "  $base FAILED"
done
echo "all done"
