#!/usr/bin/env bash
# Review-quality copies of the game footage.
#
# The source is 4K60 and ninety gigabytes across ten recordings. Nobody logging a
# box score needs that: 720p30 is comfortably enough to see a ball go through a
# hoop, and it takes the whole season to about a gigabyte, which is inside every
# free storage tier and cheap to move.
#
# Measured, not assumed: 3.7 MB per minute at crf 26.
#
# KEYFRAME SPACING IS THE SETTING THAT MATTERS HERE, and the default is wrong for
# this job. x264 defaults to a keyframe every 250 frames, which at 30fps is 8.3
# seconds, and a player seeking to any point has to fetch and decode from the
# keyframe before it. Reviewing film is nothing but seeking, so that default made
# every scrub feel broken. -g 30 puts one every second. -keyint_min 30 and
# -sc_threshold 0 stop x264 placing them anywhere else, so the spacing is
# actually regular rather than merely capped.
#
# It costs size, because a keyframe is a whole picture. That is the trade being
# made deliberately: a scrub that responds immediately is worth more here than a
# smaller file, and the free tier has room.
set -u
FF=$(py -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
OUT=${1:-out/film}
mkdir -p "$OUT"
for src in footage/*.MOV; do
  base=$(basename "$src" .MOV)
  out="$OUT/${base}.mp4"
  [ -s "$out" ] && { echo "skip $base (already done)"; continue; }
  echo "encoding $base ..."
  # -movflags +faststart so it can be streamed and scrubbed without downloading
  # the whole file first, which is the entire point of hosting it.
  "$FF" -y -loglevel error -i "$src" \
    -vf "scale=1280:-2,fps=30" -c:v libx264 -crf 26 -preset veryfast \
    -g 30 -keyint_min 30 -sc_threshold 0 \
    -pix_fmt yuv420p -movflags +faststart -an "$out" \
    && echo "  $base -> $(du -m "$out" | cut -f1) MB" \
    || echo "  $base FAILED"
done
echo "all done"
