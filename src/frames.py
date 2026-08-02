"""One frame by timestamp, without OpenCV's seek.

CAP_PROP_POS_FRAMES on these recordings is pathological. A single seek into the
15GB 4K60 file ran past ten minutes without returning, and at one seek per shot
that is not a slow step, it is a step that never finishes. ffmpeg seeks on the
container index instead and comes back in a few seconds.

Its own module because three different scripts need it and each of them
independently reached for the OpenCV version first.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def frame_at(video_path, t: float):
    """The frame at `t` seconds as a BGR array, or None."""
    import cv2
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "f.png"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
                        "-i", str(video_path), "-frames:v", "1", str(out)],
                       capture_output=True)
        return cv2.imread(str(out)) if out.exists() else None
