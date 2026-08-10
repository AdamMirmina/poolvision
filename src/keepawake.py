"""Hold the machine awake while a long scan runs, and let go when it ends.

Detection runs about an hour per minute of footage on this laptop, so a scan is
routinely a 13-hour job. the machine sleeps after 3 hours idle on AC and 10
minutes on battery, and Windows decides that on INPUT idleness, not CPU load --
so a pegged CPU does nothing to stop it. A scan left overnight stalls a few hours
in, having made no progress, with nothing to show it happened.

This asks Windows to keep the system running, the same request a video player
makes while playing. Deliberately NOT a power-plan change: the request lives only
as long as this process, so a crash, a kill, or a reboot restores normal
behavior on its own. There is nothing to remember to undo.

The display is left alone -- the screen can still blank, only the machine stays
up.

    python src/keepawake.py --while-running rimwatch
"""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time

ES_CONTINUOUS = 0x80000000        # this state persists until changed again
ES_SYSTEM_REQUIRED = 0x00000001   # keep the system up (display may still sleep)


def matching_processes(needle: str) -> int:
    """How many python processes have `needle` in their command line."""
    ps = (
        "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*{needle}*' }} | Measure-Object).Count"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60)
        return int((out.stdout or "0").strip() or 0)
    except Exception:
        # Never let a failed probe be read as "the work is gone" -- that would
        # release the lock mid-scan, which is the one thing this must not do.
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--while-running", default="rimwatch",
                    help="release once no python process matches this")
    ap.add_argument("--max-hours", type=float, default=24.0)
    args = ap.parse_args()

    if not sys.platform.startswith("win"):
        print("windows only")
        return 1

    ok = ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if not ok:
        print("could not take the wake lock")
        return 1
    print(f"holding the machine awake while any python process matches "
          f"'{args.while_running}'")

    deadline = time.time() + args.max_hours * 3600
    misses = 0
    try:
        while time.time() < deadline:
            # Two consecutive empty probes before releasing, so a single hiccup
            # between two chained scans does not drop the lock in the gap.
            misses = misses + 1 if matching_processes(args.while_running) == 0 else 0
            if misses >= 2:
                print("work finished")
                break
            time.sleep(60)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("released; normal sleep behavior restored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
