"""Credentials from .env, so they never have to be typed, pasted into a chat, or
left in shell history. Shared helper.

Environment variables win, so containers and CI keep working unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def load_env(start: Path | None = None) -> dict:
    """Parse the nearest .env walking up from `start`."""
    d = (start or Path.cwd()).resolve()
    for cand in [d, *d.parents]:
        f = cand / ".env"
        if f.exists():
            out = {}
            for raw in f.read_text(encoding="utf-8", errors="replace").splitlines():
                t = raw.strip()
                if not t or t.startswith("#") or "=" not in t:
                    continue
                k, v = t.split("=", 1)
                v = v.strip()
                if len(v) > 1 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                out[k.strip()] = v
            return out
    return {}


def need(*names: str) -> dict:
    """Required values, with a useful message rather than a KeyError."""
    e = load_env()
    out, missing = {}, []
    for n in names:
        v = os.environ.get(n) or e.get(n)
        if not v:
            missing.append(n)
        out[n] = v
    if missing:
        print("Missing credentials: " + ", ".join(missing), file=sys.stderr)
        print("Copy .env.example to .env and fill it in, or set them in the "
              "environment. See docs/CREDENTIALS.md.", file=sys.stderr)
        raise SystemExit(1)
    return out
