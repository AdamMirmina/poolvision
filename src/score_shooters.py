"""Score attribution against the shooters review NAMED, not against a rotation.

score_rotation.py infers the answer from the order people shot in. That was the
best available when nobody had labeled anything, and it has a hole it states
itself: if the pipeline learned the rotation it would score perfectly without
ever looking at a cap.

the answer key closes that. Review wrote a name on every one of the 20 shots
while watching a tape carrying no model output, so the correct answer exists
independently of both the model and the order.

The rotation is still reported, as the BASELINE to beat. In this footage the
shooters rotate strictly, so "follow the order" scores 100% without reading
anything. Any attribution result that does not clear that number is measuring
the drill, not the model.

    python src/score_shooters.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# review blue, one player pink, another player white -- the same mapping score_rotation.py uses.
CAP = {"hand": "blue", "one player": "pink", "another player": "white"}


def hms(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="labels/answerkey_IMG_2528.json")
    ap.add_argument("--shooters", default="out/shooter_IMG_2528.json")
    ap.add_argument("--tol", type=float, default=4.0)
    args = ap.parse_args()

    key = json.loads((ROOT / args.key).read_text(encoding="utf-8"))
    shots = key["shots"]
    rows = json.loads((ROOT / args.shooters).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("shots") or rows.get("rows") or []

    def when(r):
        for k in ("t", "clock_t", "descent_t"):
            if isinstance(r.get(k), (int, float)):
                return float(r[k])
        c = r.get("clock")
        if isinstance(c, str) and ":" in c:
            m, sec = c.split(":")[:2]
            return int(m) * 60 + int(sec)
        return None

    # Match each of the shots to the attribution row nearest in time. A row
    # with no answer still counts against the total -- "no answer" is a wrong
    # answer from his side of the screen, and scoring only the confident ones is
    # how every earlier number on this project flattered itself.
    used, pairs = set(), []
    for s in shots:
        best, bd = None, 1e9
        for i, r in enumerate(rows):
            t = when(r)
            if t is None or i in used:
                continue
            d = abs(t - s["t"])
            if d < bd:
                best, bd = i, d
        if best is not None and bd <= args.tol:
            used.add(best)
            pairs.append((s, rows[best]))
        else:
            pairs.append((s, None))

    named = right = color_right = no_answer = 0
    print(f"{'shot':>7}  {'truth':<6} {'picked':<10} {'cap read':<9} verdict")
    for s, r in pairs:
        truth = s["shooter"].lower()
        if r is None:
            print(f"{s['at']:>7}  {truth:<6} {'(not called)':<10} {'':<9} no call")
            continue
        pick = str(r.get("shooter") or r.get("pick") or r.get("person") or "").lower()
        cap = str(r.get("cap") or r.get("color") or r.get("color") or "").lower()
        if not pick:
            no_answer += 1
            print(f"{s['at']:>7}  {truth:<6} {'no answer':<10} {cap:<9} -")
            continue
        named += 1
        ok = pick == truth
        right += ok
        cok = cap == CAP.get(truth, "")
        color_right += cok
        print(f"{s['at']:>7}  {truth:<6} {pick:<10} {cap:<9} "
              f"{'RIGHT' if ok else 'wrong'}{'' if cok else '  (cap wrong)'}")

    total = len(shots)
    print()
    print(f"attributed        {named}/{total} shots got an answer "
          f"({no_answer} no-answer, {total - named - no_answer} never called)")
    if named:
        print(f"correct person    {right}/{named} = {100 * right / named:.0f}% of answered")
        print(f"correct cap       {color_right}/{named} = "
              f"{100 * color_right / named:.0f}% of answered")
    print(f"correct overall   {right}/{total} = {100 * right / total:.0f}% of every shot")
    print()
    print("BASELINE TO BEAT: the shooters rotate strictly here, so simply "
          "following the\norder scores 100% without reading a single cap. A "
          "number below that is not\nevidence the cap reader works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
