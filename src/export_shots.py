"""Write the detected shots where shooter.py expects to read them.

shooter.py reads labels/allshots.json, which until now came out of PocketBase.
Tonight's footage has no records yet and does not need any: the shots come
straight from the detection pass.
"""
import argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--video", default="IMG_2528")
a = ap.parse_args()
idx = json.loads((ROOT / f"out/clips_{a.video}/index.json").read_text(encoding="utf-8"))
prev = []
p = ROOT / "labels/allshots.json"
if p.exists():
    prev = [r for r in json.loads(p.read_text(encoding="utf-8")) if r["video"] != a.video + ".MOV"]
rows = [{"video": a.video + ".MOV", "n": c["n"], "t": c["t"], "tEnd": c["t_end"],
         "clock": c["clock"], "hoop": c["hoop"], "label": "",
         "clipStart": c["clip_start"], "durationS": round(c["frames"] / 60.0, 2)}
        for c in idx]
p.write_text(json.dumps(prev + rows, indent=1), encoding="utf-8")
print(f"{len(rows)} shots from {a.video} written to labels/allshots.json")
