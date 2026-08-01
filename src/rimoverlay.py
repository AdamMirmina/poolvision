"""Where the hoop sits inside each clip, as percentages.

review wants the hoop the model is scoring against marked on the clip. In the wide
view both hoops are visible, so "which one" is a real question and the answer is
currently only a word in the header.

Percentages rather than pixels, because the player scales the video to whatever
width the layout gives it and a pixel box would land in the wrong place at every
size except the one it was measured at.

Two clips, two coordinate spaces. The wide clip is the whole frame, so the rim's
share of the frame carries over unchanged. The rim clip is a crop AROUND that
hoop, so the rim's position has to be expressed relative to the crop.

    python src/rimoverlay.py
"""
import json, sys
sys.path.insert(0, "src")
import hoops, makemiss as M

PAD = 0.28      # widen the box a little so the ring frames the hoop, not hides it

out = []
for j in json.load(open("labels/judged.json")):
    if j["video"] not in M.RIMWATCH:
        continue
    try:
        rig = hoops.rig_for(j["video"])
    except KeyError:
        continue
    rx1, ry1, rx2, ry2 = rig.rims[j["hoop"]]
    w, h = rx2 - rx1, ry2 - ry1
    px, py = w * PAD, h * PAD
    bx1, by1, bx2, by2 = rx1 - px, ry1 - py, rx2 + px, ry2 + py

    wide = [bx1 / 3840 * 100, by1 / 2160 * 100, (bx2 - bx1) / 3840 * 100, (by2 - by1) / 2160 * 100]

    cx1, cy1, cx2, cy2 = rig.crops[j["hoop"]]
    cw, ch = cx2 - cx1, cy2 - cy1
    rim = [(bx1 - cx1) / cw * 100, (by1 - cy1) / ch * 100, (bx2 - bx1) / cw * 100, (by2 - by1) / ch * 100]

    out.append({"video": j["video"], "n": j["n"],
                "rimPctWide": ",".join(f"{v:.2f}" for v in wide),
                "rimPctRim": ",".join(f"{v:.2f}" for v in rim)})

print(f"{len(out)} clips given a hoop box")
seen = {}
for o in out:
    seen.setdefault(o["rimPctWide"], 0)
    seen[o["rimPctWide"]] += 1
print(f"{len(seen)} distinct hoop positions (one per hoop per camera rig, as expected)")
json.dump(out, open("out/rimoverlay.json", "w"), indent=1)
print("wrote out/rimoverlay.json")
